from typing import Callable, Dict, Union, Optional, Tuple, NamedTuple, Any
import logging
from pathlib import Path
import rich
import rich.syntax

import hydra
from omegaconf import OmegaConf, DictConfig
import torch
import lightning as L
from lightning.fabric.utilities.rank_zero import rank_zero_only
import os
import os.path as osp
from torch import nn
import colorlog
from datetime import datetime

# from forgetting_transformer.model.common import LMOutput
from transformers.modeling_outputs import ModelOutput
from forgetting_transformer.datamodule.common import DataInfo, Batch
from forgetting_transformer.checkpoint import Checkpointer
from forgetting_transformer.logger import (
    WandbLogger,
    JSONLinesLogger,
    NpzLogger,
    ConsoleLogger,
    MultiLogger,
)
from forgetting_transformer.utils import (
    check_divisible,
    safe_divide,
    is_power_of_two,
    ProgressBar,
    ThroughputMonitor,
    Timer,
    group_parameters,
)
from configs.config import Config
from collections import defaultdict, OrderedDict
import numpy as np
import time
from dataclasses import dataclass, field, asdict
from torch.distributed.fsdp import FullyShardedDataParallel
import torch.utils.flop_counter


@dataclass
class ModelInfo:
    total_params: int
    trainable_params: int
    embedding_params: int
    flops_per_token: int  # Note this depends how we train the model
    non_embedding_params: int = field(init=False)

    def __post_init__(self):
        self.non_embedding_params = self.total_params - self.embedding_params


def get_model_info(
    fabric: L.Fabric,
    model_fn: Callable[[], nn.Module],
    gradient_checkpointing: bool,
    data_info: DataInfo,
    print_model_path: Optional[Union[str, Path]],
):
    # Ideally we would use meta device. Unfortunatelly fla triton code assumes
    # the device is cuda.
    with torch.device("meta"):
        # with fabric.autocast():
            # Unfortunately some Triton code do not support meta device
        meta_model: nn.Module = model_fn()

    # TODO: to be implemented
    flops_per_token = 0
    total_params = sum(p.numel() for p in meta_model.parameters())
    trainable_params = sum(
        p.numel() for p in meta_model.parameters() if p.requires_grad
    )
    embedding_modules = [m for m in meta_model.modules() if isinstance(m, nn.Embedding)]
    assert (
        len(embedding_modules) == 1
    ), f"Model should have exactly one embedding module but got {len(embedding_modules)}"
    embedding_params = sum(p.numel() for p in embedding_modules[0].parameters())
    if print_model_path is not None:
        print_model_path = Path(print_model_path)
        assert print_model_path.name == "model.txt"
        with print_model_path.open("w") as f:
            print(meta_model, file=f)
    del meta_model
    return ModelInfo(
        total_params=total_params,
        trainable_params=trainable_params,
        embedding_params=embedding_params,
        flops_per_token=flops_per_token,
    )


def configure_root_logger(fabric: L.Fabric, log_dir: Union[Path, str], resume: bool):
    """
    Configure root logger.

    If `resume` is False, we will delete all previous log files in `log_dir`
    """
    # https://docs.python.org/3/howto/logging-cookbook.html
    # basicConfig only configures the root handler

    if hasattr(configure_root_logger, "called"):
        raise RuntimeError("`configure_root_logger` should only called once")
    configure_root_logger.called = True

    log_dir = Path(log_dir)
    # Delete previous log files is not resuming
    if not resume and fabric.is_global_zero:
        for log_file in log_dir.glob("*.log"):
            log_file.unlink()

    # Similar scheme as https://github.com/facebookresearch/hydra/blob/f4fe48442992defef7c2ddd0a6d014e3c371a073/plugins/hydra_colorlog/hydra_plugins/hydra_colorlog/conf/hydra/job_logging/colorlog.yaml#L9
    format = "[%(cyan)s%(asctime)s%(reset)s][%(purple)s%(module)s%(reset)s:%(purple)s%(lineno)d%(reset)s][%(log_color)s%(levelname)s%(reset)s] %(message)s"

    format = format.replace("[", "%(thin)s[%(reset)s")
    format = format.replace("]", "%(thin)s]%(reset)s")

    stream_handler = logging.StreamHandler()
    stream_formatter = colorlog.ColoredFormatter(
        fmt=format,
        datefmt="%Y-%m-%d %H:%M:%S",
        no_color=not stream_handler.stream.isatty(),
    )
    stream_handler.setFormatter(stream_formatter)

    filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.log")
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(filename=log_dir / filename)
    file_formatter = colorlog.ColoredFormatter(
        fmt=format, datefmt="%Y-%m-%d %H:%M:%S", no_color=True
    )
    file_handler.setFormatter(file_formatter)

    logging.basicConfig(level=logging.INFO, handlers=[stream_handler, file_handler])
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        handler.emit = rank_zero_only(handler.emit)

@torch.no_grad()
def train_eval(
    step: int,
    config: Config,
    loss_per_token: np.ndarray,
    logger: MultiLogger,
    train_data_info: DataInfo,
    extra_metrics: Dict,
):
    MIN_VAL_LENGTH = min(config.eval.min_val_length, train_data_info.seq_len)

    metrics = {
        **extra_metrics,
    }

    # Compute average loss and perplexity over different horizons
    length = train_data_info.seq_len
    while length >= MIN_VAL_LENGTH:
        loss_avg = loss_per_token[:length].mean(axis=0)
        perplexity = np.exp(loss_avg)
        metrics[f"train_eval/loss_avg_len_{length}"] = loss_avg
        metrics[f"train_eval/perplexity_len_{length}"] = perplexity
        length = length // 2

    # Regular scalar metrics
    logger.log_metrics(
        metrics=metrics,
        step=step,
        logger_names=("console", "jsonl", "wandb"),
        logger_kwargs={"jsonl": {"filename": "train_eval"}},
    )
    # Array metrics
    logger.log_metrics(
        metrics={**metrics, "train_eval/loss_per_token": loss_per_token},
        step=step,
        logger_names=("npz",),
        logger_kwargs={"npz": {"dirname": "train_eval"}},
    )

@torch.no_grad()
def validate(
    step: int,
    config: Config,
    fabric: L.Fabric,
    logger: MultiLogger,
    model: nn.Module,
    val_dataloader,
    val_data_info: DataInfo,
    extra_metrics: Dict,
):
    logging.info("Running validation...")
    MIN_VAL_LENGTH = min(config.eval.min_val_length, val_data_info.seq_len)
    start_time = time.perf_counter()
    model.eval()
    # Not sure why but litgpt has it
    # carry = None

    assert (
        val_data_info.seq_len is not None
    ), "Validation sequences must have fixed sequence length"
    batch_offset = 0
    # Do accumulation in fp64 because there could be a lot of sequences
    total_loss = torch.zeros(
        size=(val_data_info.seq_len,), dtype=torch.float64, device=fabric.device
    )
    seq_count = 0
    token_count = 0

    # carry = model.init_carry(val_data_info.local_batch_size)
    for val_data in val_dataloader:
        assert isinstance(val_data, Batch)
        val_data: Batch

        # Maybe helpful?
        val_data = Batch(
            input_ids=val_data.input_ids.contiguous().long(),
            labels=val_data.labels.contiguous().long(),
            resets=val_data.resets.contiguous().bool(),
        )

        check_batch(data_info=val_data_info, batch=val_data, batch_offset=batch_offset)

        output: ModelOutput = model(
            # carry=carry,
            input_ids=val_data.input_ids,
            labels=val_data.labels,
            # resets=val_data.resets,
        )
        # carry = output.carry
        loss = output.loss
        assert loss.size() == (
            val_data_info.local_batch_size,
            val_data_info.batch_len,
        ), loss.size()
        total_loss[
            batch_offset : batch_offset + val_data_info.batch_len
        ] += loss.sum(axis=0)

        batch_offset = (batch_offset + val_data_info.batch_len) % val_data_info.seq_len
        # Only do this after we've processed a sequence, which is when
        # batch_offset cycles back to 0
        if batch_offset == 0:
            seq_count += val_data_info.global_batch_size
        token_count += val_data_info.global_tokens_per_batch

    elapsed = time.perf_counter() - start_time
    total_loss = fabric.all_reduce(total_loss, reduce_op="sum")

    # No need to reduce for seq_count; we already use global batch size
    loss_per_token = (total_loss / seq_count).cpu().numpy()

    metrics = {
        **extra_metrics,
        "val/loss": loss_per_token.mean(axis=0),
        "val/val_token_count": token_count,
        "val/val_seq_count": seq_count,
        "val/val_time": elapsed,
        "val/val_tokens_per_second": token_count / elapsed,
    }

    # Compute average loss and perplexity over different horizons
    length = val_data_info.seq_len
    while length >= MIN_VAL_LENGTH:
        loss_avg = loss_per_token[:length].mean(axis=0)
        perplexity = np.exp(loss_avg)
        metrics[f"val/loss_avg_len_{length}"] = loss_avg
        metrics[f"val/perplexity_len_{length}"] = perplexity
        length = length // 2

    # Regular scalar metrics
    logger.log_metrics(
        metrics=metrics,
        step=step,
        logger_names=("console", "jsonl", "wandb"),
        logger_kwargs={"jsonl": {"filename": "val"}},
    )
    # Array metrics
    logger.log_metrics(
        metrics={**metrics, "val/loss_per_token": loss_per_token},
        step=step,
        logger_names=("npz",),
        logger_kwargs={"npz": {"dirname": "val"}},
    )
    model.train()


def check_data_info(fabric: L.Fabric, config: Config, data_info: DataInfo):
    assert data_info.seq_len == data_info.batch_len, "truncated BPTT not supported yet"
    assert (
        data_info.seq_len is not None
    ), "Variable sequence length is not supported yet"
    assert is_power_of_two(data_info.seq_len)
    # Interval related
    check_divisible(config.train.max_tokens, data_info.global_tokens_per_batch)
    check_divisible(config.log_interval, data_info.global_tokens_per_batch)
    check_divisible(config.train_eval_interval, data_info.global_tokens_per_batch)
    check_divisible(config.eval_interval, data_info.global_tokens_per_batch)
    check_divisible(config.checkpoint_interval, data_info.global_tokens_per_batch)

    # Otherwise batch offset will be incorrect
    check_divisible(config.checkpoint_interval, data_info.seq_len)

    check_divisible(config.checkpoint_keep_interval, config.checkpoint_interval)

    # TODO: These are actually not strictly necessary for general use cases,
    # but for our specific case where we want to ensure that different models
    # are always trained with the same set of tokens, this is necessary
    # check_divisible(config.train.max_tokens, data_info.tokens_per_stage)
    # check_divisible(config.eval_interval, data_info.tokens_per_stage)
    # check_divisible(config.checkpoint_interval, data_info.tokens_per_stage)

    # Batch shape related
    check_divisible(data_info.global_batch_size, fabric.world_size)
    assert (
        data_info.local_tokens_per_batch * fabric.world_size
        == data_info.global_tokens_per_batch
    )
    assert data_info.local_batch_size * fabric.world_size == data_info.global_batch_size
    # check_divisible(data_info.local_tokens_per_batch, config.train.grad_acc_tokens)
    check_divisible(config.train.grad_acc_tokens, data_info.batch_len)


def check_batch(data_info: DataInfo, batch: Batch, batch_offset: Optional[int]):
    assert all(
        x.size() == (data_info.local_batch_size, data_info.batch_len) for x in batch
    ), [x.size() for x in batch]
    if batch_offset is not None:
        assert 0 <= batch_offset < data_info.seq_len
        assert batch_offset % data_info.batch_len == 0
        if batch_offset == 0:
            assert batch.resets[:, :1].all() and (not batch.resets[:, 1:].any())
        else:
            assert not batch.resets.any()


def add_prefix(dictionary: Dict[str, Any], prefix: str):
    return {f"{prefix}/{key}": value for key, value in dictionary.items()}


@hydra.main(version_base=None, config_name="config", config_path="configs")
def train(config: Config):
    assert config.hf_save_dir is None
    assert config.hf_load_dir is None

    config.output_dir = osp.realpath(config.output_dir)
    config.data_dir = osp.realpath(config.data_dir)

    os.makedirs(config.output_dir, exist_ok=True)

    OmegaConf.register_new_resolver("join_path", lambda *elems: osp.join(*elems))
    OmegaConf.register_new_resolver("eval", eval)
    # Formatting, save log to file, clean log file if not resuming, and only do
    # logging in rank zero Also we do not delete previous logs even if there is
    # not checkpoint found. This is just so we can know the rerun history. The
    # log files are very small anyways.
    # Configure fabric
    fabric = L.Fabric(
        strategy=hydra.utils.instantiate(config.strategy), **config.fabric
        # strategy="ddp", **config.fabric
    )
    configure_root_logger(
        fabric=fabric,
        log_dir=Path(config.output_dir) / "logs", resume=config.resume
    )


    # logging.info(f"Fabric initialized. World size: {fabric.world_size}")
    logging.info(f"All outputs will be saved to `{osp.realpath(config.output_dir)}`")
    # Pretty print config using Rich library
    if fabric.is_global_zero:
        logging.info("Configuration:")
        rich.print(rich.syntax.Syntax(OmegaConf.to_yaml(config, resolve=True), "yaml"))
        yaml_path = Path(config.output_dir) / "config.yaml"
        with yaml_path.open("w") as f:
            f.write(OmegaConf.to_yaml(config, resolve=True))
        logging.info(f"Configuration saved to {yaml_path}.")


    # fabric.launch()
    # same seed for every process to init model (FSDP)
    fabric.seed_everything(config.seed)

    logging.info("creating datamodule")
    assert OmegaConf.is_missing(config.datamodule, "rank"), "rank should be left empty"
    assert OmegaConf.is_missing(
        config.datamodule, "world_size"
    ), "world_size should be left empty"
    datamodule: L.LightningDataModule = hydra.utils.instantiate(
        config.datamodule, world_size=fabric.world_size, rank=fabric.global_rank
    )
    # All processes wait for rank zero to finish preparing data
    with fabric.rank_zero_first():
        datamodule.prepare_data()
    datamodule.setup(stage="")
    train_dataloader, train_data_info = datamodule.train_dataloader()
    val_dataloader, val_data_info = datamodule.val_dataloader()
    train_data_info: DataInfo
    val_data_info: DataInfo
    check_data_info(fabric, config, train_data_info)
    # assert train_data_info.seq_len == val_data_info.seq_len == train_data_info.batch_len == val_data_info.batch_len, "Think twice"
    assert is_power_of_two(val_data_info.seq_len)

    # We trust the dataloaders to handle distributed sampling
    # This also sets the seed for each worker
    train_dataloader, val_dataloader = fabric.setup_dataloaders(
        train_dataloader, val_dataloader, use_distributed_sampler=False
    )
    assert hasattr(
        train_dataloader._dataloader, "state_dict"
    ), "train_dataloader must be stateful"
    assert hasattr(
        train_dataloader._dataloader, "load_state_dict"
    ), "train_dataloader must be stateful"

    logging.info("creating model")
    with fabric.init_module(empty_init=False):
        assert OmegaConf.is_missing(
            config.model.config, "vocab_size"
        ), "Vocab size should be left missing"
        config.model.config.vocab_size = train_data_info.vocab_size
        model: nn.Module = hydra.utils.instantiate(config.model)
        if config.train.gradient_checkpointing:
            model.gradient_checkpointing_enable({"use_reentrant": False})

    model = fabric.setup_module(model)
    model_info = get_model_info(
        fabric,
        lambda: hydra.utils.instantiate(config.model),
        gradient_checkpointing=config.train.gradient_checkpointing,
        data_info=train_data_info,
        print_model_path=(
            Path(config.output_dir) / "model.txt" if fabric.global_rank == 0 else None
        ),
    )

    logging.info("creating optimizer")
    param_groups, decay_list, no_decay_list = group_parameters(
        model,
        bias_weight_decay=config.train.bias_weight_decay,
        normalization_weight_decay=config.train.normalization_weight_decay,
        conv_weight_decay=config.train.conv_weight_decay,
    )
    if fabric.global_rank == 0:
        with (Path(config.output_dir) / "decay_params.txt").open("w") as f:
            print(*decay_list, file=f, sep="\n")
        with (Path(config.output_dir) / "no_decay_params.txt").open("w") as f:
            print(*no_decay_list, file=f, sep="\n")
    optimizer = hydra.utils.instantiate(config.optimizer, param_groups)
    optimizer = fabric.setup_optimizers(optimizer)

    # LR schedule
    schedule: Callable[int, float] = hydra.utils.instantiate(config.schedule)

    # TODO: the most correct thing is to is to add carry list and seq_len array here.
    # Though normally this would not be a problem.
    train_state = {
        "model": model,
        "optimizer": optimizer,
        # We actually don't use this. It doesn't work due to prefetching
        # Keeping this here just for compatibility
        "train_dataloader": train_dataloader,
        "token_count": 0,
        "batch_count": 0,  # Number of gradient updates, i.e., optimizer.step(). Or batches
        "flop_count": 0,
        "total_time": 0,  # Time since step 0. This includes validation, etc...
        "update_time": 0,  # Time actually spent on gradient updates
        "window_loss_per_token": np.zeros(shape=(train_data_info.seq_len,), dtype=np.float64),
        "window_seq_count": 0
    }
    # The true window_loss_per_token is always train_state["window_loss_per_token"] + all_reduce(local_window_loss_per_token)
    # The reason is we don't want to all reduce every step
    local_window_loss_per_token = torch.zeros(
        size=(train_data_info.seq_len,), dtype=torch.float64, device=fabric.device
    )

    progress_keys = [
        "token_count",
        "batch_count",
        "flop_count",
        "total_time",
        "update_time",
    ]

    # Checkpointer: basically saves and loads checkpoints like
    # checkpoint_dir / step-000000000016384.pt
    if config.fork_dir is not None:
       assert config.resume
    checkpointer = Checkpointer(
        fabric=fabric,
        checkpoint_dir=Path(config.output_dir) / "checkpoints",
        fork_dir=Path(config.fork_dir) / "checkpoints" if config.fork_dir is not None else None,
        fork_step=config.fork_step,
    )

    if config.resume:
        resume_step, loaded_state = checkpointer.load_checkpoint(train_state=train_state)
        assert loaded_state is None or len(loaded_state) == 0
        if resume_step is not None:
            assert train_state["token_count"] == resume_step, "What..."
    else:
        checkpointer.delete_checkpoints()
        resume_step = None

    assert "train_dataloader" in train_state
    # This is necessary because dataloader->sampler state is unreliable due to 
    # prefetching
    train_dataloader.sampler.load_state_dict({"batch_id": train_state["batch_count"]})
    # Loggers
    # We do this after checkpointer because where to resume logger state depends
    # on what checkpoint we have.
    # TODO: Note even if we do not find a checkpoint (i.e., resume_step is
    # None), if config.resume is true then we will still attempt to resume wandb
    # runs. This avoids creating redundant wandb runs. However, this might be
    # unsafe so we might consider changing this behavior to `resume =
    # resume_step is not None`
    wandb_logger = WandbLogger(
        run_id_dir=Path(config.output_dir) / "metrics" / "wandb",
        log_dir=config.wandb.log_dir,
        project=config.wandb.project,
        mode=config.wandb.mode,
        config=OmegaConf.to_container(config, resolve=True, throw_on_missing=False),
        resume=config.resume,
    )
    console_logger = ConsoleLogger()
    # Note we don't use resume_step = 0 because we only purge logs with step
    # *strictly larger* than resume_step. But if there is no checkpoint found we
    # should really just purge everything.
    jsonlines_logger = JSONLinesLogger(
        log_dir=Path(config.output_dir) / "metrics" / "jsonlines",
        resume_step=resume_step,
    )
    npz_logger = NpzLogger(
        log_dir=Path(config.output_dir) / "metrics" / "npz", resume_step=resume_step
    )

    logger = MultiLogger(
        loggers={
            "wandb": wandb_logger,
            "console": console_logger,
            "jsonl": jsonlines_logger,
            "npz": npz_logger,
        }
    )

    # Log resume history
    logger.log_metrics(
        metrics={"resume/resume_step": resume_step if resume_step is not None else 0},
        step=train_state["token_count"],
        logger_names=("wandb", "jsonl"),
        logger_kwargs={"jsonl": {"filename": "resume"}},
    )
    # Each resume results in an update. This can be used to check correctness.
    logger.log_metrics(
        add_prefix(asdict(train_data_info), "train_data_info"),
        step=train_state["token_count"],
        logger_names=("console", "wandb", "jsonl"),
        logger_kwargs={"jsonl": {"filename": "train_data_info"}},
    )
    logger.log_metrics(
        add_prefix(asdict(val_data_info), "val_data_info"),
        step=train_state["token_count"],
        logger_names=("console", "wandb", "jsonl"),
        logger_kwargs={"jsonl": {"filename": "val_data_info"}},
    )
    logger.log_metrics(
        add_prefix(asdict(model_info), "model_info"),  # pylint: disable=no-member
        step=train_state["token_count"],
        logger_names=("console", "wandb", "jsonl"),
        logger_kwargs={"jsonl": {"filename": "model_info"}},
    )

    throughput_monitor = ThroughputMonitor(fabric=fabric, window_size=100)

    # Main loop
    # Not sure why but litgpt has it
    fabric.barrier()

    # carry_list = None
    # TODO: if we save carry to checkpoint we need to change this
    # Maintains start batch_offset of the current batch in sequence
    # We only do this is sequence length is fixed
    batch_offset = 0 if train_data_info.seq_len is not None else None

    pbar = ProgressBar(
        total_steps=config.train.max_tokens,
        # total_stages=config.train.max_tokens / train_data_info.tokens_per_stage,
    )
    timer = Timer()
    timer.reset()
    for train_data in train_dataloader:
        assert isinstance(train_data, Batch)
        train_data: Batch
        if train_state["token_count"] >= config.train.max_tokens:
            break

        # Learning rate schedule
        lr = schedule(train_state["token_count"])
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Maybe helpful?
        train_data = Batch(
            input_ids=train_data.input_ids.contiguous().long(),
            labels=train_data.labels.contiguous().long(),
            resets=train_data.resets.contiguous().bool(),
        )
        check_batch(
            data_info=train_data_info, batch=train_data, batch_offset=batch_offset
        )

        grad_acc_tokens = min(
            config.train.grad_acc_tokens, train_data_info.local_tokens_per_batch
        )
        grad_acc_batch_size = safe_divide(grad_acc_tokens, train_data_info.batch_len)
        grad_acc_steps = safe_divide(
            train_data_info.local_batch_size, grad_acc_batch_size
        )

        update_start = time.perf_counter()
        # Actual gradient computation
        # It is important to do zero grad here (not later). The model could have
        # grad due to many reasons (e.g., measuring backward flops)
        optimizer.zero_grad()
        total_loss = 0.0

        # TODO: (Re)initialize carry if necessary. This could happen when batch size changes etc.
        # Though we didn't really implement that
        # if carry_list is None:
            # carry_list = [
                # model.init_carry(grad_acc_batch_size) for _ in range(grad_acc_steps)
            # ]

        # next_carry_list = []
        for step_id in range(grad_acc_steps):
            is_accumulating = step_id < grad_acc_steps - 1
            with fabric.no_backward_sync(model, enabled=is_accumulating):
                start = step_id * grad_acc_batch_size
                end = start + grad_acc_batch_size
                grad_acc_batch = Batch(*[v[start:end] for v in train_data])
                # We directly take the loss to allow fused cross entropy loss
                output: ModelOutput = model(
                    # carry=carry_list[step_id],
                    input_ids=grad_acc_batch.input_ids,
                    labels=grad_acc_batch.labels,
                    # resets=grad_acc_batch.resets,
                )
                loss = output.loss
                assert loss.size() == (
                    grad_acc_batch_size,
                    train_data_info.batch_len,
                ), loss.size()
                loss = loss.mean(axis=(0, 1)) / grad_acc_steps
                fabric.backward(loss)
            # TODO: log loss
            total_loss += loss.detach()
            with torch.no_grad():
                assert train_data_info.batch_len == train_data_info.seq_len
                local_window_loss_per_token += output.loss.sum(axis=0)

            # Note the detach carry here
            # next_carry_list.append(model.detach_carry(output.carry))

        # carry_list = next_carry_list

        # Update starting timestep for the next batch
        if batch_offset is not None:
            batch_offset = (
                batch_offset + train_data_info.batch_len
            ) % train_data_info.seq_len

        # TODO: log gradient norm
        global_grad_norm = fabric.clip_gradients(
            model, optimizer, max_norm=config.train.max_grad_norm
        )
        optimizer.step()
        update_end = time.perf_counter()

        train_state["window_seq_count"] += train_data_info.global_batch_size
        # Update various stats
        train_state["batch_count"] += 1
        # Note we use global tokens per batch here
        train_state["token_count"] += train_data_info.global_tokens_per_batch
        train_state["flop_count"] += (
            model_info.flops_per_token * train_data_info.global_tokens_per_batch
        )
        train_state["total_time"] += timer.elapsed_since_last_query()
        # This might be inaccurate due to async dispatch
        train_state["update_time"] += update_end - update_start
        throughput_monitor.update(**{key: train_state[key] for key in progress_keys})

        # Logging
        if train_state["token_count"] % config.log_interval == 0:
            total_loss = fabric.all_reduce(total_loss, reduce_op="mean")
            # Regular metrics
            train_metrics = {
                **{f"train/{key}": train_state[key] for key in progress_keys},
                # "train/stage": train_state["token_count"]
                # / train_data_info.tokens_per_stage,
                "train/lr": lr,
                "train/loss": total_loss.item(),
                "train/global_grad_norm": global_grad_norm,
            }

            logger.log_metrics(
                metrics=train_metrics,
                step=train_state["token_count"],
                logger_names=("jsonl", "wandb"),
                logger_kwargs={"jsonl": {"filename": "train"}},
            )

            # Throughput related
            if throughput_monitor.can_compute():
                throughput_metrics = {
                    **{f"throughput/{key}": train_state[key] for key in progress_keys},
                    **{
                        f"throughput/{key}": value
                        for key, value in throughput_monitor.compute().items()
                    },
                }
                logger.log_metrics(
                    metrics=throughput_metrics,
                    step=train_state["token_count"],
                    logger_names=("jsonl", "wandb"),
                    logger_kwargs={"jsonl": {"filename": "throughput"}},
                )

            # Parameter norm and gradient norm
            norm_metrics = {}
            # This summon is very important. Otherwise we will be logging
            # norms of sharded parameters
            with FullyShardedDataParallel.summon_full_params(
                model, rank0_only=True, with_grads=True, writeback=False
            ):
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        norm_metrics[f"pnorm/{name}"] = torch.linalg.vector_norm(
                            param
                        ).item()
                        if param.grad is not None:
                            norm_metrics[f"gnorm/{name}"] = torch.linalg.vector_norm(
                                param.grad
                            ).item()
            logger.log_metrics(
                metrics=norm_metrics,
                step=train_state["token_count"],
                logger_names=("jsonl", "wandb"),
                logger_kwargs={"jsonl": {"filename": "norm"}},
            )

            # Progress bar
            pbar_metrics = {"loss": train_metrics["train/loss"]}
            if throughput_monitor.can_compute():
                pbar_metrics.update({
                    "tokens/s": throughput_metrics[
                        "throughput/token_count_per_second_total_recent"
                    ],
                    "batches/s": throughput_metrics[
                        "throughput/batch_count_per_second_total_recent"
                    ],
                    "MFU": throughput_metrics["throughput/mfu_total_recent"],
                    "TFLOPS": throughput_metrics[
                        "throughput/flop_count_per_second_total_recent"
                    ]
                    / 1e12,
                })
            pbar.display(
                step=train_state["token_count"],
                elapsed=train_state["total_time"],
                metrics=pbar_metrics
            )
        # Train eval
        if train_state["token_count"] % config.train_eval_interval == 0:
            # TODO: Not sure whether we need barrier here
            fabric.barrier()
            # Sync
            with torch.no_grad():
                train_state["window_loss_per_token"] += fabric.all_reduce(local_window_loss_per_token, reduce_op="sum").cpu().numpy()
                loss_per_token = train_state["window_loss_per_token"] / train_state["window_seq_count"]
                train_eval(
                    step=train_state["token_count"],
                    loss_per_token=loss_per_token,
                    logger=logger,
                    config=config,
                    train_data_info=train_data_info,
                    extra_metrics={
                        **{f"train_eval/train_{key}": train_state[key] for key in progress_keys},
                        "train_eval/window_seq_count": train_state["window_seq_count"],
                        "train_eval/window_token_count": train_state["window_seq_count"] * train_data_info.seq_len,
                    },
                )
                local_window_loss_per_token.zero_()
                train_state["window_loss_per_token"][:] = 0.0
                train_state["window_seq_count"] = 0
            fabric.barrier()

        # Validation
        if (not config.skip_eval) and (train_state["token_count"] % config.eval_interval == 0 or (config.final_eval and train_state["token_count"] >= config.train.max_tokens)):
            # TODO: Not sure whether we need barrier here
            fabric.barrier()
            validate(
                step=train_state["token_count"],
                logger=logger,
                config=config,
                model=model,
                fabric=fabric,
                val_dataloader=val_dataloader,
                val_data_info=val_data_info,
                extra_metrics={
                    **{f"val/train_{key}": train_state[key] for key in progress_keys},
                    # "val/train_stage": train_state["token_count"]
                    # / train_data_info.tokens_per_stage,
                },
            )
            fabric.barrier()

        # Checkpointing
        # Checkpointing must be the last thing we do
        if train_state["token_count"] % config.checkpoint_interval == 0 or train_state["token_count"] >= config.train.max_tokens:
            fabric.barrier()
            save_start = time.perf_counter()
            # TODO: Disable some annoying message. If using torch>=2.4 you can delete this
            fsdp_logger = logging.getLogger("torch.distributed.fsdp._optim_utils")
            fsdp_logger.disabled = True
            fsdp_logger = logging.getLogger("torch.distributed.fsdp._debug_utils")
            fsdp_logger.disabled = True

            # All reduce loss per token
            with torch.no_grad():
                train_state["window_loss_per_token"] += fabric.all_reduce(local_window_loss_per_token, reduce_op="sum").cpu().numpy()
                local_window_loss_per_token.zero_()

            checkpointer.save_checkpoint(
                train_state=train_state, step=train_state["token_count"],
                keep=(train_state["token_count"] % config.checkpoint_keep_interval == 0 or train_state["token_count"] >= config.train.max_tokens)
            )
            # TODO: Disable some annoying message. If using torch>=2.4 you can delete this
            fsdp_logger = logging.getLogger("torch.distributed.fsdp._optim_utils")
            fsdp_logger.disabled = False
            fsdp_logger = logging.getLogger("torch.distributed.fsdp._debug_utils")
            fsdp_logger.disabled = False

            fabric.barrier()
            save_end = time.perf_counter()
            logger.log_metrics(
                metrics={"checkpoint/checkpoint_time": save_end - save_start},
                step=train_state["token_count"],
                logger_names=("jsonl", "wandb", "console"),
                logger_kwargs={"jsonl": {"filename": "checkpoint"}},
            )
    logging.info(f"Training finished with {train_state['token_count']} tokens!")


if __name__ == "__main__":
    train()  # pylint: disable=no-value-for-parameter
