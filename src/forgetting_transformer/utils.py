import os
import subprocess
import re
import logging
from lightning.fabric.utilities import rank_zero_only
import time
import datetime
from torch import nn
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings(action="ignore", message="Flash Attention is not installed")
    warnings.filterwarnings(action="ignore", message="`torch.cuda.amp")
    import fla
    try:
        import mamba_ssm
    except ImportError:
        mamba_ssm = None
import lightning as L
from collections import deque, defaultdict, OrderedDict


# https://stackoverflow.com/questions/57025836/how-to-check-if-a-given-number-is-a-power-of-two
def is_power_of_two(n):
    return (n != 0) and (n & (n - 1) == 0)


def check_divisible(a, b):
    assert a >= b
    assert a % b == 0


def safe_divide(a, b):
    check_divisible(a, b)
    return a // b



class ProgressBar:
    def __init__(self, total_steps):
        self.total_steps = total_steps
        # self.total_stages = total_stages

    @rank_zero_only
    def display(self, step, elapsed, metrics):

        entries = [
            self._get_progress_str(step),
            self._get_step_str(step),
            # self._get_stage_str(step),
        ]
        eta = (self.total_steps - step) / step * elapsed
        entries.append(f"[T: {datetime.timedelta(seconds=int(elapsed))}]")
        entries.append(f"[ETA: {datetime.timedelta(seconds=int(eta))}]")

        entries += [f"[{key}: {value:.3f}]" for (key, value) in metrics.items()]
        logging.info(" ".join(entries))

    def _get_progress_str(self, step):
        return f"[P: {step / self.total_steps * 100:.2f}%]"

    def _get_step_str(self, step):
        return f"[S: {step}/{self.total_steps}]"

    # def _get_stage_str(self, step):
        # stage = step / self.total_steps * self.total_stages
        # return f"[STG: {stage:.2f}/{self.total_stages:.2f}]"


class Timer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.start = time.perf_counter()
        self.last = self.start

    def elapsed_since_last_query(self):
        now = time.perf_counter()
        elapsed = now - self.last
        self.last = now
        return elapsed


def group_parameters(
    model: nn.Module,
    bias_weight_decay: bool,
    normalization_weight_decay: bool,
    conv_weight_decay: bool,
):
    """Group parameters based on whether weight decay should be applied.

    Disable weight decay for
    - Parameters with attribute _no_weight_decay == True
    - All linear/layernorm bias if bias_weight_decay == False
    - All normalization (rmsnorm, layernorm) parameters if normalization_weight_decay == False

    We decide to apply weight decay to embeddings. One of reasons is it is right
    before rmsnorm, meaning its norm is guaranteed to keep increasing during
    training (equivalent to smaller learning rate), which is probably not a good
    idea.

    Some examples of how people deal with this:
    - no WD for bias and layernorm:
        - gpt-neox: https://github.com/EleutherAI/gpt-neox/blob/c7863673e3c08b5886cae36cf096a0fb5789dd0e/megatron/model/utils.py#L27
        - nanogpt: https://github.com/karpathy/nanoGPT/blob/9755682b981a45507f6eb9b11eadef8cb83cebd5/model.py#L263
        - megatron-lm: https://github.com/NVIDIA/Megatron-LM/blob/50b073cc788271efb5d4fbc92d987aeed2384f4f/megatron/core/optimizer/__init__.py#L92
        - ColossalAI: https://github.com/hpcaitech/ColossalAI/blob/0d3a85d04f467d0de5e0d6fb00571dc3ef8fe023/examples/tutorial/sequence_parallel/train.py#L102
        - huggingface transformers:
            - https://github.com/huggingface/transformers/blob/174890280b340b89c5bfa092f6b4fb0e2dc2d7fc/src/transformers/trainer.py#L1045
            - https://github.com/search?q=repo%3Ahuggingface%2Ftransformers+no_decay&type=code&p=2
        - alpa: https://github.com/alpa-projects/alpa/blob/b8078a9f75cb4c90cabb4550ee48c99ef394e209/benchmark/alpa/benchmark_one_case_moe.py#L25
    - no WD for bias, layernorm and embeddings:
        - mingpt: . https://github.com/karpathy/minGPT/blob/37baab71b9abea1b76ab957409a1cc2fbfba8a26/mingpt/model.py#L215
        - levanter: https://github.com/stanford-crfm/levanter/blob/97358f9bde968d70ccd7075b848c6c6169044462/src/levanter/optim/config.py#L50
        - safari and flash-attention (they take this from mingpt so...):  https://github.com/HazyResearch/safari/blob/02220c69d247e5473616cd053a443ad99fd2559b/src/utils/optim_groups.py#L41
    - WD everything:
        - max-text: https://github.com/google/maxtext/blob/6a080c953f1a73861539e9de50a5f3ee1d61f995/MaxText/optimizers.py#L27
        - easylm: https://github.com/young-geng/EasyLM/blob/fe5b2c354e25d697fce7cd225e23bbbe72570da3/EasyLM/models/llama/llama_train.py#L86
        - litgpt: https://github.com/Lightning-AI/litgpt/blob/3c0c47930ca05d4d408a114c58b9fa167374c7bc/litgpt/pretrain.py#L211
        - torchtitan: https://github.com/pytorch/torchtitan/blob/8c497b7119863f7ea5b03675980510e597e187f1/torchtitan/optimizer.py#L35
    """

    # We use list instead of  to ensure that the order remains the same if
    # (1) (Unlikely) python internal set and dict impl changes
    # (2) If module/param name changes
    decay_dict = OrderedDict()
    no_decay_dict = OrderedDict()
    memo = set()
    # This is very verbose to avoid any unexpected behavior, especially for
    # custom parameters. However, these will still be some edge cases such as
    # duplicate modules.
    for module_name, module in model.named_modules():
        assert not hasattr(module, "no_weight_decay"), "This is not supported"
        # Modules without immediate parameters will not be checked
        for param_name, param in module.named_parameters(recurse=False):

            if param in memo:
                # Duplicated parameters. This could be due to many reasons, one
                # of the most common is weight tieing between embedding and
                # output layer. We don't do it here so this must be an error.
                raise ValueError(
                    f"Duplicate parameter {param_name} found in module {module_name}. This error is raised just in case. You can remove this line if you know what you are doing."
                )
                continue
            memo.add(param)
            should_decay = None
            assert param.requires_grad, "This does not look right man"
            if hasattr(param, "_no_weight_decay"):
                # _no_weight_decay takes precedence over anything else
                assert (
                    param._no_weight_decay
                ), "This is just in case. You can delete this if you know what you are doing."
                should_decay = not param._no_weight_decay
            elif isinstance(
                module,
                (
                    nn.LayerNorm,
                    # nn.RMSNorm, # Only pytorch 2.4 has this
                    fla.modules.RMSNorm,
                    fla.modules.FusedRMSNormSwishGate,
                    fla.modules.LayerNorm,
                    fla.modules.GroupNorm,
                ),
            ) or "RMSNorm" in str(module.__class__) or 'GroupNorm' in str(module.__class__):
                assert param.dim() == 1
                if param_name == "weight":
                    should_decay = normalization_weight_decay
                elif param_name == "bias":
                    should_decay = normalization_weight_decay and bias_weight_decay
                else:
                    raise ValueError(
                        f"Unknown param {param_name} in module {module_name}."
                    )
            elif isinstance(module, nn.Linear):
                if param_name == "weight":
                    should_decay = True
                elif param_name == "bias":
                    should_decay = bias_weight_decay
                else:
                    raise ValueError(
                        f"Unknown param {param_name} in module {module_name}."
                    )
            elif isinstance(module, nn.Conv1d):
                if param_name == "weight":
                    should_decay = conv_weight_decay
                elif param_name == "bias":
                    should_decay = conv_weight_decay and bias_weight_decay
                else:
                    raise ValueError(
                        f"Unknown param {param_name} in module {module_name}."
                    )
            elif isinstance(module, nn.Embedding):
                assert param_name == "weight"
                should_decay = True
            else:
                # For a custom module we require all its parameters be marked
                # with _no_weight_decay. If we reach this branch it means this
                # is not true
                unknown_param_list = [
                    name
                    for name, param in module.named_parameters(recurse=False)
                    if not hasattr(param, "_no_weight_decay")
                ]
                raise ValueError(
                    f"We don't know whether to apply weight decay to the following parameters from {module_name}: {unknown_param_list}. Please mark all these with `_no_weight_decay` "
                )

            assert should_decay is not None
            # module_name could be empty.
            fullname = "{module_name}.{param_name}".format(
                module_name=module_name or "ROOT", param_name=param_name
            )
            if should_decay:
                decay_dict[fullname] = param
            else:
                no_decay_dict[fullname] = param

    decay_set = set(decay_dict.values())
    no_decay_set = set(no_decay_dict.values())
    assert len(decay_set) == len(decay_dict)
    assert len(no_decay_set) == len(no_decay_dict)
    assert decay_set.intersection(no_decay_set) == set()
    assert decay_set.union(no_decay_set) == set(model.parameters())
    param_groups = [
        {"params": list(decay_dict.values())},
        {"params": list(no_decay_dict.values()), "weight_decay": 0.0},
    ]
    return param_groups, list(decay_dict.keys()), list(no_decay_dict.keys())


# https://github.com/Lightning-AI/pytorch-lightning/blob/f3f10d460338ca8b2901d5cd43456992131767ec/src/lightning/fabric/utilities/throughput.py#L241
class ThroughputMonitor:
    def __init__(self, fabric: L.Fabric, window_size: int = 100):
        # Get available flops per second FOR EACH DEVICE
        self.available_flops = L.fabric.utilities.throughput.get_available_flops(
            device=fabric.device,
            dtype=L.fabric.utilities.throughput._plugin_to_compute_dtype(
                fabric.strategy.precision
            ),
        )
        self.total_time = deque(maxlen=window_size)
        self.update_time = deque(maxlen=window_size)
        self.token_count = deque(maxlen=window_size)
        self.batch_count = deque(maxlen=window_size)
        self.flop_count = deque(maxlen=window_size)
        self.fabric = fabric

    def update(
        self,
        total_time: float,
        update_time: float,
        token_count: float,
        batch_count: float,
        flop_count: float,
    ):
        self.total_time.append(total_time)
        self.update_time.append(update_time)
        self.token_count.append(token_count)
        self.batch_count.append(batch_count)
        self.flop_count.append(flop_count)

    def can_compute(self):
        return len(self.total_time) >= 2

    def compute(self):
        assert len(self.total_time) >= 2

        metrics = {}
        for key in ["token_count", "batch_count", "flop_count"]:
            count_queue = getattr(self, key)
            for time_type in ["total", "update"]:
                time_queue = {"total": self.total_time, "update": self.update_time}[
                    time_type
                ]
                for recent in [True, False]:
                    recency = "recent" if recent else "cum"
                    metric_name = f"{key}_per_second_{time_type}_{recency}"
                    if recent:
                        count = count_queue[-1] - count_queue[0]
                        seconds = time_queue[-1] - time_queue[0]
                    else:
                        count = count_queue[-1]
                        seconds = time_queue[-1]

                    result = count / seconds
                    metrics[metric_name] = result

                    # Compute hardware utilization
                    if key == "flop_count":
                        mfu_name = f"mfu_{time_type}_{recency}"
                        metrics[mfu_name] = result / (
                            self.available_flops * self.fabric.world_size
                        )
        return metrics
