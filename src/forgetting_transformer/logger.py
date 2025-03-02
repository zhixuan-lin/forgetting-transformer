import wandb
import os
from lightning.fabric.utilities import rank_zero_only
from typing import Any, Dict, Union, Optional, Tuple, List
from pathlib import Path
import logging
import jsonlines
import numpy as np
from lightning import fabric
from collections import OrderedDict
import torch
import re
import warnings

def is_valid_filename(filename):
    # Check if the filename is not empty and does not contain illegal characters
    if not filename or re.search(r'[<>:"/\\|?*]', filename):
        return False
    else:
        return True

def convert_to_numpy(metrics: Dict[str, Any]):
    def transform(key: str, value: Any):
        if isinstance(value, torch.Tensor):
            return value.cpu().detach().numpy()
        elif isinstance(value, (np.ndarray, int, float)):
            return np.asarray(value)
        else:
            raise ValueError(f"Metric {key} with type {type(value)} cannot be converted to numpy array")

    return {k: transform(k, v) for k, v in metrics.items()}

def convert_to_scalar(metrics: Dict[str, Any]):
    def transform(key: str, value: Any):
        if isinstance(value, torch.Tensor):
            assert (
                value.numel() == 1
            ), f"Metric {key} should be a scalar but has shape {value.size()}"
            return value.item()
        elif isinstance(value, np.ndarray):
            assert (
                value.size == 1
            ), f"Metric {key} should be a scalar but has shape {value.size()}"
            return value.item()
        else:
            assert isinstance(
                value, (int, float)
            ), f"Metric {key} with type {type(value)} cannot be converted to a scalar"
            return value

    return {k: transform(k, v) for k, v in metrics.items()}


class Logger:
    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, Any], step: int, **kwargs):
        pass


class MultiLogger(Logger):
    def __init__(
        self,
        loggers: Dict[str, Logger]
    ):
        """Convenient function to log multiple logger at a time"""
        self.loggers = loggers

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, Any], step: int, logger_names: Tuple[str], logger_kwargs: Dict[str, Dict[str, Any]] = {}):
        """Log metrics to multiple loggers.

        Only the loggers whose name appear in logger_names will be logged
        """
        assert set(logger_kwargs).issubset(set(logger_names))
        assert all(name in self.loggers for name in logger_names)
        for name in logger_names:
            if name in logger_kwargs:
                kwargs = logger_kwargs[name]
            else:
                kwargs = {}
            self.loggers[name].log_metrics(metrics=metrics, step=step, **kwargs)


class NpzLogger(Logger):
    def __init__(
        self,
        log_dir: Union[str, Path],
        resume_step: Optional[int],
    ):
        self.log_dir = Path(log_dir)
        self.resume_step = resume_step
        self.setup()

    @staticmethod
    def get_step_from_filename(filename: str):
        # filename: step-000000000016384.<suffix>
        # Check format
        assert re.match(r"^step-(\d+)\..+", filename), (
            "Filename should be of format `step-<step>.<suffix> where <step> should be at"
            f" least 1 digit, but got {filename}."
        )
        stem, _, _ = filename.partition('.')
        _, _, digits = stem.partition('-')
        return int(digits)

    @staticmethod
    def get_filename(step, min_num_digits=15):
        num_digits = max(min_num_digits, len(str(step)))
        return f"step-{step:0{num_digits}d}.npz"

    @rank_zero_only
    def setup(self):
        logging.info("Setting up npz logger...")
        if not self.log_dir.is_dir():
            assert not self.log_dir.exists()
            self.log_dir.mkdir(parents=True, exist_ok=True)

        for path in self.log_dir.glob("*"):
            assert path.is_dir(), f"Path {path} is not a directory"
        # Checking files
        for path in self.log_dir.glob("*/*"):
            assert path.is_file() and path.name.endswith(".npz"), f"Path {path} is not an NPZ file"

        if self.resume_step is None:
            # Remove existing file if not resuming
            for path in self.log_dir.glob("*/*.npz"):
                logging.info(f"Deleting {path} since we are not resuming")
                path.unlink()
            # Remove empty directories
            for path in self.log_dir.glob("*"):
                assert path.is_dir(), "what?"
                path.rmdir()
        else:
            # Purge all files with step larger than resume_step
            for path in self.log_dir.glob("*/*.npz"):
                step = self.get_step_from_filename(path.name)
                if step > self.resume_step:
                    logging.info(f"Deleting {path} since its step is larger than resume step")
                    path.unlink()

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, Any], step: int, dirname: str, mode="a"):
        assert is_valid_filename(dirname), f"{dirname} is not a valid directory name"
        metrics = convert_to_numpy(metrics)
        entries = [("step", step)]
        entries.extend(metrics.items())

        directory = Path(self.log_dir) / dirname
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / self.get_filename(step)
        np.savez(path, **metrics)


class ConsoleLogger(Logger):

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, Any], step: int):
        metrics = convert_to_scalar(metrics)
        entries = [f"[step: {step}]"]

        def get_fmt_str(value):
            if isinstance(value, int):
                return "[{key}: {value}]"
            else:
                return "[{key}: {value:.3f}]"

        entries.extend(
            get_fmt_str(value).format(key=key, value=value)
            for key, value in metrics.items()
        )
        logging.info(" ".join(entries))


class JSONLinesLogger(Logger):
    def __init__(
        self,
        log_dir: Union[str, Path],
        resume_step: Optional[int],
    ):
        self.log_dir = Path(log_dir)
        self.resume_step = resume_step
        self.setup()

    @rank_zero_only
    def setup(self):
        logging.info("Setting up jsonlines logger...")
        if not self.log_dir.is_dir():
            assert not self.log_dir.exists()
            self.log_dir.mkdir(parents=True, exist_ok=True)

        # Checking file types
        for path in self.log_dir.glob("*"):
            assert path.is_file(), f"Path {path} is not file"
            assert path.name.endswith(".jsonl"), f"Path {path} does not end with `.jsonl`"

        if self.resume_step is None:
            # Remove existing file if not resuming
            for path in self.log_dir.glob("*.jsonl"):
                logging.info(f"Deleting {path} since we are not resuming")
                path.unlink()
        else:
            # Purge all lines with step larger than resume_step
            for path in self.log_dir.glob("*.jsonl"):
                with jsonlines.open(path, mode="r") as f:
                    lines = list(f.iter())
                if lines[-1]["step"] > self.resume_step:
                    logging.info(
                        f"{path}'s step is {lines[-1]['step']} which is larger"
                        f" than resume step {self.resume_step}. Truncating the file..."
                    )
                with jsonlines.open(path, mode="w") as f:
                    for line in lines:
                        if line["step"] <= self.resume_step:
                            f.write(line)

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, Any], step: int, filename: str, mode="a"):
        assert is_valid_filename(filename), f"{filename} is not a valid filename"
        metrics = convert_to_scalar(metrics)
        entries = [("step", step)]
        entries.extend(metrics.items())
        with jsonlines.open(self.log_dir / f"{filename}.jsonl", mode=mode) as writer:
            writer.write(OrderedDict(entries))


class WandbLogger(Logger):
    def __init__(
        self,
        run_id_dir: Union[str, Path],
        log_dir: Union[str, Path],
        project: str,
        config: Any,
        mode: str,
        resume: bool,
    ):
        """Wandb logger that resumes by default
        Arguments:
            - run_id_dir: the directory to save wandb id. Note this is not where
              we save the wandb log files
            - log_dir: files will be saved to log_dir / 'wandb' because that's
              just what wandb.init does
        """
        self.project = project
        self.config = config
        self.mode = mode
        self.run_id_dir = Path(run_id_dir)
        self.log_dir = Path(log_dir)
        self.resume = resume
        self.initialized = False
        self.warned = False
        self.setup()

    @rank_zero_only
    def setup(self):

        logging.info("Setting up wandb logger...")
        if not self.run_id_dir.is_dir():
            assert not self.run_id_dir.exists()
            self.run_id_dir.mkdir(parents=True, exist_ok=True)
        path = self.run_id_dir / "wandb_run_id.txt"
        if self.mode == "disabled":
            logging.info("Wandb disabled.")
            run_id = None
        elif self.resume and path.is_file():
            with path.open("r") as f:
                run_id = f.read().strip()
            logging.info(f"Resuming wandb run {run_id} from {path}.")
        else:
            if self.resume:
                logging.info(f"Wandb run id not found at {path}. Creating a new run.")
            else:
                logging.info(f"Not resuming. Creating a new wandb run.")
            run_id = None

        wandb.init(
            project=self.project,
            config=self.config,
            dir=self.log_dir,
            resume="allow",
            mode=self.mode,
            id=run_id,
            # This suppress the chunky summary at the end
            settings={"quiet": True}
        )

        if run_id is not None:
            assert wandb.run.id == run_id, f"Resumed {wandb.run.id} != previous run id {run_id}"
        logging.info(f"wandb initialized. Run id: {wandb.run.id}")

        # For some reason, if mode is disabled, wandb generates a new run id
        # regardless of the run_id we give to init. So in this case we don't
        # save run id
        if self.mode != "disabled":
            with path.open("w") as f:
                f.write(wandb.run.id)

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, Any], step):
        # TODO: We only do logging if so. The reason is wandb does not support
        # overwriting or dropping previous steps. And if we do it issues
        # extremely annoying warning.

        # If this changes in the future we could change this.
        if step >= wandb.run.step:
            metrics = convert_to_scalar(metrics)
            wandb.log(step=step, data=metrics)
        elif not self.warned:
            logging.warning(f"Due to wandb limitations, logs before step {wandb.run.step} will be dropped.")
            self.warned = True
