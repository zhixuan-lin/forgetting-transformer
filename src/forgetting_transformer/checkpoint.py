import lightning as L
import logging
from typing import Dict, Union, Optional
from pathlib import Path
import re

class Checkpointer:
    def __init__(
        self,
        fabric: L.fabric,
        checkpoint_dir: Union[Path, str],
        fork_dir: Optional[Union[Path, str]] = None,
        fork_step: Optional[int] = None
    ):
        """
        Checkpointer that can automatically load the latest checkpoint.

        If there is a checkpoint from `checkpoint_dir`, we resume from there.
        Otherwise we look in `fork_dir`. In either case new checkpoints
        will be saved to `checkpoint_dir`
        """
        assert (fork_dir is None) == (fork_step is None), "fork_dir and for_step should either both be None or not None"
        self.fabric = fabric
        self.fork_dir = Path(fork_dir) if fork_dir is not None else None
        self.fork_step = fork_step
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.check_filetype()

    def check_filetype(self):
        # Checking files
        for path in self.checkpoint_dir.glob("*"):
            assert path.is_file(), f"{path} is not file. If you use FSDP you would want to set state_dict_type='full'."
            assert path.name.endswith(".pt") or path.name.endswith(".pt.done") or path.name.endswith(".pt.keep"), f"Path {path} does not end with `.pt`, `.pt.done` or `.pt.keep`"

    def delete_checkpoints(self):
        if self.fabric.is_global_zero:
            self.check_filetype()
            logging.info("Not resuming. Deleting existing checkpoints...")
            # Delete sentinels first
            for path in self.checkpoint_dir.glob('*.pt.done'):
                path.unlink()
            for path in self.checkpoint_dir.glob('*.pt.keep'):
                path.unlink()
            # Actual checkpoint
            for path in self.checkpoint_dir.glob('*.pt'):
                path.unlink()

    @classmethod
    def get_checkpoint_path(cls, checkpoint_dir: Union[str, Path], step: Optional[int] = None):
        checkpoint_dir = Path(checkpoint_dir)
        if step is None:
            # If step is not specified we try to load the latest checkpoint
            steps = []
            for sentinel_path in checkpoint_dir.glob('*.pt.done'):
                steps.append(cls.get_step_from_filename(sentinel_path.name))
            if len(steps) > 0:
                resume_step = max(steps)
            else:
                resume_step = None
        else:
            # Check sentinel
            sentinel_path = checkpoint_dir / f"{cls.get_checkpoint_filename(step)}.done"
            if sentinel_path.is_file():
                resume_step = step
            else:
                # If step is specified but nothing found we raise an error
                raise FileNotFoundError(f"Cannot find checkpoint sentinel {sentinel_path} in {checkpoint_dir}.")


        if resume_step is not None:
            path = checkpoint_dir / cls.get_checkpoint_filename(resume_step)
        else:
            path = None

        return resume_step, path

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
    def get_checkpoint_filename(step, min_num_digits=15):
        num_digits = max(min_num_digits, len(str(step)))
        return f"step-{step:0{num_digits}d}.pt"

    def save_checkpoint(
        self,
        train_state: Dict,
        step: int,
        keep: bool
    ):
        """Save checkpoint for a step.

        The save checkpoint is like `{checkpoint_dir}/step-000000000016384.pt`

        Args:
            - keep: normally we only keep the latest checkpoint. If keep is True
              then this checkpoint is permanent
        """
        # file name is typically like "step-000000000016384.pt"
        path = self.checkpoint_dir / self.get_checkpoint_filename(step)
        logging.info(f"Saving checkpoint to {path}...")
        self.fabric.save(
            path=path,
            state=train_state
        )
        if keep and self.fabric.is_global_zero:
            keep_path = Path(f"{path}.keep")
            with keep_path.open("w"):
                pass
        # Create empty sentinel file
        # Wait for all ranks to finish saving
        self.fabric.barrier()
        if self.fabric.is_global_zero:
            sentinel_path = Path(f"{path}.done")
            with sentinel_path.open("w"):
                pass

        logging.info(f"Checkpoint saved to {path}.")

        # Delete previous checkpoints without keep flag
        if self.fabric.is_global_zero:
            for delete_path in self.checkpoint_dir.glob('*.pt'):
                delete_step = self.get_step_from_filename(delete_path.name)
                delete = False
                if not self.is_valid_checkpoint(delete_path):
                    delete = True
                elif delete_step > step :
                    raise ValueError(f"Currently saving checkpoint at step {step} but found checkpoint at step {step} at path {delete_path}. Aborting.")
                elif delete_step < step:
                    if Path(f"{delete_path}.keep").is_file():
                        delete = False
                    else:
                        delete = True
                else:
                    assert delete_step == step, "What??"

                if delete:
                    logging.info(f"Deleting invalid or outdated checkpoint {delete_path}")
                    sentinel_path = Path(f"{delete_path}.done")
                    sentinel_path.unlink()
                    delete_path.unlink()

    def is_valid_checkpoint(self, path: Path):
        return Path(f"{path}.done").is_file()

    def load_checkpoint(
        self,
        train_state: Dict
    ):
        # First we try to load the latest checkpoint from the current checkpoint directory
        resume_step, loaded_state = self.try_loading_checkpoint(train_state, self.checkpoint_dir)
        # If not we try the fork_dir
        if resume_step is None and self.fork_dir is not None:
            resume_step, loaded_state = self.try_loading_checkpoint(train_state, self.fork_dir, step=self.fork_step)
        if resume_step is None:
            if self.fork_dir is None:
                logging.info(f"No checkpoint found in {self.checkpoint_dir}. Starting from scratch")
            else:
                logging.info(f"No checkpoint found in {self.fork_dir} and {self.checkpoint_dir}. Starting from scratch")
        return resume_step, loaded_state


    def try_loading_checkpoint(
        self,
        train_state: Dict,
        directory: Union[Path, str],
        step: Optional[int] = None
    ):
        resume_step, path = self.get_checkpoint_path(directory, step)

        if resume_step is not None:
            assert path is not None
            logging.info(f"Loading latest checkpoint at step {resume_step} from {path}")
            loaded_state = self.fabric.load(path, train_state)
            logging.info(f"Checkpoint loaded from {path}")
        else:
            loaded_state = None
        return resume_step, loaded_state
