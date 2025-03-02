import torch
import zarr
import numpy as np
from torch.utils.data import IterableDataset, DataLoader, Sampler, Dataset
from typing import Callable, List, Optional, Union, Literal
from pathlib import Path
import numpy as np
import lightning as L
from forgetting_transformer.datamodule.common import DataInfo, Batch
from forgetting_transformer.utils import safe_divide, check_divisible
import logging


class SlimPajamaDataModule(L.LightningDataModule):
    VOCAB_SIZE = 32000

    def __init__(
        self,
        data_dir,
        world_size: int,
        rank: int,
        train_seq_len: int,
        train_batch_len: int,
        train_batch_size: int,
        # train_tokens_per_stage: int,
        train_num_workers: int,
        eval_tokens: int,
        eval_seq_len: int,
        eval_batch_len: int,
        eval_local_batch_size: int,
        eval_num_workers: int,

        train_doc_len: Optional[int] = None,
        eval_doc_len: Optional[int] = None,
        eval_stateful: bool = False
    ):
        """SlimPajama data module

        Note for evaluation we use local_tokens_per_batch because this is
        limited by memory. For training we use gradient accumulation so memory
        is typically not a concern.
        Arguments:
            data_dir: data_dir / 'train.zarr' should exist
        """
        super().__init__()
        # if train_reshape_list is None:
        # train_reshape_list = train_context_size_list
        self.data_dir = Path(data_dir).expanduser()
        self.train_num_workers = train_num_workers
        self.eval_num_workers = eval_num_workers

        self.train_dataset = SlimPajamaDataset(
            data_path=self.data_dir / f"slimpajama-train-token-{TRAIN_TOKENS}-len-{TRAIN_DOC_LEN}" / "data.zarr",
            split="train",
            world_size=world_size,
            rank=rank,
            batch_size=train_batch_size,
            # tokens_per_stage=train_tokens_per_stage,
            seq_len=train_seq_len,
            batch_len=train_batch_len,
            total_tokens=None,
            doc_len=train_doc_len
        )

        self.val_dataset = SlimPajamaDataset(
            data_path=self.data_dir / f"slimpajama-validation-token-{VAL_TOKENS}-len-{VAL_DOC_LEN}-ordered" / "data.zarr",
            split="validation",
            world_size=world_size,
            rank=rank,
            batch_size=eval_local_batch_size * world_size,
            # tokens_per_stage=eval_tokens,
            seq_len=eval_seq_len,
            batch_len=eval_batch_len,
            total_tokens=eval_tokens,
            doc_len=eval_doc_len
        )
        self.eval_stateful = eval_stateful

    def train_dataloader(self):
        train_sampler = StatefulSampler(self.train_dataset)
        train_dataloader = SlimPajamaDataloader(
            self.train_dataset,
            batch_size=None,
            shuffle=False,
            sampler=train_sampler,
            num_workers=self.train_num_workers,
        )
        data_info = DataInfo(
            vocab_size=self.VOCAB_SIZE,
            batch_len=self.train_dataset.batch_len,
            global_tokens_per_batch=self.train_dataset.tokens_per_batch,
            local_tokens_per_batch=self.train_dataset.local_tokens_per_batch,
            # tokens_per_stage=self.train_dataset.tokens_per_stage,
            seq_len=self.train_dataset.seq_len,
            total_tokens=self.val_dataset.total_tokens
        )
        return train_dataloader, data_info

    def val_dataloader(self):

        # We can't use stateful sampler. Otherwise the second iterator created
        # from the dataloader won't behave correctly
        # val_sampler = StatefulSampler(self.val_dataset)
        if self.eval_stateful:
            val_sampler = StatefulSampler(self.val_dataset)
        else:
            val_sampler = None
        val_dataloader = SlimPajamaDataloader(
            self.val_dataset,
            batch_size=None,
            shuffle=False,
            sampler=val_sampler,
            num_workers=self.eval_num_workers,
        )

        data_info = DataInfo(
            vocab_size=self.VOCAB_SIZE,
            batch_len=self.val_dataset.batch_len,
            global_tokens_per_batch=self.val_dataset.tokens_per_batch,
            local_tokens_per_batch=self.val_dataset.local_tokens_per_batch,
            # tokens_per_stage=self.val_dataset.tokens_per_stage,
            seq_len=self.val_dataset.seq_len,
            total_tokens=self.val_dataset.total_tokens
        )
        return val_dataloader, data_info


TRAIN_DOC_LEN = 2048
VAL_DOC_LEN = 65536

TRAIN_DOC_COUNT = 7864320
TRAIN_TOKENS = 16106127360

VAL_TOKENS = 536870912
VAL_DOC_COUNT = 8192
BOS_TOKEN = 1


class SlimPajamaDataset(Dataset):
    """Configurable dataloader for SlimPajama."""

    def __init__(
        self,
        data_path: Union[str, Path],
        world_size: int,
        rank: int,
        batch_size: int,
        batch_len: int,
        split: Literal["train", "validation"],
        seq_len: Optional[int] = None,
        # tokens_per_stage: Optional[int] = None,
        doc_len: Optional[int] = None,
        total_tokens: Optional[int] = None,
    ):
        """
        Loads a long crawl dataset of zarr array.

        A batch contains subsequences, not sequence. Also

        Arguments:
            - doc_len: this should only used for debugging. It discards
              tokens
            - tokens_per_batch: number of tokens per *global* batch. Distributed
              sampling will be handled internally
        """


        # import ipdb; ipdb.set_trace()
        self.dataset = zarr.open(data_path, mode="r")
        DOC_COUNT = dict(train=TRAIN_DOC_COUNT, validation=VAL_DOC_COUNT)[split]
        DOC_LEN = dict(train=TRAIN_DOC_LEN, validation=VAL_DOC_LEN)[split]
        assert self.dataset.shape[0] == DOC_COUNT
        assert self.dataset.shape[1] == DOC_LEN

        assert seq_len is None
        if seq_len is None:
            self.seq_len = batch_len
        else:
            self.seq_len = seq_len
        assert self.seq_len <= DOC_LEN, (self.seq_len, DOC_LEN)
        assert batch_size <= DOC_COUNT, "What?"

        assert self.dataset.shape[1] == DOC_LEN, "uh what?"
        if doc_len is not None:
            raise ValueError(f"`doc_len` is set to {doc_len} instead of `None`. This should only be used for debugging.")
            self.doc_len = doc_len
        else:
            # Discard edge tokens
            assert DOC_LEN % self.seq_len == 0, "You probably don't want this"
            self.doc_len = DOC_LEN - DOC_LEN % self.seq_len

        if total_tokens is None:
            assert split == "train"
            self.doc_count = DOC_COUNT - DOC_COUNT % batch_size
            self.total_tokens = self.doc_count * self.doc_len
        else:
            assert split == "validation"
            # In this case we what to ensure that we do not drop any token
            self.doc_count = safe_divide(total_tokens, self.doc_len)
            assert self.doc_count <= DOC_COUNT
            assert self.doc_count % batch_size == 0, "We don't want to drop any tokens"
            self.total_tokens = total_tokens


        # self.tokens_per_stage = tokens_per_stage

        # self.total_tokens = total_tokens
        # self.num_stages = safe_divide(total_tokens, tokens_per_stage)

        self.world_size = world_size
        self.rank = rank

        self.batch_len = batch_len
        self.tokens_per_batch = batch_len * batch_size
        self.global_batch_size = batch_size
        self.local_batch_size = safe_divide(self.global_batch_size, world_size)
        self.local_tokens_per_batch = self.local_batch_size * batch_len

        # self.docs_per_stage = safe_divide(tokens_per_stage, self.doc_len)
        # assert self.docs_per_stage >= self.global_batch_size
        # self.batches_per_stage = safe_divide(tokens_per_stage, self.tokens_per_batch)
        self.batch_count = safe_divide(self.total_tokens, self.tokens_per_batch)

        # A group is a sequence of regular (subsequence) batches
        # A group has shape (batch_size, seq_len)
        # while a batch has shape (batch_size, batch_len)
        self.batches_per_group = safe_divide(self.seq_len, batch_len)

        # The array containing the tokens for a stage has shape (docs_per_stage, self.doc_len).
        # Recall a group has shape (batch_size, seq_len)
        # The following counts how many groups we have per row and column in this array
        self.groups_per_column = safe_divide(self.doc_count, self.global_batch_size)

        # Not used for now
        self.groups_per_row = safe_divide(self.doc_len, self.seq_len)
        assert self.groups_per_column * self.groups_per_row * self.batches_per_group == self.batch_count

    def __len__(self):
        # return self.num_stages * self.batches_per_stage
        return self.batch_count

    def __getitem__(self, batch_id: int) -> Batch:
        assert batch_id < len(self)
        # stage = batch_id // self.batches_per_stage
        # assert (
            # stage < self.num_stages
        # ), f"what... \nbatch_id = {batch_id}\nself.__dict__ = {self.__dict__}"
        # batch_id = batch_id % self.batches_per_stage
        # doc_offset = stage * self.docs_per_stage

        group_id = batch_id // self.batches_per_group
        batch_id_in_group = batch_id % self.batches_per_group

        column_index = group_id // self.groups_per_column
        row_index = group_id % self.groups_per_column
        assert row_index < self.groups_per_column
        assert column_index < self.groups_per_row

        start_doc_id = (
            # doc_offset
            row_index * self.global_batch_size
            + self.rank * self.local_batch_size
        )
        start_pos_id = column_index * self.seq_len + batch_id_in_group * self.batch_len

        data = self.dataset[
            start_doc_id : start_doc_id + self.local_batch_size,
            start_pos_id : start_pos_id + self.batch_len,
        ]
        assert data.shape == (self.local_batch_size, self.batch_len)
        # data = data.reshape(-1, reshape_to)

        input_ids = np.array(data, dtype=np.int64)
        input_ids = np.roll(input_ids, 1, axis=-1)

        labels = np.array(data, dtype=np.int64)
        resets = np.zeros_like(data, dtype=np.bool_)

        if batch_id_in_group == 0:
            input_ids[..., 0] = BOS_TOKEN  # bos llama tokenizer
            resets[..., 0] = True
        else:
            raise ValueError("This is not supported yet, so this should never happen")
            # Last token from the last subsequence
            input_ids[..., 0] = self.dataset[
                start_doc_id : start_doc_id + self.local_batch_size, start_pos_id - 1
            ]

        return Batch(input_ids=input_ids, labels=labels, resets=resets)


class StatefulSampler(Sampler):
    def __init__(self, dataset: SlimPajamaDataset):
        super().__init__()
        self.dataset = dataset
        self.batch_id = 0

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        while self.batch_id < len(self):
            # It is crucial to do this to ensure that resuming works properly
            batch_id = self.batch_id
            self.batch_id += 1
            yield batch_id

    def state_dict(self):
        return {"batch_id": self.batch_id}

    def load_state_dict(self, state_dict):
        self.batch_id = state_dict["batch_id"]


class SlimPajamaDataloader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # assert isinstance(self.sampler, StatefulSampler), self.sampler

    def state_dict(self):
        assert isinstance(self.sampler, StatefulSampler)

        return self.sampler.state_dict()

    def load_state_dict(self, state_dict):
        assert isinstance(self.sampler, StatefulSampler)
        self.sampler.load_state_dict(state_dict)
