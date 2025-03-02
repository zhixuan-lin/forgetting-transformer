from . import DataModuleConfig
from omegaconf import MISSING
from dataclasses import dataclass
from typing import Optional

@dataclass
class LongCrawl64Config(DataModuleConfig):
    _target_: str =  'forgetting_transformer.datamodule.longcrawl64.LongCrawl64DataModule'
    # This is a custom resolver. Note inside data_dir refers to the root config node
    data_dir: str = '${join_path:${data_dir},longcrawl64}'
    rank: int = MISSING        # Should be provided programmatically
    world_size: int = MISSING  # Should be provided programmatically
    train_seq_len: Optional[int] = None
    train_batch_len: int = MISSING
    train_batch_size: int = MISSING
    # train_tokens_per_stage: int = MISSING
    train_doc_len: Optional[int] = None
    train_num_workers: int = MISSING

    eval_tokens: int = MISSING
    eval_seq_len: Optional[int] = None
    eval_batch_len: int = MISSING
    eval_local_batch_size: int = MISSING
    eval_doc_len: Optional[int] = None
    eval_num_workers: int = MISSING
