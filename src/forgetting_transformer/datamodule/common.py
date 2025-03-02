from typing import Optional, NamedTuple
import torch
from forgetting_transformer.utils import safe_divide
from dataclasses import dataclass, field


@dataclass
class DataInfo:
    vocab_size: int
    global_tokens_per_batch: int
    local_tokens_per_batch: int
    # tokens_per_stage: int
    batch_len: int
    seq_len: Optional[int]
    total_tokens: int
    global_batch_size: int = field(init=False)
    local_batch_size: int = field(init=False)
    """General dataloader information

    Each local batch has shape (local_batch_size, batch_len)

    Arguments:
        - `tokens_per_stage`: the following should always be true: as long as
          two dataloaders
            - use the same data source
            - have the same tokens_per_stage
           Then within each stage, the set of tokens they emit must be the same, even
           though the order these tokens are emitted are different.
        - `seq_len`: if None, the sequences are variable length. Otherwise all
          sequences should have the same length. The practical implication is
          that resets should either all be False, or only the first timestep is
          True.

    """
    def __post_init__(self):
        self.global_batch_size = safe_divide(self.global_tokens_per_batch, self.batch_len)
        self.local_batch_size = safe_divide(self.local_tokens_per_batch, self.batch_len)


class Batch(NamedTuple):
    input_ids: torch.LongTensor
    labels: torch.LongTensor
    resets: torch.BoolTensor
