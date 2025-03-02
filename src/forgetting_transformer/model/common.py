from typing import NamedTuple, Optional, Any
import torch


class LMOutput(NamedTuple):
    loss: torch.Tensor
    carry: Any
    logits: Optional[torch.Tensor] = None
