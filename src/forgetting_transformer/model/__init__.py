# from .mamba2 import Mamba2ForCausalLM, Mamba2Config
# from .forgetting_transformer import ForgettingTransformerForCausalLM, ForgettingTransformerConfig
# from .transformer import TransformerForCausalLM, TransformerConfig
# from .delta_net import DeltaNetForCausalLM, DeltaNetConfig
# from .hgrn2 import HGRN2ForCausalLM, HGRN2Config
# from .samba import SambaForCausalLM, SambaConfig

import importlib
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings(action="ignore", message="Flash Attention is not installed")
    warnings.filterwarnings(action="ignore", message="`torch.cuda.amp")
    from forgetting_transformer.model.forgetting_transformer import (
        ForgettingTransformerForCausalLM,
        ForgettingTransformerConfig,
    )
    from forgetting_transformer.model.forgetting_transformer.modeling_forgetting_transformer import (
        ForgettingAttentionLayer
    )

    for model in ["mamba2", "forgetting_transformer", "transformer", "delta_net", "hgrn2", "samba"]:
        # We do not want to espose the names.
            importlib.import_module(f".{model}", __name__)
