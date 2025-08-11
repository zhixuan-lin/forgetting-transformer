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
