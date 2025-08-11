import importlib
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings(action="ignore", message="Flash Attention is not installed")
    warnings.filterwarnings(action="ignore", message="`torch.cuda.amp")
    for model in ["mamba2", "forgetting_transformer", "transformer", "delta_net", "hgrn2", "samba"]:
        # We do not want to espose the names.
            importlib.import_module(f".{model}", __name__)
