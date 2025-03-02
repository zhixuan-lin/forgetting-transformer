# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_mamba2 import Mamba2Config
from .modeling_mamba2 import Mamba2ForCausalLM, Mamba2Model

AutoConfig.register(Mamba2Config.model_type, Mamba2Config, True)
AutoModel.register(Mamba2Config, Mamba2Model, True)
AutoModelForCausalLM.register(Mamba2Config, Mamba2ForCausalLM, True)


__all__ = ['Mamba2Config', 'Mamba2ForCausalLM', 'Mamba2Model']
