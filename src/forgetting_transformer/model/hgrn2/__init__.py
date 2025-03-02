# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_hgrn2 import HGRN2Config
from .modeling_hgrn2 import HGRN2ForCausalLM, HGRN2Model

AutoConfig.register(HGRN2Config.model_type, HGRN2Config)
AutoModel.register(HGRN2Config, HGRN2Model)
AutoModelForCausalLM.register(HGRN2Config, HGRN2ForCausalLM)


__all__ = ['HGRN2Config', 'HGRN2ForCausalLM', 'HGRN2Model']
