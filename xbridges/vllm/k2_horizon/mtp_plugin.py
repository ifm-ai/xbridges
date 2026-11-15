"""Registers the K2-Horizon MTP draft (modeling_k2_horizon_mtp_vllm.py) with vLLM at runtime, without changing vLLM's
files, so that a K2-Horizon export with MTP layers serves with
`--speculative-config '{"method": "mtp", "num_speculative_tokens": D}'`.

vLLM finds the MTP draft of a target model in fixed tables; register() adds K2-Horizon to them as an in-tree MTP
model is added:
- SpeculativeConfig.hf_config_override maps the target config (model_type k2_horizon, num_nextn_predict_layers > 0)
  to the draft's: model_type k2_horizon_mtp, architecture K2HorizonMTPModel, n_predict = num_nextn_predict_layers;
- k2_horizon_mtp is added to the MTP model types (MTPModelTypes) and counts num_nextn_predict_layers layers, as
  deepseek_mtp;
- the model registry gets K2HorizonMTPModel, and K2HorizonForCausalLM a vLLM K2-Horizon model that leaves the MTP
  layers of the checkpoint to the draft.

Importing this module registers. The draft's model config keeps hf_config_override below as its hf_overrides, so
vLLM's engine and worker processes import this module, and register, when they receive the config (spawned and Ray
processes too), as long as xBridges is on their PYTHONPATH.
"""
import typing

from vllm import ModelRegistry
from vllm.config import speculative

MTP_MODEL_TYPE = 'k2_horizon_mtp'
MTP_ARCHITECTURE = 'K2HorizonMTPModel'
MODELING = 'xbridges.vllm.k2_horizon.modeling_k2_horizon_mtp_vllm'

# vLLM's SpeculativeConfig.hf_config_override, which hf_config_override falls back to
_vllm_hf_config_override = None


def hf_config_override(hf_config):
    """SpeculativeConfig.hf_config_override, which also maps a K2-Horizon target config to its MTP draft's."""
    if hf_config.model_type == 'k2_horizon' and getattr(hf_config, 'num_nextn_predict_layers', 0) > 0:
        hf_config.model_type = MTP_MODEL_TYPE
        hf_config.update({'n_predict': hf_config.num_nextn_predict_layers, 'architectures': [MTP_ARCHITECTURE]})
        return hf_config
    return _vllm_hf_config_override(hf_config)


hf_config_override.k2_horizon_mtp = True


def register():
    """Registers the K2-Horizon MTP draft with vLLM; does nothing when it already is."""
    global _vllm_hf_config_override
    config_cls = speculative.SpeculativeConfig
    if getattr(config_cls.hf_config_override, 'k2_horizon_mtp', False):
        return
    _vllm_hf_config_override = config_cls.hf_config_override
    config_cls.hf_config_override = staticmethod(hf_config_override)
    if MTP_MODEL_TYPE not in typing.get_args(speculative.MTPModelTypes):
        speculative.MTPModelTypes = typing.Literal[(*typing.get_args(speculative.MTPModelTypes), MTP_MODEL_TYPE)]
    from vllm.transformers_utils import model_arch_config_convertor as convertor
    convertor.MODEL_ARCH_CONFIG_CONVERTORS.setdefault(MTP_MODEL_TYPE, convertor.DeepSeekMTPModelArchConfigConvertor)
    ModelRegistry.register_model(MTP_ARCHITECTURE, f'{MODELING}:K2HorizonMTP')
    ModelRegistry.register_model('K2HorizonForCausalLM', f'{MODELING}:K2HorizonForCausalLM')


register()
