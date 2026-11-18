"""Multi-token prediction (MTP) weights of an xLLM checkpoint (xLLM model.mtp_depth = D > 0) in the HF export.

xLLM saves MTP module k (1-based) as mtp.layers.{k-1}.{embed_norm,hidden_norm,proj}, its Transformer block as
mtp.layers.{k-1}.block.<the keys of a decoder layer>, and its final norm as output.mtp_norms.{k-1}. They are exported
as DeepSeek-V3's MTP layers are stored, as one more decoder layer model.layers.{N+k-1} (N = num_hidden_layers):
the block under its usual names, enorm (embedding norm), hnorm (hidden-state norm), eh_proj and shared_head.norm (final
norm); the embedding and the output head are shared and not repeated. config.num_nextn_predict_layers = D.

eh_proj maps [embedding; hidden state] to the hidden size; xLLM's projection takes [hidden state; embedding], so its
two input halves are swapped. As in vLLM and SGLang, module 1 takes the main model's last hidden state after its
final norm and module k > 1 module k-1's output after module k-1's final norm. The HF model does not use them.
"""
import re

import torch

# the module's own weights: xLLM key, HF name in the MTP layer, TP split dim of the xLLM tensor
MTP_PARAMS = [
    ('mtp.layers.{m}.embed_norm.weight', 'enorm.weight', -1),
    ('mtp.layers.{m}.hidden_norm.weight', 'hnorm.weight', -1),
    ('mtp.layers.{m}.proj.weight', 'eh_proj.weight', -2),
    ('output.mtp_norms.{m}.weight', 'shared_head.norm.weight', -1),
]


def is_mtp_key(key):
    return key.startswith('mtp.') or key.startswith('output.mtp_norms.')


def mtp_depth(keys):
    """The number of MTP modules in a set of xLLM keys."""
    return len({int(m) for m in re.findall(r'^mtp\.layers\.(\d+)\.', '\n'.join(keys), re.M)})


def pop_mtp_state_dict(xllm_state_dict):
    """Remove the MTP tensors from an xLLM state dict and return them."""
    return {key: xllm_state_dict.pop(key) for key in [key for key in xllm_state_dict if is_mtp_key(key)]}


def _block_as_layer(mtp_state_dict, m, layer):
    """Module m's block tensors under the names of decoder layer `layer`."""
    prefix = f'mtp.layers.{m}.block.'
    return {f'layers.{layer}.' + key[len(prefix):]: value
            for key, value in mtp_state_dict.items() if key.startswith(prefix)}


def merge_mtp_state_dicts(state_dict, tp_state_dict, hf_config):
    """Merge the MTP tensors of the next tensor-parallel rank into state_dict, as merge_state_dicts does the rest:
    the blocks with the rules of the last decoder layer, of whose kind they are."""
    from xbridges.huggingface.xllm_to_hf_main import merge_state_dicts

    last = hf_config.num_hidden_layers - 1
    for m in range(mtp_depth(state_dict.keys())):
        block = merge_state_dicts(_block_as_layer(state_dict, m, last), _block_as_layer(tp_state_dict, m, last),
                                  hf_config.num_hidden_layers, hf_config)
        for key, value in block.items():
            state_dict[f'mtp.layers.{m}.block.' + key[len(f'layers.{last}.'):]] = value
        for xllm_key, _, split_dim in MTP_PARAMS:
            xllm_key = xllm_key.format(m=m)
            state_dict[xllm_key] = torch.cat([state_dict[xllm_key], tp_state_dict[xllm_key]], dim=split_dim)
    return state_dict


def convert_mtp_state_dict_xllm_to_hf(hf_state_dict, hf_config, mtp_state_dict):
    """The HF tensors of the MTP modules (layers N, ..., N+D-1) from the merged xLLM MTP tensors."""
    from xbridges.huggingface.xllm_to_hf_main import convert_state_dict_xllm_to_hf

    num_layers, last, hidden_size = hf_config.num_hidden_layers, hf_config.num_hidden_layers - 1, hf_config.hidden_size
    depth = mtp_depth(mtp_state_dict.keys())
    if depth != hf_config.num_nextn_predict_layers:
        raise ValueError(f'{depth} MTP modules in the checkpoint, config num_nextn_predict_layers = '
                         f'{hf_config.num_nextn_predict_layers}')
    state_dict, consumed = {}, set()
    for m in range(depth):
        layer = num_layers + m
        # the block, converted as the last decoder layer (same kind), then renamed
        block = convert_state_dict_xllm_to_hf(
            hf_state_dict, hf_config, _block_as_layer(mtp_state_dict, m, last), layers=[last], include_output=False)
        for key, value in block.items():
            state_dict[key.replace(f'model.layers.{last}.', f'model.layers.{layer}.', 1)] = value
        consumed.update(key for key in mtp_state_dict if key.startswith(f'mtp.layers.{m}.block.'))
        for xllm_key, hf_name, _ in MTP_PARAMS:
            xllm_key = xllm_key.format(m=m)
            weight = mtp_state_dict[xllm_key]
            consumed.add(xllm_key)
            if hf_name == 'eh_proj.weight':
                if tuple(weight.shape) != (hidden_size, 2 * hidden_size):
                    raise ValueError(f'{xllm_key}: expected shape {(hidden_size, 2 * hidden_size)}, '
                                     f'got {tuple(weight.shape)}')
                # xLLM: [hidden state; embedding] -> HF / DeepSeek: [embedding; hidden state]
                weight = torch.cat([weight[:, hidden_size:], weight[:, :hidden_size]], dim=1)
            else:
                if tuple(weight.shape) != (hidden_size,):
                    raise ValueError(f'{xllm_key}: expected shape {(hidden_size,)}, got {tuple(weight.shape)}')
                # xLLM's RMSNorm weights are stored minus 1, as the other norms
                weight = weight + 1.
            state_dict[f'model.layers.{layer}.{hf_name}'] = weight
    unused = sorted(mtp_state_dict.keys() - consumed)
    if unused:
        raise ValueError(f'Incomplete MTP conversion: unused_source={unused}')
    return state_dict
