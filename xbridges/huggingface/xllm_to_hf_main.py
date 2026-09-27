import gc
import fire
import json
import tqdm
import math
import shutil
from pathlib import Path
import torch
from torch.distributed.checkpoint import FileSystemReader, load_state_dict
from safetensors.torch import save_file
import transformers
from transformers import AutoTokenizer

from xbridges.huggingface.k2_horizon.configuration_k2_horizon import K2HorizonConfig
from xbridges.huggingface.k2_horizon.modeling_k2_horizon import K2HorizonForCausalLM


ATTENTION_PARAMS = [
    ["layers.{layer}.attention.wq.weight", "model.layers.{layer}.self_attn.q_proj.weight", -2],
    ["layers.{layer}.attention.wk.weight", "model.layers.{layer}.self_attn.k_proj.weight", -2],
    ["layers.{layer}.attention.wv.weight", "model.layers.{layer}.self_attn.v_proj.weight", -2],
    ["layers.{layer}.attention.query_norm.weight", "model.layers.{layer}.self_attn.q_norm.weight", -1],
    ["layers.{layer}.attention.key_norm.weight", "model.layers.{layer}.self_attn.k_norm.weight", -1],
    ["layers.{layer}.attention.wr.weight", "model.layers.{layer}.self_attn.gate_proj.weight", -2],
    ["layers.{layer}.attention.wo.weight", "model.layers.{layer}.self_attn.o_proj.weight", -1],
]
MOVA_ATTENTION_PARAMS = [
    ["layers.{layer}.mova.wq.weight", "model.layers.{layer}.self_attn.q_proj.weight", -2],
    ["layers.{layer}.mova.wk.weight", "model.layers.{layer}.self_attn.k_proj.weight", -2],
    ["layers.{layer}.mova.query_norm.weight", "model.layers.{layer}.self_attn.q_norm.weight", -1],
    ["layers.{layer}.mova.key_norm.weight", "model.layers.{layer}.self_attn.k_norm.weight", -1],
    ["layers.{layer}.mova.wo.weight", "model.layers.{layer}.self_attn.o_proj.weight", -1],
    ["layers.{layer}.mova.router.weight", "model.layers.{layer}.self_attn.v_router.weight", -1],
    ["layers.{layer}.mova.router.bias", "model.layers.{layer}.self_attn.v_router.bias", None],
    ["layers.{layer}.mova.wv.weight", "model.layers.{layer}.self_attn.v_experts.{expert}.weight", -1],
    ["layers.{layer}.mova.wr.weight", "model.layers.{layer}.self_attn.gate_proj.weight", -2],
]
DENSE_PARAMS = [
    ["layers.{layer}.norm.weight", "model.layers.{layer}.input_layernorm.weight", -1],
    ["layers.{layer}.nffn.norm.weight", "model.layers.{layer}.post_attention_layernorm.weight", -1],
    ['layers.{layer}.nffn.fc1.weight', 'model.layers.{layer}.mlp.gate_proj.weight', -2],
    ['layers.{layer}.nffn.fc2.weight', 'model.layers.{layer}.mlp.down_proj.weight', -1],
    ['layers.{layer}.nffn.fc3.weight', 'model.layers.{layer}.mlp.up_proj.weight', -2],
]
MOE_PARAMS = [
    ["layers.{layer}.norm.weight", "model.layers.{layer}.input_layernorm.weight", -1],
    ["layers.{layer}.moe.norm.weight", "model.layers.{layer}.post_attention_layernorm.weight", -1],
    ["layers.{layer}.moe.router.weight", "model.layers.{layer}.mlp.gate.weight", -1],
    ["layers.{layer}.moe.router.bias", "model.layers.{layer}.mlp.gate.bias", None],
    ["layers.{layer}.moe.experts.weight1", "model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight", -3],
    ["layers.{layer}.moe.experts.weight2", "model.layers.{layer}.mlp.experts.{expert}.down_proj.weight", -3],
    ["layers.{layer}.moe.experts.weight3", "model.layers.{layer}.mlp.experts.{expert}.up_proj.weight", -3],
    ['layers.{layer}.moe.fc1.weight', 'model.layers.{layer}.mlp.shared_experts.gate_proj.weight', -1],
    ['layers.{layer}.moe.fc2.weight', 'model.layers.{layer}.mlp.shared_experts.down_proj.weight', -2],
    ['layers.{layer}.moe.fc3.weight', 'model.layers.{layer}.mlp.shared_experts.up_proj.weight', -1],
]
OTHER_PARAMS = [
    ['embed.weight', 'model.embed_tokens.weight', -1],
    ['output.final_norm.weight', 'model.norm.weight', -1],
    ['output.output.weight', 'lm_head.weight', -2]
]


def get_total_params(model):
    return sum(param.numel() for param in model.state_dict().values())


# permute for sliced rotary
def permute_qk(w, n_heads, dim1, dim2):
    if w.shape != (dim1, dim2):
        print(w.shape, dim1, dim2)
        raise ValueError
    return w.reshape(n_heads, dim1 // n_heads // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2)


def permute_qknorm(w, n_heads, head_dim):
    return w.reshape(n_heads, head_dim // 2, 2).transpose(1, 2).reshape(-1)


def _new_attention_norm_key(layer, hf_config):
    module = (
        'mova'
        if layer not in hf_config.mlp_only_layers and hf_config.mova_num_experts > 0
        else 'attention'
    )
    return f'layers.{layer}.{module}.norm.weight'


def _resolve_attention_norm_key(key, state_dict, layer, hf_config):
    """Accept block-level norms from before and after commit 5b9cf866."""
    legacy = f'layers.{layer}.norm.weight'
    if key != legacy:
        return key
    new = _new_attention_norm_key(layer, hf_config)
    present = [candidate for candidate in (legacy, new) if candidate in state_dict]
    if len(present) > 1:
        raise ValueError(
            f'Checkpoint contains both legacy and new attention norm keys: {present}'
        )
    return present[0] if present else legacy


def convert_config_xllm_to_hf(xllm_config, tokenizer):
    assert xllm_config['model']['apply_rmsnorm'] == True
    assert xllm_config['model']['swiglu'] == True
    model = xllm_config['model']
    if model.get('residual_func', 'base') not in ('base', 'add'):
        raise ValueError('IFM K2Horizon only supports additive residuals')
    if model.get('scale_emb', False):
        raise ValueError('IFM K2Horizon does not implement scale_emb')
    if model.get('attn_act_func', 'softmax') != 'softmax':
        raise ValueError('IFM K2Horizon requires softmax attention')
    if model.get('output_size', -1) not in (-1, model['vocab_size']):
        raise ValueError('IFM K2Horizon requires output_size == vocab_size')
    for option in ('two_hop_residual', 'rescale_nffn', 'causal_norm_weight'):
        if model.get(option, False):
            raise ValueError(f'IFM K2Horizon does not implement {option}')

    if xllm_config['model'].get('head_dim', None) is not None:
        head_dim = xllm_config['model']['head_dim']
    else:
        head_dim = xllm_config['model']['model_dim'] // xllm_config['model']['num_heads']

    if model['num_experts'] == 0:
        mlp_only_layers = list(range(xllm_config['model']['num_layers']))
    else:
        mlp_only_layers = list(range(model.get('num_dense_layers') or 0))
    if model.get('v_head_dim') not in (None, head_dim):
        raise ValueError('IFM K2Horizon requires v_head_dim == head_dim')

    if xllm_config['model'].get('apply_attn_gate', False):
        attention_gate_func = xllm_config['model']['attn_gate_func']
    else:
        attention_gate_func = None

    # assert xllm_config['model']['rope_head_dim'] == None
    # assert xllm_config['model']['moe_router_score_func'] == 'softmax'
    # assert xllm_config['model']['moe_router_scaling_factor'] == 1.0
    # assert xllm_config['model']['num_shared_experts'] == 0
    # assert xllm_config['model']['layernorm_num_groups'] == 1
    # assert xllm_config['model']['moe_router_bias'] == False
    # assert xllm_config['model']['num_dense_layers'] == 0
    # assert xllm_config['model']['qknorm'] == True

    return K2HorizonConfig(
        transformers_version=transformers.__version__,
        architectures=['K2HorizonForCausalLM'],
        auto_map={
            "AutoConfig": "configuration_k2_horizon.K2HorizonConfig",
            "AutoModel": "modeling_k2_horizon.K2HorizonModel",
            "AutoModelForCausalLM": "modeling_k2_horizon.K2HorizonForCausalLM"
        },
        vocab_size=xllm_config['model']['vocab_size'],
        num_hidden_layers=xllm_config['model']['num_layers'],
        mlp_only_layers=mlp_only_layers,
        hidden_size=xllm_config['model']['model_dim'],
        num_attention_heads=xllm_config['model']['num_heads'],
        num_key_value_heads=model.get('num_kv_heads') or model['num_heads'],
        head_dim=head_dim,
        rope_parameters={
            'rope_theta': xllm_config['model']['rope_base'],
            'rope_type': 'default',
        },
        intermediate_size=xllm_config['model']['ffn_hidden_dim'],
        num_experts=xllm_config['model']['num_experts'],
        num_experts_per_tok=xllm_config['model']['num_activated_experts'],
        moe_intermediate_size=xllm_config['model']['expert_inter_dim'],
        rms_norm_eps=model['rmsnorm_eps'] if 'rmsnorm_eps' in model else model['norm_eps'],
        max_position_embeddings=xllm_config['seq_len'],
        initializer_range=(model['init_std'] if model.get('init_std') is not None
                           else K2HorizonConfig.initializer_range),
        attention_dropout=xllm_config['model']['attention_dropout'],
        query_key_norm=xllm_config['model']['qknorm'],
        moe_gate_bias=xllm_config['model']['moe_router_bias'],
        layernorm_num_groups=xllm_config['model']['layernorm_num_groups'],
        num_shared_experts=xllm_config['model']['num_shared_experts'],
        router_score_func=xllm_config['model']['moe_router_score_func'],
        router_aux_loss_coef=xllm_config.get('moe_aux_loss_coeff',
                                             K2HorizonConfig.router_aux_loss_coef),
        router_scaling_factor=xllm_config['model']['moe_router_scaling_factor'],
        rope_head_dim=xllm_config['model']['rope_head_dim'],
        attention_gate_func=attention_gate_func,
        mova_num_experts=xllm_config['model'].get('num_values', 0),
        mova_num_experts_per_tok=xllm_config['model'].get('num_activated_values', 0),
        xllm_model_parallel_size=xllm_config['model_parallel_size'],
        norm_topk_prob=model['num_experts'] == 0 or model['num_activated_experts'] > 1,
        dtype='float32',
        hidden_act="silu",
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        tie_word_embeddings=False,
        attention_bias=False)


def expected_hf_keys(hf_state_dict, layers, include_output):
    layers = set(layers)
    return {
        key for key in hf_state_dict
        if (int(key.split('.')[2]) in layers if key.startswith('model.layers.') else include_output)
    }


def _restore_expert_axis(key, weight, hf_config):
    """Restore flattened native experts before TP merging or HF expert splitting."""
    is_moe = '.moe.experts.' in key
    if not is_moe and not key.endswith('.mova.wv.weight'):
        return weight
    if weight.ndim not in (2, 3):
        raise ValueError(f'{key}: expected a 2D or 3D expert tensor, got {tuple(weight.shape)}')
    if is_moe:
        rows, cols = hf_config.moe_intermediate_size, hf_config.hidden_size
        if key.endswith('.weight2'):
            rows, cols = cols, rows
        shape = (weight.numel() // (rows * cols), rows, cols)
    else:
        shape = (hf_config.mova_num_experts,
                 hf_config.num_key_value_heads * hf_config.head_dim, weight.shape[-1])
    flat_shape = (shape[0] * shape[1], shape[2])
    if min(shape) < 1 or tuple(weight.shape) not in (shape, flat_shape):
        raise ValueError(f'{key}: expected expert shape {shape} or {flat_shape}, got {tuple(weight.shape)}')
    return weight.reshape(shape)


def convert_state_dict_xllm_to_hf(
        hf_state_dict, hf_config, xllm_state_dict, layers=None, include_output=True):
    num_layers = hf_config.num_hidden_layers
    num_experts = hf_config.num_experts
    hidden_size = hf_config.hidden_size
    num_attention_heads = hf_config.num_attention_heads
    num_key_value_heads = hf_config.num_key_value_heads
    head_dim = hf_config.head_dim
    mlp_only_layers = hf_config.mlp_only_layers
    mova_num_experts = hf_config.mova_num_experts
    layers = tuple(range(num_layers) if layers is None else layers)
    consumed = set()

    state_dict = {}
    for layer in layers:
        if layer in mlp_only_layers:
            layer_params = ATTENTION_PARAMS + DENSE_PARAMS
        elif hf_config.mova_num_experts > 0:
            layer_params = MOVA_ATTENTION_PARAMS + MOE_PARAMS
        else:
            layer_params = ATTENTION_PARAMS + MOE_PARAMS

        for xllm_key, hf_key, _ in layer_params:
            xllm_key = xllm_key.format(layer=layer)
            xllm_key = _resolve_attention_norm_key(xllm_key, xllm_state_dict, layer, hf_config)
            if '{expert}' not in hf_key:
                hf_key = hf_key.format(layer=layer)

            if xllm_key not in xllm_state_dict:
                continue
            consumed.add(xllm_key)

            if xllm_key.endswith('.norm.weight'):
                state_dict[hf_key] = xllm_state_dict[xllm_key] + 1.
            elif xllm_key.endswith('.query_norm.weight'):
                state_dict[hf_key] = permute_qknorm(
                    xllm_state_dict[xllm_key],
                    n_heads=num_attention_heads,
                    head_dim=head_dim) + 1.
            elif xllm_key.endswith('.key_norm.weight'):
                state_dict[hf_key] = permute_qknorm(
                    xllm_state_dict[xllm_key],
                    n_heads=num_key_value_heads,
                    head_dim=head_dim) + 1.
            elif xllm_key.endswith('.wq.weight'):
                state_dict[hf_key] = permute_qk(
                    xllm_state_dict[xllm_key],
                    n_heads=num_attention_heads,
                    dim1=num_attention_heads * head_dim,
                    dim2=hidden_size)
            elif xllm_key.endswith('.wk.weight'):
                state_dict[hf_key] = permute_qk(
                    xllm_state_dict[xllm_key],
                    n_heads=num_key_value_heads,
                    dim1=num_key_value_heads * head_dim,
                    dim2=hidden_size)
            elif xllm_key.endswith('.mova.wv.weight'):
                weight = _restore_expert_axis(xllm_key, xllm_state_dict[xllm_key], hf_config)
                expected_shape = (mova_num_experts, num_key_value_heads * head_dim, hidden_size)
                if tuple(weight.shape) != expected_shape:
                    raise ValueError(f'{xllm_key}: expected native value-expert shape {expected_shape}')
                for expert in range(mova_num_experts):
                    state_dict[hf_key.format(layer=layer, expert=expert)] = \
                        weight[expert]
            elif '.moe.experts.' in xllm_key:
                weight = _restore_expert_axis(xllm_key, xllm_state_dict[xllm_key], hf_config)
                expected_shape = (num_experts, *hf_state_dict[hf_key.format(layer=layer, expert=0)].shape)
                if tuple(weight.shape) != expected_shape:
                    raise ValueError(f'{xllm_key}: expected native expert shape {expected_shape}')
                for expert in range(num_experts):
                    state_dict[hf_key.format(layer=layer, expert=expert)] = \
                        weight[expert]
            else:
                state_dict[hf_key] = xllm_state_dict[xllm_key]

    for xllm_key, hf_key, _ in OTHER_PARAMS if include_output else []:
        if xllm_key not in xllm_state_dict:
            continue
        consumed.add(xllm_key)
        if xllm_key.endswith('.final_norm.weight'):
            state_dict[hf_key] = xllm_state_dict[xllm_key] + 1.
        else:
            state_dict[hf_key] = xllm_state_dict[xllm_key]

    expected = expected_hf_keys(hf_state_dict, layers, include_output)
    missing = sorted(expected - state_dict.keys())
    unexpected = sorted(state_dict.keys() - expected)
    unused = sorted(xllm_state_dict.keys() - consumed)
    if missing or unexpected or unused:
        raise ValueError(f'Incomplete conversion: missing={missing}, unexpected={unexpected}, unused_source={unused}')
    for key, param in state_dict.items():
        if param.shape != hf_state_dict[key].shape:
            raise ValueError(f'{key}: source shape {tuple(param.shape)} != target shape {tuple(hf_state_dict[key].shape)}')

    return state_dict


def merge_state_dicts(state_dict, tp_state_dict, num_layers, hf_config):
    if state_dict.keys() != tp_state_dict.keys():
        raise ValueError('Tensor-parallel ranks contain different checkpoint keys')
    for layer in range(num_layers):
        if layer in hf_config.mlp_only_layers:
            layer_params = ATTENTION_PARAMS + DENSE_PARAMS
        elif hf_config.mova_num_experts > 0:
            layer_params = MOVA_ATTENTION_PARAMS + MOE_PARAMS
        else:
            layer_params = ATTENTION_PARAMS + MOE_PARAMS

        for xllm_key, _, split_dim in layer_params:
            xllm_key = xllm_key.format(layer=layer)
            xllm_key = _resolve_attention_norm_key(xllm_key, state_dict, layer, hf_config)
            if xllm_key in state_dict and split_dim is not None:
                state_dict[xllm_key] = torch.cat(
                    [_restore_expert_axis(xllm_key, state_dict[xllm_key], hf_config),
                     _restore_expert_axis(xllm_key, tp_state_dict[xllm_key], hf_config)],
                    dim=split_dim)
            elif xllm_key in state_dict and not torch.equal(state_dict[xllm_key], tp_state_dict[xllm_key]):
                raise ValueError(f'Tensor-parallel replicas differ for {xllm_key}')

    for xllm_key, _, split_dim in OTHER_PARAMS:
        if xllm_key in state_dict and split_dim is not None:
            state_dict[xllm_key] = torch.cat(
                [state_dict[xllm_key], tp_state_dict[xllm_key]], dim=split_dim)

    return state_dict


def load_xllm_state_dict(xllm_dir, hf_config, layer_l, layer_r):
    print(f'Loading layers {layer_l} to {layer_r - 1} ...')

    if not 0 <= layer_l < layer_r <= hf_config.num_hidden_layers:
        raise ValueError(f'Invalid layer range [{layer_l}, {layer_r})')
    tp_size = json.load(open(f'{xllm_dir}/config.json'))['model_parallel_size']
    if tp_size < 1:
        raise ValueError('model_parallel_size must be positive')
    checkpoints = []
    expected_keys = None
    expected_layers = set(range(hf_config.num_hidden_layers))
    # Check every rank's complete metadata before filtering or allocating tensors.
    for tp_idx in range(tp_size):
        tp_folder = f'{xllm_dir}/sharded_model.tp{tp_idx:02d}'
        reader = FileSystemReader(tp_folder)
        metadata = reader.read_metadata()
        keys = set(metadata.state_dict_metadata)
        layer_indices = set()
        for key in keys:
            if key.startswith('layers.'):
                try:
                    layer_indices.add(int(key.split('.')[1]))
                except ValueError as error:
                    raise ValueError(f'{tp_folder}: invalid layer key {key!r}') from error
        if layer_indices != expected_layers:
            raise ValueError(
                f'{tp_folder}: checkpoint layer indices disagree with config: '
                f'missing={sorted(expected_layers - layer_indices)}, '
                f'unexpected={sorted(layer_indices - expected_layers)}')
        if expected_keys is not None and keys != expected_keys:
            raise ValueError(
                f'Tensor-parallel ranks contain different checkpoint keys: {tp_folder}: '
                f'missing={sorted(expected_keys - keys)}, unexpected={sorted(keys - expected_keys)}')
        expected_keys = keys
        checkpoints.append((reader, metadata))

    xllm_state_dict = {}
    for tp_idx, (reader, metadata) in enumerate(tqdm.tqdm(checkpoints, desc='Loading XLLM model')):
        sd = {}
        # Allocate only the requested layers, not the entire checkpoint per load.
        for key, tensor in metadata.state_dict_metadata.items():
            if key.startswith('layers.'):
                layer_idx = int(key.split('.')[1])
                selected = layer_l <= layer_idx < layer_r
            else:
                selected = layer_r == hf_config.num_hidden_layers
            if selected:
                sd[key] = torch.empty(tuple(tensor.size), dtype=tensor.properties.dtype)
        load_state_dict(sd, storage_reader=reader, no_dist=True)
        tp_state_dict = sd
        del sd

        if tp_idx == 0:
            xllm_state_dict = tp_state_dict
        else:
            xllm_state_dict = merge_state_dicts(
                state_dict=xllm_state_dict,
                tp_state_dict=tp_state_dict,
                num_layers=hf_config.num_hidden_layers,
                hf_config=hf_config)

        del tp_state_dict
        gc.collect()

    return xllm_state_dict


def main(xllm_dir='ckpts/xllm-30b-a3b',
         tokenizer_dir='Qwen/Qwen3-30B-A3B',
         save_dir='ckpts/hf-30b-a3b',
         layers_per_load=None,
         dtype='bfloat16',
         safe_serialization=True):
    if dtype not in ('float32', 'bfloat16', 'float16'):
        raise ValueError(f'Unsupported output dtype: {dtype}')
    if layers_per_load is not None and layers_per_load < 1:
        raise ValueError('layers_per_load must be positive')
    output_dtype = getattr(torch, dtype)
    destination = Path(save_dir)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f'Refusing to overwrite nonempty output directory: {save_dir}')
    hf_tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir, trust_remote_code=True)
    hf_config = convert_config_xllm_to_hf(
        xllm_config=json.load(open(f'{xllm_dir}/config.json')),
        tokenizer=hf_tokenizer)
    hf_config.dtype = output_dtype
    with torch.device('meta'):
        hf_model = K2HorizonForCausalLM(config=hf_config)
    print(f'{hf_config=}')
    print(f'{hf_model=}')
    print(f'{get_total_params(hf_model)=:,}')
    print('=' * 100)

    print(f'Saving HF Model to {save_dir} ...')
    hf_config.save_pretrained(save_dir)
    hf_tokenizer.save_pretrained(save_dir)
    for source in Path(__file__).with_name('k2_horizon').glob('*.py'):
        shutil.copyfile(source, destination / source.name)

    if layers_per_load is None:
        layers_per_load = hf_config.num_hidden_layers
    n_load_times = int(math.ceil(hf_config.num_hidden_layers / layers_per_load))
    print(f'{n_load_times=}')

    weight_map = {}
    total_size = 0
    for load_idx in tqdm.trange(n_load_times, desc='Saving'):
        layer_l = layers_per_load * load_idx
        layer_r = min(layers_per_load * (load_idx + 1), hf_config.num_hidden_layers)
        xllm_state_dict = load_xllm_state_dict(
            xllm_dir=xllm_dir,
            hf_config=hf_config,
            layer_l=layer_l,
            layer_r=layer_r)

        print(f'Converting XLLM state_dict to HF format...')
        state_dict = convert_state_dict_xllm_to_hf(
            hf_state_dict=hf_model.state_dict(),
            hf_config=hf_model.config,
            xllm_state_dict=xllm_state_dict,
            layers=range(layer_l, layer_r),
            include_output=layer_r == hf_config.num_hidden_layers)

        del xllm_state_dict
        gc.collect()

        for layer_idx in range(layer_l, layer_r):
            layer_sd = {
                key: param for key, param in state_dict.items()
                if f'layers.{layer_idx}.' in key
            }
            if layer_idx == hf_config.num_hidden_layers - 1:
                layer_sd.update({
                    key: param for key, param in state_dict.items()
                    if 'model.layers.' not in key
                })

            layer_sd = {key: value.to(output_dtype).contiguous() for key, value in layer_sd.items()}
            total_size += sum(value.numel() * value.element_size() for value in layer_sd.values())
            if safe_serialization:
                filename = f'model-{layer_idx+1:05d}-of-{hf_config.num_hidden_layers:05d}.safetensors'
                save_file(layer_sd, f'{save_dir}/{filename}', metadata={'format': 'pt'})
            else:
                filename = f'pytorch_model-{layer_idx+1:05d}-of-{hf_config.num_hidden_layers:05d}.bin'
                torch.save(layer_sd, f'{save_dir}/{filename}')
            for key in layer_sd:
                weight_map[key] = filename
            print(f'{save_dir}/{filename} written.')

            del layer_sd
            gc.collect()

        del state_dict
        gc.collect()

    if weight_map.keys() != hf_model.state_dict().keys():
        raise ValueError('Exported index does not cover the complete HF model')
    index_name = 'model.safetensors.index.json' if safe_serialization else 'pytorch_model.bin.index.json'
    json.dump({
        'metadata': {'total_size': total_size},
        'weight_map': weight_map
    }, open(f'{save_dir}/{index_name}', 'w'), indent=4)


if __name__ == '__main__':
    fire.Fire(main)
