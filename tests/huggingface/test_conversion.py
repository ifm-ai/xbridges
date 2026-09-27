import json
from types import SimpleNamespace

import pytest
import torch
from torch.distributed.checkpoint import FileSystemWriter, save_state_dict
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

from xbridges.huggingface import hf_to_xllm_main as reverse_bridge
from xbridges.huggingface import xllm_to_hf_main as bridge


def source_config(kind, norm_key='norm_eps'):
    sparse = kind != 'dense'
    return {
        'model_parallel_size': 1, 'seq_len': 32,
        'model': {
            'vocab_size': 32, 'num_layers': 2, 'num_dense_layers': 1 if sparse else None,
            'model_dim': 8, 'num_heads': 4, 'num_kv_heads': 2, 'head_dim': 4,
            'rope_base': 10000, 'rope_head_dim': 2, 'ffn_hidden_dim': 16,
            'num_experts': 4 if sparse else 0, 'num_activated_experts': 2 if sparse else 0,
            'expert_inter_dim': 6 if sparse else 0, 'num_shared_experts': 1 if sparse else 0,
            'num_values': 4 if kind == 'mova' else 0, 'num_activated_values': 2 if kind == 'mova' else 0,
            'apply_rmsnorm': True, 'swiglu': True, 'attention_dropout': 0., norm_key: 1e-6,
            'qknorm': True, 'moe_router_bias': sparse, 'moe_router_score_func': 'sigmoid',
            'moe_router_scaling_factor': 2.5, 'layernorm_num_groups': 2,
            'apply_attn_gate': True, 'attn_gate_func': 'softplus',
        },
    }


def fixture_model(kind, norm_key='norm_eps'):
    torch.manual_seed(42)
    cfg = bridge.convert_config_xllm_to_hf(source_config(kind, norm_key), SimpleNamespace(
        pad_token_id=31, bos_token_id=0, eos_token_id=1))
    cfg._attn_implementation = 'eager'
    model = bridge.K2HorizonForCausalLM(cfg).eval()
    sd = {}
    mappings = []
    for layer in range(2):
        params = (bridge.ATTENTION_PARAMS + bridge.DENSE_PARAMS if layer in cfg.mlp_only_layers else
                  (bridge.MOVA_ATTENTION_PARAMS if kind == 'mova' else bridge.ATTENTION_PARAMS) + bridge.MOE_PARAMS)
        mappings.extend((k.format(layer=layer), v.replace('{layer}', str(layer)), axis) for k, v, axis in params)
    mappings.extend(bridge.OTHER_PARAMS)
    targets = model.state_dict()
    for src, dst, axis in mappings:
        if '{expert}' in dst:
            count = cfg.mova_num_experts if '.mova.' in src else cfg.num_experts
            shape = (count, *targets[dst.format(expert=0)].shape)
        elif dst in targets:
            shape = targets[dst].shape
        else:
            continue
        sd[src] = torch.randn(shape) * .03
    return cfg, model, sd, mappings


@pytest.mark.parametrize('kind', ['dense', 'moe', 'mova'])
@pytest.mark.parametrize('norm_key', ['norm_eps', 'rmsnorm_eps'])
def test_complete_conversion_and_tp_merge(kind, norm_key):
    cfg, model, sd, mappings = fixture_model(kind, norm_key)
    tp0, tp1 = {}, {}
    for src, _, axis in mappings:
        if src not in sd:
            continue
        if axis is None:
            tp0[src] = sd[src].clone()
            tp1[src] = sd[src].clone()
        else:
            tp0[src], tp1[src] = [p.clone() for p in sd[src].chunk(2, dim=axis)]
    merged = bridge.merge_state_dicts(tp0, tp1, 2, cfg)
    for key in sd:
        torch.testing.assert_close(merged[key], sd[key], rtol=0, atol=0)
    result = bridge.convert_state_dict_xllm_to_hf(model.state_dict(), cfg, merged)
    new_sd = dict(sd)
    for layer in range(cfg.num_hidden_layers):
        legacy_key = f'layers.{layer}.norm.weight'
        if legacy_key in new_sd:
            new_sd[bridge._new_attention_norm_key(layer, cfg)] = new_sd.pop(legacy_key)
    axis_by_key = {src: axis for src, _, axis in mappings if src in sd}
    for layer in range(cfg.num_hidden_layers):
        legacy_key = f'layers.{layer}.norm.weight'
        if legacy_key in axis_by_key:
            axis_by_key[bridge._new_attention_norm_key(layer, cfg)] = axis_by_key.pop(legacy_key)
    new_tp0, new_tp1 = {}, {}
    for key, value in new_sd.items():
        axis = axis_by_key[key]
        if axis is None:
            new_tp0[key], new_tp1[key] = value.clone(), value.clone()
        else:
            new_tp0[key], new_tp1[key] = [part.clone() for part in value.chunk(2, dim=axis)]
    new_merged = bridge.merge_state_dicts(new_tp0, new_tp1, 2, cfg)
    for key in new_sd:
        torch.testing.assert_close(new_merged[key], new_sd[key], rtol=0, atol=0)
    new_result = bridge.convert_state_dict_xllm_to_hf(model.state_dict(), cfg, new_merged)
    assert new_result.keys() == result.keys()
    for key in result:
        torch.testing.assert_close(new_result[key], result[key], rtol=0, atol=0)
    model.load_state_dict(result, strict=True)
    torch.testing.assert_close(result['model.layers.0.input_layernorm.weight'], sd['layers.0.norm.weight'] + 1)
    with torch.no_grad():
        out = model(torch.tensor([[0, 4, 7]]), use_cache=True)
        cached = model(torch.tensor([[9]]), past_key_values=out.past_key_values, use_cache=True)
        full = model(torch.tensor([[0, 4, 7, 9]]), use_cache=False)
    assert torch.isfinite(out.logits).all()
    torch.testing.assert_close(cached.logits[:, -1], full.logits[:, -1], atol=1e-5, rtol=1e-5)


def test_sparse_without_dense_prefix():
    config = source_config('mova')
    config['model']['num_dense_layers'] = None
    cfg = bridge.convert_config_xllm_to_hf(config, SimpleNamespace(pad_token_id=31, bos_token_id=0, eos_token_id=1))
    assert cfg.mlp_only_layers == []


def test_top_one_router_keeps_native_probability():
    config = source_config('moe')
    config['model']['num_activated_experts'] = 1
    cfg = bridge.convert_config_xllm_to_hf(config, SimpleNamespace(pad_token_id=31, bos_token_id=0, eos_token_id=1))
    assert cfg.norm_topk_prob is False


@pytest.mark.parametrize('kind', ['dense', 'moe', 'mova'])
@pytest.mark.parametrize('eps_fields', [
    {'norm_eps': 3e-5},
    {'rmsnorm_eps': 3e-5, 'layernorm_eps': 1e-4},
    {'rmsnorm_eps': 3e-5, 'norm_eps': 1e-6},
])
def test_rms_norm_epsilon_follows_checkpoint_config(kind, eps_fields):
    config = source_config(kind)
    del config['model']['norm_eps']
    config['model'].update(eps_fields)
    cfg = bridge.convert_config_xllm_to_hf(
        config, SimpleNamespace(pad_token_id=31, bos_token_id=0, eos_token_id=1))
    assert cfg.rms_norm_eps == 3e-5


@pytest.mark.parametrize('field,value', [
    ('residual_func', 'delta'), ('scale_emb', True), ('two_hop_residual', True),
    ('rescale_nffn', True), ('causal_norm_weight', True), ('attn_act_func', 'silu'),
    ('v_head_dim', 8), ('output_size', 64),
])
def test_unsupported_semantics_rejected(field, value):
    config = source_config('dense')
    config['model'][field] = value
    with pytest.raises(ValueError):
        bridge.convert_config_xllm_to_hf(config, SimpleNamespace(pad_token_id=31, bos_token_id=0, eos_token_id=1))


@pytest.mark.parametrize('fault', ['missing', 'unused', 'shape', 'extra_expert', 'extra_value'])
def test_invalid_checkpoint_rejected(fault):
    cfg, model, sd, _ = fixture_model('mova')
    if fault == 'missing':
        del sd['layers.1.norm.weight']
    elif fault == 'unused':
        sd['layers.1.mova.residual.weight'] = torch.ones(8)
    elif fault == 'shape':
        sd['layers.1.norm.weight'] = torch.ones(9)
    else:
        key = 'layers.1.moe.experts.weight1' if fault == 'extra_expert' else 'layers.1.mova.wv.weight'
        sd[key] = torch.cat([sd[key], sd[key][:1]])
    with pytest.raises(ValueError):
        bridge.convert_state_dict_xllm_to_hf(model.state_dict(), cfg, sd)


@pytest.mark.parametrize('safe,dtype', [(False, 'float32'), (True, 'float32'), (True, 'bfloat16')])
@pytest.mark.parametrize('norm_key', ['norm_eps', 'rmsnorm_eps'])
def test_checkpoint_export_and_auto_reload(tmp_path, safe, dtype, norm_key):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    cfg, model, sd, _ = fixture_model('mova', norm_key)
    root, output, tokdir = tmp_path / 'source', tmp_path / 'hf', tmp_path / 'tokenizer'
    root.mkdir()
    (root / 'config.json').write_text(json.dumps(source_config('mova', norm_key)))
    save_state_dict(sd, storage_writer=FileSystemWriter(root / 'sharded_model.tp00'), no_dist=True)
    raw = Tokenizer(WordLevel({f't{i}': i for i in range(32)}, unk_token='t2'))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, bos_token='t0', eos_token='t1', pad_token='t31')
    tokenizer.save_pretrained(tokdir)
    bridge.main(str(root), str(tokdir), str(output), layers_per_load=1, dtype=dtype, safe_serialization=safe)
    restored, info = AutoModelForCausalLM.from_pretrained(
        output, trust_remote_code=True, dtype=getattr(torch, dtype),
        attn_implementation='eager', output_loading_info=True)
    assert not any(info.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs'))
    expected = bridge.convert_state_dict_xllm_to_hf(model.state_dict(), cfg, sd)
    for key, param in restored.state_dict().items():
        torch.testing.assert_close(param, expected[key].to(param.dtype), atol=0, rtol=0)
    index = json.loads(next(output.glob('*.index.json')).read_text())
    assert index['metadata']['total_size'] == sum(p.numel() * p.element_size() for p in restored.state_dict().values())
    with torch.no_grad():
        assert torch.isfinite(restored(torch.tensor([[0, 4, 7]])).logits).all()
    with pytest.raises(ValueError, match='overwrite'):
        bridge.main(str(root), str(tokdir), str(output))
    if not safe:
        from xbridges.huggingface import xllm_to_hf_parallel as parallel
        other = tmp_path / 'parallel'
        parallel.prepare_staging_directory(other)
        _, meta_model = parallel._save_static_artifacts(xllm_dir=root, tokenizer_dir=tokdir, save_dir=other)
        results = [parallel._convert_one_layer(parallel.WorkerRequest(
            xllm_dir=str(root), tokenizer_dir=str(tokdir), save_dir=str(other),
            layer=task.layer, num_layers=2, filename=task.filename))
            for task in parallel.build_layer_tasks(2)]
        parallel._write_index(save_dir=other, model=meta_model, results=results)
        for result in results:
            parallel_state = torch.load(other / result.filename, weights_only=True)
            serial_state = torch.load(output / result.filename, weights_only=True)
            assert parallel_state.keys() == serial_state.keys()
            for key in serial_state:
                torch.testing.assert_close(serial_state[key], parallel_state[key], atol=0, rtol=0)


def test_tp_bias_mismatch_rejected():
    cfg, _, sd, _ = fixture_model('mova')
    other = {k: v.clone() for k, v in sd.items()}
    other['layers.1.mova.router.bias'].add_(1)
    with pytest.raises(ValueError, match='replicas differ'):
        bridge.merge_state_dicts(sd, other, 2, cfg)


@pytest.mark.parametrize('kind', ['dense', 'moe', 'mova'])
def test_initialization_and_aux_loss_config_preserved(kind):
    config = source_config(kind)
    config['model']['init_std'] = 0.015
    config['moe_aux_loss_coeff'] = 0.004
    cfg = bridge.convert_config_xllm_to_hf(
        config, SimpleNamespace(pad_token_id=31, bos_token_id=0, eos_token_id=1))
    assert cfg.initializer_range == 0.015
    assert cfg.router_aux_loss_coef == 0.004


@pytest.mark.parametrize('init_std', [None, 0.0])
def test_optional_initialization_config(init_std):
    config = source_config('dense')
    config['model']['init_std'] = init_std
    cfg = bridge.convert_config_xllm_to_hf(
        config, SimpleNamespace(pad_token_id=31, bos_token_id=0, eos_token_id=1))
    expected = bridge.K2HorizonConfig.initializer_range if init_std is None else init_std
    assert cfg.initializer_range == expected
    assert cfg.router_aux_loss_coef == 0.0


def _tp_shard(sd, mappings, tp_size, rank):
    return {
        src: (sd[src] if axis is None else sd[src].chunk(tp_size, dim=axis)[rank]).clone()
        for src, _, axis in mappings if src in sd
    }


def _native_expert_state(monkeypatch, kind, tp_size, rank):
    from xllm.modules.moe import expert
    from xllm.modules.model_parallel import layers

    with monkeypatch.context() as patch, torch.random.fork_rng(devices=[]):
        for module in (expert, layers):
            patch.setattr(module, 'get_model_parallel_world_size', lambda: tp_size)
            patch.setattr(module, 'get_model_parallel_rank', lambda: rank)
        torch.manual_seed(123)
        moe = expert.Expert(model_dim=8, expert_inter_dim=6, num_experts=4, init_std=0.03)
        state = {f'layers.1.moe.experts.{key}': value for key, value in moe.state_dict().items()}
        if kind == 'mova':
            torch.manual_seed(456)
            value = layers.GroupRowParallelLinear(in_features=8, out_features=8, num_groups=4)
            state.update({f'layers.1.mova.wv.{key}': value for key, value in value.state_dict().items()})
    assert all(value.ndim == 2 for value in state.values())
    return state


@pytest.mark.parametrize('kind', ['moe', 'mova'])
@pytest.mark.parametrize('tp_size', [1, 2])
@pytest.mark.parametrize('layout', ['flat', 'legacy'])
def test_native_expert_dcp_export(tmp_path, monkeypatch, kind, tp_size, layout):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from xbridges.huggingface import xllm_to_hf_parallel as parallel

    cfg, model, sd, mappings = fixture_model(kind, 'rmsnorm_eps')
    full_native = _native_expert_state(monkeypatch, kind, 1, 0)
    for key, value in full_native.items():
        sd[key] = value.unflatten(0, (4, -1))
    expected = bridge.convert_state_dict_xllm_to_hf(model.state_dict(), cfg, sd)
    root, output, tokdir = tmp_path / 'source', tmp_path / 'hf', tmp_path / 'tokenizer'
    root.mkdir()
    config = source_config(kind, 'rmsnorm_eps')
    config['model_parallel_size'] = tp_size
    (root / 'config.json').write_text(json.dumps(config))
    for rank in range(tp_size):
        shard = _tp_shard(sd, mappings, tp_size, rank)
        # Expert/value weights come from native modules, not HF target shapes.
        native = _native_expert_state(monkeypatch, kind, tp_size, rank)
        for key, value in native.items():
            count = 4 if '.mova.' in key else 4 // tp_size
            shard[key] = value if layout == 'flat' else value.unflatten(0, (count, -1))
        save_state_dict(shard, storage_writer=FileSystemWriter(root / f'sharded_model.tp{rank:02d}'), no_dist=True)

    raw = Tokenizer(WordLevel({f't{i}': i for i in range(32)}, unk_token='t2'))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, bos_token='t0', eos_token='t1', pad_token='t31')
    tokenizer.save_pretrained(tokdir)
    bridge.main(
        str(root), str(tokdir), str(output), layers_per_load=1,
        dtype='float32', safe_serialization=False,
    )
    exported = {}
    for file in sorted(output.glob('*.bin')):
        exported.update(torch.load(file, weights_only=True))
    assert exported.keys() == expected.keys()
    for key in expected:
        torch.testing.assert_close(exported[key], expected[key], rtol=0, atol=0)
    model.load_state_dict(exported, strict=True)
    with torch.no_grad():
        assert torch.isfinite(model(torch.tensor([[0, 4, 7]])).logits).all()

    state = parallel._load_layer_state(xllm_dir=root, hf_config=cfg, layer=1)
    converted = bridge.convert_state_dict_xllm_to_hf(model.state_dict(), cfg, state, layers=[1])
    for key in converted:
        torch.testing.assert_close(converted[key], expected[key], rtol=0, atol=0)


@pytest.mark.parametrize('kind', ['moe', 'mova'])
@pytest.mark.parametrize('tp_size', [2, 4])
def test_flat_experts_can_be_merged_directly(kind, tp_size):
    cfg, model, sd, mappings = fixture_model(kind)
    shards = [_tp_shard(sd, mappings, tp_size, rank) for rank in range(tp_size)]
    for shard in shards:
        for key, value in shard.items():
            if '.moe.experts.' in key or key.endswith('.mova.wv.weight'):
                shard[key] = value.flatten(0, 1)
    merged = shards[0]
    for shard in shards[1:]:
        merged = bridge.merge_state_dicts(merged, shard, 2, cfg)
    expected = bridge.convert_state_dict_xllm_to_hf(model.state_dict(), cfg, sd)
    actual = bridge.convert_state_dict_xllm_to_hf(model.state_dict(), cfg, merged)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


@pytest.mark.parametrize('key', [
    'layers.1.moe.experts.weight1', 'layers.1.moe.experts.weight2',
    'layers.1.moe.experts.weight3', 'layers.1.mova.wv.weight',
])
@pytest.mark.parametrize('fault', ['transpose', 'extra_row', 'scalar'])
def test_invalid_flat_expert_shape_rejected(key, fault):
    cfg, model, sd, _ = fixture_model('mova')
    flat = sd[key].flatten(0, 1)
    if fault == 'transpose':
        sd[key] = flat.T.contiguous()
    elif fault == 'extra_row':
        sd[key] = torch.cat([flat, flat[:1]])
    else:
        sd[key] = torch.tensor(1.)
    with pytest.raises(ValueError, match=key):
        bridge.convert_state_dict_xllm_to_hf(model.state_dict(), cfg, sd)


@pytest.mark.parametrize('fault', ['extra_layer', 'missing_layer', 'tp_missing_key', 'tp_extra_key'])
@pytest.mark.parametrize('layer_l,layer_r', [(0, 1), (1, 2), (0, 2)])
def test_full_metadata_checked_before_loading(tmp_path, monkeypatch, fault, layer_l, layer_r):
    cfg, _, sd, mappings = fixture_model('mova')
    config = source_config('mova')
    config['model_parallel_size'] = 2
    (tmp_path / 'config.json').write_text(json.dumps(config))
    for rank in range(2):
        shard = _tp_shard(sd, mappings, 2, rank)
        if fault == 'extra_layer':
            shard.update({key.replace('layers.1.', 'layers.2.'): value.clone()
                          for key, value in list(shard.items()) if key.startswith('layers.1.')})
        elif fault == 'missing_layer':
            shard = {key: value for key, value in shard.items() if not key.startswith('layers.1.')}
        elif rank == 1:
            if fault == 'tp_missing_key':
                del shard['layers.1.mova.router.bias']
            else:
                shard['layers.1.unexpected.weight'] = torch.ones(8)
        save_state_dict(shard, storage_writer=FileSystemWriter(tmp_path / f'sharded_model.tp{rank:02d}'), no_dist=True)

    def no_tensor_load(*args, **kwargs):
        pytest.fail('Invalid complete metadata must be rejected before loading tensors')

    monkeypatch.setattr(bridge, 'load_state_dict', no_tensor_load)
    message = 'layer indices' if fault in ('extra_layer', 'missing_layer') else 'different checkpoint keys'
    with pytest.raises(ValueError, match=message):
        bridge.load_xllm_state_dict(str(tmp_path), cfg, layer_l, layer_r)


@pytest.mark.parametrize('layer_l,layer_r', [(0, 1), (1, 2)])
def test_valid_metadata_keeps_layer_selective_loading(tmp_path, monkeypatch, layer_l, layer_r):
    cfg, _, sd, _ = fixture_model('mova')
    (tmp_path / 'config.json').write_text(json.dumps(source_config('mova')))
    save_state_dict(sd, storage_writer=FileSystemWriter(tmp_path / 'sharded_model.tp00'), no_dist=True)
    original_load = bridge.load_state_dict
    loaded = []

    def record_load(state, **kwargs):
        loaded.append(set(state))
        return original_load(state, **kwargs)

    monkeypatch.setattr(bridge, 'load_state_dict', record_load)
    bridge.load_xllm_state_dict(str(tmp_path), cfg, layer_l, layer_r)
    expected = {key for key in sd if key.startswith(f'layers.{layer_l}.')
                or (layer_r == 2 and not key.startswith('layers.'))}
    assert loaded == [expected]


@pytest.mark.parametrize('fmt,shape', [('2d', (12, 8)), ('3d', (2, 6, 8))])
def test_hf_to_xllm_expert_weight_format(fmt, shape):
    params = [(
        'layers.{layer}.moe.experts.weight1',
        'model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight',
        -3,
    )]
    source = {
        f'model.layers.1.mlp.experts.{expert}.gate_proj.weight': torch.full((6, 8), float(expert))
        for expert in range(4)
    }
    expected = {key: torch.empty_like(value) for key, value in source.items()}
    config = SimpleNamespace(num_experts=4, mova_num_experts=0)

    state, specs = reverse_bridge._convert_params(
        source, expected, params, 1, config,
        1, 2, True, torch.float32, 'new', fmt,
    )
    key = 'layers.1.moe.experts.weight1'
    stacked = torch.stack(list(source.values())[2:])
    wanted = stacked.flatten(0, 1) if fmt == '2d' else stacked
    assert tuple(state[key].shape) == shape
    assert specs[key] == (shape, torch.float32)
    torch.testing.assert_close(state[key], wanted)


def test_invalid_expert_weight_format_rejected():
    with pytest.raises(ValueError, match='expert_weight_format'):
        reverse_bridge._validate_expert_weight_format('legacy')
