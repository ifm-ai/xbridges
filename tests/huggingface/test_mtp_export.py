"""Export of the multi-token prediction (MTP) modules of an xLLM checkpoint (xbridges/huggingface/mtp.py)."""
import json

import pytest
import torch
from safetensors.torch import load_file
from torch.distributed.checkpoint import FileSystemWriter, save_state_dict
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

from xbridges.huggingface import mtp
from xbridges.huggingface import xllm_to_hf_main as bridge
from tests.huggingface.test_conversion import fixture_model, source_config

DEPTH = 2


def mtp_fixture(kind):
    """The fixture model with DEPTH MTP modules: blocks of the last layer's kind, their own weights, and the TP split
    dim of every xLLM tensor."""
    config = source_config(kind)
    config['model']['mtp_depth'] = DEPTH
    cfg, model, sd, mappings = fixture_model(kind)
    cfg.num_nextn_predict_layers = DEPTH
    last, dim = cfg.num_hidden_layers - 1, cfg.hidden_size
    axis = {src: a for src, _, a in mappings if src in sd}
    torch.manual_seed(7)
    for m in range(DEPTH):
        for key in [key for key in sd if key.startswith(f'layers.{last}.')]:
            name = f'mtp.layers.{m}.block.' + key[len(f'layers.{last}.'):]
            sd[name], axis[name] = torch.randn_like(sd[key]) * .03, axis[key]
        for xllm_key, _, split_dim in mtp.MTP_PARAMS:
            name = xllm_key.format(m=m)
            shape = (dim, 2 * dim) if name.endswith('proj.weight') else (dim,)
            sd[name], axis[name] = torch.randn(shape) * .03, split_dim
    return config, cfg, model, sd, axis


@pytest.mark.parametrize('kind', ['dense', 'moe', 'mova'])
def test_mtp_tp_merge_and_conversion(kind):
    _, cfg, model, sd, axis = mtp_fixture(kind)
    tp0, tp1 = {}, {}
    for key, value in sd.items():
        if axis[key] is None:
            tp0[key], tp1[key] = value.clone(), value.clone()
        else:
            tp0[key], tp1[key] = [part.clone() for part in value.chunk(2, dim=axis[key])]
    merged = bridge.merge_state_dicts(tp0, tp1, cfg.num_hidden_layers, cfg)
    for key in sd:
        torch.testing.assert_close(merged[key], sd[key], rtol=0, atol=0)

    mtp_sd = mtp.pop_mtp_state_dict(merged)
    assert mtp.mtp_depth(mtp_sd.keys()) == DEPTH and not any(mtp.is_mtp_key(key) for key in merged)
    main = bridge.convert_state_dict_xllm_to_hf(model.state_dict(), cfg, merged)
    model.load_state_dict(main, strict=True)
    out = mtp.convert_mtp_state_dict_xllm_to_hf(model.state_dict(), cfg, mtp_sd)

    n, last, dim = cfg.num_hidden_layers, cfg.num_hidden_layers - 1, cfg.hidden_size
    for m in range(DEPTH):
        layer = n + m
        # the block: as the last decoder layer would convert, under layer n+m
        block = {f'layers.{last}.' + key[len(f'mtp.layers.{m}.block.'):]: value
                 for key, value in mtp_sd.items() if key.startswith(f'mtp.layers.{m}.block.')}
        expected = bridge.convert_state_dict_xllm_to_hf(model.state_dict(), cfg, block, layers=[last],
                                                        include_output=False)
        for key, value in expected.items():
            torch.testing.assert_close(out[key.replace(f'model.layers.{last}.', f'model.layers.{layer}.')], value,
                                       rtol=0, atol=0)
        # the module's own weights: norms + 1, projection halves swapped to [embedding; hidden state]
        proj = mtp_sd[f'mtp.layers.{m}.proj.weight']
        torch.testing.assert_close(out[f'model.layers.{layer}.eh_proj.weight'][:, :dim], proj[:, dim:], rtol=0, atol=0)
        torch.testing.assert_close(out[f'model.layers.{layer}.eh_proj.weight'][:, dim:], proj[:, :dim], rtol=0, atol=0)
        for xllm_key, hf_name in (('mtp.layers.{m}.embed_norm.weight', 'enorm.weight'),
                                  ('mtp.layers.{m}.hidden_norm.weight', 'hnorm.weight'),
                                  ('output.mtp_norms.{m}.weight', 'shared_head.norm.weight')):
            torch.testing.assert_close(out[f'model.layers.{layer}.{hf_name}'], mtp_sd[xllm_key.format(m=m)] + 1,
                                       rtol=0, atol=0)
    assert len(out) == DEPTH * (len(expected) + len(mtp.MTP_PARAMS))


def test_mtp_depth_must_match_config():
    _, cfg, model, sd, _ = mtp_fixture('moe')
    mtp_sd = mtp.pop_mtp_state_dict(sd)
    cfg.num_nextn_predict_layers = DEPTH - 1
    with pytest.raises(ValueError, match='num_nextn_predict_layers'):
        mtp.convert_mtp_state_dict_xllm_to_hf(model.state_dict(), cfg, mtp_sd)


def test_unknown_mtp_tensor_rejected():
    _, cfg, model, sd, _ = mtp_fixture('moe')
    mtp_sd = mtp.pop_mtp_state_dict(sd)
    mtp_sd['mtp.layers.0.extra.weight'] = torch.zeros(1)
    with pytest.raises(ValueError, match='unused_source'):
        mtp.convert_mtp_state_dict_xllm_to_hf(model.state_dict(), cfg, mtp_sd)


@pytest.mark.parametrize('safe', [False, True])
def test_mtp_checkpoint_export_and_reload(tmp_path, safe):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    config, cfg, model, sd, _ = mtp_fixture('moe')
    root, output, tokdir = tmp_path / 'source', tmp_path / 'hf', tmp_path / 'tokenizer'
    root.mkdir()
    (root / 'config.json').write_text(json.dumps(config))
    save_state_dict(sd, storage_writer=FileSystemWriter(root / 'sharded_model.tp00'), no_dist=True)
    raw = Tokenizer(WordLevel({f't{i}': i for i in range(32)}, unk_token='t2'))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, bos_token='t0', eos_token='t1', pad_token='t31')
    tokenizer.save_pretrained(tokdir)
    bridge.main(str(root), str(tokdir), str(output), layers_per_load=1, dtype='float32', safe_serialization=safe)

    assert json.loads((output / 'config.json').read_text())['num_nextn_predict_layers'] == DEPTH
    index = json.loads(next(output.glob('*.index.json')).read_text())
    mtp_file = 'model-mtp.safetensors' if safe else 'pytorch_model-mtp.bin'
    mtp_saved = load_file(output / mtp_file) if safe else torch.load(output / mtp_file, weights_only=True)
    expected_mtp = mtp.convert_mtp_state_dict_xllm_to_hf(model.state_dict(), cfg, mtp.pop_mtp_state_dict(dict(sd)))
    assert mtp_saved.keys() == expected_mtp.keys()
    assert {key for key, name in index['weight_map'].items() if name == mtp_file} == expected_mtp.keys()
    for key in expected_mtp:
        torch.testing.assert_close(mtp_saved[key], expected_mtp[key], rtol=0, atol=0)

    # the HF model loads the decoder and ignores the MTP layers
    restored, info = AutoModelForCausalLM.from_pretrained(
        output, trust_remote_code=True, dtype=torch.float32, attn_implementation='eager', output_loading_info=True)
    assert not any(info.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs'))
    without_mtp = mtp.pop_mtp_state_dict(dict(sd)) and {k: v for k, v in sd.items() if not mtp.is_mtp_key(k)}
    expected = bridge.convert_state_dict_xllm_to_hf(model.state_dict(), cfg, without_mtp)
    for key, param in restored.state_dict().items():
        torch.testing.assert_close(param, expected[key], atol=0, rtol=0)


def test_parallel_converter_refuses_mtp(tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from xbridges.huggingface import xllm_to_hf_parallel as parallel
    config, *_ = mtp_fixture('moe')
    root, tokdir = tmp_path / 'source', tmp_path / 'tokenizer'
    root.mkdir()
    (root / 'config.json').write_text(json.dumps(config))
    raw = Tokenizer(WordLevel({f't{i}': i for i in range(32)}, unk_token='t2'))
    PreTrainedTokenizerFast(tokenizer_object=raw, bos_token='t0', eos_token='t1', pad_token='t31').save_pretrained(tokdir)
    with pytest.raises(NotImplementedError, match='multi-token prediction'):
        parallel._build_hf_model(xllm_dir=root, tokenizer_dir=tokdir)
