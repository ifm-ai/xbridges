"""Small, deterministic round-trip test for the HuggingFace/XLLM bridge."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.default_planner import _EmptyStateDictLoadPlanner
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict

from xbridges.huggingface import compare_xllm_checkpoints
from xbridges.huggingface import hf_to_xllm_main as hf_to_xllm
from xbridges.huggingface import xllm_to_hf_main as xllm_to_hf


def _toy_config():
    return {
        "model_parallel_size": 2,
        "seq_len": 32,
        "model": {
            "vocab_size": 32,
            "num_layers": 2,
            "num_dense_layers": None,
            "model_dim": 8,
            "num_heads": 4,
            "num_kv_heads": 2,
            "head_dim": 2,
            "rope_base": 10_000,
            "rope_head_dim": 2,
            "ffn_hidden_dim": 16,
            "num_experts": 0,
            "num_activated_experts": 0,
            "expert_inter_dim": 0,
            "num_shared_experts": 0,
            "apply_rmsnorm": True,
            "swiglu": True,
            "attention_dropout": 0.0,
            "norm_eps": 1e-6,
            "qknorm": True,
            "moe_router_bias": False,
            "moe_router_score_func": "softmax",
            "moe_router_scaling_factor": 1.0,
            "layernorm_num_groups": 1,
        },
    }


def _load_dcp(path):
    state = {}
    _load_state_dict(
        state_dict=state,
        storage_reader=FileSystemReader(path),
        planner=_EmptyStateDictLoadPlanner(),
        no_dist=True,
    )
    return state


class HFToXLLMToyTest(unittest.TestCase):
    def test_grouped_weight_layout_canonicalization(self):
        legacy = torch.arange(24).reshape(2, 3, 4)
        flattened = legacy.flatten(0, 1)
        suffixes = [
            "moe.experts.weight1", "moe.experts.weight2",
            "moe.experts.weight3", "mova.wv.weight",
        ]
        for suffix in suffixes:
            key = f"layers.1.{suffix}"
            with self.subTest(key=key):
                self.assertEqual(
                    compare_xllm_checkpoints.canonical_shape(key, legacy.shape), (6, 4)
                )
                torch.testing.assert_close(
                    compare_xllm_checkpoints.canonical_tensor(key, legacy),
                    compare_xllm_checkpoints.canonical_tensor(key, flattened),
                )
        dense_key = "layers.1.nffn.fc1.weight"
        self.assertEqual(
            compare_xllm_checkpoints.canonical_shape(dense_key, legacy.shape),
            tuple(legacy.shape),
        )

    def test_dense_tp2_round_trip(self):
        torch.manual_seed(7)
        tokenizer = SimpleNamespace(pad_token_id=31, bos_token_id=0, eos_token_id=1)
        hf_config = xllm_to_hf.convert_config_xllm_to_hf(_toy_config(), tokenizer)
        hf_config._attn_implementation = "eager"
        model = xllm_to_hf.K2HorizonForCausalLM(hf_config)

        mappings = []
        for layer in range(hf_config.num_hidden_layers):
            mappings.extend(
                (x.format(layer=layer), h.format(layer=layer), axis)
                for x, h, axis in xllm_to_hf.ATTENTION_PARAMS + xllm_to_hf.DENSE_PARAMS
            )
        mappings.extend(xllm_to_hf.OTHER_PARAMS)

        xllm_state = {}
        hf_targets = model.state_dict()
        mappings = [mapping for mapping in mappings if mapping[1] in hf_targets]
        for xllm_key, hf_key, _ in mappings:
            xllm_state[xllm_key] = torch.randn(hf_targets[hf_key].shape)

        hf_state = xllm_to_hf.convert_state_dict_xllm_to_hf(
            hf_targets, hf_config, xllm_state
        )

        new_xllm_state = {
            (
                xllm_to_hf._new_attention_norm_key(int(key.split(".")[1]), hf_config)
                if key.startswith("layers.") and key.endswith(".norm.weight")
                and key.count(".") == 3
                else key
            ): value
            for key, value in xllm_state.items()
        }
        hf_state_from_new = xllm_to_hf.convert_state_dict_xllm_to_hf(
            hf_targets, hf_config, new_xllm_state
        )
        for key in hf_state:
            torch.testing.assert_close(hf_state_from_new[key], hf_state[key])

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            hf_dir = root / "hf"
            output_dir, new_output_dir = root / "legacy", root / "new"
            bf16_output_dir = root / "bf16"
            hf_config.save_pretrained(hf_dir)
            save_file(hf_state, hf_dir / "model.safetensors")

            hf_to_xllm.convert_checkpoint(
                hf_dir=str(hf_dir),
                save_dir=str(output_dir),
                tp_rank=-1,
                sync_files=False,
                dtype="float32",
                norm_key_format="legacy",
            )
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                hf_to_xllm.convert_checkpoint(
                    hf_dir=str(hf_dir),
                    save_dir=str(output_dir),
                    tp_rank=0,
                    sync_files=False,
                    dtype="float32",
                    norm_key_format="legacy",
                )

            split_axes = {key: axis for key, _, axis in mappings}
            for rank in range(2):
                actual = _load_dcp(output_dir / f"full_model.tp{rank:02d}")
                self.assertEqual(actual.keys(), xllm_state.keys())
                for key, full_tensor in xllm_state.items():
                    axis = split_axes[key]
                    expected = full_tensor if axis is None else full_tensor.chunk(2, dim=axis)[rank]
                    with self.subTest(rank=rank, key=key):
                        torch.testing.assert_close(actual[key], expected, rtol=1e-6, atol=1e-7)

            hf_to_xllm.convert_checkpoint(
                hf_dir=str(hf_dir),
                save_dir=str(new_output_dir),
                tp_rank=-1,
                sync_files=False,
                dtype="float32",
            )
            new_keys = FileSystemReader(
                new_output_dir / "full_model.tp00"
            ).read_metadata().state_dict_metadata.keys()
            self.assertIn("layers.0.attention.norm.weight", new_keys)
            self.assertNotIn("layers.0.norm.weight", new_keys)
            for rank in range(2):
                compare_xllm_checkpoints.compare(
                    output_dir / f"full_model.tp{rank:02d}",
                    new_output_dir / f"full_model.tp{rank:02d}",
                )

            hf_to_xllm.convert_checkpoint(
                hf_dir=str(hf_dir),
                save_dir=str(bf16_output_dir),
                tp_rank=-1,
                sync_files=False,
                dtype="bfloat16",
            )
            bf16_state = _load_dcp(bf16_output_dir / "full_model.tp00")
            self.assertTrue(all(tensor.dtype == torch.bfloat16
                                for tensor in bf16_state.values()))


if __name__ == "__main__":
    unittest.main()
