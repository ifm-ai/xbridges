from __future__ import annotations

import json
import types
from pathlib import Path

import pytest


def _load_converter():
    import xbridges.huggingface.xllm_to_hf_parallel as converter
    return converter


def test_parallel_converter_compares_one_shard_byte_for_byte(tmp_path: Path) -> None:
    converter = _load_converter()
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    reference.mkdir()
    candidate.mkdir()
    filename = "pytorch_model-00001-of-00002.bin"
    (reference / filename).write_bytes(b"reference weights")
    (candidate / filename).write_bytes(b"reference weights")

    assert hasattr(converter, "assert_matching_shard"), (
        "the parallel converter must expose per-shard parity validation"
    )
    assert converter.assert_matching_shard(
        reference=reference, candidate=candidate, filename=filename
    )

    (candidate / filename).write_bytes(b"different weights")
    with pytest.raises(RuntimeError, match=f"MD5 mismatch: {filename}"):
        converter.assert_matching_shard(
            reference=reference, candidate=candidate, filename=filename
        )


def test_parallel_converter_compares_all_reference_artifacts(tmp_path: Path) -> None:
    converter = _load_converter()
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    reference.mkdir()
    candidate.mkdir()
    for root in (reference, candidate):
        (root / "config.json").write_text('{"model_type":"k2_horizon"}\n')
        (root / "pytorch_model-00001-of-00001.bin").write_bytes(b"weights")

    assert hasattr(converter, "assert_matching_artifacts"), (
        "the parallel converter must expose full conversion parity validation"
    )
    assert converter.assert_matching_artifacts(reference=reference, candidate=candidate)

    (candidate / "config.json").write_text('{"model_type":"different"}\n')
    with pytest.raises(RuntimeError, match="MD5 mismatch: config.json"):
        converter.assert_matching_artifacts(reference=reference, candidate=candidate)


def test_parallel_converter_index_matches_serial_converter_bytes(tmp_path: Path) -> None:
    converter = _load_converter()
    results = [
        converter.WorkerResult(
            layer=0,
            filename="pytorch_model-00001-of-00002.bin",
            keys=("model.layers.0.self_attn.q_proj.weight",),
        ),
        converter.WorkerResult(
            layer=1,
            filename="pytorch_model-00002-of-00002.bin",
            keys=("lm_head.weight",),
        ),
    ]
    import torch
    model = types.SimpleNamespace(state_dict=lambda: {
        key: torch.empty(6) for result in results for key in result.keys
    })
    converter._write_index(save_dir=tmp_path, model=model, results=results)

    expected = {
        "metadata": {"total_size": 48},
        "weight_map": {
            "model.layers.0.self_attn.q_proj.weight": "pytorch_model-00001-of-00002.bin",
            "lm_head.weight": "pytorch_model-00002-of-00002.bin",
        },
    }
    expected_path = tmp_path / "expected.json"
    with expected_path.open("w") as handle:
        json.dump(expected, handle, indent=4)

    assert (tmp_path / "pytorch_model.bin.index.json").read_bytes() == expected_path.read_bytes()


def test_parallel_converter_configures_worker_threads_once(monkeypatch: pytest.MonkeyPatch) -> None:
    converter = _load_converter()
    calls: list[tuple[str, int]] = []
    fake_torch = types.SimpleNamespace(
        set_num_threads=lambda value: calls.append(("threads", value)),
        set_num_interop_threads=lambda value: calls.append(("interop", value)),
    )
    monkeypatch.setattr(converter, "torch", fake_torch)
    monkeypatch.setattr(converter, "_WORKER_THREADS_CONFIGURED", False, raising=False)

    assert hasattr(converter, "_configure_worker_threads"), (
        "worker thread configuration must be reusable across multiple layer tasks"
    )
    converter._configure_worker_threads()
    converter._configure_worker_threads()

    assert calls == [("threads", 1), ("interop", 1)]
