#!/usr/bin/env python3
"""Convert an XLLM checkpoint to Hugging Face shards in parallel.

Use ``prepare``, one or more ``layer`` invocations, and ``finalize`` when a
Slurm allocation distributes layers across processes. ``all`` uses local
worker processes for a single-host conversion. The caller owns validation and
atomic publication of the staging directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from xbridges.huggingface.k2_horizon.modeling_k2_horizon import K2HorizonForCausalLM
from xbridges.huggingface.xllm_to_hf_main import (
    convert_config_xllm_to_hf,
    convert_state_dict_xllm_to_hf,
    load_xllm_state_dict,
)


PLAN_FILENAME = ".xllm-to-hf-plan.json"
_WORKER_THREADS_CONFIGURED = False


@dataclass(frozen=True)
class LayerTask:
    layer: int
    filename: str


@dataclass(frozen=True)
class WorkerRequest:
    xllm_dir: str
    tokenizer_dir: str
    save_dir: str
    layer: int
    num_layers: int
    filename: str
    reference_dir: str | None = None
    write_metadata: bool = False


@dataclass(frozen=True)
class WorkerResult:
    layer: int
    filename: str
    keys: tuple[str, ...]


@dataclass(frozen=True)
class ConversionPlan:
    xllm_dir: str
    tokenizer_dir: str
    num_layers: int


def build_layer_tasks(num_layers: int) -> tuple[LayerTask, ...]:
    if num_layers < 1:
        raise ValueError("num_layers must be positive")
    return tuple(
        LayerTask(
            layer=layer,
            filename=f"pytorch_model-{layer + 1:05d}-of-{num_layers:05d}.bin",
        )
        for layer in range(num_layers)
    )


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)  # nosec B324 - artifact parity only.
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_md5s(root: Path) -> dict[str, str]:
    if not root.is_dir():
        raise RuntimeError(f"artifact directory does not exist: {root}")
    return {
        path.relative_to(root).as_posix(): _md5(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def assert_matching_artifacts(*, reference: Path, candidate: Path) -> dict[str, str]:
    reference_md5s = _artifact_md5s(reference)
    candidate_md5s = _artifact_md5s(candidate)
    missing = sorted(set(reference_md5s) - set(candidate_md5s))
    unexpected = sorted(set(candidate_md5s) - set(reference_md5s))
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append("missing=" + ", ".join(missing))
        if unexpected:
            details.append("unexpected=" + ", ".join(unexpected))
        raise RuntimeError("artifact filenames differ: " + "; ".join(details))
    for relative_path in sorted(reference_md5s):
        if reference_md5s[relative_path] != candidate_md5s[relative_path]:
            raise RuntimeError(f"MD5 mismatch: {relative_path}")
    return candidate_md5s


def assert_matching_shard(*, reference: Path, candidate: Path, filename: str) -> str:
    reference_shard = reference / filename
    candidate_shard = candidate / filename
    if not reference_shard.is_file():
        raise RuntimeError(f"reference shard does not exist: {reference_shard}")
    if not candidate_shard.is_file():
        raise RuntimeError(f"converted shard does not exist: {candidate_shard}")
    reference_md5 = _md5(reference_shard)
    candidate_md5 = _md5(candidate_shard)
    if reference_md5 != candidate_md5:
        raise RuntimeError(f"MD5 mismatch: {filename}")
    return candidate_md5


def _resolve_directory(path: Path) -> str:
    if not path.is_dir():
        raise RuntimeError(f"required directory does not exist: {path}")
    return str(path.resolve())


def _read_num_layers(xllm_dir: Path) -> int:
    source_config = json.loads((xllm_dir / "config.json").read_text())
    try:
        num_layers = int(source_config["model"]["num_layers"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("XLLM checkpoint config does not define model.num_layers") from error
    if num_layers < 1:
        raise RuntimeError("XLLM checkpoint config model.num_layers must be positive")
    return num_layers


def _plan_path(save_dir: Path) -> Path:
    return save_dir / PLAN_FILENAME


def _write_json(destination: Path, payload: dict[str, Any]) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, sort_keys=True))
    os.replace(temporary, destination)


def prepare_staging_directory(save_dir: Path) -> None:
    if save_dir.exists():
        if not save_dir.is_dir() or any(save_dir.iterdir()):
            raise RuntimeError(f"refusing to overwrite conversion directory: {save_dir}")
        return
    save_dir.mkdir(parents=True)


def write_conversion_plan(
    *, save_dir: Path, xllm_dir: Path, tokenizer_dir: Path, num_layers: int
) -> None:
    _write_json(
        _plan_path(save_dir),
        {
            "xllm_dir": _resolve_directory(xllm_dir),
            "tokenizer_dir": _resolve_directory(tokenizer_dir),
            "num_layers": num_layers,
        },
    )


def require_conversion_plan(
    *, save_dir: Path, xllm_dir: Path, tokenizer_dir: Path, num_layers: int
) -> None:
    path = _plan_path(save_dir)
    if not path.is_file():
        raise RuntimeError(f"conversion plan does not exist: {path}")
    try:
        payload = json.loads(path.read_text())
        plan = ConversionPlan(
            xllm_dir=str(payload["xllm_dir"]),
            tokenizer_dir=str(payload["tokenizer_dir"]),
            num_layers=int(payload["num_layers"]),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid conversion plan: {path}") from error
    requested = ConversionPlan(
        xllm_dir=_resolve_directory(xllm_dir),
        tokenizer_dir=_resolve_directory(tokenizer_dir),
        num_layers=num_layers,
    )
    if plan != requested:
        raise RuntimeError("conversion inputs do not match the staged conversion plan")


def _layer_result_path(save_dir: Path, layer: int) -> Path:
    return save_dir / f".xllm-to-hf-layer-{layer:05d}.json"


def _write_layer_result(*, save_dir: Path, result: WorkerResult) -> None:
    _write_json(
        _layer_result_path(save_dir, result.layer),
        {"filename": result.filename, "keys": list(result.keys)},
    )


def _collect_layer_results(*, save_dir: Path, tasks: tuple[LayerTask, ...]) -> list[WorkerResult]:
    results: list[WorkerResult] = []
    for task in tasks:
        shard = save_dir / task.filename
        sidecar = _layer_result_path(save_dir, task.layer)
        if not shard.is_file():
            raise RuntimeError(f"missing converted layer shard: {shard}")
        if not sidecar.is_file():
            raise RuntimeError(f"missing conversion metadata: {sidecar}")
        payload = json.loads(sidecar.read_text())
        result = WorkerResult(
            layer=task.layer,
            filename=str(payload["filename"]),
            keys=tuple(payload["keys"]),
        )
        if result.filename != task.filename:
            raise RuntimeError(f"conversion metadata names the wrong shard: {sidecar}")
        results.append(result)
    return results


def _remove_layer_results(*, save_dir: Path, tasks: tuple[LayerTask, ...]) -> None:
    for task in tasks:
        _layer_result_path(save_dir, task.layer).unlink(missing_ok=True)


def _load_layer_state(*, xllm_dir: Path, hf_config: Any, layer: int) -> dict[str, Any]:
    return load_xllm_state_dict(str(xllm_dir), hf_config, layer, layer + 1)


def _tokenizer(tokenizer_dir: Path) -> Any:
    return AutoTokenizer.from_pretrained(
        tokenizer_dir,
        trust_remote_code=True,  # nosec B615 - conversion intentionally executes checkpoint tokenizer code.
    )


def _build_hf_model(*, xllm_dir: Path, tokenizer_dir: Path) -> tuple[Any, Any, Any]:
    tokenizer = _tokenizer(tokenizer_dir)
    hf_config = convert_config_xllm_to_hf(
        xllm_config=json.loads((xllm_dir / "config.json").read_text()),
        tokenizer=tokenizer,
    )
    with torch.device("meta"):
        model = K2HorizonForCausalLM(config=hf_config)
    return tokenizer, hf_config, model


def _temporary_shard_path(destination: Path) -> Path:
    return destination.parent / f".{destination.name}.tmp-{os.getpid()}" / destination.name


def _configure_worker_threads() -> None:
    """Limit a reusable worker process to one CPU thread exactly once."""
    global _WORKER_THREADS_CONFIGURED
    if _WORKER_THREADS_CONFIGURED:
        return
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _WORKER_THREADS_CONFIGURED = True


def _convert_one_layer(request: WorkerRequest) -> WorkerResult:
    _configure_worker_threads()
    xllm_dir = Path(request.xllm_dir)
    tokenizer, hf_config, model = _build_hf_model(
        xllm_dir=xllm_dir,
        tokenizer_dir=Path(request.tokenizer_dir),
    )
    del tokenizer
    if hf_config.num_hidden_layers != request.num_layers:
        raise RuntimeError("worker layer count differs from the conversion plan")
    converted = convert_state_dict_xllm_to_hf(
        hf_state_dict=model.state_dict(),
        hf_config=hf_config,
        xllm_state_dict=_load_layer_state(xllm_dir=xllm_dir, hf_config=hf_config, layer=request.layer),
        layers=[request.layer],
        include_output=request.layer == request.num_layers - 1,
    )
    layer_state = {
        key: value for key, value in converted.items() if f"layers.{request.layer}." in key
    }
    if request.layer == request.num_layers - 1:
        layer_state.update({key: value for key, value in converted.items() if "model.layers." not in key})
    layer_state = {key: value.float().contiguous() for key, value in layer_state.items()}
    destination = Path(request.save_dir) / request.filename
    temporary = _temporary_shard_path(destination)
    temporary.parent.mkdir()
    torch.save(layer_state, temporary)
    os.replace(temporary, destination)
    temporary.parent.rmdir()
    result = WorkerResult(layer=request.layer, filename=request.filename, keys=tuple(layer_state))
    if request.reference_dir is not None:
        assert_matching_shard(
            reference=Path(request.reference_dir),
            candidate=Path(request.save_dir),
            filename=request.filename,
        )
    if request.write_metadata:
        _write_layer_result(save_dir=Path(request.save_dir), result=result)
    return result


def _save_static_artifacts(*, xllm_dir: Path, tokenizer_dir: Path, save_dir: Path) -> tuple[Any, Any]:
    tokenizer, hf_config, model = _build_hf_model(xllm_dir=xllm_dir, tokenizer_dir=tokenizer_dir)
    hf_config.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    source_directory = Path(__file__).with_name("k2_horizon")
    for source in sorted(source_directory.glob("*.py")):
        shutil.copyfile(source, save_dir / source.name)
    return hf_config, model


def _write_index(*, save_dir: Path, model: Any, results: list[WorkerResult]) -> None:
    weight_map = {
        key: result.filename
        for result in sorted(results, key=lambda result: result.layer)
        for key in result.keys
    }
    expected = model.state_dict()
    if weight_map.keys() != expected.keys():
        raise ValueError('Exported index does not cover the complete HF model')
    total_size = sum(value.numel() * value.element_size() for value in expected.values())
    destination = save_dir / "pytorch_model.bin.index.json"
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    with temporary.open("w") as handle:
        json.dump(
            {"metadata": {"total_size": total_size}, "weight_map": weight_map},
            handle,
            indent=4,
        )
    os.replace(temporary, destination)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("all", "prepare", "layer", "finalize"), default="all")
    parser.add_argument("--xllm-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--layer", type=int)
    parser.add_argument("--verify-identical-to", type=Path)
    parser.add_argument("--verify-shard-identical-to", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    _resolve_directory(args.xllm_dir)
    _resolve_directory(args.tokenizer_dir)
    if args.verify_identical_to is not None and not args.verify_identical_to.is_dir():
        raise RuntimeError(f"reference directory does not exist: {args.verify_identical_to}")
    if args.verify_shard_identical_to is not None and not args.verify_shard_identical_to.is_dir():
        raise RuntimeError(f"reference directory does not exist: {args.verify_shard_identical_to}")
    num_layers = _read_num_layers(args.xllm_dir)

    if args.mode == "prepare":
        prepare_staging_directory(args.save_dir)
        hf_config, _ = _save_static_artifacts(
            xllm_dir=args.xllm_dir,
            tokenizer_dir=args.tokenizer_dir,
            save_dir=args.save_dir,
        )
        write_conversion_plan(
            save_dir=args.save_dir,
            xllm_dir=args.xllm_dir,
            tokenizer_dir=args.tokenizer_dir,
            num_layers=hf_config.num_hidden_layers,
        )
        print(f"prepared conversion for {hf_config.num_hidden_layers} layer shards", flush=True)
        return 0

    if args.mode == "layer":
        if args.layer is None or not 0 <= args.layer < num_layers:
            raise RuntimeError(f"--layer must be in [0, {num_layers})")
        require_conversion_plan(
            save_dir=args.save_dir,
            xllm_dir=args.xllm_dir,
            tokenizer_dir=args.tokenizer_dir,
            num_layers=num_layers,
        )
        task = build_layer_tasks(num_layers)[args.layer]
        result = _convert_one_layer(
            WorkerRequest(
                xllm_dir=str(args.xllm_dir),
                tokenizer_dir=str(args.tokenizer_dir),
                save_dir=str(args.save_dir),
                layer=task.layer,
                num_layers=num_layers,
                filename=task.filename,
                reference_dir=(
                    str(args.verify_shard_identical_to)
                    if args.verify_shard_identical_to is not None
                    else None
                ),
                write_metadata=True,
            )
        )
        print(f"converted {result.filename} keys={len(result.keys)}", flush=True)
        return 0

    if args.mode == "finalize":
        require_conversion_plan(
            save_dir=args.save_dir,
            xllm_dir=args.xllm_dir,
            tokenizer_dir=args.tokenizer_dir,
            num_layers=num_layers,
        )
        _, hf_config, model = _build_hf_model(
            xllm_dir=args.xllm_dir,
            tokenizer_dir=args.tokenizer_dir,
        )
        tasks = build_layer_tasks(hf_config.num_hidden_layers)
        _write_index(save_dir=args.save_dir, model=model, results=_collect_layer_results(save_dir=args.save_dir, tasks=tasks))
        _remove_layer_results(save_dir=args.save_dir, tasks=tasks)
        _plan_path(args.save_dir).unlink()
        if args.verify_identical_to is not None:
            checked = assert_matching_artifacts(
                reference=args.verify_identical_to, candidate=args.save_dir
            )
            print(f"verified {len(checked)} matching files against {args.verify_identical_to}", flush=True)
        print(f"finalized index from {len(tasks)} converted layer shards", flush=True)
        return 0

    if args.workers is None or args.workers < 1:
        raise RuntimeError("--workers must be positive with --mode all")
    prepare_staging_directory(args.save_dir)
    hf_config, model = _save_static_artifacts(
        xllm_dir=args.xllm_dir,
        tokenizer_dir=args.tokenizer_dir,
        save_dir=args.save_dir,
    )
    tasks = build_layer_tasks(hf_config.num_hidden_layers)
    worker_count = min(args.workers, len(tasks))
    requests = [
        WorkerRequest(
            xllm_dir=str(args.xllm_dir),
            tokenizer_dir=str(args.tokenizer_dir),
            save_dir=str(args.save_dir),
            layer=task.layer,
            num_layers=hf_config.num_hidden_layers,
            filename=task.filename,
            reference_dir=(
                str(args.verify_shard_identical_to)
                if args.verify_shard_identical_to is not None
                else None
            ),
        )
        for task in tasks
    ]
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        results = list(executor.map(_convert_one_layer, requests))
    _write_index(save_dir=args.save_dir, model=model, results=results)
    if args.verify_identical_to is not None:
        checked = assert_matching_artifacts(reference=args.verify_identical_to, candidate=args.save_dir)
        print(f"verified {len(checked)} matching files against {args.verify_identical_to}", flush=True)
    print(f"converted {len(results)} layer shards with {worker_count} workers", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
