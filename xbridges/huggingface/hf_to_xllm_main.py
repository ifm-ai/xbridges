"""Convert K2-Horizon HuggingFace checkpoints to XLLM TP checkpoints.

The tensor mapping is intentionally shared with :mod:`xllm_to_hf_main`: that
module defines the XLLM/HF names and the axis sharded by tensor parallelism;
this module only applies those rules in the opposite direction.  Large HF
checkpoints are read a layer at a time and one ``full_model.tpNN`` checkpoint
is produced per TP rank.

The command line also contains rank batching, Slurm-array, automatic memory
budgeting, resume, and per-rank logging features that used to live in the 375B
shell script. Run ``python -m xbridges.huggingface.hf_to_xllm_main --help``
for the complete interface.
"""

from __future__ import annotations

import gc
import json
import logging
import math
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import torch
from safetensors import SafetensorError, safe_open
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter
from torch.distributed.checkpoint.metadata import TensorStorageMetadata
from torch.distributed.checkpoint.state_dict_saver import _save_state_dict

from xbridges.huggingface.k2_horizon.configuration_k2_horizon import (
    K2HorizonConfig,
)
from xbridges.huggingface.k2_horizon.modeling_k2_horizon import (
    K2HorizonForCausalLM,
)
from xbridges.huggingface.xllm_to_hf_main import (
    ATTENTION_PARAMS,
    DENSE_PARAMS,
    MOE_PARAMS,
    MOVA_ATTENTION_PARAMS,
    OTHER_PARAMS,
    _new_attention_norm_key,
)


logger = logging.getLogger(__name__)

WEIGHT_FILES = (
    ("model.safetensors.index.json", "safetensors"),
    ("pytorch_model.bin.index.json", "bin"),
    ("model.safetensors", "safetensors"),
    ("pytorch_model.bin", "bin"),
)
TargetSpec = tuple[tuple[int, ...], torch.dtype]


def _bool(value: object, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.lower() in ("true", "false", "1", "0"):
        return value.lower() in ("true", "1")
    raise ValueError(f"{name} must be true or false, got {value!r}")


def _config_value(config, name, default=None):
    value = getattr(config, name, default)
    return default if value is None else value


class HFCheckpointReader:
    """Read only the tensors needed for the current layer."""

    def __init__(self, checkpoint_dir: Path):
        self.root = checkpoint_dir.resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"HuggingFace checkpoint not found: {self.root}")
        with (self.root / "config.json").open() as fp:
            config_dict = json.load(fp)
        # Older bridge exports used the pre-Transformers-5 ``rope_theta``
        # field. Normalize it without requiring an in-place checkpoint edit.
        if "rope_parameters" not in config_dict and "rope_theta" in config_dict:
            config_dict["rope_parameters"] = {
                "rope_type": "default",
                "rope_theta": config_dict.pop("rope_theta"),
            }
        self.config = K2HorizonConfig.from_dict(config_dict)

        found = [(self.root / name, kind) for name, kind in WEIGHT_FILES if (self.root / name).is_file()]
        if not found:
            raise FileNotFoundError(
                f"No HF weights found in {self.root}; expected one of "
                + ", ".join(name for name, _ in WEIGHT_FILES)
            )
        path, self.checkpoint_format = found[0]
        if len(found) > 1:
            logger.warning("Multiple HF weight formats found; using %s", path.name)

        if path.name.endswith(".index.json"):
            with path.open() as fp:
                self.weight_map = {str(k): str(v) for k, v in json.load(fp)["weight_map"].items()}
        elif self.checkpoint_format == "safetensors":
            try:
                with safe_open(path, framework="pt", device="cpu") as fp:
                    self.weight_map = {key: path.name for key in fp.keys()}
            except SafetensorError as exc:
                raise RuntimeError(f"Cannot read {path}") from exc
        else:
            state = self._load_bin(path.name)
            self.weight_map = {key: path.name for key in state}
            del state
            gc.collect()

        for filename in set(self.weight_map.values()):
            self._weight_path(filename)
        logger.info(
            "Detected %s checkpoint: %d tensors in %d file(s)",
            self.checkpoint_format,
            len(self.weight_map),
            len(set(self.weight_map.values())),
        )

    def _weight_path(self, filename: str) -> Path:
        path = (self.root / filename).resolve()
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise ValueError(f"Weight index escapes checkpoint directory: {filename}") from exc
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def _load_bin(self, filename: str) -> dict[str, torch.Tensor]:
        path = self._weight_path(filename)
        try:
            state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Cannot memory-map {path}; rewrite it with torch.save zip serialization"
            ) from exc
        if not isinstance(state, dict):
            raise TypeError(f"Expected a state dict in {path}, got {type(state)}")
        return state

    @contextmanager
    def tensors(self, keys: Sequence[str]) -> Iterator[Mapping[str, torch.Tensor]]:
        missing = set(keys) - self.weight_map.keys()
        if missing:
            raise KeyError(f"HF index is missing: {sorted(missing)[:20]}")
        by_file: dict[str, list[str]] = {}
        for key in keys:
            by_file.setdefault(self.weight_map[key], []).append(key)

        opened = []
        selected: dict[str, torch.Tensor] = {}
        try:
            for filename, file_keys in by_file.items():
                path = self._weight_path(filename)
                if path.suffix == ".safetensors":
                    handle = safe_open(path, framework="pt", device="cpu")
                    opened.append(handle)
                    selected.update({key: handle.get_tensor(key) for key in file_keys})
                else:
                    state = self._load_bin(filename)
                    opened.append(state)
                    absent = set(file_keys) - state.keys()
                    if absent:
                        raise KeyError(f"{filename} is missing {sorted(absent)[:20]}")
                    selected.update({key: state[key] for key in file_keys})
            yield selected
        finally:
            selected.clear()
            opened.clear()
            gc.collect()


def _model_state(config: K2HorizonConfig) -> Mapping[str, torch.Tensor]:
    with torch.device("meta"):
        return K2HorizonForCausalLM(config).state_dict()


def _layer_params(config: K2HorizonConfig, layer: int):
    if layer in config.mlp_only_layers:
        return ATTENTION_PARAMS + DENSE_PARAMS
    if config.mova_num_experts > 0:
        return MOVA_ATTENTION_PARAMS + MOE_PARAMS
    return ATTENTION_PARAMS + MOE_PARAMS


def _hf_keys(params, layer: int, config: K2HorizonConfig) -> list[str]:
    keys = []
    for xllm_key, hf_key, _ in params:
        count = (
            config.mova_num_experts
            if xllm_key.endswith(".mova.wv.weight")
            else config.num_experts
        )
        if "{expert}" in hf_key:
            keys.extend(hf_key.format(layer=layer, expert=i) for i in range(count))
        else:
            keys.append(hf_key.format(layer=layer))
    return keys


def _inverse_permute_qk(weight, heads, head_dim):
    return (
        weight.reshape(heads, 2, head_dim // 2, weight.shape[1])
        .transpose(1, 2)
        .reshape_as(weight)
    )


def _inverse_permute_qknorm(weight, heads, head_dim):
    return (
        weight.reshape(heads, 2, head_dim // 2)
        .transpose(1, 2)
        .reshape_as(weight)
    )


def _transform(xllm_key: str, weight: torch.Tensor, config: K2HorizonConfig):
    if xllm_key.endswith(".wq.weight"):
        return _inverse_permute_qk(weight, config.num_attention_heads, config.head_dim)
    if xllm_key.endswith(".wk.weight"):
        return _inverse_permute_qk(weight, config.num_key_value_heads, config.head_dim)
    if xllm_key.endswith(".query_norm.weight"):
        return _inverse_permute_qknorm(
            weight, config.num_attention_heads, config.head_dim
        ).sub(1)
    if xllm_key.endswith(".key_norm.weight"):
        return _inverse_permute_qknorm(
            weight, config.num_key_value_heads, config.head_dim
        ).sub(1)
    if xllm_key.endswith((".norm.weight", ".final_norm.weight")):
        return weight.sub(1)
    return weight


def _slice(weight: torch.Tensor, rank: int, size: int, dim: int | None):
    if dim is None:
        return weight.clone(memory_format=torch.contiguous_format)
    dim %= weight.ndim
    if weight.shape[dim] % size:
        raise ValueError(f"Cannot split shape {tuple(weight.shape)} on dim {dim} over TP={size}")
    width = weight.shape[dim] // size
    return weight.narrow(dim, rank * width, width).clone(memory_format=torch.contiguous_format)


def _convert_params(
    source: Mapping[str, torch.Tensor],
    expected: Mapping[str, torch.Tensor],
    params,
    layer: int | None,
    config: K2HorizonConfig,
    tp_rank: int,
    tp_size: int,
    materialize: bool,
    target_dtype: torch.dtype,
    norm_key_format: str,
    expert_weight_format: str,
) -> tuple[dict[str, torch.Tensor], dict[str, TargetSpec]]:
    output, specs = {}, {}
    for x_template, hf_template, split_dim in params:
        fields = {} if layer is None else {"layer": layer}
        xllm_key = x_template.format(**fields)
        if norm_key_format == "new" and layer is not None and xllm_key == f"layers.{layer}.norm.weight":
            xllm_key = _new_attention_norm_key(layer, config)
        count = (
            config.mova_num_experts
            if xllm_key.endswith(".mova.wv.weight")
            else config.num_experts
        )
        is_expert_weight = "{expert}" in hf_template
        if is_expert_weight:
            hf_keys = [hf_template.format(**fields, expert=i) for i in range(count)]
            tensors = [source[key] for key in hf_keys]
            weight = torch.stack(tensors) if materialize else None
            shape = (count, *expected[hf_keys[0]].shape)
            dtype = tensors[0].dtype
            if any(tuple(t.shape) != tuple(expected[key].shape) for key, t in zip(hf_keys, tensors)):
                raise ValueError(f"Invalid expert shape for {xllm_key}")
            if any(t.dtype != dtype for t in tensors):
                raise ValueError(f"Mixed expert dtypes for {xllm_key}")
        else:
            hf_key = hf_template.format(**fields)
            tensor = source[hf_key]
            if tuple(tensor.shape) != tuple(expected[hf_key].shape):
                raise ValueError(
                    f"{hf_key}: shape {tuple(tensor.shape)} != {tuple(expected[hf_key].shape)}"
                )
            shape, dtype = tuple(tensor.shape), tensor.dtype
            weight = tensor if materialize else None

        split_axis = split_dim if split_dim is None else split_dim % len(shape)
        if split_axis is not None:
            if shape[split_axis] % tp_size:
                raise ValueError(f"{xllm_key}: shape {shape} is not divisible by TP={tp_size}")
            shape = (*shape[:split_axis], shape[split_axis] // tp_size, *shape[split_axis + 1 :])
        if is_expert_weight and expert_weight_format == "2d":
            shape = (shape[0] * shape[1], *shape[2:])
        specs[xllm_key] = (tuple(shape), target_dtype)

        if materialize:
            weight = _transform(xllm_key, weight, config)
            weight = _slice(weight, tp_rank, tp_size, split_dim)
            if is_expert_weight and expert_weight_format == "2d":
                weight = weight.flatten(0, 1)
            if weight.dtype != target_dtype:
                weight = weight.to(target_dtype)
            output[xllm_key] = weight
    return output, specs


def build_tp_state_dict(
    reader: HFCheckpointReader,
    tp_rank: int,
    tp_size: int,
    materialize: bool,
    target_dtype: torch.dtype,
    norm_key_format: str,
    expert_weight_format: str,
):
    config = reader.config
    expected = _model_state(config)
    state, specs = {}, {}
    started = time.monotonic()
    for layer in range(config.num_hidden_layers):
        params = [
            param
            for param in _layer_params(config, layer)
            if param[1].format(layer=layer, expert=0) in expected
        ]
        with reader.tensors(_hf_keys(params, layer, config)) as source:
            converted, converted_specs = _convert_params(
                source, expected, params, layer, config, tp_rank, tp_size,
                materialize, target_dtype, norm_key_format, expert_weight_format,
            )
        state.update(converted)
        specs.update(converted_specs)
        logger.info("TP %d/%d: layer %d/%d%s", tp_rank, tp_size, layer + 1,
                    config.num_hidden_layers, " materialized" if materialize else " validated")

    keys = [hf for _, hf, _ in OTHER_PARAMS]
    with reader.tensors(keys) as source:
        converted, converted_specs = _convert_params(
            source, expected, OTHER_PARAMS, None, config, tp_rank, tp_size,
            materialize, target_dtype, norm_key_format, expert_weight_format,
        )
    state.update(converted)
    specs.update(converted_specs)
    size = sum(math.prod(shape) * torch.empty((), dtype=dtype).element_size()
               for shape, dtype in specs.values())
    logger.info("TP %d/%d: %d tensors, %s parameters, %.2f GiB in %.1fs",
                tp_rank, tp_size, len(specs),
                f"{sum(math.prod(s) for s, _ in specs.values()):,}",
                size / 2**30, time.monotonic() - started)
    return state, specs


def _validate_hf(reader: HFCheckpointReader, strict_keys: bool):
    config = reader.config
    if _config_value(config, "attention_bias", False):
        raise ValueError("HF attention biases cannot be represented in XLLM")
    if _config_value(config, "tie_word_embeddings", False):
        raise ValueError("Tied embeddings are not supported by this bridge")
    expected = set(_model_state(config))
    actual = set(reader.weight_map)
    missing, extra = expected - actual, actual - expected
    if missing or (strict_keys and extra):
        raise ValueError(
            f"HF key mismatch: missing={sorted(missing)[:20]}, extra={sorted(extra)[:20]}"
        )
    if extra:
        logger.warning("Ignoring %d unexpected HF tensors", len(extra))


def _validate_dcp(path: Path, specs: Mapping[str, TargetSpec]):
    actual = FileSystemReader(path).read_metadata().state_dict_metadata
    if actual.keys() != specs.keys():
        raise ValueError("Written DCP keys differ from the conversion plan")
    for key, (shape, dtype) in specs.items():
        metadata = actual[key]
        if not isinstance(metadata, TensorStorageMetadata):
            raise TypeError(f"{key} is not tensor metadata")
        if tuple(metadata.size) != shape or metadata.properties.dtype != dtype:
            raise ValueError(f"Written DCP metadata differs for {key}")


def _write_dcp(state, specs, save_dir: Path, rank: int, threads: int, sync: bool):
    destination = save_dir / f"full_model.tp{rank:02d}"
    temporary = save_dir / f".full_model.tp{rank:02d}.tmp-{os.getpid()}"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing checkpoint: {destination}")
    if temporary.exists():
        shutil.rmtree(temporary)
    try:
        writer = FileSystemWriter(temporary, single_file_per_rank=True,
                                  sync_files=sync, thread_count=threads, overwrite=False)
        _save_state_dict(state, storage_writer=writer, no_dist=True)
        _validate_dcp(temporary, specs)
        os.replace(temporary, destination)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return destination


def _xllm_config(config: K2HorizonConfig, hf_dir: Path, tp_size: int,
                 dtype: str, router_bias_update_rate: float):
    rope = config.rope_parameters
    if rope is None:
        raise ValueError("HF config must define rope_parameters")
    rope_theta = rope["rope_theta"] if isinstance(rope, dict) else rope.rope_theta
    dense = len(config.mlp_only_layers) if config.num_experts else None
    return {
        "model_parallel_size": tp_size,
        "context_parallel_size": 1,
        "slurm": {"is_slurm_job": False, "global_rank": 0, "world_size": tp_size},
        "dtype": dtype,
        "seq_len": config.max_position_embeddings,
        "tokenizer": {"type": "huggingface", "path": str(hf_dir.resolve()),
                      "num_reserved_special_tokens": 0, "data_tokenized": False},
        "model": {
            "arch": "transformer", "num_layers": config.num_hidden_layers,
            "model_dim": config.hidden_size, "num_heads": config.num_attention_heads,
            "num_kv_heads": config.num_key_value_heads, "head_dim": config.head_dim,
            "qknorm": config.query_key_norm, "ffn_hidden_dim": config.intermediate_size,
            "num_experts": config.num_experts,
            "num_activated_experts": config.num_experts_per_tok,
            "num_shared_experts": config.num_shared_experts, "num_dense_layers": dense,
            "expert_inter_dim": config.moe_intermediate_size,
            "moe_router_score_func": config.router_score_func,
            "moe_router_bias": config.moe_gate_bias,
            "moe_router_bias_update_rate": router_bias_update_rate if config.moe_gate_bias else None,
            "moe_router_scaling_factor": config.router_scaling_factor,
            "vocab_size": config.vocab_size,
            "layernorm_num_groups": config.layernorm_num_groups,
            "norm_eps": config.rms_norm_eps, "apply_rmsnorm": True,
            "rope_base": rope_theta, "rope_head_dim": config.rope_head_dim,
            "attention_dropout": config.attention_dropout, "swiglu": True,
            "num_values": config.mova_num_experts,
            "num_activated_values": config.mova_num_experts_per_tok,
            "apply_attn_gate": config.attention_gate_func is not None,
            "attn_gate_func": config.attention_gate_func,
            "init_std": config.initializer_range,
            "fused_block": False, "init_mode": "none",
        },
    }


def _dtype_name(specs: Mapping[str, TargetSpec]):
    dtypes = {dtype for _, dtype in specs.values()}
    names = {torch.float32: "fp32", torch.float16: "fp16", torch.bfloat16: "bf16"}
    if len(dtypes) != 1 or next(iter(dtypes)) not in names:
        raise ValueError(f"XLLM requires one supported dtype, got {dtypes}")
    return names[next(iter(dtypes))]


def _target_dtype(name: str) -> torch.dtype:
    aliases = {
        "float32": torch.float32, "fp32": torch.float32,
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
        "float16": torch.float16, "fp16": torch.float16,
    }
    try:
        return aliases[name.lower()]
    except KeyError as exc:
        raise ValueError(
            f'dtype must be one of {sorted(aliases)}, got {name!r}'
        ) from exc


def _validate_expert_weight_format(value: str) -> str:
    if value not in {"2d", "3d"}:
        raise ValueError(
            f'expert_weight_format must be "2d" or "3d", got {value!r}'
        )
    return value


def _tp_size(config: K2HorizonConfig, requested: int | None) -> int:
    inferred = getattr(config, "xllm_model_parallel_size", None)
    size = requested if requested is not None else (inferred or 1)
    if size < 1:
        raise ValueError("tp_size must be positive")
    if requested is None:
        if inferred is None:
            logger.info("HF config has no XLLM TP metadata; defaulting to TP=1")
        else:
            logger.info("Inferred TP=%d from the HF config", size)
    return size


def _write_json(path: Path, value: Mapping[str, object]):
    if path.exists():
        with path.open() as fp:
            if json.load(fp) != value:
                raise FileExistsError(f"Existing {path} belongs to a different conversion")
        return
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def convert_checkpoint(
    hf_dir: str,
    save_dir: str,
    tp_size: int | None = None,
    tp_rank: int = -1,
    dry_run: bool = False,
    strict_keys: bool = True,
    writer_threads: int = 1,
    sync_files: bool = True,
    write_config: bool = True,
    router_bias_update_rate: float = 1e-3,
    dtype: str = "float32",
    norm_key_format: str = "new",
    expert_weight_format: str = "2d",
) -> None:
    """Convert one TP rank, or every rank serially when ``tp_rank=-1``."""
    dry_run = _bool(dry_run, "dry_run")
    strict_keys = _bool(strict_keys, "strict_keys")
    sync_files = _bool(sync_files, "sync_files")
    write_config = _bool(write_config, "write_config")
    if writer_threads < 1:
        raise ValueError("writer_threads must be positive")
    if norm_key_format not in {"new", "legacy"}:
        raise ValueError(f'norm_key_format must be "new" or "legacy", got {norm_key_format!r}')
    expert_weight_format = _validate_expert_weight_format(expert_weight_format)
    reader = HFCheckpointReader(Path(hf_dir))
    _validate_hf(reader, strict_keys)
    tp_size = _tp_size(reader.config, tp_size)
    output_dtype = _target_dtype(dtype)
    if tp_rank != -1 and not 0 <= tp_rank < tp_size:
        raise ValueError(f"tp_rank must be -1 or in [0, {tp_size})")
    ranks: Iterable[int] = range(tp_size) if tp_rank == -1 else (tp_rank,)
    if dry_run:
        ranks = (0,)
    output = Path(save_dir)
    if not dry_run:
        output.mkdir(parents=True, exist_ok=True)

    for rank in ranks:
        state, specs = build_tp_state_dict(
            reader, rank, tp_size, not dry_run, output_dtype, norm_key_format,
            expert_weight_format)
        if dry_run:
            continue
        dtype = _dtype_name(specs)
        if write_config:
            _write_json(output / "config.json", _xllm_config(
                reader.config, Path(hf_dir), tp_size, dtype, router_bias_update_rate))
        dcp = _write_dcp(state, specs, output, rank, writer_threads, sync_files)
        del state
        gc.collect()
        _write_json(output / f"conversion_info.tp{rank:02d}.json", {
            "source_hf_dir": str(Path(hf_dir).resolve()),
            "output_dcp_dir": str(dcp.resolve()), "tp_size": tp_size, "tp_rank": rank,
            "norm_key_format": norm_key_format,
            "expert_weight_format": expert_weight_format,
            "num_tensors": len(specs),
            "num_parameters": sum(math.prod(shape) for shape, _ in specs.values()),
            "source_format": reader.checkpoint_format, "output_dtype": dtype,
            "target_dtype": dtype,
        })


def _rank_worker(kwargs):
    rank = kwargs.pop("tp_rank")
    log_path = Path(kwargs.pop("log_path"))
    threads = str(kwargs.pop("threads_per_rank"))
    os.environ.update(OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads, MALLOC_ARENA_MAX="4")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        handlers=[logging.FileHandler(log_path), logging.StreamHandler()], force=True)
    convert_checkpoint(tp_rank=rank, **kwargs)
    return rank


def _slurm_ranks(tp_size: int, concurrent_ranks: int) -> list[int]:
    task = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task is None:
        return list(range(tp_size))
    groups = math.ceil(tp_size / concurrent_ranks)
    task_id = int(task)
    task_count = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", groups))
    if task_count != groups or not 0 <= task_id < groups:
        raise ValueError(f"Use a Slurm array with {groups} elements: --array=0-{groups - 1}%{groups}")
    first = task_id * concurrent_ranks
    return list(range(first, min(first + concurrent_ranks, tp_size)))


def _available_memory_gib() -> int:
    with open("/proc/meminfo") as fp:
        for line in fp:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024 // 1024
    raise RuntimeError("MemAvailable is absent from /proc/meminfo")


def _memory_limited_concurrency(requested: int, rank_bytes: int) -> int:
    """Use at most 90% of currently available RAM for materialized TP ranks."""
    if rank_bytes <= 0:
        raise ValueError("rank checkpoint size must be positive")
    available_bytes = _available_memory_gib() * 2**30
    affordable = max(1, int(available_bytes * 0.9) // rank_bytes)
    active = min(requested, affordable)
    if active < requested:
        logger.info(
            "Capping rank concurrency from %d to %d (%.1f GiB/rank, %.1f GiB available)",
            requested, active, rank_bytes / 2**30, available_bytes / 2**30,
        )
    return active


def main(
    hf_dir: str,
    save_dir: str,
    tp_size: int | None = None,
    dtype: str = "float32",
    writer_threads: int = 8,
    concurrent_ranks: int | None = None,
    norm_key_format: str = "new",
    expert_weight_format: str = "2d",
) -> None:
    """Convert HF weights, optionally batching ranks in parallel.

    TP is inferred from bridge-produced HF checkpoints and otherwise defaults
    to one. All selected ranks are converted; a Slurm array selects adjacent
    groups. Rank concurrency defaults to the TP size and is capped by currently
    available host memory.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    if writer_threads < 1:
        raise ValueError("writer_threads must be positive")
    reader = HFCheckpointReader(Path(hf_dir))
    _validate_hf(reader, strict_keys=True)
    tp_size = _tp_size(reader.config, tp_size)
    if concurrent_ranks is None:
        concurrent_ranks = tp_size
    if concurrent_ranks < 1:
        raise ValueError("concurrent_ranks must be positive")
    output_dtype = _target_dtype(dtype)
    expert_weight_format = _validate_expert_weight_format(expert_weight_format)
    ranks = _slurm_ranks(tp_size, concurrent_ranks)
    output = Path(save_dir)
    existing = [output / f"full_model.tp{rank:02d}" for rank in ranks
                if (output / f"full_model.tp{rank:02d}").exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing checkpoint(s): "
            + ", ".join(map(str, existing))
        )

    common = dict(hf_dir=hf_dir, save_dir=save_dir, tp_size=tp_size,
                  strict_keys=True, writer_threads=writer_threads, sync_files=True,
                  write_config=True, router_bias_update_rate=1e-3, dtype=dtype,
                  norm_key_format=norm_key_format,
                  expert_weight_format=expert_weight_format)
    logger.info("Planning TP rank memory")
    _, planned_specs = build_tp_state_dict(
        reader, 0, tp_size, False, output_dtype, norm_key_format,
        expert_weight_format)
    rank_bytes = sum(
        math.prod(shape) * torch.empty((), dtype=spec_dtype).element_size()
        for shape, spec_dtype in planned_specs.values()
    )
    active = _memory_limited_concurrency(
        min(concurrent_ranks, len(ranks)), rank_bytes)
    output.mkdir(parents=True, exist_ok=True)
    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    threads = max(1, cpus // active)
    logger.info("Converting ranks %s with concurrency=%d, threads/rank=%d", ranks, active, threads)

    jobs = []
    for rank in ranks:
        jobs.append(dict(common, tp_rank=rank, dry_run=False, threads_per_rank=threads,
                         log_path=str(output / f"convert.tp{rank:02d}.log")))
    if active == 1:
        for job in jobs:
            _rank_worker(job)
        return
    with ProcessPoolExecutor(max_workers=active) as pool:
        futures = {pool.submit(_rank_worker, job): job["tp_rank"] for job in jobs}
        for future in as_completed(futures):
            rank = futures[future]
            future.result()
            logger.info("TP rank %d completed", rank)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
