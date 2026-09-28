"""Compare two XLLM DCP checkpoints one tensor at a time."""

import argparse
import gc
import re
from pathlib import Path

import torch
from torch.distributed.checkpoint import FileSystemReader, load_state_dict
from torch.distributed.checkpoint.metadata import TensorStorageMetadata


GROUPED_WEIGHT_PATTERN = re.compile(
    r"^layers\.\d+\.(?:moe\.experts\.weight[123]|mova\.wv\.weight)$"
)


def canonical_shape(key, shape):
    """Flatten the explicit expert axis used by legacy grouped weights."""
    shape = tuple(shape)
    if GROUPED_WEIGHT_PATTERN.match(key) and len(shape) == 3:
        return (shape[0] * shape[1], shape[2])
    return shape


def canonical_tensor(key, tensor):
    if GROUPED_WEIGHT_PATTERN.match(key) and tensor.ndim == 3:
        return tensor.flatten(0, 1)
    return tensor


def tensor_metadata(reader):
    metadata = reader.read_metadata().state_dict_metadata
    invalid = [key for key, value in metadata.items()
               if not isinstance(value, TensorStorageMetadata)]
    if invalid:
        raise TypeError(f"Non-tensor DCP entries: {invalid[:20]}")
    return metadata


def canonical_key(key):
    """Treat pre/post-5b9cf866 block input norm names as equivalent."""
    return re.sub(
        r"^(layers\.\d+)\.(?:attention|mova)\.norm\.(weight|bias)$",
        r"\1.norm.\2",
        key,
    )


def canonical_metadata(reader, label):
    result = {}
    for physical_key, metadata in tensor_metadata(reader).items():
        key = canonical_key(physical_key)
        if key in result:
            raise ValueError(
                f"{label} contains duplicate legacy/new keys for {key}: "
                f"{result[key][0]} and {physical_key}"
            )
        result[key] = (physical_key, metadata)
    return result


def load_one(reader, metadata, key):
    state = {key: torch.empty(tuple(metadata.size), dtype=metadata.properties.dtype)}
    load_state_dict(state, storage_reader=reader, no_dist=True)
    return state[key]


def compare(source_dir, roundtrip_dir, rtol=1e-6, atol=1e-7):
    source_reader = FileSystemReader(Path(source_dir))
    roundtrip_reader = FileSystemReader(Path(roundtrip_dir))
    source = canonical_metadata(source_reader, "source")
    roundtrip = canonical_metadata(roundtrip_reader, "roundtrip")
    if source.keys() != roundtrip.keys():
        raise AssertionError(
            f"DCP key mismatch: missing={sorted(source.keys() - roundtrip.keys())[:20]}, "
            f"extra={sorted(roundtrip.keys() - source.keys())[:20]}"
        )

    keys = sorted(source, key=lambda key: source[key][1].size.numel())
    for index, key in enumerate(keys, 1):
        left_key, left_meta = source[key]
        right_key, right_meta = roundtrip[key]
        left_shape = canonical_shape(key, left_meta.size)
        right_shape = canonical_shape(key, right_meta.size)
        if left_shape != right_shape:
            raise AssertionError(
                f"{key}: shape mismatch: {tuple(left_meta.size)} vs "
                f"{tuple(right_meta.size)}"
            )
        if left_meta.properties.dtype != right_meta.properties.dtype:
            raise AssertionError(f"{key}: dtype mismatch")
        left = load_one(source_reader, left_meta, left_key)
        right = load_one(roundtrip_reader, right_meta, right_key)
        left = canonical_tensor(key, left)
        right = canonical_tensor(key, right)
        try:
            torch.testing.assert_close(
                left, right,
                rtol=rtol if left.is_floating_point() else 0,
                atol=atol if left.is_floating_point() else 0,
            )
        except AssertionError as exc:
            raise AssertionError(f"Round-trip mismatch for {key}: {exc}") from exc
        del left, right
        gc.collect()
        if index % 25 == 0 or index == len(keys):
            print(f"Compared {index}/{len(keys)} tensors", flush=True)
    print(f"PASS: {len(keys)} tensors match (rtol={rtol}, atol={atol})", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", required=True)
    parser.add_argument("--roundtrip_dir", required=True)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--atol", type=float, default=1e-7)
    compare(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
