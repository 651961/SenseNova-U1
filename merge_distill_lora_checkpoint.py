#!/usr/bin/env python3
"""Merge LoRA tensors directly into a safetensors checkpoint directory."""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


def load_lora(path: Path):
    result = {}
    stats = Counter()
    with safe_open(str(path), framework="pt", device="cpu") as f:
        keys = set(f.keys())
        for down_name in sorted(k for k in keys if k.endswith(".lora_down.weight")):
            up_name = down_name.replace(".lora_down.weight", ".lora_up.weight")
            alpha_name = down_name.replace(".lora_down.weight", ".alpha")
            if up_name not in keys or alpha_name not in keys:
                raise ValueError(f"incomplete LoRA entry: {down_name}")
            down = f.get_tensor(down_name).float()
            up = f.get_tensor(up_name).float()
            if down.ndim != 2 or up.ndim != 2 or down.shape[0] != up.shape[1]:
                raise ValueError(f"bad LoRA shapes for {down_name}: {tuple(down.shape)}, {tuple(up.shape)}")
            alpha = float(f.get_tensor(alpha_name).item())
            rank = int(down.shape[0])
            base_name = down_name[: -len(".lora_down.weight")] + ".weight"
            if base_name.startswith("diffusion_model."):
                base_name = base_name[len("diffusion_model.") :]
            result[base_name] = (up, down, alpha, rank)
            stats[(alpha, rank)] += 1
    return result, stats


def load_and_validate_index(base: Path) -> tuple[dict, dict[str, str], set[str]]:
    index_path = base / "model.safetensors.index.json"
    if not index_path.is_file():
        raise ValueError(f"missing checkpoint index: {index_path}")
    index = json.loads(index_path.read_text())
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"invalid or empty weight_map in {index_path}")

    expected_by_shard: dict[str, set[str]] = defaultdict(set)
    for tensor_name, shard_name in weight_map.items():
        expected_by_shard[shard_name].add(tensor_name)

    actual_shards = {p.name for p in base.glob("model-*.safetensors")}
    referenced_shards = set(expected_by_shard)
    unreferenced = actual_shards - referenced_shards
    for shard_name in sorted(unreferenced):
        with safe_open(str(base / shard_name), framework="pt", device="cpu") as f:
            if f.keys():
                raise ValueError(
                    "checkpoint shard layout does not match the index: "
                    f"unreferenced shard contains tensors: {shard_name}"
                )
    missing = referenced_shards - actual_shards
    if missing:
        raise ValueError(
            "checkpoint shard layout does not match the index: "
            f"missing={sorted(missing)}"
        )

    for shard_name, expected_names in sorted(expected_by_shard.items()):
        source = base / shard_name
        if not source.is_file():
            raise ValueError(f"index references missing shard: {source}")
        with safe_open(str(source), framework="pt", device="cpu") as f:
            actual_names = set(f.keys())
        if actual_names != expected_names:
            missing = sorted(expected_names - actual_names)
            extra = sorted(actual_names - expected_names)
            raise ValueError(
                f"shard/index mismatch in {source}: "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )

    return index, weight_map, unreferenced


def merge(base: Path, output: Path, lora: dict, weight_map: dict[str, str]) -> int:
    missing = sorted(set(lora) - set(weight_map))
    if missing:
        raise ValueError(
            f"matched {len(lora) - len(missing)}/{len(lora)} LoRA tensors; "
            f"missing examples: {missing[:5]}"
        )

    lora_by_shard: dict[str, dict] = defaultdict(dict)
    for tensor_name, entry in lora.items():
        lora_by_shard[weight_map[tensor_name]][tensor_name] = entry

    matched = 0
    # copytree() has already preserved the original checkpoint byte-for-byte.
    # Rewrite only shards which actually contain LoRA targets.
    for shard_name, shard_lora in sorted(lora_by_shard.items()):
        source = base / shard_name
        with safe_open(str(source), framework="pt", device="cpu") as f:
            tensors = {}
            for name in f.keys():
                value = f.get_tensor(name)
                if name in shard_lora:
                    up, down, alpha, rank = shard_lora[name]
                    delta = (alpha / rank) * (up @ down)
                    if tuple(delta.shape) != tuple(value.shape):
                        raise ValueError(f"shape mismatch for {name}: {tuple(delta.shape)} vs {tuple(value.shape)}")
                    value = (value.float() + delta).to(value.dtype)
                    matched += 1
                tensors[name] = value.contiguous()
            save_file(tensors, str(output / shard_name), metadata=f.metadata())
    if matched != len(lora):
        raise AssertionError(f"internal error: merged {matched}/{len(lora)} LoRA tensors")
    return matched


def validate_output(
    base: Path, output: Path, weight_map: dict[str, str], referenced_shards: set[str]
) -> None:
    # A merged native SenseNova checkpoint must retain the base checkpoint's
    # layout. In particular, do not create a Diffusers model_index.json.
    structural_files = [
        "config.json",
        "model.safetensors.index.json",
        "tokenizer_config.json",
    ]
    for name in structural_files:
        source = base / name
        target = output / name
        if source.exists() != target.exists():
            raise ValueError(f"output layout changed for {name}")
        if source.is_file() and source.read_bytes() != target.read_bytes():
            raise ValueError(f"output unexpectedly modified {name}")

    output_shards = {p.name for p in output.glob("model-*.safetensors")}
    if output_shards != referenced_shards:
        raise ValueError(
            "output shard layout differs from base: "
            f"added={sorted(output_shards - referenced_shards)}, "
            f"removed={sorted(referenced_shards - output_shards)}"
        )

    expected_by_shard: dict[str, set[str]] = defaultdict(set)
    for tensor_name, shard_name in weight_map.items():
        expected_by_shard[shard_name].add(tensor_name)
    for shard_name, expected_names in sorted(expected_by_shard.items()):
        with safe_open(str(output / shard_name), framework="pt", device="cpu") as f:
            actual_names = set(f.keys())
        if actual_names != expected_names:
            raise ValueError(f"merged shard/index mismatch: {shard_name}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_path", required=True)
    p.add_argument("--lora_path", required=True)
    p.add_argument("--output_path", required=True)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    base = Path(args.model_path).expanduser().resolve()
    lora_path = Path(args.lora_path).expanduser().resolve()
    output = Path(args.output_path).expanduser().resolve()
    if not base.is_dir() or not lora_path.is_file():
        raise SystemExit("model_path must be a directory and lora_path must be a file")
    if output == base:
        raise SystemExit("output_path must differ from model_path")
    if output.exists() and not args.overwrite:
        raise SystemExit(f"output exists: {output}; use --overwrite")
    _, weight_map, unreferenced_shards = load_and_validate_index(base)
    lora, stats = load_lora(lora_path)
    if not lora:
        raise SystemExit(f"no LoRA tensors found in {lora_path}")
    print(f"[lora] tensors={len(lora)}")
    for (alpha, rank), count in sorted(stats.items()):
        print(f"[lora] alpha={alpha:g}, rank={rank}, alpha/rank={alpha / rank:g}, tensors={count}")
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=str(output.parent)) as tmp:
        staging = Path(tmp) / output.name
        shutil.copytree(base, staging)
        # Some exporters leave empty, unreferenced safetensors placeholders.
        # They are not part of the indexed checkpoint and must not be carried
        # into the merged output.
        for shard_name in unreferenced_shards:
            (staging / shard_name).unlink()
        matched = merge(base, staging, lora, weight_map)
        validate_output(base, staging, weight_map, set(weight_map.values()))
        if output.exists():
            shutil.rmtree(output)
        staging.replace(output)
    print(f"[done] merged {matched} tensors; native checkpoint layout preserved")


if __name__ == "__main__":
    main()
"""
python /datasets/codes_zsqiao/SenseNova-U1/merge_lora_checkpoint.py \
    --model_path /models/SenseNova-U1.5-8B-MoT-EMA-1500step\
    --lora_path /models/SenseNova-U1.5-8B-MoT-LoRAs/SenseNova-U1.5-8B-MoT-LoRA-8step.safetensors \
    --output_path /models/ckpt_zsqiao/SenseNova-U1.5-8B-MoT-EMA-1500step-distill
"""
