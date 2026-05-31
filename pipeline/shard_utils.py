"""Helpers for deterministic row sharding."""

from __future__ import annotations


def select_shard(rows: list[dict], num_shards: int, shard_index: int) -> list[dict]:
    if num_shards < 1:
        raise SystemExit("--num-shards must be at least 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise SystemExit("--shard-index must be between 0 and --num-shards - 1")
    if num_shards == 1:
        return rows
    return [row for index, row in enumerate(rows) if index % num_shards == shard_index]
