from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from redis.asyncio import Redis

DEFAULT_SEQ_KEY_FORMAT = "seq:{id}"


class TargetFetcher(ABC):
    """Returns target sequences for a set of UniRef-like ids.

    The orchestrator fetches neighbor sequences after a VDB lookup but before
    re-embedding + alignment. Real deployments back this with the UniRef50
    shard tree; tests use the in-memory `DictTargetFetcher`.
    """

    @abstractmethod
    async def fetch(self, collection: str, ids: Sequence[str]) -> dict[str, str]:
        """Map requested ids to sequences. Unknown ids are silently dropped."""
        ...


class DictTargetFetcher(TargetFetcher):
    """In-memory fetcher. Unknown ids are dropped."""

    def __init__(self, id_to_seq: Mapping[str, str] | None = None) -> None:
        self._data: dict[str, str] = dict(id_to_seq or {})

    async def fetch(self, collection: str, ids: Sequence[str]) -> dict[str, str]:
        return {i: self._data[i] for i in ids if i in self._data}


class RedisTargetFetcher(TargetFetcher):
    """Redis-backed fetcher.

    Keys follow `key_format`, which may reference `{id}` and `{collection}`.
    Values are raw sequence bytes (utf-8 encoded). The complementary
    populator is `plmmsa.tools.build_sequence_cache`, which streams a FASTA
    into the same key space.
    """

    def __init__(
        self,
        redis: Redis,
        *,
        key_format: str = DEFAULT_SEQ_KEY_FORMAT,
    ) -> None:
        self._redis = redis
        self._key_format = key_format

    async def fetch(self, collection: str, ids: Sequence[str]) -> dict[str, str]:
        if not ids:
            return {}
        keys = [self._key_format.format(collection=collection, id=i) for i in ids]
        raw = await self._redis.mget(keys)  # pyright: ignore[reportGeneralTypeIssues]
        result: dict[str, str] = {}
        for ident, value in zip(ids, raw, strict=True):
            if value is None:
                continue
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            result[ident] = value
        return result


class FastaTargetFetcher(TargetFetcher):
    """Loads a FASTA file into memory once; returns sequences by id.

    For production shard access (UniRef50 tree), add a streaming / indexed
    fetcher that avoids loading the whole corpus at init.
    """

    def __init__(self, fasta_path: Path | str) -> None:
        self._data = _parse_fasta(Path(fasta_path))

    async def fetch(self, collection: str, ids: Sequence[str]) -> dict[str, str]:
        return {i: self._data[i] for i in ids if i in self._data}


def _parse_fasta(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    current_id: str | None = None
    current_parts: list[str] = []

    def flush() -> None:
        if current_id is not None:
            data[current_id] = "".join(current_parts)

    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            flush()
            current_id = line[1:].split()[0]
            current_parts = []
        else:
            current_parts.append(line)
    flush()
    return data


def dict_fetcher_from_fasta_text(text: str) -> DictTargetFetcher:
    """Helper for tests: parse a FASTA string into a DictTargetFetcher."""
    data: dict[str, str] = {}
    current_id: str | None = None
    current_parts: list[str] = []

    def flush() -> None:
        if current_id is not None:
            data[current_id] = "".join(current_parts)

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            flush()
            current_id = line[1:].split()[0]
            current_parts = []
        else:
            current_parts.append(line)
    flush()
    return DictTargetFetcher(data)


def _coerce_ids(ids: Iterable[str]) -> list[str]:
    return list(ids)
