from __future__ import annotations

import numpy as np
import pytest

from plmmsa.align.base import Aligner, Alignment
from plmmsa.align.plmalign import PLMAlign

_RNG = np.random.default_rng(42)
_DIM = 8
_LETTER_CACHE: dict[str, np.ndarray] = {}


def _one_hot_embeddings(seq: str) -> np.ndarray:
    """Map residues to a fixed random embedding per letter — identical letters
    produce identical rows across calls, so cosine similarity is a clean proxy
    for a match/mismatch signal."""
    rows: list[np.ndarray] = []
    for r in seq:
        if r not in _LETTER_CACHE:
            _LETTER_CACHE[r] = _RNG.standard_normal(_DIM).astype(np.float32)
        rows.append(_LETTER_CACHE[r])
    return np.stack(rows, axis=0)


def test_plmalign_is_aligner_subclass() -> None:
    assert issubclass(PLMAlign, Aligner)
    assert PLMAlign.id == "plmalign"


def test_local_identity_is_full_length() -> None:
    seq = "MKTIIALSYIFCLVFA"
    emb = _one_hot_embeddings(seq)
    aligner = PLMAlign()

    results = aligner.align(emb, [emb], mode="local")
    assert len(results) == 1
    a = results[0]

    assert a.mode == "local"
    assert a.length == len(seq)
    assert a.query_start == 0 and a.query_end == len(seq)
    assert a.target_start == 0 and a.target_end == len(seq)
    assert a.identity(seq, seq) == pytest.approx(1.0)
    # Every column is a non-gap match.
    assert all(qi >= 0 and ti >= 0 for qi, ti in a.columns)


def test_local_finds_substring_without_flanks() -> None:
    query = "LSYIFCL"
    target = "XXXX" + query + "YYY"
    q_emb = _one_hot_embeddings(query)
    t_emb = _one_hot_embeddings(target)
    aligner = PLMAlign()

    a = aligner.align(q_emb, [t_emb], mode="local")[0]

    assert a.length == len(query)
    assert a.query_start == 0 and a.query_end == len(query)
    assert a.target_start == 4  # skipped the XXXX prefix
    assert a.target_end == 4 + len(query)
    assert a.identity(query, target) == pytest.approx(1.0)


def test_global_end_to_end_with_gaps() -> None:
    query = "AGCTA"
    target = "AGTA"  # missing C — expect a single gap in target
    q_emb = _one_hot_embeddings(query)
    t_emb = _one_hot_embeddings(target)
    aligner = PLMAlign()

    a = aligner.align(q_emb, [t_emb], mode="global", gap_open=1.0, gap_extend=0.5)[0]

    assert a.mode == "global"
    assert a.query_start == 0 and a.query_end == len(query)
    assert a.target_start == 0 and a.target_end == len(target)
    rendered_q, rendered_t = a.render(query, target)
    assert len(rendered_q) == len(rendered_t)
    assert rendered_q.replace("-", "") == query
    assert rendered_t.replace("-", "") == target
    # At least one gap is introduced on the target side.
    assert rendered_t.count("-") >= 1


def test_high_gap_penalty_prefers_ungapped() -> None:
    query = "AGCTA"
    target = "AGCA"  # one-residue deletion relative to query
    q_emb = _one_hot_embeddings(query)
    t_emb = _one_hot_embeddings(target)
    aligner = PLMAlign()

    cheap = aligner.align(q_emb, [t_emb], mode="local", gap_open=0.1, gap_extend=0.1)[0]
    expensive = aligner.align(q_emb, [t_emb], mode="local", gap_open=50.0, gap_extend=50.0)[0]

    # Cheap gaps should produce a longer alignment (more columns used).
    assert cheap.length >= expensive.length


def test_kwargs_pass_through_unknown_keys() -> None:
    """The Aligner interface accepts extra kwargs — unknown ones are ignored
    by PLMAlign so callers can forward a single dict to any backend."""
    aligner = PLMAlign()
    emb = _one_hot_embeddings("AGC")
    result = aligner.align(emb, [emb], mode="local", made_up_knob=123, another=True)[0]
    assert isinstance(result, Alignment)
    assert result.length == 3


def test_batch_preserves_order() -> None:
    aligner = PLMAlign()
    q = _one_hot_embeddings("AGC")
    targets = [
        _one_hot_embeddings("AGC"),
        _one_hot_embeddings("XXAGCYY"),
        _one_hot_embeddings("QQQ"),
    ]
    results = aligner.align(q, targets, mode="local")
    assert len(results) == 3
    assert results[0].length == 3
    assert results[1].length == 3
    assert results[1].target_start == 2
    # Totally unrelated target should yield a shorter / lower-scoring alignment.
    assert results[2].score <= results[0].score
