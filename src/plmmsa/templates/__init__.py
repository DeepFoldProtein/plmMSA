"""Templates re-alignment pipeline.

See `PLAN_TEMPLATES_REALIGN.md`. Re-aligns hmmsearch-style A3M records
against the query under OTalign / Ankh-Large / glocal, with query-side
insertions stripped from the output (no lowercase letters in rendered
rows).
"""

from __future__ import annotations

from plmmsa.templates.a3m_parser import (
    DroppedRecord,
    ParseResult,
    Record,
    columns_from_a3m_row,
    parse_hmmsearch_a3m,
)

__all__ = [
    "DroppedRecord",
    "ParseResult",
    "Record",
    "columns_from_a3m_row",
    "parse_hmmsearch_a3m",
]
