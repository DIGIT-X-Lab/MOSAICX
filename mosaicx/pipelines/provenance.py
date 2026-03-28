# mosaicx/pipelines/provenance.py
"""Coordinate-based source provenance utilities."""

from __future__ import annotations

import bisect
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mosaicx.documents.models import LoadedDocument


def locate_in_document(
    doc: LoadedDocument,
    start: int,
    end: int,
) -> list[dict] | None:
    """Map a char range to page bboxes in a LoadedDocument.

    Uses binary search to find all TextBlocks that overlap [start, end),
    groups them by page, and returns the union bbox for each page.

    Parameters
    ----------
    doc:
        A LoadedDocument with a populated ``text_blocks`` list.
    start:
        Inclusive start char offset in the full document text.
    end:
        Exclusive end char offset in the full document text.

    Returns
    -------
    list[dict] | None
        One entry per page touched by the range, each with keys
        ``page`` (int), ``bbox`` (tuple), ``start`` (int), ``end`` (int).
        Returns ``None`` if ``doc.text_blocks`` is empty.
    """
    if not doc.text_blocks:
        return None

    blocks = doc.text_blocks

    # Build a sorted list of block start offsets for binary search.
    # Blocks are assumed to be sorted by start; if not, sort defensively.
    starts = [b.start for b in blocks]

    # Find the first block whose *end* is > query start (i.e., could overlap).
    # We look for blocks where b.end > start AND b.start < end.
    # Binary search to skip blocks that end before our range.
    # bisect_left finds first index where starts[i] >= start;
    # but blocks before that index might still have end > start, so
    # we walk back one position to catch partial overlaps.
    right_idx = bisect.bisect_left(starts, end)  # first block starting at or after end
    left_idx = bisect.bisect_right(starts, start) - 1  # last block starting <= start
    left_idx = max(left_idx, 0)

    overlapping: list = []
    for block in blocks[left_idx:right_idx]:
        # Overlap condition: block.start < end AND block.end > start
        if block.start < end and block.end > start:
            overlapping.append(block)

    if not overlapping:
        return None

    # Group by page and compute union bbox per page.
    by_page: dict[int, list] = {}
    for block in overlapping:
        by_page.setdefault(block.page, []).append(block)

    result: list[dict] = []
    for page_num in sorted(by_page):
        page_blocks = by_page[page_num]
        x0 = min(b.bbox[0] for b in page_blocks)
        y0 = min(b.bbox[1] for b in page_blocks)
        x1 = max(b.bbox[2] for b in page_blocks)
        y1 = max(b.bbox[3] for b in page_blocks)
        result.append(
            {
                "page": page_num,
                "bbox": (x0, y0, x1, y1),
                "start": start,
                "end": end,
            }
        )

    return result
