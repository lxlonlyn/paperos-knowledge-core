"""Deterministic DP partition of sentence units inside one major section."""

from __future__ import annotations

from typing import Any

from paperos_core.ingestion.sentence_units import SentenceUnit

TINY_TOKEN_THRESHOLD = 250


def partition_units(
    units: list[SentenceUnit],
    *,
    target_tokens: int,
    hard_max_tokens: int,
    count: Any,
) -> list[tuple[int, int]]:
    """Return half-open index ranges ``[start, end)`` covering all units."""
    n = len(units)
    if n == 0:
        return []
    if n == 1:
        return [(0, 1)]

    prefix = [0]
    for unit in units:
        prefix.append(prefix[-1] + unit.tokens)

    def span_tokens(start: int, end: int) -> int:
        return prefix[end] - prefix[start]

    inf = float("inf")
    costs = [inf] * (n + 1)
    prev = [-1] * (n + 1)
    costs[0] = 0.0
    for end in range(1, n + 1):
        for start in range(end):
            tokens = span_tokens(start, end)
            if tokens > hard_max_tokens:
                continue
            edge = _edge_cost(
                units[start:end],
                tokens=tokens,
                target_tokens=target_tokens,
                is_final=(end == n),
            )
            candidate = costs[start] + edge
            if candidate < costs[end] - 1e-12:
                costs[end] = candidate
                prev[end] = start
            elif abs(candidate - costs[end]) <= 1e-12:
                if _better_tie_break(
                    units,
                    start,
                    end,
                    prev_start=prev[end],
                    prev_end=end,
                    target_tokens=target_tokens,
                ):
                    prev[end] = start
    if prev[n] < 0:
        # Fallback: one unit per chunk (each must fit hard_max)
        return [(index, index + 1) for index in range(n)]

    ranges: list[tuple[int, int]] = []
    cursor = n
    while cursor > 0:
        start = prev[cursor]
        if start < 0:
            start = cursor - 1
        ranges.append((start, cursor))
        cursor = start
    ranges.reverse()
    return ranges


def _edge_cost(
    units: list[SentenceUnit],
    *,
    tokens: int,
    target_tokens: int,
    is_final: bool,
) -> float:
    ratio = tokens / max(target_tokens, 1)
    size_cost = (ratio - 1.0) ** 2
    last = units[-1]
    if last.subsection_end:
        boundary_cost = 0.0
    elif last.paragraph_end:
        boundary_cost = 0.15
    else:
        boundary_cost = 0.35
    tiny_cost = 0.0
    if tokens < TINY_TOKEN_THRESHOLD and not is_final:
        tiny_cost = ((TINY_TOKEN_THRESHOLD - tokens) / TINY_TOKEN_THRESHOLD) ** 2
    emergency = sum(1 for unit in units if unit.emergency_split)
    return size_cost + boundary_cost + tiny_cost + emergency * 0.5


def _better_tie_break(
    units: list[SentenceUnit],
    start: int,
    end: int,
    *,
    prev_start: int,
    prev_end: int,
    target_tokens: int,
) -> bool:
    if prev_start < 0:
        return True
    new_tokens = sum(unit.tokens for unit in units[start:end])
    old_tokens = sum(unit.tokens for unit in units[prev_start:prev_end])
    new_dist = abs(new_tokens - target_tokens)
    old_dist = abs(old_tokens - target_tokens)
    if new_dist != old_dist:
        return new_dist < old_dist
    new_boundary = units[end - 1].subsection_end
    old_boundary = units[prev_end - 1].subsection_end
    if new_boundary != old_boundary:
        return new_boundary
    return start < prev_start
