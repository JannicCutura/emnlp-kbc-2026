from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from statistics import median
from typing import Iterable


NUMERIC_RELATIONS = {"hasArea", "hasCapacity"}
ABSTENTION_RELATIONS = {
    "companyTradesAtStockExchange",
    "personHasCityOfDeath",
}
EXPLICIT_NONE_MARKERS = {"none", "null", "unknown", "no answer", "n/a"}
_APOSTROPHE_LIKE = set("'’‘ʻʼʹ`´")
_ASCII_SYMBOLS = set("+$<=>|~^")
_NUMERIC_PATTERN = re.compile(
    r"(?<![\w.])"
    r"(?P<number>[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
    r"(?:e[-+]?\d+)?)"
    r"(?:\s*(?P<suffix>k|thousand|million|billion))?"
    r"(?![\w.])",
    re.IGNORECASE,
)
_MAGNITUDE = {
    "k": 1_000,
    "thousand": 1_000,
    "million": 1_000_000,
    "billion": 1_000_000_000,
}


def evaluator_normalize(value: str) -> str:
    """Normalize exactly like the official string evaluator."""
    value = "".join(char for char in value.strip() if char not in _APOSTROPHE_LIKE)
    value = unicodedata.normalize("NFKD", value).casefold()
    output: list[str] = []
    for char in value:
        if char in _APOSTROPHE_LIKE or unicodedata.combining(char):
            continue
        if char in _ASCII_SYMBOLS or unicodedata.category(char).startswith("P"):
            output.append(" ")
        else:
            output.append(char)
    return " ".join("".join(output).split())


def normalize(value: str) -> str:
    """Canonical project normalizer; kept as a short public name."""
    return evaluator_normalize(value)


def parse_json_array(text: str, relation: str) -> list[str]:
    """Extract the first JSON array and clean it for one challenge relation."""
    parsed = extract_json_array(text)
    if parsed is None:
        return []

    values = [
        str(item).strip()
        for item in parsed
        if isinstance(item, (str, int, float)) and not isinstance(item, bool)
    ]
    values = [value for value in values if value]
    if (
        relation in ABSTENTION_RELATIONS
        and len(values) == 1
        and normalize(values[0]) in EXPLICIT_NONE_MARKERS
    ):
        return []
    if relation in NUMERIC_RELATIONS:
        if not values:
            return []
        match = _NUMERIC_PATTERN.search(values[0])
        if not match:
            return []
        number = match.group("number").replace(",", "")
        suffix = match.group("suffix")
        if suffix or "e" in number.casefold():
            multiplier = _MAGNITUDE[suffix.casefold()] if suffix else 1
            number = _format_number(float(number) * multiplier)
        return [number]
    return _deduplicate(values)


def extract_json_array(text: str) -> list[object] | None:
    """Return the first real JSON array, distinguishing it from no array."""
    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char != "[":
            continue
        try:
            candidate, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, list):
            return candidate
    return None


def parse_json_object(text: str) -> dict[str, object]:
    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            return candidate
    return {}


def build_unambiguous_alias_maps(
    train_rows: Iterable[dict[str, object]],
) -> dict[str, dict[str, str]]:
    """Build relation-specific alias maps solely from nested train aliases."""
    candidates: dict[str, dict[str, dict[str, str]]] = {}
    for row in train_rows:
        relation = row.get("Relation")
        entities = row.get("ObjectEntities")
        if not isinstance(relation, str) or not isinstance(entities, list):
            continue
        relation_candidates = candidates.setdefault(relation, {})
        for entity in entities:
            # Flat SyntheticCoT answers contain no alias groups and are ignored.
            if not isinstance(entity, list):
                continue
            aliases = [
                value.strip()
                for value in entity
                if isinstance(value, str) and value.strip()
            ]
            if not aliases:
                continue
            canonical = aliases[0]
            canonical_key = normalize(canonical)
            for alias in aliases:
                alias_key = normalize(alias)
                if alias_key and canonical_key:
                    relation_candidates.setdefault(alias_key, {}).setdefault(
                        canonical_key, canonical
                    )
    return {
        relation: {
            alias: next(iter(canonical_surfaces.values()))
            for alias, canonical_surfaces in relation_candidates.items()
            if len(canonical_surfaces) == 1
        }
        for relation, relation_candidates in candidates.items()
    }


def canonicalize_candidates(
    values: Iterable[str], alias_map: dict[str, str]
) -> tuple[list[str], list[dict[str, str]]]:
    """Apply unambiguous train aliases and evaluator-equivalent deduplication."""
    result: list[str] = []
    rewrites: list[dict[str, str]] = []
    seen: set[str] = set()
    for value in values:
        canonical = alias_map.get(normalize(value), value)
        key = normalize(canonical)
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(canonical)
        if canonical != value:
            rewrites.append({"from": value, "to": canonical})
    return result, rewrites


def median_numeric_candidates(passes: Iterable[list[str]]) -> list[str]:
    numbers: list[float] = []
    for values in passes:
        if not values:
            continue
        try:
            numbers.append(float(values[0]))
        except ValueError:
            continue
    if not numbers:
        return []
    return [_format_number(float(median(numbers)))]


def support_fraction_candidates(
    passes: Iterable[list[str]], threshold: float
) -> list[str]:
    materialized = list(passes)
    _validate_fraction(threshold, "threshold")
    if not materialized:
        return []
    counts, surfaces = _candidate_counts_and_surfaces(materialized)
    return [
        surfaces[key]
        for key, count in counts.items()
        if count / len(materialized) >= threshold
    ]


def union_candidates(passes: Iterable[list[str]]) -> list[str]:
    materialized = list(passes)
    if not materialized:
        return []
    counts, surfaces = _candidate_counts_and_surfaces(materialized)
    return [surfaces[key] for key in counts]


def majority_single_with_none(
    passes: Iterable[list[str]], none_threshold: float
) -> list[str]:
    """Choose one plurality entity; empty wins only by strict plurality."""
    materialized = list(passes)
    _validate_fraction(none_threshold, "none_threshold")
    if not materialized:
        return []
    counts, surfaces = _candidate_counts_and_surfaces(materialized)
    none_count = sum(not _normalized_unique(values) for values in materialized)
    if not counts:
        return []
    winner, winner_count = max(counts.items(), key=lambda item: item[1])
    if (
        none_count > winner_count
        and none_count / len(materialized) >= none_threshold
    ):
        return []
    return [surfaces[winner]]


def threshold_multi_with_none(
    passes: Iterable[list[str]],
    candidate_threshold: float,
    none_threshold: float,
) -> list[str]:
    """Threshold entities unless empty is the strict plurality candidate."""
    materialized = list(passes)
    _validate_fraction(candidate_threshold, "candidate_threshold")
    _validate_fraction(none_threshold, "none_threshold")
    if not materialized:
        return []
    counts, surfaces = _candidate_counts_and_surfaces(materialized)
    none_count = sum(not _normalized_unique(values) for values in materialized)
    if (
        none_count > max(counts.values(), default=0)
        and none_count / len(materialized) >= none_threshold
    ):
        return []
    return [
        surfaces[key]
        for key, count in counts.items()
        if count / len(materialized) >= candidate_threshold
    ]


def _format_number(value: float) -> str:
    return str(int(value)) if value.is_integer() else f"{value:g}"


def _validate_fraction(value: float, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not 0 <= value <= 1
    ):
        raise ValueError(f"{name} must be a number between 0 and 1")


def _deduplicate(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = normalize(value)
        if key and key not in seen:
            seen.add(key)
            result.append(value)
    return result


def _normalized_unique(values: Iterable[str]) -> list[tuple[str, str]]:
    unique: list[tuple[str, str]] = []
    seen: set[str] = set()
    for value in values:
        key = normalize(value)
        if key and key not in seen:
            seen.add(key)
            unique.append((key, value))
    return unique


def _candidate_counts_and_surfaces(
    passes: Iterable[list[str]],
) -> tuple[Counter[str], dict[str, str]]:
    counts: Counter[str] = Counter()
    surfaces: dict[str, str] = {}
    for values in passes:
        for key, value in _normalized_unique(values):
            surfaces.setdefault(key, value)
            counts[key] += 1
    return counts, surfaces
