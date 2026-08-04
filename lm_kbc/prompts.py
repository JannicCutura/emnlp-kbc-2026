from __future__ import annotations

import json
from typing import Any


RELATION_GUIDANCE = {
    "hasArea": (
        "Return total surface area in square kilometres as exactly one plain "
        "numeric string without a unit."
    ),
    "hasCapacity": (
        "Return maximum spectator capacity as exactly one integer string without "
        "commas or a unit."
    ),
    "countryLandBordersCountry": (
        "Return every current land-bordering country or comparable territory. "
        "Include integral overseas territories and enclaves; exclude maritime "
        "borders and non-integral dependencies."
    ),
    "personHasCityOfDeath": (
        "Return the city or most specific publicly known locality of death. Return "
        "an empty list if the person is living or the locality is unknown."
    ),
    "companyTradesAtStockExchange": (
        "Return every current exchange on which this exact company's own shares "
        "trade. Do not transfer a parent listing to a subsidiary; return an empty "
        "list if it is not independently public."
    ),
    "awardWonBy": (
        "Return recipient entities, never winning works. Exclude rescinded awards "
        "and similarly named predecessor or successor awards."
    ),
}


ANCHOR_GUIDANCE = {
    "hasArea": "Identify geographic type, location, scope, namesakes, and rough scale.",
    "hasCapacity": "Identify the exact venue, location, type, names, and configuration.",
    "countryLandBordersCountry": (
        "Identify exact geographic scope, including integral overseas territories, "
        "enclaves, and exclaves."
    ),
    "personHasCityOfDeath": (
        "Identify the exact person by profession, nationality, era, and lifespan; "
        "distinguish namesakes."
    ),
    "companyTradesAtStockExchange": (
        "Identify the exact legal entity, country, industry, parent/subsidiary "
        "status, and whether it is independently listed."
    ),
    "awardWonBy": (
        "Identify the exact award, granting body, field, category, era, and any "
        "similarly named predecessor or successor."
    ),
}


FOCI = {
    "hasArea": (
        "Recall the preferred total-area figure, including inland water.",
        "Resolve the exact subject rather than a containing region or namesake.",
        "Independently recall the value in square kilometres.",
        "Check land-only versus total area and sanity-check the magnitude.",
    ),
    "hasCapacity": (
        "Recall the official maximum spectator capacity.",
        "Resolve the exact venue and location rather than a namesake.",
        "Recall the largest published capacity after renovations.",
        "Distinguish total maximum from a smaller seated configuration and check scale.",
    ),
    "personHasCityOfDeath": (
        "Recall the most specific publicly reported death locality.",
        "Independently recall biographies, obituaries, and the person's final place.",
        "Look for positive evidence of death and its locality; an old description "
        "as alive is not proof the person remained living through 1 July 2026.",
    ),
    "companyTradesAtStockExchange": (
        "Recall the primary current listing and exchange.",
        "Check current secondary or dual listings; return exchanges, not tickers.",
        "Ensure each listing belongs to this legal entity, not a parent or subsidiary.",
        "Check whether an earlier listing was delisted, renamed, or transferred.",
    ),
    "countryLandBordersCountry": (
        "Enumerate all neighbours clockwise around the boundary.",
        "Independently recall all current land neighbours.",
        "Check enclaves, exclaves, integral overseas territories, and short borders.",
    ),
    "awardWonBy": (
        "Return recipients from editions before 1960 only.",
        "Return recipients from 1960 through 1979 only.",
        "Return recipients from 1980 through 1999 only.",
        "Return recipients from 2000 through 2014 only.",
        "Return recipients from 2015 through 1 July 2026 only.",
        "Recall omitted recipients by profession, nationality, and recipient type.",
    ),
}


def select_examples(
    train_rows: list[dict[str, Any]],
    relation: str,
    limit: int,
    *,
    exclude_subject: str,
    offset: int,
) -> list[dict[str, Any]]:
    """Select rotating leave-one-subject-out examples, preferring SyntheticCoT.

    The empty/non-empty mix mirrors the relation's own training distribution.
    Demonstrating more empty answers than the data contains biases the model
    toward abstention, which costs recall on exactly the relations where empty
    is a legitimate answer (death 42% empty in train, exchange 34%, borders
    18%).
    """
    candidates = [
        row
        for row in train_rows
        if row.get("Relation") == relation
        and row.get("SubjectEntity") != exclude_subject
    ]
    if limit <= 0 or not candidates:
        return []

    def rotate(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not values:
            return []
        shift = offset % len(values)
        return values[shift:] + values[:shift]

    def has_rationale(row: dict[str, Any]) -> bool:
        value = row.get("Rationale")
        return isinstance(value, str) and bool(value.strip())

    def deduplicate_subjects(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # SyntheticCoT rows are prepended to train rows, so one subject can
        # appear twice; the reasoned copy comes first and wins.
        seen: set[str] = set()
        unique: list[dict[str, Any]] = []
        for row in rows:
            subject = row.get("SubjectEntity")
            if subject in seen:
                continue
            seen.add(subject)
            unique.append(row)
        return unique

    pools: dict[bool, list[dict[str, Any]]] = {}
    for is_empty in (True, False):
        group = [
            row
            for row in candidates
            if (not row.get("ObjectEntities")) is is_empty
        ]
        pools[is_empty] = deduplicate_subjects(
            rotate([row for row in group if has_rationale(row)])
            + rotate([row for row in group if not has_rationale(row)])
        )

    empty_rate = len(pools[True]) / len(candidates)
    empty_target = min(round(limit * empty_rate), len(pools[True]))
    nonempty_target = min(limit - empty_target, len(pools[False]))
    # Backfill from the other class when one pool cannot fill its share.
    empty_target = min(limit - nonempty_target, len(pools[True]))
    empties = pools[True][:empty_target]
    nonempties = pools[False][:nonempty_target]

    total = len(empties) + len(nonempties)
    selected: list[dict[str, Any]] = []
    used_empty = 0
    for index in range(total):
        remaining_nonempty = len(nonempties) - (len(selected) - used_empty)
        # Space the empty-answer demonstrations evenly rather than blocking
        # them at either end, where position alone would weight them.
        take_empty = used_empty < len(empties) and (
            remaining_nonempty == 0
            or (used_empty + 0.5) / len(empties) <= (index + 0.5) / total
        )
        if take_empty:
            selected.append(empties[used_empty])
            used_empty += 1
        else:
            selected.append(nonempties[len(selected) - used_empty])
    return selected


def description_anchor_messages(
    subject: str, relation: str, output_mode: str
) -> list[dict[str, str]]:
    output = (
        "Return only the constrained object."
        if output_mode == "json_schema"
        else (
            "Return only JSON with exactly `description` and `identity_checks`, "
            'for example {"description":"...","identity_checks":["..."]}.'
        )
    )
    return [
        {
            "role": "system",
            "content": (
                "Using only knowledge stored in your pretrained weights, identify "
                "the exact subject before answering the relation. Do not browse, "
                "use tools, or request retrieval. Keep the description short and "
                f"admit uncertainty. {output}\n\nRelation: {relation}\n"
                f"Definition: {RELATION_GUIDANCE[relation]}\n"
                f"Identity focus: {ANCHOR_GUIDANCE[relation]}"
            ),
        },
        {"role": "user", "content": f"Subject: {subject}"},
    ]


def candidate_messages(
    subject: str,
    relation: str,
    examples: list[dict[str, Any]],
    focus: str,
    anchor: str,
    output_mode: str,
) -> list[dict[str, str]]:
    award = relation == "awardWonBy"
    if output_mode == "json_schema":
        output = (
            "Return only the constrained object with a concise `rationale` and "
            "recipient objects in `candidates`; a work title belongs in `work`, "
            "never `recipient`."
            if award
            else (
                "Return only the constrained object with a concise `rationale` "
                "and object strings in `answers`."
            )
        )
    else:
        output = (
            "Return only JSON with exactly `rationale` and `candidates`. Each "
            "candidate must contain `recipient`, `work`, and `year`; never put a "
            "work title in `recipient`."
            if award
            else "Return only JSON with exactly `rationale` and `answers`."
        )
    system = (
        "Answer from factual knowledge stored in your pretrained weights. This is "
        "closed-book: do not browse, use tools, or request retrieval. Use the world "
        f"state as of 1 July 2026. {output}\n\nRelation: {relation}\n"
        f"Definition: {RELATION_GUIDANCE[relation]}\nPass focus: {focus}\n\n"
        "Uncertain same-model identity anchor (not ground truth):\n"
        f"{anchor}\nCorrect or ignore the anchor if it conflicts with your knowledge."
    )
    messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    for example in examples:
        answers = _canonical_answers(example)
        rationale = example.get("Rationale")
        if not isinstance(rationale, str) or not rationale.strip():
            rationale = (
                "Verified challenge training example; the answer set follows "
                "the supplied relation definition."
            )
        response: dict[str, Any]
        if award:
            response = {
                "rationale": rationale,
                "candidates": [
                    {"recipient": answer, "work": None, "year": None}
                    for answer in answers
                ],
            }
        else:
            response = {"rationale": rationale, "answers": answers}
        messages.extend(
            [
                {"role": "user", "content": f"Subject: {example['SubjectEntity']}"},
                {
                    "role": "assistant",
                    "content": json.dumps(response, ensure_ascii=False),
                },
            ]
        )
    messages.append({"role": "user", "content": f"Subject: {subject}"})
    return messages


def completion_messages(
    subject: str,
    relation: str,
    current: list[str],
    anchor: str,
    output_mode: str,
) -> list[dict[str, str]]:
    award = relation == "awardWonBy"
    if award:
        focus = (
            "Check every era, recipient type, profession, nationality, joint award, "
            "honorary recipient, and less famous recipient. Return only correct "
            "recipients missing from the current set."
        )
        output = (
            "Return only the constrained candidate object."
            if output_mode == "json_schema"
            else (
                "Return only JSON with `candidates`; every item has `recipient`, "
                "`work`, and `year`."
            )
        )
    else:
        focus = (
            "Check enclaves, exclaves, integral overseas territories, and easily "
            "omitted short land borders. Return only missing neighbours."
        )
        output = (
            "Return only the constrained answer object."
            if output_mode == "json_schema"
            else "Return only JSON with an `answers` array."
        )
    return [
        {
            "role": "system",
            "content": (
                "Find omissions using only pretrained knowledge; do not browse or "
                f"use tools. {output}\nRelation: {relation}\n"
                f"Definition: {RELATION_GUIDANCE[relation]}\nCheck: {focus}\n"
                f"Uncertain identity anchor: {anchor}"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Subject: {subject}\nExisting candidates: "
                + json.dumps(current, ensure_ascii=False)
            ),
        },
    ]


def border_verification_messages(
    subject: str, candidates: list[str], output_mode: str
) -> list[dict[str, str]]:
    output = (
        "Return only the constrained answer object."
        if output_mode == "json_schema"
        else "Return only JSON with an `answers` array."
    )
    output += (
        " Copy each accepted answer exactly from the supplied candidate strings; "
        "do not rename, translate, or paraphrase it."
    )
    return [
        {
            "role": "system",
            "content": (
                "Verify land-border candidates using only knowledge in your "
                "pretrained weights. Keep every correct candidate and reject only "
                f"clear errors. {output}\nDefinition: "
                f"{RELATION_GUIDANCE['countryLandBordersCountry']}"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Country: {subject}\nCandidates: "
                + json.dumps(candidates, ensure_ascii=False)
            ),
        },
    ]


def _canonical_answers(row: dict[str, Any]) -> list[str]:
    answers: list[str] = []
    for entity in row.get("ObjectEntities", []):
        if isinstance(entity, list) and entity:
            answers.append(str(entity[0]))
        elif isinstance(entity, str):
            answers.append(entity)
    return answers
