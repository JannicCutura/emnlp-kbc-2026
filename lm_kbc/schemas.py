from __future__ import annotations

import json

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AnswerSet(StrictModel):
    answers: list[str] = Field(description="Object strings satisfying the relation")


class IdentityAnchor(StrictModel):
    description: str = Field(
        min_length=1,
        max_length=400,
        description="Concise description of the exact subject entity",
    )
    identity_checks: list[str] = Field(
        min_length=1,
        max_length=4,
        description=(
            "Short checks that distinguish the subject from similarly named or "
            "related entities"
        ),
    )


class ReasonedAnswerSet(StrictModel):
    rationale: str = Field(
        min_length=1, description="Concise factual rationale for the answer set"
    )
    answers: list[str] = Field(description="Object strings satisfying the relation")


class AwardCandidate(StrictModel):
    recipient: str = Field(description="Recipient entity name, never a work title")
    work: str | None = Field(description="Associated winning work, if applicable")
    year: int | None = Field(description="Award year, if known")


class AwardCandidateSet(StrictModel):
    candidates: list[AwardCandidate]


class ReasonedAwardCandidateSet(StrictModel):
    rationale: str = Field(
        min_length=1,
        description="Concise factual rationale about the recipient set"
    )
    candidates: list[AwardCandidate]


def response_format(model: type[BaseModel], name: str) -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": model.model_json_schema(),
        },
    }


def validate_json(text: str, model: type[BaseModel]) -> BaseModel | None:
    """Validate a structured response, tolerating outer runtime token noise."""
    try:
        return model.model_validate_json(text.strip())
    except ValidationError:
        pass
    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[start:])
            return model.model_validate(value)
        except (json.JSONDecodeError, ValidationError):
            continue
    return None
