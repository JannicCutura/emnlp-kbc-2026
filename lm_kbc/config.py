from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml


RELATIONS = (
    "awardWonBy",
    "companyTradesAtStockExchange",
    "countryLandBordersCountry",
    "hasArea",
    "hasCapacity",
    "personHasCityOfDeath",
)

_ROOT_CONFIG_KEYS = {
    "lm_studio",
    "generation",
    "train_data_file",
    "synthetic_cot_file",
}
_ROOT_METADATA_KEYS = {"calibration_provenance"}


def _default_samples() -> dict[str, int]:
    return {
        "hasArea": 20,
        "hasCapacity": 15,
        "personHasCityOfDeath": 20,
        "companyTradesAtStockExchange": 10,
        "countryLandBordersCountry": 3,
        "awardWonBy": 20,
    }


@dataclass(frozen=True)
class LMStudioConfig:
    base_url: str = "http://localhost:1234/v1"
    model: str = ""
    model_parameters_billion: float = 0.0
    api_key: str = "lm-studio"
    # Name of an environment variable holding the key for a hosted endpoint.
    # When set it wins over api_key, so no secret is ever stored in the config.
    api_key_env: str | None = None
    timeout_seconds: int = 300
    max_retries: int = 5
    retry_backoff_seconds: float = 2.0
    # Optional gateway routing block (OpenRouter `provider`), recorded in the
    # run manifest so a hosted run names the backend that produced it.
    provider_routing: dict[str, Any] | None = None


@dataclass(frozen=True)
class GenerationConfig:
    output_mode: str = "prompt_json"
    temperature: float = 0.6
    top_p: float = 0.95
    max_tokens: int = 2048
    seed: int | None = 42
    samples: dict[str, int] = field(default_factory=_default_samples)
    # Samples within a row are independent draws, so they can be issued at once.
    # 1 keeps the strictly serial behaviour a local single-model server wants.
    concurrency: int = 1
    few_shot: int = 5
    few_shot_by_relation: dict[str, int] = field(
        default_factory=lambda: {"awardWonBy": 1}
    )
    candidate_thresholds: dict[str, float] = field(
        default_factory=lambda: {
            "companyTradesAtStockExchange": 0.30,
            "awardWonBy": 0.05,
        }
    )
    abstention_thresholds: dict[str, float] = field(
        default_factory=lambda: {
            "personHasCityOfDeath": 0.50,
            "companyTradesAtStockExchange": 0.50,
        }
    )


@dataclass(frozen=True)
class AppConfig:
    lm_studio: LMStudioConfig
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    train_data_file: str = "data/train.jsonl"
    synthetic_cot_file: str | None = None


def _merged_generation(raw: dict[str, Any] | None) -> GenerationConfig:
    values = raw or {}
    if not isinstance(values, dict):
        raise ValueError("generation must be a map")
    try:
        return GenerationConfig(**values)
    except TypeError as exc:
        raise ValueError(f"invalid generation configuration: {exc}") from exc


def load_config(path: str | Path) -> AppConfig:
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a map")
    unknown = set(raw) - _ROOT_CONFIG_KEYS - _ROOT_METADATA_KEYS
    if unknown:
        key = sorted(unknown, key=repr)[0]
        raise ValueError(f"configuration has unknown top-level key {key!r}")
    for key in _ROOT_METADATA_KEYS:
        if key in raw and not isinstance(raw[key], dict):
            raise ValueError(f"{key} must be a map")
    lm_values = raw.get("lm_studio")
    if not isinstance(lm_values, dict):
        raise ValueError("lm_studio must be a map")
    train_file = raw.get("train_data_file", "data/train.jsonl")
    if not isinstance(train_file, str):
        raise ValueError("train_data_file must be a path string")
    try:
        lm_studio = LMStudioConfig(**lm_values)
    except TypeError as exc:
        raise ValueError(f"invalid lm_studio configuration: {exc}") from exc
    config = AppConfig(
        lm_studio=lm_studio,
        generation=_merged_generation(raw.get("generation")),
        train_data_file=train_file,
        synthetic_cot_file=raw.get("synthetic_cot_file"),
    )
    validate_config(config)
    return config


def validate_config(config: AppConfig) -> None:
    model = config.lm_studio
    if not isinstance(model.model, str) or not model.model.strip():
        raise ValueError("lm_studio.model must name the model loaded in LM Studio")
    if not isinstance(model.base_url, str) or not model.base_url.strip():
        raise ValueError("lm_studio.base_url must be a non-empty string")
    if not isinstance(model.api_key, str):
        raise ValueError("lm_studio.api_key must be a string")
    if model.api_key_env is not None and (
        not isinstance(model.api_key_env, str) or not model.api_key_env.strip()
    ):
        raise ValueError("lm_studio.api_key_env must be a non-empty string or null")
    if (
        isinstance(model.max_retries, bool)
        or not isinstance(model.max_retries, int)
        or not 0 <= model.max_retries <= 20
    ):
        raise ValueError("lm_studio.max_retries must be an integer from 0 to 20")
    if (
        isinstance(model.retry_backoff_seconds, bool)
        or not isinstance(model.retry_backoff_seconds, (int, float))
        or not 0 < model.retry_backoff_seconds <= 60
    ):
        raise ValueError(
            "lm_studio.retry_backoff_seconds must be greater than 0 and at most 60"
        )
    if model.provider_routing is not None and not isinstance(
        model.provider_routing, dict
    ):
        raise ValueError("lm_studio.provider_routing must be a map or null")
    if (
        isinstance(model.model_parameters_billion, bool)
        or not isinstance(model.model_parameters_billion, (int, float))
        or not 0 < model.model_parameters_billion <= 32
    ):
        raise ValueError(
            "lm_studio.model_parameters_billion must be greater than zero and "
            "at most the challenge maximum of 32B"
        )
    if (
        isinstance(model.timeout_seconds, bool)
        or not isinstance(model.timeout_seconds, int)
        or model.timeout_seconds <= 0
    ):
        raise ValueError("lm_studio.timeout_seconds must be positive")

    gen = config.generation
    if gen.output_mode not in {"prompt_json", "json_schema"}:
        raise ValueError(
            "generation.output_mode must be 'prompt_json' or 'json_schema'"
        )
    if (
        isinstance(gen.temperature, bool)
        or not isinstance(gen.temperature, (int, float))
        or not 0 <= gen.temperature <= 2
    ):
        raise ValueError("generation.temperature must be between 0 and 2")
    if (
        isinstance(gen.top_p, bool)
        or not isinstance(gen.top_p, (int, float))
        or not 0 < gen.top_p <= 1
    ):
        raise ValueError("generation.top_p must be greater than 0 and at most 1")
    if (
        isinstance(gen.max_tokens, bool)
        or not isinstance(gen.max_tokens, int)
        or not 1 <= gen.max_tokens <= 8192
    ):
        raise ValueError("generation.max_tokens must be between 1 and 8192")
    if gen.seed is not None and (
        isinstance(gen.seed, bool) or not isinstance(gen.seed, int)
    ):
        raise ValueError("generation.seed must be an integer or null")

    if not isinstance(gen.samples, dict) or set(gen.samples) != set(RELATIONS):
        raise ValueError(
            "generation.samples must contain exactly the six challenge relations"
        )
    for relation, count in gen.samples.items():
        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 20:
            raise ValueError(
                f"generation.samples[{relation!r}] must be an integer from 1 to 20"
            )

    if (
        isinstance(gen.concurrency, bool)
        or not isinstance(gen.concurrency, int)
        or not 1 <= gen.concurrency <= 32
    ):
        raise ValueError("generation.concurrency must be an integer from 1 to 32")
    if (
        isinstance(gen.few_shot, bool)
        or not isinstance(gen.few_shot, int)
        or not 0 <= gen.few_shot <= 20
    ):
        raise ValueError("generation.few_shot must be an integer from 0 to 20")
    _validate_relation_int_map(
        "few_shot_by_relation", gen.few_shot_by_relation, minimum=0, maximum=20
    )
    _validate_threshold_map(
        "candidate_thresholds",
        gen.candidate_thresholds,
        {"companyTradesAtStockExchange", "awardWonBy"},
    )
    _validate_threshold_map(
        "abstention_thresholds",
        gen.abstention_thresholds,
        {"personHasCityOfDeath", "companyTradesAtStockExchange"},
    )
    if not config.train_data_file.strip():
        raise ValueError("train_data_file must be a non-empty path")
    if config.synthetic_cot_file is not None:
        if not isinstance(config.synthetic_cot_file, str) or not (
            config.synthetic_cot_file.strip()
        ):
            raise ValueError("synthetic_cot_file must be a non-empty path or null")


def _validate_relation_int_map(
    name: str, values: object, *, minimum: int, maximum: int
) -> None:
    if not isinstance(values, dict):
        raise ValueError(f"generation.{name} must be a map")
    unknown = set(values) - set(RELATIONS)
    if unknown:
        raise ValueError(f"generation.{name} has unknown relation {sorted(unknown)[0]!r}")
    for relation, value in values.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not minimum <= value <= maximum
        ):
            raise ValueError(
                f"generation.{name}[{relation!r}] must be an integer from "
                f"{minimum} to {maximum}"
            )


def _validate_threshold_map(
    name: str, values: object, expected_relations: set[str]
) -> None:
    if not isinstance(values, dict) or set(values) != expected_relations:
        expected = ", ".join(sorted(expected_relations))
        raise ValueError(f"generation.{name} must contain exactly: {expected}")
    for relation, value in values.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0 <= value <= 1
        ):
            raise ValueError(
                f"generation.{name}[{relation!r}] must be between 0 and 1"
            )


def smoke_config(config: AppConfig) -> AppConfig:
    """Return the same pipeline policy with one sample per relation.

    The completion cap stays generous enough for a long award enumeration: a
    smoke that truncates its own output tests the token limit, not the model.
    """
    smoke_generation = replace(
        config.generation,
        temperature=0.2,
        max_tokens=min(config.generation.max_tokens, 2048),
        samples={relation: 1 for relation in RELATIONS},
    )
    smoke = replace(config, generation=smoke_generation)
    validate_config(smoke)
    return smoke
