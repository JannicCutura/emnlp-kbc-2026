from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .checkpoint import RowCheckpointStore
from .client import LMStudioClient
from .config import AppConfig, RELATIONS
from .io import atomic_write_jsonl, atomic_write_text
from .parsing import (
    NUMERIC_RELATIONS,
    build_unambiguous_alias_maps,
    canonicalize_candidates,
    extract_json_array,
    majority_single_with_none,
    median_numeric_candidates,
    normalize,
    parse_json_array,
    parse_json_object,
    support_fraction_candidates,
    threshold_multi_with_none,
    union_candidates,
)
from .prompts import (
    FOCI,
    border_verification_messages,
    candidate_messages,
    completion_messages,
    description_anchor_messages,
    select_examples,
)
from .schemas import (
    AnswerSet,
    AwardCandidateSet,
    IdentityAnchor,
    ReasonedAnswerSet,
    ReasonedAwardCandidateSet,
    response_format,
    validate_json,
)


METHOD_VERSION = "improvement-plan-v2"
POLICY_FILES = (
    "pipeline.py",
    "prompts.py",
    "parsing.py",
    "schemas.py",
    "client.py",
)


class PredictionPipeline:
    """One opinionated implementation of the challenge improvement plan."""

    def __init__(
        self,
        config: AppConfig,
        client: LMStudioClient,
        train_rows: list[dict[str, Any]],
    ) -> None:
        self.config = config
        self.client = client
        self.train_rows = train_rows
        self._alias_cache: dict[tuple[str, str], dict[str, str]] = {}

    def _alias_map(self, subject: str, relation: str) -> dict[str, str]:
        """Build train-only aliases without using the target row's gold aliases."""
        key = (subject, relation)
        if key not in self._alias_cache:
            leave_one_out = [
                row
                for row in self.train_rows
                if not (
                    row.get("SubjectEntity") == subject
                    and row.get("Relation") == relation
                )
            ]
            self._alias_cache[key] = build_unambiguous_alias_maps(
                leave_one_out
            ).get(relation, {})
        return self._alias_cache[key]

    def _call(
        self,
        messages: list[dict[str, str]],
        *,
        seed_offset: int,
        schema_model: type,
        schema_name: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> tuple[str, dict[str, Any]]:
        gen = self.config.generation
        seed = gen.seed + seed_offset if gen.seed is not None else None
        return self.client.chat(
            messages,
            temperature=gen.temperature if temperature is None else temperature,
            top_p=gen.top_p,
            max_tokens=gen.max_tokens if max_tokens is None else max_tokens,
            seed=seed,
            response_format=(
                response_format(schema_model, schema_name)
                if gen.output_mode == "json_schema"
                else None
            ),
        )

    def _anchor(
        self,
        trace: dict[str, Any],
        subject: str,
        relation: str,
        seed_offset: int,
    ) -> str:
        text, response = self._call(
            description_anchor_messages(
                subject, relation, self.config.generation.output_mode
            ),
            seed_offset=seed_offset,
            schema_model=IdentityAnchor,
            schema_name=f"{relation}_identity",
            max_tokens=min(self.config.generation.max_tokens, 256),
            temperature=0.2,
        )
        validated = validate_json(text, IdentityAnchor)
        if isinstance(validated, IdentityAnchor):
            description = validated.description.strip()
            checks = [check.strip() for check in validated.identity_checks if check.strip()]
        else:
            obj = parse_json_object(text)
            raw_description = obj.get("description")
            raw_checks = obj.get("identity_checks")
            description = (
                raw_description.strip() if isinstance(raw_description, str) else ""
            )
            checks = (
                [
                    check.strip()
                    for check in raw_checks
                    if isinstance(check, str) and check.strip()
                ]
                if isinstance(raw_checks, list)
                else []
            )
        if not description:
            description = f"Exact identity of {subject}; details are uncertain."
        anchor = description
        if checks:
            anchor += " Identity checks: " + "; ".join(checks)
        trace["anchor"] = {
            "description": description,
            "identity_checks": checks,
            "raw_text": text,
            "schema_valid": isinstance(validated, IdentityAnchor),
            "provider": response.get("provider"),
            "usage": response.get("usage"),
        }
        return anchor

    def _answer_step(
        self,
        trace: dict[str, Any],
        *,
        stage: str,
        messages: list[dict[str, str]],
        relation: str,
        seed_offset: int,
        reasoned: bool,
        award: bool,
        metadata: dict[str, Any] | None = None,
    ) -> list[str] | None:
        """Run one answer stage and record it in the trace."""
        parsed, step = self._answer_call(
            stage=stage,
            messages=messages,
            relation=relation,
            seed_offset=seed_offset,
            reasoned=reasoned,
            award=award,
            metadata=metadata,
        )
        trace["workflow_steps"].append(step)
        return parsed

    def _answer_call(
        self,
        *,
        stage: str,
        messages: list[dict[str, str]],
        relation: str,
        seed_offset: int,
        reasoned: bool,
        award: bool,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[list[str] | None, dict[str, Any]]:
        """Issue one request and return its objects plus its trace step.

        This touches no shared state, so concurrent samples can call it and the
        caller appends the steps in a deterministic order afterwards.

        The objects are None when the response yielded no usable JSON. Callers
        decide what that means; it must never be read as an empty answer, since
        for the abstention relations that would turn a parser failure into a
        vote for "no objects".
        """
        if reasoned and award:
            schema_model = ReasonedAwardCandidateSet
        elif reasoned:
            schema_model = ReasonedAnswerSet
        elif award:
            schema_model = AwardCandidateSet
        else:
            schema_model = AnswerSet
        text, response = self._call(
            messages,
            seed_offset=seed_offset,
            schema_model=schema_model,
            schema_name=f"{relation}_{stage.replace(':', '_')}",
        )
        validated = validate_json(text, schema_model)
        rationale = ""
        if isinstance(validated, (AnswerSet, ReasonedAnswerSet)):
            answers = validated.answers
            if isinstance(validated, ReasonedAnswerSet):
                rationale = validated.rationale.strip()
            parsed = parse_json_array(
                json.dumps(answers, ensure_ascii=False), relation
            )
        elif isinstance(validated, (AwardCandidateSet, ReasonedAwardCandidateSet)):
            if isinstance(validated, ReasonedAwardCandidateSet):
                rationale = validated.rationale.strip()
            parsed = parse_json_array(
                json.dumps(
                    [candidate.recipient for candidate in validated.candidates],
                    ensure_ascii=False,
                ),
                relation,
            )
        else:
            parsed, rationale, usable = self._fallback_answer(
                text, relation, award, reasoned
            )
            if not usable:
                parsed = None

        step = {
            "stage": stage,
            "text": text,
            "parsed": parsed,
            "usable": parsed is not None,
            "rationale": rationale,
            "schema_valid": validated is not None,
            "output_mode": self.config.generation.output_mode,
            # Hosted gateways route one model name across backends at differing
            # quantizations, so the serving provider is part of the result.
            "provider": response.get("provider"),
            "usage": response.get("usage"),
        }
        if metadata:
            step.update(metadata)
        return parsed, step

    @staticmethod
    def _fallback_answer(
        text: str, relation: str, award: bool, reasoned: bool
    ) -> tuple[list[str], str, bool]:
        obj = parse_json_object(text)
        rationale = obj.get("rationale", "") if reasoned else ""
        if not isinstance(rationale, str):
            rationale = ""
        key = "candidates" if award else "answers"
        raw_values = obj.get(key)
        if award and isinstance(raw_values, list):
            values = [
                candidate["recipient"].strip()
                for candidate in raw_values
                if isinstance(candidate, dict)
                and isinstance(candidate.get("recipient"), str)
                and candidate["recipient"].strip()
            ]
            if values:
                return (
                    parse_json_array(json.dumps(values), relation),
                    rationale.strip(),
                    True,
                )
        if isinstance(raw_values, list):
            return (
                parse_json_array(json.dumps(raw_values, ensure_ascii=False), relation),
                rationale.strip(),
                True,
            )
        return (
            parse_json_array(text, relation),
            rationale.strip(),
            extract_json_array(text) is not None,
        )

    def _sample(
        self,
        trace: dict[str, Any],
        subject: str,
        relation: str,
        anchor: str,
        base_seed: int,
    ) -> list[list[str]]:
        gen = self.config.generation
        count = gen.samples[relation]
        few_shot = gen.few_shot_by_relation.get(relation, gen.few_shot)

        calls: list[dict[str, Any]] = []
        for index in range(count):
            focus = FOCI[relation][index % len(FOCI[relation])]
            examples = select_examples(
                self.train_rows,
                relation,
                few_shot,
                exclude_subject=subject,
                offset=index * max(few_shot, 1),
            )
            calls.append(
                {
                    "stage": f"sample:{index + 1}",
                    "messages": candidate_messages(
                        subject,
                        relation,
                        examples,
                        focus,
                        anchor,
                        gen.output_mode,
                    ),
                    "relation": relation,
                    "seed_offset": base_seed + index,
                    "reasoned": True,
                    "award": relation == "awardWonBy",
                    "metadata": {
                        "focus": focus,
                        "example_subjects": [
                            example["SubjectEntity"] for example in examples
                        ],
                    },
                }
            )

        workers = min(gen.concurrency, len(calls))
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(lambda call: self._answer_call(**call), calls))
        else:
            results = [self._answer_call(**call) for call in calls]

        passes: list[list[str]] = []
        unusable = 0
        # Steps are appended in request order regardless of completion order, so
        # a concurrent run produces the same trace as a serial one.
        for values, step in results:
            trace["workflow_steps"].append(step)
            # A sample the parser cannot read carries no evidence either way, so
            # it drops out and the row aggregates over the remaining samples.
            if values is None:
                unusable += 1
                continue
            passes.append(values)
        trace["unusable_samples"] = unusable
        if unusable and not passes:
            tqdm.write(
                f"row {trace['row_index']} ({subject}, {relation}): all "
                f"{unusable} samples were unparseable; predicting no objects",
                file=sys.stderr,
            )
        return passes

    def _canonicalize_passes(
        self, passes: list[list[str]], alias_map: dict[str, str]
    ) -> tuple[list[list[str]], list[dict[str, str]]]:
        effective: list[list[str]] = []
        rewrites: list[dict[str, str]] = []
        for values in passes:
            canonical, changed = canonicalize_candidates(values, alias_map)
            effective.append(canonical)
            rewrites.extend(changed)
        return effective, rewrites

    def _aggregate(
        self,
        trace: dict[str, Any],
        relation: str,
        raw_passes: list[list[str]],
        alias_map: dict[str, str],
    ) -> list[str]:
        gen = self.config.generation
        passes, rewrites = self._canonicalize_passes(raw_passes, alias_map)
        details: dict[str, Any] = {
            "passes": passes,
            "samples": len(passes),
            "empty_samples": sum(not values for values in passes),
            "unusable_samples": trace.get("unusable_samples", 0),
            "alias_rewrites": rewrites,
        }
        if passes != raw_passes:
            details["raw_passes"] = raw_passes

        if relation in NUMERIC_RELATIONS:
            result = median_numeric_candidates(passes)
            details["method"] = "median"
        elif relation == "personHasCityOfDeath":
            threshold = gen.abstention_thresholds[relation]
            result = majority_single_with_none(passes, threshold)
            details.update({"method": "majority-with-none", "none_threshold": threshold})
        elif relation == "companyTradesAtStockExchange":
            candidate_threshold = gen.candidate_thresholds[relation]
            none_threshold = gen.abstention_thresholds[relation]
            result = threshold_multi_with_none(
                passes, candidate_threshold, none_threshold
            )
            details.update(
                {
                    "method": "threshold-with-none",
                    "candidate_threshold": candidate_threshold,
                    "none_threshold": none_threshold,
                }
            )
        elif relation == "countryLandBordersCountry":
            result = union_candidates(passes)
            details["method"] = "union"
        elif relation == "awardWonBy":
            threshold = gen.candidate_thresholds[relation]
            result = support_fraction_candidates(passes, threshold)
            details.update({"method": "threshold-union", "candidate_threshold": threshold})
        else:
            raise ValueError(f"Unsupported relation: {relation}")
        details["result"] = result
        trace["self_consistency"] = details
        return result

    def _complete(
        self,
        trace: dict[str, Any],
        subject: str,
        relation: str,
        candidates: list[str],
        anchor: str,
        seed_offset: int,
        alias_map: dict[str, str],
    ) -> list[str]:
        missing = self._answer_step(
            trace,
            stage="completion",
            messages=completion_messages(
                subject,
                relation,
                candidates,
                anchor,
                self.config.generation.output_mode,
            ),
            relation=relation,
            seed_offset=seed_offset,
            reasoned=False,
            award=relation == "awardWonBy",
        )
        # An unreadable completion means no omissions were found, never that the
        # aggregated candidates should be discarded.
        combined = union_candidates([candidates, missing or []])
        canonical, rewrites = canonicalize_candidates(combined, alias_map)
        trace["completion"] = {
            "missing": missing,
            "usable": missing is not None,
            "alias_rewrites": rewrites,
        }
        return canonical

    def _verify_borders(
        self,
        trace: dict[str, Any],
        subject: str,
        candidates: list[str],
        seed_offset: int,
        alias_map: dict[str, str],
    ) -> list[str]:
        accepted = self._answer_step(
            trace,
            stage="border-verification",
            messages=border_verification_messages(
                subject, candidates, self.config.generation.output_mode
            ),
            relation="countryLandBordersCountry",
            seed_offset=seed_offset,
            reasoned=False,
            award=False,
        )
        # A verifier whose answer cannot be read must not silently reject every
        # neighbour; keep the candidates it was asked to judge.
        if accepted is None:
            trace["verification"] = {
                "accepted": list(candidates),
                "fallback": "malformed-verifier-response",
            }
            return list(candidates)
        accepted, rewrites = canonicalize_candidates(accepted, alias_map)
        accepted_keys = {normalize(value) for value in accepted}
        verified = [
            candidate for candidate in candidates if normalize(candidate) in accepted_keys
        ]
        trace["verification"] = {"accepted": verified}
        if rewrites:
            trace["verification"]["alias_rewrites"] = rewrites
        return verified

    def predict_row(
        self, row: dict[str, Any], row_index: int
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        subject = row.get("SubjectEntity")
        relation = row.get("Relation")
        if not isinstance(subject, str) or not subject.strip():
            raise ValueError("Every input row needs a non-empty SubjectEntity")
        if relation not in RELATIONS:
            raise ValueError(f"Unsupported relation: {relation!r}")
        alias_map = self._alias_map(subject, relation)
        trace: dict[str, Any] = {
            "row_index": row_index,
            "SubjectEntity": subject,
            "Relation": relation,
            "method": METHOD_VERSION,
            "model": self.config.lm_studio.model,
            "model_parameters_billion": (
                self.config.lm_studio.model_parameters_billion
            ),
            "workflow_steps": [],
            "unusable_samples": 0,
            "completion": None,
            "verification": None,
        }
        base_seed = row_index * 1000
        anchor = self._anchor(trace, subject, relation, base_seed)
        passes = self._sample(
            trace, subject, relation, anchor, base_seed + 10
        )
        candidates = self._aggregate(trace, relation, passes, alias_map)
        if relation in {"countryLandBordersCountry", "awardWonBy"}:
            # An omission pass presupposes a set to have omissions from. When
            # every sample agreed the answer is empty, asking "what is missing?"
            # is a leading question, and the model answers it: on validation it
            # invented maritime neighbours for Malta, Samoa, Australia, and the
            # Bahamas from a unanimous and correct empty aggregate.
            if candidates:
                candidates = self._complete(
                    trace,
                    subject,
                    relation,
                    candidates,
                    anchor,
                    base_seed + 100,
                    alias_map,
                )
            else:
                trace["completion"] = {"skipped": "unanimous-empty-aggregate"}
        if relation == "countryLandBordersCountry" and candidates:
            candidates = self._verify_borders(
                trace, subject, candidates, base_seed + 110, alias_map
            )
        candidates, final_rewrites = canonicalize_candidates(candidates, alias_map)
        trace["final_alias_rewrites"] = final_rewrites
        trace["final"] = candidates
        prediction = {
            "SubjectEntity": subject,
            "Relation": relation,
            "ObjectEntities": candidates,
        }
        return prediction, trace

    def run(
        self,
        input_rows: list[dict[str, Any]],
        output_path: str | Path,
        *,
        resume: bool = False,
        runtime_dir: str | Path | None = None,
    ) -> list[dict[str, Any]]:
        output_path = Path(output_path)
        raw_path = output_path.parent / "raw" / f"{output_path.stem}.raw.json"
        checkpoint_key = sha256(
            str(output_path.resolve()).encode("utf-8")
        ).hexdigest()[:16]
        state_parent = (
            Path(runtime_dir)
            if runtime_dir is not None
            else output_path.parent / ".akbc-state"
        )
        state_root = state_parent / (
            f"{output_path.stem}-{checkpoint_key}.{METHOD_VERSION}.state"
        )
        manifest_config = asdict(self.config)
        manifest_config["lm_studio"].pop("api_key", None)
        manifest_config["method"] = METHOD_VERSION
        manifest_config["policy_sha256"] = _policy_digest()
        manifest_config["prompt_examples_sha256"] = _stable_digest(self.train_rows)
        store = RowCheckpointStore(state_root, manifest_config, input_rows)

        if resume:
            predictions, raw_records = store.load()
        else:
            if store.manifest_path.exists():
                raise ValueError(
                    f"Existing row state found at {state_root}; use --resume or "
                    "choose a different output/runtime path"
                )
            predictions, raw_records = [], []
        store.initialize()

        progress = tqdm(
            total=len(input_rows),
            initial=len(predictions),
            desc="Predicting",
        )
        try:
            for index in range(len(predictions), len(input_rows)):
                prediction, raw = self.predict_row(input_rows[index], index)
                _validate_prediction(prediction)
                store.write(index, input_rows[index], prediction, raw)
                predictions.append(prediction)
                raw_records.append(raw)
                progress.update(1)
        finally:
            progress.close()

        atomic_write_jsonl(predictions, output_path)
        atomic_write_text(
            raw_path,
            json.dumps(raw_records, ensure_ascii=False, indent=2),
        )
        atomic_write_text(
            store.root / "complete.json",
            json.dumps(
                {"completed_rows": len(predictions), "total_rows": len(input_rows)},
                ensure_ascii=False,
                indent=2,
            ),
        )
        return predictions


def _stable_digest(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _policy_digest() -> str:
    """Bind resumable rows to the exact prompt, parser, schema, and call policy."""
    digest = sha256()
    module_dir = Path(__file__).resolve().parent
    for name in POLICY_FILES:
        digest.update(name.encode("utf-8"))
        digest.update((module_dir / name).read_bytes())
    return digest.hexdigest()


def _validate_prediction(row: dict[str, Any]) -> None:
    values = row.get("ObjectEntities")
    if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
        raise ValueError("Pipeline produced a non-flat ObjectEntities list")
