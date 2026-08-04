from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from lm_kbc.artifacts import (
    merge_prediction_files,
    package_prediction_file,
    select_relation_file,
)
from lm_kbc.calibration import (
    CALIBRATABLE_RELATIONS,
    calibrate_files,
    load_verified_train_rows,
    reject_obvious_nontrain_path,
)
from lm_kbc.client import LMStudioClient
from lm_kbc.config import RELATIONS, load_config, smoke_config, validate_config
from lm_kbc.io import read_jsonl, write_jsonl
from lm_kbc.pipeline import PredictionPipeline
from lm_kbc.synthetic_cot import (
    build_synthetic_cot_file,
    load_synthetic_cot_file,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the single AKBC 2026 improvement-plan pipeline and tools."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    predict = commands.add_parser("predict", help="Generate model predictions")
    predict.add_argument("-c", "--config", required=True)
    predict.add_argument("-i", "--input", required=True)
    predict.add_argument("-o", "--output", required=True)
    predict.add_argument("--resume", action="store_true")
    predict.add_argument("--runtime-dir")
    predict.add_argument("--limit", type=int)
    predict.add_argument("--dry-run", action="store_true")
    predict.add_argument(
        "--smoke",
        action="store_true",
        help="Use one sample per relation and a 512-token completion cap",
    )
    predict.add_argument(
        "--output-mode", choices=("prompt_json", "json_schema")
    )
    predict.add_argument("--model", help="Override the LM Studio model identifier")
    predict.add_argument(
        "--model-parameters-billion",
        type=float,
        help="Published total parameters for --model",
    )
    predict.add_argument("--max-restarts", type=int, default=10)
    predict.add_argument("--restart-delay", type=float, default=10.0)

    calibrate = commands.add_parser(
        "calibrate", help="Calibrate thresholds from complete train traces"
    )
    calibrate.add_argument("--train", required=True)
    calibrate.add_argument("--traces", required=True)
    calibrate.add_argument("--output", required=True)
    calibrate.add_argument(
        "--relations", nargs="+", choices=CALIBRATABLE_RELATIONS
    )
    calibrate.add_argument("--allow-partial", action="store_true")

    build_cot = commands.add_parser(
        "build-cot", help="Build evaluator-filtered train-only SyntheticCoT"
    )
    build_cot.add_argument("--train", required=True)
    build_cot.add_argument("--traces", required=True)
    build_cot.add_argument("--output", required=True)

    select = commands.add_parser(
        "select", help="Create answer-blanked input for selected relations"
    )
    select.add_argument("-i", "--input", required=True)
    select.add_argument("-o", "--output", required=True)
    select.add_argument("-r", "--relations", nargs="+", required=True)

    merge = commands.add_parser("merge", help="Merge partial predictions by row key")
    merge.add_argument("-b", "--base", required=True)
    merge.add_argument("-p", "--partial", nargs="+", required=True)
    merge.add_argument("-o", "--output", required=True)

    package = commands.add_parser(
        "package", help="Validate and create a Codabench predictions ZIP"
    )
    package.add_argument("-i", "--input", required=True)
    package.add_argument("-o", "--output", required=True)
    package.add_argument("--expected-rows", type=int)
    return parser


def _predict(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if (args.model is None) != (args.model_parameters_billion is None):
        raise ValueError(
            "--model and --model-parameters-billion must be supplied together"
        )
    if args.model is not None:
        config = replace(
            config,
            lm_studio=replace(
                config.lm_studio,
                model=args.model,
                model_parameters_billion=args.model_parameters_billion,
            ),
        )
    if args.output_mode is not None:
        config = replace(
            config,
            generation=replace(
                config.generation, output_mode=args.output_mode
            ),
        )
    if args.smoke:
        config = smoke_config(config)
    validate_config(config)

    protected_inputs = [Path(args.input), Path(config.train_data_file)]
    if config.synthetic_cot_file:
        protected_inputs.append(Path(config.synthetic_cot_file))
    resolved_output = Path(args.output).resolve()
    for protected in protected_inputs:
        if resolved_output == protected.resolve():
            raise ValueError(
                f"prediction output must not overwrite input data: {protected}"
            )

    rows = read_jsonl(args.input)
    if args.limit is not None:
        if args.limit < 0:
            raise ValueError("--limit cannot be negative")
        rows = rows[: args.limit]
    _validate_input_rows(rows)
    if args.dry_run:
        predictions = [
            {
                "SubjectEntity": row["SubjectEntity"],
                "Relation": row["Relation"],
                "ObjectEntities": [],
            }
            for row in rows
        ]
        write_jsonl(predictions, args.output)
        print(f"Validated {len(rows)} rows; no LM Studio request was made.")
        return 0

    train_rows = load_verified_train_rows(config.train_data_file)
    example_rows = train_rows
    if config.synthetic_cot_file:
        reject_obvious_nontrain_path(
            config.synthetic_cot_file, label="SyntheticCoT data"
        )
        synthetic_rows = load_synthetic_cot_file(
            config.synthetic_cot_file,
            train_rows,
            expected_model=config.lm_studio.model,
            expected_model_parameters_billion=(
                config.lm_studio.model_parameters_billion
            ),
        )
        if not synthetic_rows:
            raise ValueError("synthetic_cot_file contains no usable examples")
        example_rows = synthetic_rows + train_rows

    if args.max_restarts < 0:
        raise ValueError("--max-restarts cannot be negative")
    if args.restart_delay < 0:
        raise ValueError("--restart-delay cannot be negative")
    attempt = 0
    while True:
        pipeline = PredictionPipeline(
            config, LMStudioClient(config.lm_studio), example_rows
        )
        try:
            predictions = pipeline.run(
                rows,
                args.output,
                resume=args.resume or attempt > 0,
                runtime_dir=args.runtime_dir,
            )
            break
        except ValueError:
            raise
        except Exception as exc:
            if attempt >= args.max_restarts:
                raise
            attempt += 1
            print(
                f"Run failed ({type(exc).__name__}: {exc}); restart "
                f"{attempt}/{args.max_restarts} in {args.restart_delay:g}s.",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(args.restart_delay)
    print(f"Wrote {len(predictions)} predictions to {Path(args.output)}")
    return 0


def _validate_input_rows(rows: list[dict]) -> None:
    seen: set[tuple[str, str]] = set()
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"input row {index} must be a JSON object")
        subject = row.get("SubjectEntity")
        relation = row.get("Relation")
        if not isinstance(subject, str) or not subject.strip():
            raise ValueError(f"input row {index} needs a non-empty SubjectEntity")
        if relation not in RELATIONS:
            raise ValueError(f"input row {index} has unsupported relation {relation!r}")
        key = (subject, relation)
        if key in seen:
            raise ValueError(f"input row {index} duplicates key {key!r}")
        seen.add(key)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "predict":
            return _predict(args)
        if args.command == "calibrate":
            calibrate_files(
                args.train,
                args.traces,
                args.output,
                relations=args.relations,
                allow_partial=args.allow_partial,
            )
            print(f"Wrote {args.output}")
        elif args.command == "build-cot":
            rows = build_synthetic_cot_file(args.train, args.traces, args.output)
            print(f"Retained {len(rows)} examples in {args.output}")
        elif args.command == "select":
            rows = select_relation_file(args.input, args.output, args.relations)
            print(f"Wrote {len(rows)} answer-blanked rows to {args.output}")
        elif args.command == "merge":
            rows = merge_prediction_files(args.base, args.partial, args.output)
            print(f"Wrote {len(rows)} merged rows to {args.output}")
        elif args.command == "package":
            count = package_prediction_file(
                args.input, args.output, expected_rows=args.expected_rows
            )
            print(f"Packaged {count} predictions in {args.output}")
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
