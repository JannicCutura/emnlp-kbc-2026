# Literature-review implementation notes

Reviewed `agents/literature-review.md` on 31 July 2026 against the current
pipeline and challenge rules. The review is a useful methodological synthesis,
but it is not presently citation-ready: the claimed `paper/literature/`
bibliography, source matrix, and English related-work draft are absent, and the
remaining document uses opaque `turn...` citation markers.

## Implemented hard-cutover method

The useful recommendations were consolidated into the one supported pipeline,
`akbc.py`, with `configs/improvement-plan.yaml`. It implements:

- description-first subject identity anchoring;
- relation-wise rationale-bearing self-consistency;
- numeric median aggregation for area and capacity;
- generation-first empty voting for death city and stock exchange;
- low-threshold award union followed by one omission pass;
- train-only threshold calibration and optional, model-bound SyntheticCoT;
- conservative completion and verification only for the strong border path.

The earlier numeric selector, repeated award completion, hard empty gates,
generic verifier, and ablation-specific configurations were deliberately
removed. They either failed empirically or duplicated the fixed method without
adding a clean scientific comparison.

## Deferred ideas

- Finer per-period award continuation with early stopping remains promising,
  but should be added only after the one-pass system has a measured baseline.
- Candidate and empty thresholds should be calibrated on complete training
  traces, never validation gold.
- Model and output-mode comparisons should reuse the same CLI and fixed smoke
  set rather than creating more configs or orchestration paths.

Retrieval-backed verification, relation-specific fine-tuning, another blanket
same-model verifier, and broad changes to the already strong border workflow
are not appropriate next steps.
