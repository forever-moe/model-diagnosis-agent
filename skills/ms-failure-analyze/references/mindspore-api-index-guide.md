# MindSpore API Index Consumption Guide

Use this guide when the failure is about MindSpore API semantics, Primitive mapping,
backend support, ACLNN paths, KBK support, wrappers, `func_op`, or
scenario-dependent `mint` behavior.

## When To Read The Index

Do not load the API index in Stage 0.

Do not load it by default in Stage 1 if `failure-showcase.md` or `error-codes.md`
already gives a clear diagnosis.

Start reading the API index in Stage 2 when any of these is true:

- the report mentions `mindspore.mint`, `mindspore.ops`, or `mindspore.nn`
- the logs mention a Primitive name or an `aclnnXxx` symbol
- the problem is about backend support, GRAPH vs PYNATIVE differences, wrapper
  resolution, `func_op`, view/composite semantics, or ACLNN adaptation
- you need to know which Primitive or ACLNN interface a public API actually reaches

## What To Read

Read in this order:

1. `docs/mindspore_api_index/mindspore_api_analysis_methodology.md`
   - focus on `How To Read The Index`
   - focus on `Error Priority Rules`
2. `docs/mindspore_api_index/mint_api_index.yaml`
   - read only the relevant API record
3. `docs/mindspore_api_index/mint_api_evidence.yaml`
   - read only if the main record is still insufficient

Do not inject the entire YAML files into the prompt. Read only the minimum
records needed for the active diagnosis.

## How To Read A Record

Follow this field order:

1. `semantic_kind`
2. `trust_level`
3. `primitive` or `possible_primitives`
4. `pynative_support` and `graph_kbk_o0_support`
5. `aclnn`
6. `llm_summary`
7. `llm_warning`

Interpretation rules:

- If `unknown_reason=not_applicable`, stop operator mapping analysis.
- If `unknown_reason=func_op_expansion`, do not read GRAPH `unknown` as unsupported.
- If `unknown_reason=scenario_dependent`, continue with parameter- or branch-based reasoning.
- If `fact_origin=inherited_from_construct`, treat primitive/support facts as inherited facts.

## When To Read Evidence

Read a single evidence record only when you must confirm one of these:

- `terminal_symbol`
- `construct` inheritance chain
- `aclnn.effective_interfaces`
- `func_op_expands_to`
- why `llm_warning` or `unknown_reason` is present

If the evidence record is still insufficient, then fall back to source code.

## Version Drift

Before trusting the index, compare the user's MindSpore version with:

- `meta.mindspore_version_hint`
- `meta.repo_commit_hint`

If they are clearly far apart:

- keep using the index as a hint
- explicitly warn that the index may be stale for the user's version
- prefer evidence or source verification before making a hard conclusion

## Update Triggers

The index should be rebuilt when any of these happens:

- changes under `mindspore/python/mindspore/mint/**`
- changes under `mindspore/python/mindspore/ops/function/**`
- changes under `mindspore/python/mindspore/ops/auto_generate/**`
- changes under `mindspore/ops/op_def/**` or `mindspore/ops/api_def/**`
- changes under `mindspore/ccsrc/frontend/operator/meta_dsl/func_op/**`
- changes under `mindspore/ops/kernel/ascend/aclnn/**`
- changes under `mindspore/ops/kernel/cpu/**`
- changes under `mindspore/ops/kernel/gpu/**`
- changes under `mindspore/ops/fallback/**`
- a diagnosis proves the current index wrong or incomplete

Use periodic drift checks only as a fallback. Do not rebuild on every diagnosis.

## Typical Read Patterns

### `mindspore.mint.randint_like`

- expect a direct operator record
- expect `primitive=RandIntLike`
- expect `aclnn.effective_interfaces=aclnnInplaceRandom`
- if needed, read evidence to confirm `terminal_symbol` and `prelude_calls`

### `mindspore.mint.add`

- expect `multi_overload_op`
- do not bind support or ACLNN conclusions to a single primitive

### `mindspore.mint.Conv1d`

- expect `scenario_dependent_module`
- read `possible_primitives`
- do not force a single backend conclusion without branch conditions

### `mindspore.mint.CosineEmbeddingLoss`

- expect `func_op_operator`
- interpret GRAPH through `func_op_expansion`, not direct kernel registration

### `mindspore.mint.distributed.get_rank`

- expect `runtime_utility`
- do not continue operator mapping
