---
name: ms-failure-analyze
description: "MindSpore failure analyzer for Ascend/GPU/CPU backends. Use whenever users report ANY MindSpore error, crash, or unexpected behavior — including error codes (CANN, ACLNN, CUDA, NCCL), Python exceptions, device issues (OOM, heartbeat, ECC), graph compilation failures, operator errors, mint/ops API issues, ACLNN adaptation problems (gen_ops.py, PyBoost, KBK, BPROP), or MindSporeTest ST/UT failures. Also use when users paste error logs, stack traces, or CANN debug output."
---

# MS (MindSpore) Failure Analyzer

## Stage 0: Gather Context

Before diving into diagnosis, ensure you have the minimum context. If any of these are missing from the user's report, ask for them upfront — it saves multiple round-trips later.

| Info | Why It Matters | How to Get |
|------|---------------|------------|
| MindSpore version | API/behavior changes across versions | `python -c "import mindspore; print(mindspore.__version__)"` |
| Backend & device | Determines which error code system applies | `device_target` in `set_context()` or `CONTEXT_DEVICE_TARGET` env var |
| CANN version (Ascend) or CUDA version (GPU) | Version mismatch is the #1 silent failure source | `cat /usr/local/Ascend/ascend-toolkit/version` or `nvcc --version` |
| Execution mode | GRAPH_MODE vs PYNATIVE_MODE have different failure patterns | `mode` in `set_context()` or `CONTEXT_MODE` env var |
| Complete error message | Truncated logs hide the real root cause | Ask for full traceback if only a snippet is provided |

If the user provides a clear error with sufficient context, skip directly to Stage 1.

## Stage 1: Find Similar Problem

### 1. Check Failure Showcase (check history FIRST)

Before analyzing the error in detail, search [Failure Showcase](references/failure-showcase.md) for matching patterns:
- Error keywords (OOM, timeout, ECC, link error, compile failed, etc.)
- Call stack patterns (operator names, functions, graph nodes)
- Backend + module combinations (Ascend+HCCL, GPU+NCCL, graph compile, etc.)
- Execution mode issues (GRAPH_MODE vs PYNATIVE_MODE)

If matching failure found:
1. Show historical failure info
2. Provide previously successful solution
3. Ask user: "Does this solution work for your issue?"
4. If yes → END Stage 1 (skip error code analysis entirely)
5. If no → continue to step 2 (Identify Error Pattern)

If no match found → continue to step 2.

### 2. Identify Error Pattern

Parse error message to determine error source. MindSpore errors come in several forms:

> **Reference:** [Diagnosis Guide — Section 12: Quick Reference Error Code Mapping](references/diagnosis-guide.md#12-quick-reference-error-code-mapping) provides a quick lookup table for error codes and their corresponding problem domains. For systematic problem classification, see [Section 1: Problem Classification Overview](references/diagnosis-guide.md#1-problem-classification-overview).

**Python Exceptions (MindSpore Framework):**
- `RuntimeError` — device errors, operator execution failures, graph compilation errors
- `ValueError` — parameter validation, shape mismatches, invalid configurations
- `TypeError` — wrong data types, incompatible tensor types
- `MemoryError` — host or device memory exhaustion
- `NotImplementedError` — unsupported features on specific backend

**CANN Error Codes (Ascend backend, numeric):**
- `1xxxxx` — environment/logic errors (user-fixable)
- `2xxxxx` — resource exhaustion (memory, streams, devices)
- `3xxxxx` — business exceptions (queue full/empty, storage limits)
- `5xxxxx` — internal software/hardware errors (need support)

**CANN Inner Error Codes (Ascend backend, alphanumeric):**
- `E[5-B]xxxx` — AICORE errors (TBE compilation, operator spec issues)
- `E3xxxx` — AICPU kernel execution failures
- `EExxxx` — Runtime errors (HBM OOM, task failure)
- `EI/EJxxxx` — HCCL/HCCP distributed errors
- `EKxxxx` — Profiling errors
- `EZxxxx` — AICORE execution failures (rtStreamSynchronize, model execution)

**ACLNN Error Codes (Ascend operator API):**
- `161xxx` — parameter errors (nullptr, invalid params)
- `361xxx` — runtime call errors
- `561xxx` — internal errors (shape inference, tiling, kernel finding)

If error code found:
1. Extract error code/exception type and context
2. Check [Error Codes reference](references/error-codes.md) for known solutions
3. If direct match → provide solution, then ask user: "Did this resolve your issue?" (same verification loop as Stage 2 Step 3 — do not skip validation just because the match looks obvious)
4. If partial match → proceed to Stage 2

## Stage 2: Analyze Failure

**Failure Orientation Strategy:** Platform → Scripts → MindSpore Framework → Backend (CANN/CUDA/CPU)

### Quick Route (skip levels when evidence is clear)

Based on Stage 1 error pattern, jump directly to the most likely level:

> **Reference:** [Diagnosis Guide — Section 1.2: Decision Tree for Problem Classification](references/diagnosis-guide.md#12-decision-tree-for-problem-classification) provides a decision tree for automated problem routing based on error keywords and patterns.

| Stage 1 Finding | Start At | Rationale |
|-----------------|----------|-----------|
| CANN error code (1xxxxx-5xxxxx) | Backend (Ascend) | Error originates from CANN runtime |
| ACLNN error code (161xxx/361xxx/561xxx) | Backend (Ascend) | Operator-level ACLNN failure |
| CANN inner code (E[x]xxxx) | Backend (Ascend) | TBE/AICPU/HCCL internal error |
| Hardware keywords (heartbeat, ECC, link error) | Platform | Hardware fault signals |
| CUDA error / GPU OOM | Backend (GPU) | GPU runtime error |
| NCCL error | Backend (GPU) | GPU distributed communication |
| Python exception only (ValueError, TypeError) | Scripts | Likely user code or API misuse |
| Graph compilation error / type inference | Framework | MindSpore graph compiler issue |
| gen_ops.py / YAML error | Backend (Ascend) | ACLNN adaptation build error |
| `set_context` / mode / device error | Scripts | Configuration issue |
| No clear signal | Platform (top-down) | Default: scan from top |

When jumping to a specific level, still collect basic evidence from upper levels (version info, device status) to rule out environmental factors.

### Step 1: Collect Evidence

For each orientation layer, collect:
- Platform: HW type, driver version, CANN version (Ascend), CUDA version (GPU), OS info
- Scripts: User code patterns, context configuration, execution mode, library calls
- MindSpore Framework: Operator calls, graph structure, parameter validation, debug logs
- Backend: Error codes from ACL/HCCL/GE (Ascend), CUDA errors (GPU), system errors (CPU)

### Step 2: Orient and Diagnose

Apply orientation strategy (start at the level identified by Quick Route, then expand if needed):

**Platform Level:**
- Ascend: Check NPU device health with `npu-smi info`, verify CANN version compatibility
- GPU: Check GPU status with `nvidia-smi`, verify CUDA/cuDNN version compatibility
- CPU: Check system resources (memory, CPU usage)
- **Version compatibility check** (very common failure source):
  - MindSpore ↔ CANN: each MindSpore version requires a specific CANN version range. Check:
    ```bash
    python -c "import mindspore; print(mindspore.__version__)"
    cat /usr/local/Ascend/ascend-toolkit/version
    ```
    Cross-reference with MindSpore release notes for supported CANN versions.
  - MindSpore ↔ Python: verify Python version matches MindSpore requirements
  - MindSpore ↔ CUDA/cuDNN: for GPU, check CUDA toolkit and cuDNN version match
  - Driver version: Ascend driver (`npu-smi info`) or NVIDIA driver (`nvidia-smi`) must meet minimum requirements
  - If version mismatch detected → this is likely the root cause; advise version alignment before further diagnosis
- Check for hardware errors:
  - Ascend: 507010 heartbeat lost, 507053 memory UCE, 507054 HBM ECC, 507056 link error
  - GPU: CUDA device errors, ECC errors
- If hardware issue → Provide hardware-specific fix → Validate with user

**Script Level:**
- Check `mindspore.set_context()` configuration:
  - `device_target` matches available hardware ("Ascend"/"GPU"/"CPU")
  - `mode` is appropriate (GRAPH_MODE=0, PYNATIVE_MODE=1)
  - `ascend_config` / `gpu_config` settings are valid
- Check environment variables (ASCEND_OPP_PATH, LD_LIBRARY_PATH, CUDA paths)
- **MindSporeTest repo cases** (the following env vars are only effective when test cases originate from the MindSporeTest repo):
  - Global device, mode and jit_level are configured via environment variables, not in-script `set_context()`:
    - `CONTEXT_DEVICE_TARGET` — determines test device ("Ascend"/"GPU"/"CPU")
    - `CONTEXT_MODE` — determines execution mode ("0"=GRAPH_MODE, "1"=PYNATIVE_MODE)
    - `CONTEXT_JIT_LEVEL` — determines global jit_level ("O0"/"O1"/"O2")
  - When reproducing test failures from MindSporeTest, must set these env vars to match the CI/test environment
  - Test scripts may also use `set_context_mode(mode='pynative'|'kbk'|'ge')` from the ST framework:
    - `pynative` = PYNATIVE_MODE (eager execution)
    - `kbk` = KernelByKernel (GRAPH_MODE + jit_level O0, operators execute one-by-one in graph)
    - `ge` = GraphEngine (GRAPH_MODE + jit_level O1/O2, full graph optimization and offload)
- Analyze user code for misuse:
  - Wrong device placement, cross-device tensor operations
  - Shape/dtype mismatches in network definition
  - Improper `construct()` / `@ms_function` usage in GRAPH_MODE
  - Dynamic control flow in static graph mode
- Review script patterns (repeated context init, improper resource cleanup)
- If script issue → Provide code fix → Validate with user

**MindSpore Framework Level:**
- Check operator availability on target backend (`Supported Platforms: Ascend GPU CPU`)
- Verify operator parameter constraints (types, shapes, value ranges)
- Review graph compilation errors (IR optimization, type inference failures)
- For `mindspore.mint` / Primitive / ACLNN / backend-support / wrapper / `func_op` / scenario-dependent questions, read the API index before falling back to source code:
  - first read [MindSpore API Index Consumption Guide](references/mindspore-api-index-guide.md)
  - then read the relevant record from `docs/mindspore_api_index/mint_api_index.yaml`
  - only read `mint_api_evidence.yaml` if you must confirm `terminal_symbol`, `construct` inheritance, `aclnn.effective_interfaces`, `func_op_expands_to`, or explain `llm_warning` / `unknown_reason`
- Identify which API layer the user is using and check accordingly:
  - **`mindspore.mint`** (preferred high-level API, PyTorch-compatible):
    - `mint.*` functions — check view-vs-copy semantics (`@jit_view_unsupported` on squeeze/flatten/reshape/t/narrow/split/broadcast_to)
    - `mint.nn.*` layers — Conv1d/2d/3d, BatchNorm, LayerNorm, GroupNorm, Dropout, loss functions, activations
    - `mint.nn.functional.*` — functional versions of all nn layers
    - `mint.optim.*` — AdamW, Adam, SGD, FusedAdamW optimizers
    - `mint.linalg.*` — inv, norm, vector_norm, matrix_norm, qr
    - `mint.distributed.*` — init_process_group, all_reduce, barrier, etc.
    - `mint.special.*` — erfc, expm1, exp2, log1p, log_softmax
    - Note: many mint APIs are marked "experimental" and may change between versions
  - **`mindspore.ops`** (lower-level operator API):
    - `ops.function.*` — functional API wrappers
    - `ops.operations.*` (Primitive) — low-level operator primitives
  - **`mindspore.nn`** (traditional high-level layers)
  - Check for mint-specific pitfalls:
    - `mint.equal()` returns Python `bool` (not Tensor) — differs from `ops.equal()`
    - `mint.item()` requires single-element Tensor, raises RuntimeError otherwise
    - View ops in mint may not work in JIT/GRAPH_MODE (decorated with `@jit_view_unsupported`)
    - mint wraps `ops.*_ext` variants which are the newer implementations
- Read [MindSpore API reference](references/mindspore-api.md) for API details (includes mint, ops, nn)
- If framework issue → Provide framework fix → Validate with user

**Backend Level (Ascend / GPU / CPU):**

Read the matching section in [Backend Diagnosis Reference](references/backend-diagnosis.md) for detailed steps:

> **Reference:** [Diagnosis Guide](references/diagnosis-guide.md) provides domain-specific diagnosis guidance:
> - [Section 2: Memory Issues](references/diagnosis-guide.md#2-memory-issues) — OOM, memory corruption, HBM/VRAM exhaustion
> - [Section 3: Hardware/Device Issues](references/diagnosis-guide.md#3-hardwaredevice-issues) — heartbeat lost, ECC errors, device timeout
> - [Section 4: Distributed Communication Issues](references/diagnosis-guide.md#4-distributed-communication-issues) — HCCL/NCCL errors, timeout, rank configuration
> - [Section 5: Operator Issues](references/diagnosis-guide.md#5-operator-issues) — TBE/ACLNN errors, operator not supported
> - [Section 6: Graph Compilation Issues](references/diagnosis-guide.md#6-graph-compilation-issues) — static graph errors, type inference, context empty
> - [Section 7: Precision Issues](references/diagnosis-guide.md#7-precision-issues) — float16 instability, seed-dependent errors, backend differences, MindSpore vs Torch_NPU comparison
> - [Section 8: Environment/Configuration Issues](references/diagnosis-guide.md#8-environmentconfiguration-issues) — CANN missing, version mismatch, profiler issues
> - [Section 9: API Usage Issues](references/diagnosis-guide.md#9-api-usage-issues) — mint API pitfalls, return type confusion, parameter validation
> - [Section 10: ACLNN Adaptation Issues](references/diagnosis-guide.md#10-aclnn-adaptation-issues) — gen_ops.py, PyBoost, KBK, BPROP errors
> - [Section 11: Parallel/Distributed Training Issues](references/diagnosis-guide.md#11-parallel-distributed-training-issues) — pipeline parallel, semi-auto parallel, optimizer parallel

- **Ascend** — CANN/ACLNN error codes, CANN logs, TBE/AKG compilation, ACLNN adaptation-level diagnosis (gen_ops.py → GeneralInfer → PyBoost/KBK → BPROP → View/Composite)
  - **ACLNN API Documentation:** Only read third-party docs from [aclnn_api_docs/](../../../docs/cann/aclnn_api_docs/) when one of these is true:
    1. Error info is still insufficient after checking `failure-showcase.md`, `error-codes.md`, and `diagnosis-guide.md`
    2. The log explicitly contains an `aclnnXxx` API name
    3. The failure is clearly about dtype support, parameter support, shape/layout constraints, or optional parameter semantics
    4. You need to verify whether MindSpore parameter handling matches the underlying ACLNN interface
  - When reading third-party ACLNN docs:
    1. Extract the API name from stack traces, adaptation code, or nearby logs
    2. Search [aclnn_api_docs/](../../../docs/cann/aclnn_api_docs/) directly by API name or API stem
    3. Focus only on function prototype, parameter constraints, dtype support, shape/layout rules, and failure conditions
    4. In the answer, include the ACLNN API name, concrete doc path, extracted constraints, and how they map to the current error
  - For the read workflow only, see [CANN API Reference](references/cann-api-reference.md)
- **GPU** — CUDA errors (`CUDA_LAUNCH_BLOCKING=1`), NCCL distributed, cuDNN, compute capability
- **CPU** — System resources, operator availability, threading config, segfault diagnosis

**MindSpore Framework Log Analysis:**

Read [Backend Diagnosis Reference — Log Analysis](references/backend-diagnosis.md#mindspore-framework-log-analysis) for GLOG patterns and graph dump analysis.

**Show Fix Advice:**

When root cause is **confirmed**:
```
Analysis: [Failure type identified]
Backend: [Ascend/GPU/CPU]
Root Cause: [Specific cause]
Solution: [Actionable steps]
```

When root cause is **uncertain** — provide further location means instead of guessing:
```
Analysis: [Failure type identified]
Backend: [Ascend/GPU/CPU]
Root Cause: Unable to confirm — further debugging required
Further Location:
  [Specific debug steps with env vars, log commands, or debug patches]
```

**Further location techniques:** Read [Backend Diagnosis Reference — Further Location Techniques](references/backend-diagnosis.md#further-location-techniques) for CANN debug log analysis and MindSpore debug patch guidelines.

### Step 3: Validate and Iterate

After showing fix advice, ask user: "Did this resolve your issue?"

Handle response:
- **User confirms fixed** → Proceed to Stage 3
- **User says not fixed** → Loop back to Step 2 with new evidence
- **User provides debug output** → Analyze, refine, show updated advice, ask again
- **User asks to implement fix** → Execute the fix, then **re-ask** validation question (see below)
- **User changes topic** → Ask about original issue first, then handle new topic

**Why verification matters — do not skip this step:**

Writing an unverified fix into failure-showcase.md creates false knowledge that misleads all future diagnoses. A proposed fix may look correct but fail in the user's actual environment due to version differences, edge cases, or misidentified root causes. The cost of recording a wrong solution far exceeds the cost of one extra confirmation round-trip.

Therefore, after showing fix advice or executing a code change at the user's request, always re-ask the validation question: "Change applied. Please verify in your test environment — did this resolve the issue?" Only proceed to Stage 3 (Accumulate Experience) after the user explicitly confirms resolution (e.g., "resolved", "passed", "fixed", "works now"). Until then:
- Do not write or update failure-showcase.md
- Do not declare the diagnosis complete
- Do not move on to a different problem
- If the user ignores the question, ask again

## Stage 3: Accumulate Experience

### Step 1: Report Analysis Summary

Review analysis and extract key points:
- **Failure Info:** Error message, context, environment
- **Backend:** Ascend/GPU/CPU
- **Failure Type:** Platform/Scripts/Framework/Backend
- **Root Cause:** Specific issue identified
- **Solution:** Steps that resolved the issue

Report to user as structured summary.

### Step 2: Update Failure Showcase

First, search [Failure Showcase reference](references/failure-showcase.md) for matching existing entries.

**Matching criteria** — an existing entry "matches" if ANY of these hold:
- Same error code (e.g., both have `507010` or `561003`)
- Same exception type AND similar root cause (e.g., both are `RuntimeError` from dynamic shape in bprop)
- Overlapping error keywords in `failure_info` (≥ 2 significant keywords match)

**If matching entry exists** — update it in place (do NOT create a duplicate):
- Enrich `failure_info` / `root_cause` / `solution` with any new details learned
- Refresh `last_seen` to current date
- Increment `occurrences` count
- **Fill `observed_at`** if it was empty — always record at least one concrete observation location

**If no matching entry exists** — create a new entry (all fields required, `observed_at` must NOT be left empty):
```
- failure_info: "[error keywords/context]"
- observed_at: "[file:function or test location where observed]"
- backend: "ascend|gpu|cpu|all"
- failure_type: "platform|scripts|framework|backend"
- root_cause: "[specific cause]"
- solution: "[actionable steps]"
- last_seen: "[timestamp]"
- occurrences: [count]
```

### Step 3: Update Diagnosis Guide

After updating `failure-showcase.md`, evaluate whether `diagnosis-guide.md` needs synchronization:

> **Reference:** [Diagnosis Guide — Document Update Mechanism](references/diagnosis-guide.md#document-update-mechanism) provides complete update triggers, workflow, and quality assurance checklist.

**Quick evaluation checklist:**
- [ ] Does this case introduce a **new error code**? → Update Section 12: Error Code Mapping
- [ ] Does this case represent a **new problem domain**? → Consider adding new section
- [ ] Does this case require **new diagnosis steps**? → Update corresponding section
- [ ] Does this case have a **novel solution approach**? → Update solutions table
- [ ] Does this case **consolidate similar patterns**? → Refactor related sections

**If any checklist item is YES**, update the corresponding section in `diagnosis-guide.md`:
1. Add/update error patterns and identification features
2. Add/update diagnosis steps
3. Add/update solutions table
4. Update case reference in Appendix: Case Index
5. Increment version number following [Version Increment Rules](references/diagnosis-guide.md#version-increment-rules)
6. Update `Last Updated` date

## Stage 4: Validate and Close

**Default assumption**: The local machine does NOT have Ascend/GPU hardware or the required runtime environment to execute verification tests. Do NOT attempt to run test commands locally unless the user explicitly confirms the local machine can run them.

After giving the diagnosis and proposed fix:
1. Ask the user whether they need help deploying the fix to a **remote server** for compilation and verification.
2. If the user confirms remote verification → follow the [Remote Deploy & Verify](references/remote-deploy-verify.md) workflow.
3. If the user prefers to verify on their own (locally or remotely) → wait for them to report the result.
4. After remote verification completes, return to this stage to confirm final resolution.

**Remote Deploy Workflow Overview:**

```
collect → gather context → preflight → sync (dry-run → confirm → execute)
  → build (if C++ changed) → install → run → summarize → return to Stage 4
```

**Quick path for test-only changes:**

```
sync-run   (= sync + run in one step, skipping build)
```

See [Remote Deploy & Verify Reference](references/remote-deploy-verify.md) for detailed workflow and configuration.

## Quick References

- [Error Codes](references/error-codes.md) - Complete error code mappings (MindSpore + CANN + ACLNN)
- [Failure Showcase](references/failure-showcase.md) - Historical failures and solutions
- [Diagnosis Guide](references/diagnosis-guide.md) - Systematic problem classification, error pattern identification, and diagnosis direction mapping (v1.0)
- [MindSpore API](references/mindspore-api.md) - API system (mint/ops/nn), backend registration, operator patterns
- [MindSpore API Index Guide](references/mindspore-api-index-guide.md) - When and how to consume the Mint API index and methodology docs
- [CANN API Reference](references/cann-api-reference.md) - How and when to read third-party ACLNN API docs
- [Backend Diagnosis](references/backend-diagnosis.md) - Per-backend (Ascend/GPU/CPU) detailed diagnosis, log analysis, further location techniques
- [Remote Deploy & Verify](references/remote-deploy-verify.md) - Remote deployment and verification workflow for MindSpore development

## Skill Directory and Tool Paths

This skill's root directory is the folder containing this `SKILL.md` file.
All tool scripts live under `<skill_dir>/tools/` and all references under `<skill_dir>/references/`.

When executing tool scripts via Shell, always resolve the **full absolute path** first. For example, if this SKILL.md is at `/repo/skills/ms-failure-analyze/SKILL.md`, then:

```bash
python /repo/skills/ms-failure-analyze/tools/remote_deploy_verify.py collect --local-root /path/to/mindspore
```

Do NOT rely on relative paths like `python tools/remote_deploy_verify.py` — the shell working directory is usually the workspace root, not the skill directory.

## Diagnostic Commands

```bash
# Ascend
npu-smi info                                     # NPU device status
cat /usr/local/Ascend/ascend-toolkit/version      # CANN version
ls /usr/local/Ascend/ascend-toolkit/latest/opp/   # Operator packages
tail -f /var/log/npu/slog/*/device-*/plog/*.log   # CANN device logs

# GPU
nvidia-smi                                        # GPU device status
nvcc --version                                    # CUDA version
nvidia-smi topo -m                                # GPU topology (distributed)

# MindSpore
python -c "import mindspore; print(mindspore.__version__)"
python -c "import mindspore; print(mindspore.get_context('device_target'))"
python -c "import mindspore; print(mindspore.get_context('mode'))"
python -c "import mindspore; print(mindspore.hal.device_count())"  # Available devices
python -c "import mindspore; print(mindspore.get_context('jit_config'))"  # JIT config

# System
free -h                                           # Memory status
ulimit -a                                         # System limits
```

## Environment Variables

- `ASCEND_OPP_PATH` — Operator compiler path (Ascend)
- `ASCEND_GLOBAL_LOG_LEVEL` — CANN log level (0=debug, 1=info, 2=warning, 3=error, 4=null)
- `ASCEND_SLOG_PRINT_TO_STDOUT` — Print CANN slog to stdout (1=enable, 0=disable)
- `GLOG_v` — MindSpore log level (0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR, 4=CRITICAL)
- `MS_EXCEPTION_DISPLAY_LEVEL` — Exception display (0=full, 1=hide framework stack)
- `CUDA_VISIBLE_DEVICES` — GPU device visibility
- `NCCL_DEBUG` — NCCL debug level (GPU distributed)
- `MS_DEV_FORCE_ACL` — Force ACL kernel execution (Ascend debugging)
- `CONTEXT_DEVICE_TARGET` — MindSporeTest repo: global device target ("Ascend"/"GPU"/"CPU")
- `CONTEXT_MODE` — MindSporeTest repo: global execution mode ("0"=GRAPH_MODE, "1"=PYNATIVE_MODE)
- `CONTEXT_JIT_LEVEL` — MindSporeTest repo: global jit_level ("O0"/"O1"/"O2")
