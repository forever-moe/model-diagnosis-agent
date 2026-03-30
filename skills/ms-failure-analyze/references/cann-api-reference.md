# CANN API Reference

This document is a narrow routing guide for third-party ACLNN API docs under [docs/cann/aclnn_api_docs/](../../../docs/cann/aclnn_api_docs/).

Do not use it as the default ACLNN diagnosis entrypoint. For most failures, start with:
- [Failure Showcase](failure-showcase.md)
- [Error Codes](error-codes.md)
- [Diagnosis Guide](diagnosis-guide.md)

Use the third-party ACLNN docs only when you need interface-contract details that are not already clear from those documents.

## When to Read ACLNN API Docs

Read `aclnn_api_docs/` only in these cases:
- Error information is still insufficient after checking `failure-showcase.md`, `error-codes.md`, and `diagnosis-guide.md`
- The log explicitly contains an `aclnnXxx` API name
- The failure is clearly about parameter support, dtype support, shape/layout constraints, or optional parameter semantics
- You need to verify whether MindSpore's parameter handling matches the underlying ACLNN interface

Do not read the third-party ACLNN docs by default just because the error is ACLNN-related.

## How to Find the Right Doc

### 1. Extract the ACLNN API Name

Preferred sources, in order:
- Error stack lines such as `LAUNCH_ACLNN`, `GetWorkspaceSize`, or direct `aclnnXxx` mentions
- MindSpore op names when the mapping is obvious, for example `ops.add` -> `aclnnAdd`
- Nearby adaptation code or logs when the failure is in PyBoost/KBK/customize layers

### 2. Locate the File Directly

Do not use an index file. Search the third-party docs directory directly:
- Exact API name, for example `aclnnAdd`
- API family name when the file groups multiple related APIs, for example `aclnnAdd&aclnnInplaceAdd.md`
- If exact lookup fails, search the directory for the API stem and inspect the closest match

### 3. Read Only the Relevant Sections

Focus on:
- Function prototype
- Parameter constraints
- Supported data types
- Shape/layout requirements
- Return values and failure conditions

Ignore the rest unless the diagnosis specifically requires it.

## How to Apply the Doc to Diagnosis

After reading the third-party API doc:
1. State the ACLNN API name you checked
2. State the concrete doc path you read
3. Extract the specific constraints that matter
4. Compare those constraints with the user's actual inputs or error text
5. Explain whether the issue is:
   - a user/input problem
   - a MindSpore adaptation problem
   - still inconclusive

Recommended answer shape when using third-party ACLNN docs:

```text
ACLNN API: aclnnXxx
Doc: docs/cann/aclnn_api_docs/<file>.md
Relevant Constraints:
- ...
- ...
Mapping to Current Error:
- ...
Conclusion:
- ...
```

## Notes

- `aclnn_api_docs/` is third-party crawled content and should not be edited locally.
- If the doc content is unclear or noisy, use it only as a constraint reference, not as the primary diagnosis narrative.
- For ACLNN adaptation workflow, PyBoost/KBK/BPROP/View/composite diagnosis, use [Diagnosis Guide](diagnosis-guide.md) instead.
- For ACLNN error code interpretation and first actions, use [Error Codes](error-codes.md).

## See Also

- [Error Codes](error-codes.md) - High-frequency ACLNN error entrypoint and first actions
- [Diagnosis Guide](diagnosis-guide.md) - Systematic ACLNN adaptation diagnosis
- [Backend Diagnosis](backend-diagnosis.md) - Per-backend diagnosis flow
