# MindSpore Failure Diagnosis Guide

This document provides systematic guidance for diagnosing MindSpore failures based on historical patterns. It enables LLM-based failure analyzers to quickly identify problem types and determine appropriate investigation directions.

---

## Table of Contents

1. [Problem Classification Overview](#1-problem-classification-overview)
2. [Memory Issues](#2-memory-issues)
3. [Hardware/Device Issues](#3-hardwaredevice-issues)
4. [Distributed Communication Issues](#4-distributed-communication-issues)
5. [Operator Issues](#5-operator-issues)
6. [Graph Compilation Issues](#6-graph-compilation-issues)
7. [Precision Issues](#7-precision-issues)
8. [Environment/Configuration Issues](#8-environmentconfiguration-issues)
9. [API Usage Issues](#9-api-usage-issues)
10. [ACLNN Adaptation Issues](#10-aclnn-adaptation-issues)
11. [Parallel/Distributed Training Issues](#11-parallel-distributed-training-issues)
12. [Quick Reference: Error Code Mapping](#12-quick-reference-error-code-mapping)
13. [Document Update Mechanism](#document-update-mechanism)

---

## 1. Problem Classification Overview

### 1.1 Classification Dimensions

| Dimension | Categories | Description |
|-----------|------------|-------------|
| **failure_type** | platform, scripts, framework, backend | Root cause attribution |
| **backend** | ascend, gpu, cpu, all | Target hardware platform |
| **problem_domain** | memory, hardware, communication, operator, graph, precision, environment, api | Problem domain classification |

### 1.2 Decision Tree for Problem Classification

```
Error Occurred
    │
    ├─ Contains error codes (ELxxxx, EZxxxx, 50xxxx)?
    │   └─ YES → Check Error Code Mapping (Section 12)
    │
    ├─ Contains "memory", "OOM", "exhausted"?
    │   └─ YES → Memory Issues (Section 2)
    │
    ├─ Contains "HCCL", "NCCL", "distributed", "timeout"?
    │   └─ YES → Distributed Communication Issues (Section 4)
    │
    ├─ Contains operator name + "not supported"/"compile failed"?
    │   └─ YES → Operator Issues (Section 5)
    │
    ├─ Contains "graph compile", "type inference", "construct"?
    │   └─ YES → Graph Compilation Issues (Section 6)
    │
    ├─ Contains "precision", "loss", "float16", "float32"?
    │   └─ YES → Precision Issues (Section 7)
    │
    ├─ Contains "CANN", "libascendcl", "environment"?
    │   └─ YES → Environment Issues (Section 8)
    │
    ├─ Contains "mint.", "mint.nn"?
    │   └─ YES → API Usage Issues (Section 9)
    │
    └─ Contains "aclnn", "PyBoost", "KBK", "BPROP"?
        └─ YES → ACLNN Adaptation Issues (Section 10)
```

---

## 2. Memory Issues

### 2.1 Ascend HBM Out of Memory

**Error Patterns:**
```
EL0004, 200000, 207018, RuntimeError, out of memory, device memory exhausted
```

**Identification Features:**
- Error codes: `EL0004`, `200000`, `207018`
- Keywords: `out of memory`, `memory exhausted`, `HBM`
- Backend: `ascend`

**Root Cause Analysis:**
1. Large tensor allocation exceeds available HBM
2. Batch size too large for model
3. Graph memory fragmentation
4. Memory leak from incomplete cleanup

**Diagnosis Steps:**
1. Check `npu-smi info` for memory usage
2. Review model size and batch size configuration
3. Check for memory fragmentation in long-running processes
4. Verify gradient checkpointing settings

**Solutions:**
| Solution | When to Apply | Implementation |
|----------|---------------|----------------|
| Reduce batch size | Model fits but batch too large | Decrease batch_size parameter |
| Gradient checkpointing | Large model training | Use `ms.nn.CellRecompute` |
| Memory cache cleanup | Memory fragmentation | Call `ms.hal.memory.empty_cache()` |
| Limit device memory | Need memory headroom | Set `max_device_memory` in context |

**Case Reference:** Common Failure Patterns - Out of Memory (Ascend HBM)

---

### 2.2 GPU VRAM Out of Memory

**Error Patterns:**
```
CUDA out of memory, RuntimeError, CUDA error
```

**Identification Features:**
- Keywords: `CUDA out of memory`, `CUDA error`
- Backend: `gpu`

**Root Cause Analysis:**
1. Model too large for GPU VRAM
2. Batch size exceeds available memory
3. Other processes consuming VRAM

**Diagnosis Steps:**
1. Run `nvidia-smi` to check VRAM usage
2. Identify other processes using GPU memory
3. Review model architecture and batch size

**Solutions:**
| Solution | When to Apply | Implementation |
|----------|---------------|----------------|
| Reduce batch size | Training OOM | Decrease batch_size |
| Gradient checkpointing | Large model | Enable gradient checkpointing |
| Check GPU visibility | Wrong GPU selected | Set `CUDA_VISIBLE_DEVICES` |
| Kill competing processes | Other processes using VRAM | Identify and terminate |

**Case Reference:** Common Failure Patterns - Out of Memory (GPU VRAM)

---

### 2.3 Memory Corruption / Dirty Memory

**Error Patterns:**
```
memory corruption, dirty memory, silent corruption, incorrect output
```

**Identification Features:**
- Keywords: `memory corruption`, `dirty memory`, `silent corruption`
- Often manifests as incorrect results rather than explicit errors
- May be intermittent

**Root Cause Analysis:**
1. CANN operator backward has memory stomp behavior
2. View operator strides calculation error
3. Inplace operation on view tensor
4. Memory access out of bounds

**Diagnosis Steps:**
1. Isolate the failing operator with minimal reproduction
2. Check if issue is deterministic or intermittent
3. Compare with CPU/GPU backend results
4. Review CANN version compatibility

**Solutions:**
| Solution | When to Apply |
|----------|---------------|
| Upgrade CANN version | CANN operator bug |
| Use alternative operator | Specific operator has known issue |
| Avoid inplace on view tensors | View inplace scenario |
| Add memory barriers | Race condition in memory access |

**Case References:**
- PadV3 Dynamic Shape Memory Corruption
- mint.max Empty Tensor Dirty Memory
- View Op Strides Calculation Error

---

## 3. Hardware/Device Issues

### 3.1 Device Heartbeat Lost

**Error Patterns:**
```
507010, lost heartbeat, RuntimeError, task scheduler, device hang
```

**Identification Features:**
- Error code: `507010`
- Keywords: `lost heartbeat`, `device hang`, `task scheduler`
- Backend: `ascend`

**Root Cause Analysis:**
1. NPU device unresponsive
2. Thermal throttling
3. Hardware failure
4. Driver crash

**Diagnosis Steps:**
1. Run `npu-smi info` to check device health
2. Check device temperature
3. Review system logs for hardware errors
4. Test with different device if available

**Solutions:**
| Solution | When to Apply |
|----------|---------------|
| Device reset | Device unresponsive |
| Check cooling | Thermal throttling suspected |
| Hardware support | Persistent hardware failure |
| Driver reinstall | Driver crash |

**Case Reference:** Common Failure Patterns - Device Heartbeat Lost

---

### 3.2 HBM ECC Error

**Error Patterns:**
```
507054, HBM ECC, multi-bit, hardware error, memory fault
```

**Identification Features:**
- Error code: `507054`
- Keywords: `HBM ECC`, `multi-bit`, `hardware error`
- Backend: `ascend`

**Root Cause Analysis:**
HBM memory hardware ECC fault - this is a hardware failure.

**Diagnosis Steps:**
1. Check `npu-smi info` for ECC error count
2. Run hardware diagnostics
3. Test on different device

**Solutions:**
- **Hardware fault** - Contact hardware support
- Try different device if available
- Replace faulty hardware

**Case Reference:** Common Failure Patterns - HBM ECC Error

---

### 3.3 AI Core Execution Timeout

**Error Patterns:**
```
507014, AI Core timeout, AICORE_TIMEOUT, execution timeout
```

**Identification Features:**
- Error code: `507014`
- Keywords: `AI Core timeout`, `execution timeout`
- Backend: `ascend`

**Root Cause Analysis:**
1. Infinite loop in operator
2. Operator computation too complex
3. Hardware issue

**Diagnosis Steps:**
1. Check CANN logs for specific operator
2. Try reducing input size
3. Check if operator has known timeout issues in current CANN version

**Solutions:**
| Solution | When to Apply |
|----------|---------------|
| Reduce input size | Operator computation heavy |
| Upgrade CANN | Known timeout issue in version |
| Use alternative operator | Specific operator problematic |
| Contact support | Persistent unknown cause |

**Case Reference:** Common Failure Patterns - AI Core Execution Timeout

---

## 4. Distributed Communication Issues

### 4.1 HCCL Communication Error (Ascend)

**Error Patterns:**
```
HCCL, timeout, EI0002, EI0006, notify wait, socket build, distributed
```

**Identification Features:**
- Error codes: `EI0002`, `EI0006`
- Keywords: `HCCL`, `timeout`, `distributed`, `notify wait`
- Backend: `ascend`

**Root Cause Analysis:**
1. Network timeout between devices
2. Rank configuration error
3. HCCS link issue
4. Communication group not created
5. Execution sequence mismatch between ranks

**Diagnosis Steps:**
1. Check network connectivity between devices
2. Verify ranktable configuration
3. Check `HCCL_WHITELIST_DISABLE` setting
4. Ensure all ranks start correctly
5. Check if communication domain is initialized

**Solutions:**
| Solution | When to Apply | Implementation |
|----------|---------------|----------------|
| Check network | Connection timeout | Verify network connectivity |
| Fix ranktable | Configuration error | Correct rank_table_file |
| Set HCCL_WHITELIST_DISABLE | Network restriction | `export HCCL_WHITELIST_DISABLE=1` |
| Synchronize ranks | Rank startup timing | Ensure all ranks start before communication |
| Skip communication for group_size=1 | Single card scenario | Add logic to skip when group_size equals 1 |

**Case References:**
- Common Failure Patterns - HCCL Communication Error
- HCCL CCOOL RootInfo Cleanup
- HCCL destroy_process_group Async Send/Recv
- vllm ops.AllReduce group_size=1

---

### 4.2 NCCL Communication Error (GPU)

**Error Patterns:**
```
NCCL error, timeout, unhandled system error, distributed, GPU
```

**Identification Features:**
- Keywords: `NCCL`, `timeout`, `distributed`
- Backend: `gpu`

**Root Cause Analysis:**
1. Network issue
2. GPU topology problem
3. NCCL version mismatch
4. Firewall blocking

**Diagnosis Steps:**
1. Check GPU connectivity with `nvidia-smi topo`
2. Set `NCCL_DEBUG=INFO` for detailed logs
3. Verify NCCL version compatibility
4. Check firewall settings

**Solutions:**
| Solution | When to Apply |
|----------|---------------|
| Check GPU topology | Topology issue | Run `nvidia-smi topo` |
| Enable NCCL debug | Need detailed logs | `export NCCL_DEBUG=INFO` |
| Update NCCL | Version mismatch | Install compatible NCCL version |
| Configure firewall | Network blocking | Open required ports |

**Case Reference:** Common Failure Patterns - NCCL Communication Error

---

### 4.3 Distributed Initialization Failure

**Error Patterns:**
```
init_process_group, TCPStore, connection refused, distributed, mint.distributed
```

**Identification Features:**
- Keywords: `init_process_group`, `TCPStore`, `connection refused`
- Backend: `all`

**Root Cause Analysis:**
1. Master address/port unreachable
2. Firewall blocking
3. Incorrect rank configuration
4. Master process not started before workers

**Diagnosis Steps:**
1. Verify `MASTER_ADDR` and `MASTER_PORT` environment variables
2. Check firewall rules
3. Ensure all ranks use consistent `world_size`
4. Check if master process started before workers

**Solutions:**
| Solution | When to Apply |
|----------|---------------|
| Verify environment variables | Address/port issue | Check MASTER_ADDR/MASTER_PORT |
| Configure firewall | Connection blocked | Open required ports |
| Fix rank configuration | Configuration error | Ensure consistent world_size |
| Start master first | Timing issue | Ensure master starts before workers |

**Case References:**
- Common Failure Patterns - mint.distributed init_process_group Failure
- DistributedMeta local_rank_id Device ID

---

## 5. Operator Issues

### 5.1 Operator Not Supported on Backend

**Error Patterns:**
```
RuntimeError, not supported, operator, backend, fallback
```

**Identification Features:**
- Keywords: `not supported`, `operator`, `backend`, `fallback`
- Backend: `all`

**Root Cause Analysis:**
1. Operator not implemented for target backend
2. Current MindSpore version doesn't support the operator
3. CANN version doesn't support the operator

**Diagnosis Steps:**
1. Check operator's Supported Platforms documentation
2. Verify MindSpore and CANN versions
3. Search for alternative operators

**Solutions:**
| Solution | When to Apply |
|----------|---------------|
| Check supported platforms | First step | Review operator documentation |
| Upgrade MindSpore/CANN | Version issue | Update to newer version |
| Use alternative operator | Not available | Find equivalent operator |
| CPU fallback | Specific ops | Use CPU for unsupported operators |

**Case Reference:** Common Failure Patterns - Operator Not Supported on Backend

---

### 5.2 TBE Operator Compilation Error

**Error Patterns:**
```
TBE, compile failed, E9xxxx, EBxxxx, operator compilation, UB overflow
```

**Identification Features:**
- Error codes: `E9xxxx`, `EBxxxx`
- Keywords: `TBE`, `compile failed`, `UB overflow`
- Backend: `ascend`

**Root Cause Analysis:**
1. Unsupported shape/dtype for TBE operator
2. UB memory overflow during compilation
3. CANN version incompatibility

**Diagnosis Steps:**
1. Check input shapes/dtypes match TBE operator constraints
2. Check CANN version compatibility
3. Review CANN logs for detailed compilation error

**Solutions:**
| Solution | When to Apply | Implementation |
|----------|---------------|----------------|
| Fix input shapes/dtypes | Constraint violation | Match operator requirements |
| Disable JIT compile | Compilation issue | Set `jit_compile=False` in ascend_config |
| Upgrade CANN | Version incompatibility | Use compatible CANN version |
| Check CANN logs | Need detailed error | Review compilation logs |

**Case Reference:** Common Failure Patterns - TBE Operator Compilation Error

---

### 5.3 ACLNN Operator Error

**Error Patterns:**
```
EZ1001, EZ9999, aclnn, GetWorkspaceSize, RuntimeError
```

**Identification Features:**
- Error codes: `EZ1001`, `EZ9999`, `EZ9903`
- Keywords: `aclnn`, `GetWorkspaceSize`, `LAUNCH_ACLNN`
- Backend: `ascend`

**Root Cause Analysis:**
1. Invalid parameters passed to ACLNN API
2. Input tensor shape/dtype not supported
3. Empty tensor handling issue
4. ACLNN interface not properly adapted

**Diagnosis Steps:**
1. Check ACLNN API parameter requirements
2. Verify input tensor properties
3. Check for empty tensor edge cases
4. Compare with PTA implementation

**Solutions:**
| Solution | When to Apply |
|----------|---------------|
| Fix parameter validation | Invalid parameters | Check parameter constraints |
| Handle empty tensors | Empty tensor issue | Add empty tensor handling |
| Check input constraints | Shape/dtype issue | Verify against ACLNN spec |
| Add L2_DFX_PHASE1 macro | Missing macro | Add required macro before L2 interface |

**Case References:**
- aclnnAll/aclnnReduceSum Duplicate Dims in Axis
- mint.nn.functional.max_pool2d L2_DFX_PHASE1 Missing
- EZ1001 GroupedMatmul Internal Input Count

---

### 5.4 AICPU Operator Issues

**Error Patterns:**
```
aicpu, AICPU, eigen, precision, complex
```

**Identification Features:**
- Keywords: `aicpu`, `AICPU`, `eigen`
- Often involves complex types or special operators

**Root Cause Analysis:**
1. AICPU implementation uses eigen library
2. CANN eigen library upgrade changed behavior
3. Complex type handling differences

**Diagnosis Steps:**
1. Check if operator uses AICPU implementation
2. Compare with previous CANN version behavior
3. Review precision requirements

**Solutions:**
| Solution | When to Apply |
|----------|---------------|
| Adjust precision standard | Precision change | Update test tolerance |
| Use alternative implementation | AICPU issue | Find ACLNN equivalent |
| Pin CANN version | Behavior change | Use specific CANN version |

**Case Reference:** tensor.sigmoid Complex Type Precision

---

## 6. Graph Compilation Issues

### 6.1 Static Graph Compilation Error

**Error Patterns:**
```
RuntimeError, graph compile, type inference, abstract type, infer failed
```

**Identification Features:**
- Keywords: `graph compile`, `type inference`, `infer failed`, `construct`
- Occurs in `GRAPH_MODE`
- Backend: `all`

**Root Cause Analysis:**
1. Unsupported Python syntax in `construct()`
2. Dynamic control flow in static graph
3. Type inference failure
4. Custom class not properly decorated

**Diagnosis Steps:**
1. Switch to `PYNATIVE_MODE` to isolate issue
2. Review `construct()` method for unsupported syntax
3. Check for dynamic control flow (if/for with tensor conditions)
4. Verify custom classes use `@ms.jit_class`

**Solutions:**
| Solution | When to Apply | Implementation |
|----------|---------------|----------------|
| Switch to PYNATIVE_MODE | Debugging | `set_context(mode=PYNATIVE_MODE)` |
| Remove dynamic control flow | Static graph requirement | Use static conditions |
| Use @ms.jit_class | Custom class issue | Decorate custom classes |
| Check supported platforms | Operator issue | Verify operator availability |

**Case Reference:** Common Failure Patterns - Graph Compilation Error (Static Graph)

---

### 6.2 Context Empty Error

**Error Patterns:**
```
107002, context is empty, aclrtSetContext, aclrtSetDevice, RuntimeError
```

**Identification Features:**
- Error code: `107002`
- Keywords: `context is empty`, `aclrtSetContext`, `aclrtSetDevice`
- Backend: `ascend`

**Root Cause Analysis:**
NPU context not initialized before calling NPU operations.

**Diagnosis Steps:**
1. Check initialization sequence
2. Verify `set_context` is called before tensor operations
3. Check for multi-process context sharing issues

**Solutions:**
| Solution | When to Apply |
|----------|---------------|
| Initialize context first | Missing initialization | Call `set_context(device_target='Ascend', device_id=N)` before operations |
| Fix initialization sequence | Order issue | Ensure context created before any NPU operations |

**Case Reference:** Common Failure Patterns - Context Empty (Ascend)

---

### 6.3 View Operation in GRAPH_MODE

**Error Patterns:**
```
jit_view_unsupported, view, squeeze, flatten, reshape, GRAPH_MODE, graph compile
```

**Identification Features:**
- Keywords: `jit_view_unsupported`, `view`, `GRAPH_MODE`
- Involves mint view operations
- Backend: `all`

**Root Cause Analysis:**
mint view operations (squeeze, unsqueeze, flatten, reshape, t, narrow, split, broadcast_to, permute, transpose) are decorated with `@jit_view_unsupported` and may fail in GRAPH_MODE.

**Diagnosis Steps:**
1. Identify mint view operations in code
2. Check if running in GRAPH_MODE
3. Test in PYNATIVE_MODE

**Solutions:**
| Solution | When to Apply |
|----------|---------------|
| Switch to PYNATIVE_MODE | Debugging | Use dynamic graph mode |
| Use ops.* equivalents | GRAPH_MODE required | Replace mint.view ops with ops.* |
| Use .copy() | Need materialization | Add `.copy()` after view ops |

**Case Reference:** Common Failure Patterns - mint View Op in GRAPH_MODE / JIT

---

## 7. Precision Issues

### 7.1 Float16 Precision Instability

**Error Patterns:**
```
precision, float16, fp16, loss, seed, random
```

**Identification Features:**
- Keywords: `precision`, `float16`, `fp16`, `loss`
- Often seed-dependent
- Backend: `all`

**Root Cause Analysis:**
1. Float16 has limited precision range
2. Accumulation errors in long computations
3. Critical value edge cases
4. Benchmark reference mismatch

**Diagnosis Steps:**
1. Check if issue is seed-dependent
2. Compare with float32 results
3. Identify accumulation count in computation
4. Review benchmark reference implementation

**Solutions:**
| Solution | When to Apply | Implementation |
|----------|---------------|----------------|
| Adjust loss tolerance | Accumulation error | Increase tolerance based on accumulation count |
| Use float32 | Precision critical | Convert to float32 for computation |
| Limit random range | Critical values | Clip random values away from critical points |
| Use dual benchmark | Reference mismatch | Compare with multiple references |

**Case References:**
- mint.cumsum Float16 Precision Standard
- ops.logcumsumexp Float16 Precision
- CPU floor_mod fp16 Precision With Fixed Seed

---

### 7.2 Seed-Dependent Precision Issues

**Error Patterns:**
```
precision, seed, intermittent, random, specific seed
```

**Identification Features:**
- Issue appears only with specific random seeds
- Intermittent failures
- Often related to edge case inputs

**Root Cause Analysis:**
1. Random data generates edge case values
2. Critical values trigger precision boundaries
3. Division by small values amplifies errors
4. Gradient accumulation with specific patterns

**Diagnosis Steps:**
1. Identify the failing seed
2. Analyze the generated data at that seed
3. Check for edge case values (very small, very large, critical points)
4. Review computation for error amplification points

**Solutions:**
| Solution | When to Apply |
|----------|---------------|
| Limit random range | Edge case values | Clip random generation range |
| Use fixed seed | Reproducibility | Set specific seed for testing |
| Adjust tolerance | Accumulation error | Increase loss tolerance |
| Skip critical values | Boundary issues | Avoid generating critical values |

**Case References:**
- nn.tanh Ascend Intermittent Precision Issue
- mint.nn.functional.hardswish Intermittent Precision Issue
- CPU ops.prod Precision With Small Values

---

### 7.3 Backend Precision Differences

**Error Patterns:**
```
precision, CPU, Ascend, GPU, backend, different result
```

**Identification Features:**
- Different results on different backends
- Precision mismatch between implementations

**Root Cause Analysis:**
1. Different operator implementations across backends
2. Different numerical libraries (eigen vs others)
3. Different accumulation orders
4. Architecture-specific optimizations (AVX, ARM)

**Diagnosis Steps:**
1. Compare results across backends
2. Check operator implementation differences
3. Review numerical library versions
4. Check for architecture-specific code paths

**Solutions:**
| Solution | When to Apply |
|----------|---------------|
| Use backend-specific tolerance | Different implementations | Set different loss for each backend |
| Align implementations | Consistency required | Modify implementation to match |
| Use common reference | Benchmark issue | Use numpy or torch as reference |

**Case References:**
- AVX/AVX512 vs ARM/SSE MatMul Precision Inconsistency
- CPU cummax fp32 vs fp64 Precision
- ops.gelu Ascend Backend Intermittent Precision Issue

---

### 7.4 MindSpore vs Torch_NPU Precision Comparison (Ascend)

**Error Patterns:**
```
precision, MindSpore, Torch_NPU, ACLNN, binary mismatch, Ascend
```

**Identification Features:**
- Keywords: `precision`, `MindSpore vs Torch_NPU`, `ACLNN`, `binary mismatch`
- Backend: `ascend`
- Precision difference between MindSpore and Torch_NPU on same hardware

**Root Cause Analysis:**
1. MindSpore ACLNN interface integration differs from Torch_NPU implementation
2. Parameter passing differences to ACLNN APIs
3. Data type conversion inconsistencies
4. Memory management or tensor layout differences

**Systematic Diagnosis Workflow:**

**Step 1: Build Controlled Comparison Environment**

Create a test environment with identical conditions for both frameworks:

```python
import torch
import torch_npu
import mindspore as ms
import numpy as np

ms.set_context(device_target='Ascend', device_id=0)

np.random.seed(42)
input_data = np.random.randn(2, 3, 4).astype(np.float32)
```

**Step 2: Ensure Input Consistency**

Both frameworks must receive identical input data:

| Aspect | Requirement | Verification Method |
|--------|-------------|---------------------|
| Input tensor values | Binary identical | Compare numpy arrays before framework conversion |
| Data type | Same dtype (float32/float16) | Check tensor.dtype |
| Shape | Same dimensions | Check tensor.shape |
| Device placement | Same NPU device | Check device_id |

**Step 3: Binary-Level Output Comparison**

Compare outputs at binary level, not just numerical tolerance:

```python
import struct

def to_binary_bytes(tensor):
    """Convert tensor to binary bytes for exact comparison."""
    if isinstance(tensor, torch.Tensor):
        return tensor.cpu().numpy().tobytes()
    elif isinstance(tensor, ms.Tensor):
        return tensor.asnumpy().tobytes()
    return None

def compare_binary(ms_output, torch_output):
    """Binary-level comparison of outputs."""
    ms_bytes = to_binary_bytes(ms_output)
    torch_bytes = to_binary_bytes(torch_output)
    
    if ms_bytes == torch_bytes:
        print("✓ Binary identical - ACLNN integration is correct")
        return True
    else:
        print("✗ Binary mismatch - ACLNN integration may differ")
        # Find first difference position
        for i, (b1, b2) in enumerate(zip(ms_bytes, torch_bytes)):
            if b1 != b2:
                print(f"  First difference at byte offset {i}")
                break
        return False
```

**Step 4: Operator-Level Isolation**

If network-level comparison fails, isolate to specific operators:

```python
def test_single_operator(op_name, input_shape, dtype=np.float32):
    """Test single operator comparison between MindSpore and Torch_NPU."""
    np.random.seed(42)
    input_data = np.random.randn(*input_shape).astype(dtype)
    
    # MindSpore execution
    ms_input = ms.Tensor(input_data)
    ms_output = getattr(ms.ops, op_name)(ms_input)
    
    # Torch_NPU execution
    torch_input = torch.from_numpy(input_data).npu()
    torch_output = getattr(torch, op_name)(torch_input)
    
    return compare_binary(ms_output, torch_output)
```

**Step 5: ACLNN Interface Verification**

When binary mismatch is detected, verify ACLNN interface calls:

| Check Point | MindSpore | Torch_NPU |
|-------------|-----------|-----------|
| ACLNN API name | Check operator registration | Check torch_npu implementation |
| Parameter order | Compare with CANN docs | Compare with PTA source |
| Parameter types | Verify type conversions | Verify type handling |
| Optional parameters | Check None handling | Check default values |
| Workspace size | Check memory allocation | Check workspace calculation |

**Diagnosis Decision Tree:**

```
Binary Comparison Result
    │
    ├─ Binary Identical
    │   └─ ACLNN integration is correct
    │       → Precision issue is from other sources (data, algorithm)
    │
    └─ Binary Mismatch
        │
        ├─ Single Operator Test
        │   ├─ Pass → Issue in network composition, not ACLNN
        │   └─ Fail → Issue in ACLNN integration
        │
        └─ ACLNN Integration Check
            ├─ Parameter mismatch → Fix parameter passing
            ├─ Type conversion issue → Align type handling
            └─ Memory/layout issue → Check tensor layout
```

**Solutions:**

| Solution | When to Apply | Implementation |
|----------|---------------|----------------|
| Fix ACLNN parameter passing | Parameter mismatch | Compare with Torch_NPU source code for exact parameter handling |
| Align type conversions | Type mismatch | Ensure same dtype conversions as Torch_NPU |
| Check tensor layout | Memory issue | Verify NCHW vs NHWC, contiguous vs strided |
| Verify optional param handling | None/default issue | Match Torch_NPU's handling of optional parameters |

**Reference Implementation Checklist:**

When investigating ACLNN integration differences, compare these aspects with Torch_NPU (PyTorch Ascend):

1. **ACLNN API Call Sequence:**
   - `aclnnXxxGetWorkspaceSize` parameters
   - `aclnnXxx` call parameters
   - Workspace allocation and management

2. **Parameter Conversion:**
   - Scalar to tensor conversion
   - Optional parameter handling (None vs default)
   - Array/list to vector conversion
   - String to enum mapping

3. **Tensor Handling:**
   - Contiguous vs non-contiguous tensors
   - Memory format (NCHW, NHWC, ND)
   - Device synchronization points

4. **Type System:**
   - Implicit type promotion rules
   - Precision mode settings
   - Mixed precision handling

**Case References:**
- as_strided bfloat16 Precision Issue
- mint.nn.functional.batch_norm Backward Dtype
- PTA-MindSpore Parameter Mismatch

---

## 8. Environment/Configuration Issues

### 8.1 CANN Environment Missing

**Error Patterns:**
```
libascendcl.so not found, ImportError, cannot find CANN, ASCEND_OPP_PATH
```

**Identification Features:**
- Keywords: `libascendcl.so`, `CANN`, `ASCEND_OPP_PATH`
- ImportError type
- Backend: `ascend`

**Root Cause Analysis:**
CANN environment variables not set or CANN not installed.

**Diagnosis Steps:**
1. Check if CANN is installed
2. Verify environment variables are set
3. Check CANN version compatibility

**Solutions:**
| Solution | When to Apply |
|----------|---------------|
| Source CANN environment | Environment not set | `source /usr/local/Ascend/ascend-toolkit/set_env.sh` |
| Install CANN | Not installed | Install compatible CANN version |
| Verify installation | Check installation | `cat /usr/local/Ascend/ascend-toolkit/version` |

**Case Reference:** Common Failure Patterns - Missing CANN Environment

---

### 8.2 Device Target Mismatch

**Error Patterns:**
```
RuntimeError, device_target, should be one of, Ascend GPU CPU, invalid device
```

**Identification Features:**
- Keywords: `device_target`, `invalid device`
- Backend: `all`

**Root Cause Analysis:**
`set_context` device_target does not match available hardware or is misspelled.

**Diagnosis Steps:**
1. Check available hardware
2. Verify device_target string spelling
3. Check device availability

**Solutions:**
| Solution | When to Apply |
|----------|---------------|
| Use correct device_target | Misspelling | Use 'Ascend', 'GPU', or 'CPU' |
| Check hardware availability | Device not found | Run `npu-smi info` or `nvidia-smi` |

**Case Reference:** Common Failure Patterns - Device Target Mismatch

---

### 8.3 Version Mismatch

**Error Patterns:**
```
version mismatch, ImportError, libmindspore_extension.so, symbol missing, compatibility
```

**Identification Features:**
- Keywords: `version mismatch`, `symbol missing`, `compatibility`
- Often after package upgrades
- ImportError or symbol lookup errors

**Root Cause Analysis:**
1. MindSpore and CANN version incompatibility
2. Custom operator built with different MindSpore version
3. Dependency package version conflict

**Diagnosis Steps:**
1. Check MindSpore version
2. Check CANN version
3. Verify dependency compatibility
4. Check custom operator build environment

**Solutions:**
| Solution | When to Apply |
|----------|---------------|
| Align versions | Version mismatch | Use compatible MindSpore and CANN versions |
| Rebuild custom operators | Custom op issue | Rebuild with current MindSpore version |
| Update dependencies | Dependency conflict | Update dependent packages |

**Case References:**
- vllm-mindspore Version Mismatch
- MindSpore-CANN Compatibility Symbol Missing
- custom_ops Header File Compatibility

---

### 8.4 Profiler Issues

**Error Patterns:**
```
profiler, core dump, CANN, collection
```

**Identification Features:**
- Keywords: `profiler`, `core dump`
- Occurs during profiling

**Root Cause Analysis:**
1. CANN version has profiler bug
2. Permission issues for data collection
3. Resource conflict during profiling

**Diagnosis Steps:**
1. Check CANN version for known profiler issues
2. Verify user permissions
3. Check for resource conflicts

**Solutions:**
| Solution | When to Apply |
|----------|---------------|
| Upgrade CANN | Known bug | Use CANN version with profiler fix |
| Check permissions | Permission issue | Run with appropriate permissions |
| Disable problematic collection | Specific feature issue | Skip disk/osrt collection for non-root |

**Case References:**
- Profiler Core Dump on CANN 8.3.RC1.B020
- profiler Collecting Profiler Data, Non-root User

---

## 9. API Usage Issues

### 9.1 mint API Return Type Confusion

**Error Patterns:**
```
TypeError, AttributeError, mint.equal, bool, Tensor expected, item
```

**Identification Features:**
- Keywords: `mint.equal`, `bool`, `Tensor expected`
- Return type mismatch

**Root Cause Analysis:**
`mint.equal()` returns Python bool (not Tensor), unlike `ops.equal()` which returns Tensor.

**Diagnosis Steps:**
1. Check if using mint.equal() expecting Tensor result
2. Compare with ops.equal() behavior

**Solutions:**
| Solution | When to Apply |
|----------|---------------|
| Use ops.equal() | Need Tensor result | Replace mint.equal() with ops.equal() |
| Handle bool return | Scalar comparison OK | Use mint.equal() for bool comparison |

**Case Reference:** Common Failure Patterns - mint.equal() Return Type Confusion

---

### 9.2 mint.item() on Multi-Element Tensor

**Error Patterns:**
```
RuntimeError, cannot be converted to Scalar, mint.item, elements
```

**Identification Features:**
- Keywords: `mint.item`, `cannot be converted to Scalar`
- Backend: `all`

**Root Cause Analysis:**
`mint.item()` requires single-element Tensor; calling on multi-element Tensor raises error.

**Diagnosis Steps:**
1. Check tensor size before calling item()
2. Verify tensor has exactly 1 element

**Solutions:**
| Solution | When to Apply |
|----------|---------------|
| Ensure single element | Before item() | Check tensor.size == 1 |
| Use indexing | Need specific element | Use tensor[0] or tensor.sum() first |

**Case Reference:** Common Failure Patterns - mint.item() on Multi-Element Tensor

---

### 9.3 mint.nn Parameter Validation

**Error Patterns:**
```
ValueError, TypeError, in_channels, out_channels, groups, Validator, mint.nn.Conv, mint.nn.BatchNorm
```

**Identification Features:**
- Keywords: `Validator`, `in_channels`, `out_channels`, `groups`
- mint.nn layer instantiation

**Root Cause Analysis:**
mint.nn layers use strict parameter validation; invalid parameters raise ValueError.

**Common Validation Rules:**
- `in_channels % groups == 0`
- `out_channels % groups == 0`
- Positive values for dimensions

**Diagnosis Steps:**
1. Check parameter constraints in mint.nn documentation
2. Verify parameter values meet constraints

**Solutions:**
| Solution | When to Apply |
|----------|---------------|
| Fix parameter values | Constraint violation | Ensure parameters meet constraints |
| Check groups divisibility | Conv layer issue | Verify channels divisible by groups |

**Case Reference:** Common Failure Patterns - mint.nn Layer Parameter Validation

---

### 9.4 mint Experimental API Changes

**Error Patterns:**
```
AttributeError, ImportError, module mint has no attribute, API removed, experimental
```

**Identification Features:**
- Keywords: `experimental`, `API removed`, `has no attribute`
- After MindSpore version upgrade

**Root Cause Analysis:**
Many mint APIs are marked 'experimental' and may be removed or renamed across versions.

**Diagnosis Steps:**
1. Check MindSpore release notes for API changes
2. Search for alternative APIs

**Solutions:**
| Solution | When to Apply |
|----------|---------------|
| Check release notes | After upgrade | Review API changes in new version |
| Use stable ops.* | Need stability | Replace mint.* with ops.* |
| Pin MindSpore version | Stability required | Use specific MindSpore version |

**Case Reference:** Common Failure Patterns - mint Experimental API Removed After Upgrade

---

## 10. ACLNN Adaptation Issues

> **ACLNN API Documentation Reference:** Third-party ACLNN API docs live under [aclnn_api_docs/](../../../docs/cann/aclnn_api_docs/).
>
> **When to read ACLNN docs:** Only when error info is still insufficient, parameter constraints are unclear, dtype support is unknown, or you need to verify ACLNN behavior vs MindSpore implementation.
>
> **How to use them:** Extract `aclnnXxx` from the stack or adaptation code, locate the matching file directly in `aclnn_api_docs/`, then read only the function prototype, parameter constraints, dtype support, shape/layout rules, and failure conditions. See [CANN API Reference](cann-api-reference.md).

### 10.1 gen_ops.py YAML Build Error

**Error Patterns:**
```
gen_ops.py, KeyError, YAML, keys structure, py_method missing, function_doc
```

**Identification Features:**
- Keywords: `gen_ops.py`, `YAML`, `py_method missing`, `function_doc`
- Backend: `ascend`

**Root Cause Analysis:**
ACLNN operator YAML definition has incorrect field hierarchy or missing entries.

**Diagnosis Steps:**
1. Compare YAML structure with working operator (e.g., add)
2. Check for missing fields: `op_def`, `api_def`, `function_doc`
3. Check for encoding issues (Chinese characters in English YAML)

**Solutions:**
| Solution | When to Apply |
|----------|---------------|
| Fix YAML structure | Structure error | Align with working operator YAML |
| Add missing fields | Missing entries | Add py_method, function_doc as needed |
| Fix encoding | Encoding issue | Remove non-ASCII characters |

**Case Reference:** Common Failure Patterns - gen_ops.py YAML Build Error

---

### 10.x ACLNN Backend Type Support Verification

**Error Patterns:**
```
TypeError, not supported, input type, PowTensorScalar, Complex64, ACLNN
```

**Identification Features:**
- Keywords: `not supported`, `input type`, ACLNN, type is not supported
- Backend: `ascend`
- Error occurs in InferType/CheckAndInferType function

**Root Cause Analysis:**
When MindSpore reports "type is not supported" for an operator, it could be either:
1. ACLNN backend doesn't support this type (need adaptation)
2. MindSpore framework type check is too strict (false positive)

**Diagnosis Steps:**
1. Check if the error occurs in MindSpore framework layer (e.g., `ops/infer/ops_func_impl/*.cc`)
2. **Verify ACLNN backend support**:
   - Find ACLNN interface documentation (relative to skill root, i.e., `<skill_dir>/../../docs/cann/`):
     - **Third-party API docs**: `../../docs/cann/aclnn_api_docs/`
     - Search directly by API name or API stem, for example `aclnnPowTensorScalar`
   - Check the "数据类型" (Data Type) column in the documentation table
   - For example: `aclnnPowTensorScalar&aclnnInplacePowTensorScalar.md` shows COMPLEX64/COMPLEX128 are supported
3. If ACLNN supports the type but MindSpore rejects it → Framework bug (type check too strict)
4. If ACLNN doesn't support the type → Need to implement ACLNN support or use alternative

**Common Locations:**
- Type check in InferType: `mindspore/ops/infer/ops_func_impl/{op_name}.cc`
- PyBoost customize: `mindspore/ops/kernel/ascend/aclnn/pyboost_impl/customize/{op_name}.cc`
- API YAML definition: `mindspore/ops/api_def/{op_name}.yaml`

**Solutions:**
| Solution | When to Apply | Implementation |
|----------|---------------|----------------|
| Fix framework type check | ACLNN supports type but MS rejects | Add type to valid_types in InferType |
| Add ACLNN support | ACLNN doesn't support type | Adapt operator for ACLNN backend |
| Use alternative API | Workaround needed | Use tensor^tensor form instead of tensor^scalar |

**Case Reference:** mint.pow Complex Type Not Supported (fixed by adding Complex64/Complex128 to valid_types)

---

### 10.2 GeneralInfer Dynamic Shape Crash

**Error Patterns:**
```
RuntimeError, InferShape, dynamic shape, unknown, kShapeDimAny, GetScalarValue, has_value
```

**Identification Features:**
- Keywords: `InferShape`, `dynamic shape`, `kShapeDimAny`, `GetScalarValue`
- Backend: `ascend`

**Root Cause Analysis:**
C++ GeneralInfer doesn't handle dynamic shape/rank fallback properly.

**Diagnosis Steps:**
1. Check if GetScalarValue is used without checking has_value()
2. Verify fallback for unknown dimensions exists

**Solutions:**
| Solution | When to Apply |
|----------|---------------|
| Check has_value() | Before GetScalarValue | Add has_value() check |
| Return kShapeDimAny | Unknown dims | Return unknown dimension marker |
| Use GetArrayValue | Per-element check | Check IsValueUnknown for each element |

**Case Reference:** Common Failure Patterns - GeneralInfer Dynamic Shape Crash

---

### 10.3 PyBoost Parameter Conversion Failure

**Error Patterns:**
```
RuntimeError, TypeError, PyBoost, tuple, vector, Optional, None, LAUNCH_ACLNN, parameter
```

**Identification Features:**
- Keywords: `PyBoost`, `tuple`, `vector`, `Optional`, `LAUNCH_ACLNN`
- Backend: `ascend`

**Root Cause Analysis:**
PyBoost customize fails to convert MindSpore parameters to ACLNN-expected types.

**Common Conversion Issues:**
- `tuple[int]` not converted to `std::vector<int64_t>`
- `Optional None` not handled
- String not converted to enum

**Diagnosis Steps:**
1. Compare with PTA source for exact conversion logic
2. Check parameter type expectations

**Solutions:**
| Solution | When to Apply |
|----------|---------------|
| Add tuple→vector conversion | Tuple parameter | Implement vector conversion |
| Handle None case | Optional parameter | Convert None to empty tensor or default |
| Add str→enum mapping | String parameter | Implement enum conversion |

**Case Reference:** Common Failure Patterns - PyBoost Parameter Conversion Failure

---

### 10.4 KBK Kernel Registration Error

**Error Patterns:**
```
RuntimeError, undeclared, undefined, MS_ACLNN_KERNEL_FACTORY_REG, namespace, KBK, Graph kernel
```

**Identification Features:**
- Keywords: `MS_ACLNN_KERNEL_FACTORY_REG`, `namespace`, `KBK`
- Backend: `ascend`

**Root Cause Analysis:**
KBK kernel has namespace mismatch or wrong registration macro.

**Diagnosis Steps:**
1. Check namespace alignment in header and .cc files
2. Verify MS_ACLNN_KERNEL_FACTORY_REG class name matches
3. Check if forward/backward are in separate files

**Solutions:**
| Solution | When to Apply |
|----------|---------------|
| Align namespaces | Namespace mismatch | Ensure consistent namespace |
| Fix registration macro | Wrong class name | Use correct class name in macro |
| Separate forward/backward | Registration conflict | Use separate files for each |

**Case Reference:** Common Failure Patterns - KBK Kernel Registration / Namespace Error

---

### 10.5 BPROP Input/Output Count Mismatch

**Error Patterns:**
```
RuntimeError, bprop, gradient, input count, output count, REG_BPROP_BUILDER, backward
```

**Identification Features:**
- Keywords: `bprop`, `input count`, `output count`, `REG_BPROP_BUILDER`
- Backend: `ascend`

**Root Cause Analysis:**
BPROP builder has wrong input/output count.

**BPROP Count Rules:**
- Backward inputs = forward inputs + out + dout
- Backward outputs = forward input count
- Each forward input gets one gradient output
- Non-differentiable inputs return `ib->OutZeros(x)`

**Diagnosis Steps:**
1. Count forward inputs
2. Verify backward input count = forward inputs + 2
3. Verify backward output count = forward input count

**Solutions:**
| Solution | When to Apply |
|----------|---------------|
| Fix input count | Input mismatch | Add out and dout to inputs |
| Fix output count | Output mismatch | Return gradient for each forward input |
| Use OutZeros | Non-differentiable | Return zeros for non-differentiable inputs |

**Case Reference:** Common Failure Patterns - BPROP Input/Output Count Mismatch

---

### 10.6 BPROP Dynamic Value in Graph Mode

**Error Patterns:**
```
RuntimeError, bprop, Conditional, ShapeCalc, dynamic, ValueAny, graph mode, backward compile
```

**Identification Features:**
- Keywords: `bprop`, `Conditional`, `ShapeCalc`, `ValueAny`, `graph mode`
- Backend: `ascend`

**Root Cause Analysis:**
BPROP builder uses C++ if/else on scalar values that may be unknown at compile time.

**Diagnosis Steps:**
1. Check if BuildValue()->ContainsValueAny() for scalar inputs
2. Identify C++ conditionals on scalar values

**Solutions:**
| Solution | When to Apply |
|----------|---------------|
| Use ib->Conditional() | Unknown scalars | Replace C++ if with Conditional |
| Use ShapeCalc | Dynamic shape deps | Use DEF_PURE_SHAPE_CALC + ib->ShapeCalc() |
| Check IsDynamicRank | Dynamic rank | Add dedicated dynamic path function |

**Case Reference:** Common Failure Patterns - BPROP Dynamic Value in Graph Mode

---

### 10.7 ACLNN Composite Op Missing Sub-Operator

**Error Patterns:**
```
RuntimeError, aclnn, composite, sub-operator, missing, LAUNCH_ACLNN, call chain
```

**Identification Features:**
- Keywords: `composite`, `sub-operator`, `call chain`
- Backend: `ascend`

**Root Cause Analysis:**
Composite operator fails because sub-operators are not adapted.

**Diagnosis Steps:**
1. Analyze PTA C++ implementation for all EXEC_NPU_CMD/LAUNCH_ACLNN calls
2. Check each sub-operator status in MindSpore

**Solutions:**
| Solution | When to Apply |
|----------|---------------|
| Implement missing sub-ops | Sub-op missing | Adapt all sub-operators first |
| Check YAML + Infer + PyBoost + KBK | Complete adaptation | Ensure all components present |

**Case Reference:** Common Failure Patterns - ACLNN Composite Op Missing Sub-Operator

---

### 10.8 View Op Strides Calculation Error

**Error Patterns:**
```
RuntimeError, view, strides, transpose, reshape, zero-copy, silent corruption, incorrect output
```

**Identification Features:**
- Keywords: `view`, `strides`, `zero-copy`, `silent corruption`
- Backend: `ascend`

**Root Cause Analysis:**
View operator's strides calculation produces incorrect output.

**Diagnosis Steps:**
1. Verify strides calculation against PyTorch TensorShape.cpp
2. Check YAML has correct `view: True` / `graph_view: True` flags
3. Verify REG_VIEW_STRIDES_CALC_FUN registration

**Solutions:**
| Solution | When to Apply |
|----------|---------------|
| Fix strides calculation | Calculation error | Align with PyTorch reference |
| Add view flags | Missing flags | Add view: True to YAML |
| Test non-contiguous inputs | Edge cases | Verify with various input layouts |

**Case Reference:** Common Failure Patterns - View Op Strides Calculation Error

---

### 10.9 PTA-MindSpore Parameter Mismatch

**Error Patterns:**
```
RuntimeError, ValueError, parameter mismatch, PTA, forward backward, different names, hidden parameter
```

**Identification Features:**
- Keywords: `parameter mismatch`, `PTA`, `hidden parameter`
- Backend: `ascend`

**Root Cause Analysis:**
MindSpore ACLNN adaptation has parameter alignment issues with PTA.

**Common Mismatches:**
- Forward/backward parameter name differences
- Hidden hardcoded parameters in backward
- Optional None handling divergence
- Output tensor count mismatch

**Diagnosis Steps:**
1. Audit PTA source: check op_plugin_functions.yaml for signatures
2. Check derivatives.yaml for backward registration
3. Review C++ implementation for hidden defaults

**Solutions:**
| Solution | When to Apply |
|----------|---------------|
| Align parameter names | Name mismatch | Match PTA naming |
| Handle hidden parameters | Hidden defaults | Add missing parameters |
| Align None handling | Optional divergence | Match PTA None behavior |

**Case Reference:** Common Failure Patterns - PTA-MindSpore Parameter Mismatch

---

## 11. Parallel/Distributed Training Issues

### 11.1 Pipeline Parallel Issues

**Error Patterns:**
```
pipeline, lazy_inline, ValueError, AddN, shape must be same
```

**Identification Features:**
- Keywords: `pipeline`, `lazy_inline`, `pp`
- Shape mismatch errors in pipeline

**Root Cause Analysis:**
1. Missing @lazy_inline tag
2. LazyInline position incorrect
3. require_grad=False weight handling issue

**Diagnosis Steps:**
1. Check if @lazy_inline is added to network
2. Verify lazy_inline tag position
3. Check weight require_grad settings

**Solutions:**
| Solution | When to Apply |
|----------|---------------|
| Add @lazy_inline | Missing tag | Add decorator to network |
| Fix lazy_inline position | Position error | Correct tag placement |
| Handle require_grad=False | Weight issue | Process non-trainable weights |

**Case References:**
- Pipeline Parallel LazyInline Required
- TP+DP+MOE_TP LazyInline Position Error
- Pipeline(stages=4) + HSDP Graph Cycle Error

---

### 11.2 Semi-Auto Parallel Issues

**Error Patterns:**
```
semi_auto_parallel, strategy, layout, format, AllGather, BN
```

**Identification Features:**
- Keywords: `semi_auto_parallel`, `strategy`, `layout`
- Format mismatch errors

**Root Cause Analysis:**
1. Parameter format mismatch between graphs
2. Strategy configuration error
3. Layout propagation issue

**Diagnosis Steps:**
1. Check parameter formats across graphs
2. Verify strategy configuration
3. Check layout propagation

**Solutions:**
| Solution | When to Apply |
|----------|---------------|
| Fix format propagation | Format mismatch | Don't pass special format for certain shapes |
| Correct strategy | Strategy error | Fix in_strategy configuration |
| Check layout consistency | Layout issue | Verify layout propagation |

**Case References:**
- BN gamma 5HD Format AllGather
- BatchNorm Semi-Auto Parallel Parameter Format Mismatch

---

### 11.3 Data Parallel Issues

**Error Patterns:**
```
data_parallel, mirror, gradient, feature value detection
```

**Identification Features:**
- Keywords: `data_parallel`, `mirror`, `gradient`
- Gradient synchronization issues

**Root Cause Analysis:**
1. Feature not supporting data parallel mode
2. Mirror operator not inserted
3. Gradient synchronization timing

**Diagnosis Steps:**
1. Check if feature supports data parallel mode
2. Verify mirror operator insertion
3. Check gradient synchronization

**Solutions:**
| Solution | When to Apply |
|----------|---------------|
| Add warning prompt | Unsupported mode | Warn about data parallel limitation |
| Use auto/semi-auto parallel | Need feature support | Switch parallel mode |

**Case Reference:** Feature Value Detection Not Supporting Data Parallel

---

### 11.4 Optimizer Parallel Issues

**Error Patterns:**
```
optimizer_parallel, precision, O2, O0, SEG
```

**Identification Features:**
- Keywords: `optimizer_parallel`, `precision`, `O2`, `O0`

**Root Cause Analysis:**
1. O2 optimization cases not maintained
2. 910A doesn't support O0
3. Precision differences between optimization levels

**Diagnosis Steps:**
1. Check optimization level (O0/O1/O2)
2. Verify device support for optimization level
3. Compare precision across levels

**Solutions:**
| Solution | When to Apply |
|----------|---------------|
| Switch to O0 | O2 not maintained | Use O0 optimization level |
| Use 910B | 910A limitation | Use device that supports O0 |

**Case References:**
- test_optimizer_parallel_semi_auto_matmul Loss Difference
- test_semi_auto_parallel_avgpool3d Precision Issue

---

### 11.5 DryRun Issues

**Error Patterns:**
```
dryrun, mock acl, IsCompileSimulation, KBK, strategy
```

**Identification Features:**
- Keywords: `dryrun`, `mock acl`, `IsCompileSimulation`

**Root Cause Analysis:**
1. Mock ACL interface behavior incorrect
2. KBK mode check missing
3. Resource allocation issue

**Diagnosis Steps:**
1. Check if IsCompileSimulation checks KBK mode
2. Verify mock ACL behavior
3. Check resource allocation

**Solutions:**
| Solution | When to Apply |
|----------|---------------|
| Use UseSimulationApi() | Mode check issue | Replace IsCompileSimulation with UseSimulationApi |
| Fix resource allocation | Memory issue | Correct null pointer handling |

**Case References:**
- KBK IsCompileSimulation Mock ACL
- IsCompileSimulation KBK Mode Mock ACL Issue

---

## 12. Quick Reference: Error Code Mapping

### 12.1 Ascend Error Codes

| Error Code | Category | Description | Section |
|------------|----------|-------------|---------|
| EL0004 | Memory | HBM out of memory | 2.1 |
| 107002 | Context | Context is empty | 6.2 |
| 507010 | Hardware | Device heartbeat lost | 3.1 |
| 507014 | Hardware | AI Core execution timeout | 3.3 |
| 507033 | Device | aclrtSetDevice repeated | 8.2 |
| 507054 | Hardware | HBM ECC error | 3.2 |
| EI0002 | Distributed | HCCL communication error | 4.1 |
| EI0006 | Distributed | HCCL timeout | 4.1 |
| EZ1001 | Operator | ACLNN parameter error | 5.3 |
| EZ9903 | Operator | Missing L2_DFX_PHASE1 macro | 5.3 |
| EZ9999 | Operator | ACLNN execution error | 5.3 |
| E9xxxx | Operator | TBE compilation error | 5.2 |
| EBxxxx | Operator | TBE compilation error | 5.2 |

### 12.2 Common Exception Types

| Exception Type | Common Causes | Section |
|----------------|---------------|---------|
| RuntimeError | Operator execution, memory, device issues | Multiple |
| ValueError | Parameter validation, shape mismatch | 5, 9 |
| TypeError | Type mismatch, wrong parameter type | 5, 9 |
| ImportError | Missing library, environment issue | 8 |
| AttributeError | Missing attribute, API change | 9 |
| IndexError | Index out of bounds, tensor indexing | 6 |

### 12.3 Keyword Quick Reference

| Keyword | Problem Domain | Section |
|---------|----------------|---------|
| out of memory, OOM, exhausted | Memory | 2 |
| heartbeat, ECC, device hang | Hardware | 3 |
| HCCL, NCCL, distributed, timeout | Communication | 4 |
| not supported, compile failed, TBE | Operator | 5 |
| graph compile, type inference, construct | Graph | 6 |
| precision, loss, float16, seed | Precision | 7 |
| CANN, environment, version | Environment | 8 |
| mint., mint.nn, jit_view | API | 9 |
| aclnn, PyBoost, KBK, BPROP | ACLNN Adaptation | 10 |
| pipeline, semi_auto, data_parallel | Parallel | 11 |

---

## Appendix: Case Index

### Memory Issues
- Out of Memory (Ascend HBM) - Common Failure Patterns
- Out of Memory (GPU VRAM) - Common Failure Patterns
- PadV3 Dynamic Shape Memory Corruption
- mint.max Empty Tensor Dirty Memory
- View Op Strides Calculation Error

### Hardware Issues
- Device Heartbeat Lost - Common Failure Patterns
- HBM ECC Error - Common Failure Patterns
- AI Core Execution Timeout - Common Failure Patterns
- Profiler Core Dump on CANN 8.3.RC1.B020

### Distributed Communication
- HCCL Communication Error - Common Failure Patterns
- NCCL Communication Error - Common Failure Patterns
- mint.distributed init_process_group Failure - Common Failure Patterns
- HCCL CCOOL RootInfo Cleanup
- HCCL destroy_process_group Async Send/Recv

### Operator Issues
- Operator Not Supported on Backend - Common Failure Patterns
- TBE Operator Compilation Error - Common Failure Patterns
- aclnnAll/aclnnReduceSum Duplicate Dims in Axis
- mint.nn.functional.max_pool2d L2_DFX_PHASE1 Missing
- EZ1001 GroupedMatmul Internal Input Count

### Graph Compilation
- Graph Compilation Error (Static Graph) - Common Failure Patterns
- Context Empty (Ascend) - Common Failure Patterns
- mint View Op in GRAPH_MODE / JIT - Common Failure Patterns

### Precision Issues
- mint.cumsum Float16 Precision Standard
- ops.logcumsumexp Float16 Precision
- CPU floor_mod fp16 Precision With Fixed Seed
- nn.tanh Ascend Intermittent Precision Issue
- mint.nn.functional.hardswish Intermittent Precision Issue
- MindSpore vs Torch_NPU Binary Comparison (Section 7.4)

### Environment Issues
- Missing CANN Environment - Common Failure Patterns
- Device Target Mismatch - Common Failure Patterns
- vllm-mindspore Version Mismatch
- MindSpore-CANN Compatibility Symbol Missing

### API Usage
- mint.equal() Return Type Confusion - Common Failure Patterns
- mint.item() on Multi-Element Tensor - Common Failure Patterns
- mint.nn Layer Parameter Validation - Common Failure Patterns
- mint Experimental API Removed After Upgrade - Common Failure Patterns

### ACLNN Adaptation
- gen_ops.py YAML Build Error - Common Failure Patterns
- GeneralInfer Dynamic Shape Crash - Common Failure Patterns
- PyBoost Parameter Conversion Failure - Common Failure Patterns
- KBK Kernel Registration / Namespace Error - Common Failure Patterns
- BPROP Input/Output Count Mismatch - Common Failure Patterns
- BPROP Dynamic Value in Graph Mode - Common Failure Patterns
- ACLNN Composite Op Missing Sub-Operator - Common Failure Patterns
- View Op Strides Calculation Error - Common Failure Patterns
- PTA-MindSpore Parameter Mismatch - Common Failure Patterns

### Parallel Training
- Pipeline Parallel LazyInline Required
- BN gamma 5HD Format AllGather
- Feature Value Detection Not Supporting Data Parallel
- test_optimizer_parallel_semi_auto_matmul Loss Difference
- KBK IsCompileSimulation Mock ACL

---

## Document Information

- **Version:** 1.0
- **Last Updated:** 2026-03-30
- **Source:** failure-showcase.md historical cases
- **Purpose:** Standardized reference for LLM-based MindSpore failure diagnosis

---

## Document Update Mechanism

This document must be kept synchronized with the evolving MindSpore ecosystem. The following mechanisms ensure the diagnosis guidance remains accurate and effective.

### Update Triggers

| Trigger | Condition | Action Required |
|---------|-----------|-----------------|
| **New Failure Pattern** | A verified solution is added to `failure-showcase.md` that introduces a new problem domain or diagnosis approach | Add corresponding section or update existing section |
| **Error Code Change** | New CANN/MindSpore error codes introduced, or existing codes deprecated | Update Section 12: Error Code Mapping |
| **API Deprecation/Addition** | MindSpore API changes (mint.*, ops.*, nn.*) that affect diagnosis patterns | Update Section 9: API Usage Issues |
| **Version Compatibility** | New MindSpore/CANN/CUDA version released with different behavior | Update Section 8: Environment/Configuration Issues |
| **Solution Obsolescence** | A previously documented solution no longer works due to framework changes | Mark as deprecated, provide updated solution |
| **Pattern Consolidation** | Multiple similar patterns can be generalized into a unified diagnosis approach | Refactor and consolidate sections |

### Update Workflow

```
New Verified Failure Case
         │
         ▼
┌─────────────────────────────────────┐
│  Stage 1: Check failure-showcase.md │
│  - Is this a new pattern?           │
│  - Does it match existing entry?    │
└─────────────────────────────────────┘
         │
         ├─ Matches existing → Update occurrence count in failure-showcase.md
         │                        │
         │                        ▼
         │                  Check if diagnosis-guide.md needs update:
         │                  - Does the solution differ from documented approach?
         │                  - Are there new diagnosis steps to add?
         │                  - Is the error code mapping complete?
         │
         └─ New pattern → Add to failure-showcase.md
                              │
                              ▼
                        Evaluate diagnosis-guide.md impact:
                        ┌──────────────────────────────────────┐
                        │ 1. Which section does this belong to? │
                        │ 2. Is this a new error code?          │
                        │ 3. Does it require new diagnosis step?│
                        │ 4. Should it be added to case index?  │
                        └──────────────────────────────────────┘
                              │
                              ▼
                        Update diagnosis-guide.md:
                        - Add/update error pattern
                        - Add/update diagnosis steps
                        - Add/update solutions table
                        - Update case reference
                        - Update error code mapping (if applicable)
                        - Increment version number
                        - Update last_seen date
```

### Section-Specific Update Guidelines

**Section 2-3 (Memory/Hardware):**
- Add new error codes from CANN releases
- Update device-specific behaviors (910A vs 910B differences)
- Document new memory management features

**Section 4 (Distributed Communication):**
- Track HCCL/NCCL version changes
- Document new distributed training patterns
- Update timeout and configuration recommendations

**Section 5 (Operator Issues):**
- Add new ACLNN error codes (EZxxxx series)
- Document operator deprecation/migration paths
- Update TBE compilation error patterns

**Section 6 (Graph Compilation):**
- Track GRAPH_MODE behavior changes
- Document new static graph limitations
- Update type inference error patterns

**Section 7 (Precision Issues):**
- Add new precision comparison methods
- Document dtype-specific behaviors
- Update benchmark reference standards

**Section 8 (Environment/Configuration):**
- Track version compatibility matrix changes
- Document new environment variables
- Update profiler and debugging tool changes

**Section 9 (API Usage Issues):**
- Track mint API changes (experimental → stable)
- Document API deprecation notices
- Add new API-specific pitfalls

**Section 10 (ACLNN Adaptation):**
- Document new ACLNN adaptation patterns
- Update PyBoost/KBK/BPROP guidelines
- Add new parameter conversion rules

**Section 11 (Parallel Training):**
- Track parallel strategy changes
- Document new parallel modes
- Update optimizer parallel behaviors

**Section 12 (Error Code Mapping):**
- Add new error codes immediately when discovered
- Mark deprecated codes
- Cross-reference with CANN release notes

### Version Increment Rules

| Change Type | Version Increment | Example |
|-------------|-------------------|---------|
| Major restructure or new section | Major version | 1.0 → 2.0 |
| New error pattern or diagnosis step | Minor version | 1.0 → 1.1 |
| Case reference update or typo fix | Patch version | 1.0 → 1.0.1 |

### Synchronization with failure-showcase.md

The `diagnosis-guide.md` must maintain consistency with `failure-showcase.md`:

| failure-showcase.md Change | diagnosis-guide.md Action |
|----------------------------|---------------------------|
| New failure type added | Evaluate if new section needed |
| New error code documented | Update Section 12 mapping |
| Solution updated | Update corresponding solution table |
| Pattern consolidated | Refactor related sections |
| Entry marked deprecated | Mark corresponding guidance as deprecated |

### Quality Assurance Checklist

Before committing updates to diagnosis-guide.md, verify:

- [ ] All error codes in new entries are mapped in Section 12
- [ ] Case references link to existing failure-showcase.md entries
- [ ] Solution tables are actionable and specific
- [ ] Diagnosis steps are ordered and testable
- [ ] Version number is incremented appropriately
- [ ] Last Updated date reflects current change
- [ ] No duplicate patterns across sections
- [ ] Cross-references between sections are valid

### Automated Sync Points

The following events should trigger a review of diagnosis-guide.md:

1. **After Stage 3 completion** in ms-failure-analyze skill workflow
2. **Monthly review** of failure-showcase.md for pattern consolidation
3. **MindSpore major release** for API and behavior changes
4. **CANN major release** for error code and operator changes
5. **User feedback** indicating outdated or incorrect guidance
