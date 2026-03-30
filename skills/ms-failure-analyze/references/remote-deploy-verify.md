# Remote Deploy & Verify Reference (Simplified)

This document describes the simplified remote deployment verification workflow for MindSpore development. All operations produce structured JSON output for LLM consumption.

> **Path resolution**: All `tools/` paths are relative to the skill directory. Resolve to absolute paths before execution.

## Core Features (6 Steps)

1. **SSH Login** - Verify SSH key-based authentication
2. **Sync Files** - Distinguish mindspore code vs external files
3. **Environment Init** - Source env.sh (proxy, CANN, conda)
4. **Build MindSpore** - Compile based on backend type
5. **Install Product** - pip install or PYTHONPATH mode
6. **Run Tests** - Execute verification commands

## Workflow

```
SSH → Sync → Env → Build → Install → Run → Rollback (optional)
```

## Configuration File (deploy_context.json)

```json
{
  "remote": {
    "host": "10.0.1.100",
    "user": "dev",
    "port": 22
  },
  "paths": {
    "local_root": "/home/user/mindspore",
    "remote_root": "/home/dev/mindspore"
  },
  "env_script": "/home/dev/env.sh",
  "build": {
    "platform": "ascend",
    "nproc": 96
  },
  "install": {
    "method": "pip"
  },
  "timeouts": {
    "build": 3600,
    "install": 300,
    "run": 600
  }
}
```

### Configuration Fields

| Field | Required | Description |
|-------|----------|-------------|
| `remote.host` | Yes | Remote server hostname or IP |
| `remote.user` | Yes | SSH username |
| `remote.port` | No | SSH port (default: 22) |
| `paths.local_root` | Yes | Local mindspore repository path |
| `paths.remote_root` | Yes | Remote mindspore repository path |
| `env_script` | Yes | Path to env.sh on remote server |
| `build.platform` | Yes | Build backend: `ascend`, `gpu`, or `cpu` |
| `build.nproc` | No | Parallel build jobs (default: 96) |
| `install.method` | No | `pip` (default) or `pythonpath` |
| `timeouts.*` | No | Timeout in seconds for each operation |

### env.sh Requirements

The env.sh script should contain:

```bash
#!/bin/bash
# Proxy settings (optional)
export http_proxy="http://proxy.example.com:8080"
export https_proxy="http://proxy.example.com:8080"

# CANN environment
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Conda environment
conda activate ms_dev
```

## Commands

### Step 1: Check SSH Connection

```bash
python tools/remote_deploy_verify.py check-ssh --config deploy_context.json
```

**Output:**
```json
{
  "success": true,
  "message": "SSH connection successful",
  "host": "10.0.1.100",
  "user": "dev"
}
```

**On failure:** Run `ssh-copy-id user@host` to set up key-based authentication.

### Step 2: Sync Files

```bash
python tools/remote_deploy_verify.py sync --config deploy_context.json --files file1.py file2.cpp
```

**File Classification:**
- **mindspore files**: Under `local_root` → synced to `remote_root`, can be rolled back
- **external files**: Outside `local_root` → synced to temp directory, deleted after task

**Output:**
```json
{
  "success": true,
  "files_synced": [
    {"file": "mindspore/ops/operations/array_ops.py", "type": "mindspore", "success": true}
  ],
  "summary": {
    "mindspore_files": 1,
    "external_files": 0,
    "is_git_repo": true
  }
}
```

### Step 3: Verify Environment

```bash
python tools/remote_deploy_verify.py verify-env --config deploy_context.json
```

**Checks:**
- Proxy: GitHub accessibility
- CANN: `acl.get_soc_name()` returns chip type
- Conda: Active environment name
- Python: Version check

**Output:**
```json
{
  "success": true,
  "checks": {
    "proxy": {"success": true, "message": "GitHub accessible"},
    "cann": {"success": true, "message": "Chip: Ascend910B1"},
    "conda": {"success": true, "message": "Environment: ms_dev"},
    "python": {"success": true, "message": "Python 3.9.18"}
  }
}
```

### Step 4: Build MindSpore

```bash
python tools/remote_deploy_verify.py build --config deploy_context.json
```

**Build Commands by Platform:**

| Platform | Command | Includes |
|----------|---------|----------|
| `ascend` | `bash build.sh -e ascend -j96` | Ascend + CPU |
| `gpu` | `bash build.sh -e gpu -j96` | GPU + CPU |
| `cpu` | `bash build.sh -e cpu -j96` | CPU only |

**Important:** Ascend and GPU cannot be built together. Choose based on the problem's backend type.

**Output:**
```json
{
  "success": true,
  "platform": "ascend",
  "nproc": 96,
  "elapsed_seconds": 1234.5,
  "log_file": "/path/to/deploy_logs/build_20260331_120000.log"
}
```

### Step 5: Install MindSpore

```bash
python tools/remote_deploy_verify.py install --config deploy_context.json
```

**Installation Methods:**

| Method | Description |
|--------|-------------|
| `pip` | Install wheel package (recommended) |
| `pythonpath` | Set PYTHONPATH environment variable |

**pip mode output:**
```json
{
  "success": true,
  "method": "pip",
  "wheel_path": "/home/dev/mindspore/build/package/mindspore-2.5.0-cp39-cp39-linux_aarch64.whl",
  "elapsed_seconds": 43.1
}
```

**pythonpath mode output:**
```json
{
  "success": true,
  "method": "pythonpath",
  "message": "PYTHONPATH mode does not require installation"
}
```

### Step 6: Run Tests

```bash
python tools/remote_deploy_verify.py run --config deploy_context.json --command "pytest tests/ut/ops/test_xxx.py -xvs"
```

**Output:**
```json
{
  "success": true,
  "command": "pytest tests/ut/ops/test_xxx.py -xvs",
  "exit_code": 0,
  "verdict": "pass",
  "pytest_summary": {"passed": 5, "failed": 0, "total": 5},
  "output_tail": ["..."],
  "log_file": "/path/to/deploy_logs/run_20260331_120000.log"
}
```

### Rollback

```bash
python tools/remote_deploy_verify.py rollback --config deploy_context.json
```

**Rollback Actions:**
- Git repo: `git checkout -- .` and `git stash pop`
- Non-git: Restore original file contents
- External files: Delete temp directory

### One-Click Deploy

```bash
python tools/remote_deploy_verify.py deploy --config deploy_context.json --files file1.py --command "pytest test.py" --auto-rollback
```

Executes all 6 steps in sequence. Use `--auto-rollback` to automatically rollback after successful verification.

## MindSpore-Specific Notes

### Build Backend Selection

| Backend | When to Use |
|---------|-------------|
| `ascend` | Error occurs on Ascend NPU |
| `gpu` | Error occurs on GPU |
| `cpu` | CPU-only issues |

### Wheel Package Location

MindSpore builds produce wheels in `build/package/`, not `dist/`:
- **Correct**: `build/package/mindspore-*.whl`
- **Incorrect**: `dist/mindspore-*.whl`

### Environment Variables for Tests

| Variable | Description | Values |
|----------|-------------|--------|
| `CONTEXT_DEVICE_TARGET` | Target device | `Ascend` / `GPU` / `CPU` |
| `CONTEXT_MODE` | Execution mode | `0` (GRAPH_MODE) / `1` (PYNATIVE_MODE) |
| `GLOG_v` | Log level | `0` (DEBUG) / `1` (INFO) / `2` (WARNING) |

## Troubleshooting

| Problem | Solution |
|---------|----------|
| SSH connection failed | Run `ssh-copy-id user@host` for key-based auth |
| GitHub not accessible | Check proxy settings in env.sh |
| CANN not working | Verify `source set_env.sh` in env.sh |
| Build fails | Check build log for errors |
| Wrong wheel location | Use `build/package/` not `dist/` |
| Backend mismatch | Remember: Ascend and GPU cannot be built together |
