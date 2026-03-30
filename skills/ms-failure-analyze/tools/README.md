# MindSpore Remote Deploy & Verify Tools

This directory contains utility scripts for the `ms-failure-analyze` skill.

## remote_deploy_verify.py

Remote deployment and verification tool for MindSpore development. Syncs local code changes to a remote server, builds, runs tests, and reports results — all via structured JSON output for LLM consumption.

### Usage

```bash
# Collect local git changes
python tools/remote_deploy_verify.py collect --local-root /path/to/mindspore

# Run pre-flight checks
python tools/remote_deploy_verify.py preflight --config deploy_context.json

# Sync code to remote
python tools/remote_deploy_verify.py sync --config deploy_context.json [--dry-run]

# Build on remote
python tools/remote_deploy_verify.py build --config deploy_context.json [--mode MODE]

# Install built wheel
python tools/remote_deploy_verify.py install --config deploy_context.json

# Run verification
python tools/remote_deploy_verify.py run --config deploy_context.json

# Quick sync + run
python tools/remote_deploy_verify.py sync-run --config deploy_context.json

# Execute arbitrary command
python tools/remote_deploy_verify.py exec --config deploy_context.json "ls -la"

# Generate report
python tools/remote_deploy_verify.py report --config deploy_context.json

# Rollback changes
python tools/remote_deploy_verify.py rollback --config deploy_context.json
```

### Key Features

- **MindSpore-specific module classification**: Automatically categorizes changed files based on MindSpore project structure
- **Multi-backend support**: Handles Ascend, GPU, and CPU backends (note: Ascend and GPU cannot be built together)
- **Docker build mode**: Supports building in Docker containers for reproducible environments
- **Structured JSON output**: All commands produce machine-readable JSON for LLM consumption
- **Automatic rollback**: Can automatically revert remote changes after successful verification

### MindSpore-Specific Notes

1. **Build backend selection**: Choose `build.backend` based on the problem's backend type
   - `ascend`: Builds Ascend + CPU backends
   - `gpu`: Builds GPU + CPU backends
   - `cpu`: Builds CPU only

2. **Wheel location**: MindSpore builds produce wheels in `build/package/`, not `dist/`

3. **Environment variables**: MindSporeTest requires specific environment variables:
   - `CONTEXT_DEVICE_TARGET`: Target device (Ascend/GPU/CPU)
   - `CONTEXT_MODE`: Execution mode (0=GRAPH_MODE, 1=PYNATIVE_MODE)
   - `GLOG_v`: Log level

### Configuration

Create a `deploy_context.json` file:

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
  "exec_context": {
    "type": "bare",
    "env_setup": "source /home/dev/env.sh && conda activate ms_dev"
  },
  "build": {
    "python_version": "3.9",
    "backend": "ascend"
  },
  "verify": {
    "command": "pytest tests/ut/ops/test_xxx.py -xvs",
    "timeout": 600,
    "env_vars": {
      "CONTEXT_DEVICE_TARGET": "Ascend",
      "CONTEXT_MODE": "0"
    }
  }
}
```

### Documentation

See [remote-deploy-verify.md](../references/remote-deploy-verify.md) for detailed documentation.
