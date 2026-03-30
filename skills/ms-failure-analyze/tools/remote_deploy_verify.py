#!/usr/bin/env python3
"""Remote deploy & verify tool for MindSpore development.

This tool provides a simple and reliable way to sync local code changes to a remote server,
compile MindSpore, and run verification tests. All operations produce structured JSON output
for LLM consumption.

Core Features (6 Steps):
1. SSH connection verification
2. File synchronization (distinguish mindspore code vs external files)
3. Environment initialization (source env.sh)
4. MindSpore compilation (optional)
5. Product installation (pip or PYTHONPATH)
6. Test execution

Usage:
    python tools/remote_deploy_verify.py check-ssh --config deploy_context.json
    python tools/remote_deploy_verify.py sync --config deploy_context.json --files file1.py file2.cpp
    python tools/remote_deploy_verify.py verify-env --config deploy_context.json
    python tools/remote_deploy_verify.py build --config deploy_context.json
    python tools/remote_deploy_verify.py install --config deploy_context.json
    python tools/remote_deploy_verify.py run --config deploy_context.json --command "pytest test.py"
    python tools/remote_deploy_verify.py rollback --config deploy_context.json
    python tools/remote_deploy_verify.py deploy --config deploy_context.json --files file1.py --command "pytest test.py"
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
SKILL_DIR = TOOLS_DIR.parent
DEFAULT_LOG_DIR = SKILL_DIR / "deploy_logs"

DEFAULT_TIMEOUTS = {
    "ssh": 15,
    "build": 3600,
    "install": 300,
    "run": 600,
}


def _json_output(data: dict):
    """Output JSON result to stdout."""
    print(json.dumps(data, ensure_ascii=False, indent=2))


def _load_config(path: str) -> dict:
    """Load configuration from JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: str, data: dict):
    """Save data to JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _timestamp() -> str:
    """Get current timestamp string."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _ensure_log_dir(config: dict) -> Path:
    """Ensure log directory exists."""
    log_dir = Path(config.get("log_dir", str(DEFAULT_LOG_DIR)))
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def _get_timeout(config: dict, operation: str) -> int:
    """Get timeout for specific operation."""
    return config.get("timeouts", {}).get(operation, DEFAULT_TIMEOUTS.get(operation, 60))


def _get_ssh_base(config: dict) -> list:
    """Get base SSH command with connection parameters."""
    remote = config["remote"]
    cmd = ["ssh"]
    port = remote.get("port", 22)
    if port != 22:
        cmd += ["-p", str(port)]
    cmd += ["-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10"]
    cmd.append(f"{remote['user']}@{remote['host']}")
    return cmd


def _build_remote_command(config: dict, user_cmd: str) -> str:
    """Build remote command with environment setup prefix.
    
    Each remote command automatically includes:
    1. source env.sh - Set up proxy, CANN, conda environment
    2. PYTHONPATH setup (if using pythonpath install method)
    3. cd to remote_root
    """
    env_script = config["env_script"]
    remote_root = config["paths"]["remote_root"]
    install_method = config.get("install", {}).get("method", "pip")
    
    prefix_parts = [f"source {env_script}"]
    
    if install_method == "pythonpath":
        prefix_parts.extend([
            f"export MINDSPORE_PACKPATH={remote_root}",
            "export PYTHONPATH=$MINDSPORE_PACKPATH:$PYTHONPATH",
            "export PYTHONPATH=$MINDSPORE_PACKPATH/build/package:$PYTHONPATH",
            "export PYTHONPATH=$MINDSPORE_PACKPATH/build/package/build/lib:$PYTHONPATH",
        ])
    
    prefix_parts.append(f"cd {remote_root}")
    prefix = " && ".join(prefix_parts)
    return f"{prefix} && {user_cmd}"


def _run_ssh(config: dict, remote_cmd: str, timeout: int = 600) -> dict:
    """Execute command on remote server via SSH."""
    ssh_base = _get_ssh_base(config)
    full_cmd = ssh_base + [remote_cmd]
    
    start = time.time()
    try:
        proc = subprocess.run(
            full_cmd,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        elapsed = time.time() - start
        return {
            "exit_code": proc.returncode,
            "stdout": proc.stdout or "",
            "stderr": proc.stderr or "",
            "elapsed_seconds": round(elapsed, 1),
            "timed_out": False,
        }
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        return {
            "exit_code": -1,
            "stdout": "",
            "stderr": f"Command timed out after {timeout}s",
            "elapsed_seconds": round(elapsed, 1),
            "timed_out": True,
        }
    except Exception as e:
        elapsed = time.time() - start
        return {
            "exit_code": -1,
            "stdout": "",
            "stderr": str(e),
            "elapsed_seconds": round(elapsed, 1),
            "timed_out": False,
        }


def _run_scp(config: dict, local_path: str, remote_path: str) -> dict:
    """Copy file to remote server via SCP."""
    remote = config["remote"]
    port = remote.get("port", 22)
    
    cmd = ["scp"]
    if port != 22:
        cmd += ["-P", str(port)]
    cmd += [local_path, f"{remote['user']}@{remote['host']}:{remote_path}"]
    
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return {
            "success": proc.returncode == 0,
            "error": proc.stderr[:200] if proc.returncode != 0 else "",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def _parse_pytest_output(output: str) -> dict:
    """Parse pytest output to extract test summary."""
    import re
    summary_re = r"=+\s*(.*?)\s*in\s*[\d.]+s\s*=+"
    m = re.search(summary_re, output)
    if not m:
        return {"passed": 0, "failed": 0, "error": 0, "total": 0, "parse_failed": True}
    
    summary_text = m.group(1)
    item_re = re.compile(
        r"(\d+)\s+(passed|failed|errors?|warnings?|deselected|skipped|xfailed|xpassed)"
    )
    result = {"passed": 0, "failed": 0, "error": 0}
    
    for im in item_re.finditer(summary_text):
        count = int(im.group(1))
        key = im.group(2)
        if key in ("error", "errors"):
            result["error"] = count
        elif key.startswith("warning"):
            result["warnings"] = count
        else:
            result[key] = count
    
    result["total"] = result.get("passed", 0) + result.get("failed", 0) + result.get("error", 0)
    return result


def cmd_check_ssh(args):
    """Step 1: Check SSH connection to remote server."""
    config = _load_config(args.config)
    remote = config["remote"]
    
    result = _run_ssh(config, "echo OK", timeout=15)
    
    if result["exit_code"] == 0 and "OK" in result["stdout"]:
        _json_output({
            "success": True,
            "message": "SSH connection successful",
            "host": remote["host"],
            "user": remote["user"],
            "elapsed_seconds": result["elapsed_seconds"],
        })
    else:
        _json_output({
            "success": False,
            "message": "SSH connection failed",
            "host": remote["host"],
            "user": remote["user"],
            "error": result["stderr"][:300] if result["stderr"] else "Unknown error",
            "suggestion": f"Please run: ssh-copy-id {remote['user']}@{remote['host']}",
        })


def cmd_verify_env(args):
    """Step 3: Verify remote environment (proxy, CANN, conda)."""
    config = _load_config(args.config)
    env_script = config["env_script"]
    remote_root = config["paths"]["remote_root"]
    
    checks = {}
    
    proxy_cmd = _build_remote_command(
        config,
        "curl -s -o /dev/null -w '%{http_code}' https://github.com --connect-timeout 5"
    )
    result = _run_ssh(config, proxy_cmd, timeout=15)
    checks["proxy"] = {
        "success": result["stdout"].strip() in ("200", "301", "302"),
        "message": "GitHub accessible" if result["stdout"].strip() in ("200", "301", "302") 
                   else "GitHub not accessible, check proxy settings",
    }
    
    cann_cmd = _build_remote_command(
        config,
        "python3 -c \"import acl; print(acl.get_soc_name())\" 2>&1 || echo CANN_ERROR"
    )
    result = _run_ssh(config, cann_cmd, timeout=15)
    checks["cann"] = {
        "success": "CANN_ERROR" not in result["stdout"] and result["exit_code"] == 0,
        "message": f"Chip: {result['stdout'].strip()}" if "CANN_ERROR" not in result["stdout"] 
                   else "CANN environment not properly set",
    }
    
    conda_cmd = _build_remote_command(config, "echo $CONDA_DEFAULT_ENV")
    result = _run_ssh(config, conda_cmd, timeout=10)
    checks["conda"] = {
        "success": bool(result["stdout"].strip()),
        "message": f"Environment: {result['stdout'].strip()}" if result["stdout"].strip() 
                   else "No conda environment activated",
    }
    
    python_cmd = _build_remote_command(config, "python3 --version")
    result = _run_ssh(config, python_cmd, timeout=10)
    checks["python"] = {
        "success": result["exit_code"] == 0,
        "message": result["stdout"].strip(),
    }
    
    all_success = all(c["success"] for c in checks.values())
    
    _json_output({
        "success": all_success,
        "checks": checks,
        "env_script": env_script,
        "remote_root": remote_root,
    })


def cmd_sync(args):
    """Step 2: Sync files to remote server."""
    config = _load_config(args.config)
    log_dir = _ensure_log_dir(config)
    local_root = Path(config["paths"]["local_root"]).resolve()
    remote_root = config["paths"]["remote_root"]
    
    files = args.files or []
    if not files:
        _json_output({"success": True, "message": "No files to sync", "files_synced": []})
        return
    
    mindspore_files = []
    external_files = []
    
    for f in files:
        abs_path = Path(f).resolve()
        try:
            rel_path = abs_path.relative_to(local_root)
            mindspore_files.append(str(rel_path).replace("\\", "/"))
        except ValueError:
            external_files.append(str(abs_path).replace("\\", "/"))
    
    sync_state = {
        "timestamp": _timestamp(),
        "mindspore_files": mindspore_files,
        "external_files": external_files,
        "is_git_repo": False,
        "stash_id": None,
        "temp_dir": None,
        "original_files": {},
    }
    
    is_git_result = _run_ssh(config, f"cd {remote_root} && git rev-parse --git-dir 2>/dev/null", timeout=10)
    is_git = is_git_result["exit_code"] == 0
    sync_state["is_git_repo"] = is_git
    
    if is_git and mindspore_files:
        stash_result = _run_ssh(config, f"cd {remote_root} && git stash", timeout=30)
        if stash_result["exit_code"] == 0 and "stash" in stash_result["stdout"].lower():
            import re
            m = re.search(r"stash@\{(\d+)\}", stash_result["stdout"])
            if m:
                sync_state["stash_id"] = m.group(0)
    
    synced = []
    
    for f in mindspore_files:
        local_path = str(local_root / f)
        remote_path = f"{remote_root}/{f}"
        
        if not is_git:
            check_result = _run_ssh(config, f"test -f {shlex.quote(remote_path)} && echo EXISTS", timeout=10)
            if "EXISTS" in check_result["stdout"]:
                content_result = _run_ssh(config, f"cat {shlex.quote(remote_path)}", timeout=30)
                sync_state["original_files"][f] = content_result["stdout"]
        
        mkdir_result = _run_ssh(config, f"mkdir -p $(dirname {shlex.quote(remote_path)})", timeout=10)
        scp_result = _run_scp(config, local_path, remote_path)
        synced.append({
            "file": f,
            "type": "mindspore",
            "success": scp_result["success"],
            "error": scp_result.get("error", ""),
        })
    
    if external_files:
        temp_dir = f"{remote_root}/.deploy_temp_{_timestamp()}"
        sync_state["temp_dir"] = temp_dir
        _run_ssh(config, f"mkdir -p {temp_dir}", timeout=10)
        
        for f in external_files:
            local_path = f
            remote_path = f"{temp_dir}/{Path(f).name}"
            scp_result = _run_scp(config, local_path, remote_path)
            synced.append({
                "file": f,
                "type": "external",
                "success": scp_result["success"],
                "error": scp_result.get("error", ""),
            })
    
    state_file = log_dir / "sync_state.json"
    _save_json(str(state_file), sync_state)
    
    all_success = all(s["success"] for s in synced)
    
    _json_output({
        "success": all_success,
        "files_synced": synced,
        "sync_state_file": str(state_file),
        "summary": {
            "mindspore_files": len(mindspore_files),
            "external_files": len(external_files),
            "is_git_repo": is_git,
        },
    })


def cmd_build(args):
    """Step 4: Build MindSpore on remote server."""
    config = _load_config(args.config)
    log_dir = _ensure_log_dir(config)
    
    platform = config["build"]["platform"]
    nproc = config["build"].get("nproc", 96)
    timeout = _get_timeout(config, "build")
    
    build_cmd = f"bash build.sh -e {platform} -j{nproc}"
    full_cmd = _build_remote_command(config, build_cmd)
    
    result = _run_ssh(config, full_cmd, timeout=timeout)
    
    log_file = log_dir / f"build_{_timestamp()}.log"
    log_file.write_text(result["stdout"] + "\n" + result["stderr"], encoding="utf-8")
    
    output_lines = (result["stdout"] + "\n" + result["stderr"]).strip().splitlines()
    
    _json_output({
        "success": result["exit_code"] == 0,
        "platform": platform,
        "nproc": nproc,
        "elapsed_seconds": result["elapsed_seconds"],
        "log_file": str(log_file),
        "output_tail": output_lines[-20:] if output_lines else [],
        "error": result["stderr"][-500:] if result["stderr"] and result["exit_code"] != 0 else "",
        "timed_out": result.get("timed_out", False),
    })


def cmd_install(args):
    """Step 5: Install MindSpore wheel package."""
    config = _load_config(args.config)
    remote_root = config["paths"]["remote_root"]
    timeout = _get_timeout(config, "install")
    install_method = config.get("install", {}).get("method", "pip")
    
    if install_method == "pythonpath":
        _json_output({
            "success": True,
            "method": "pythonpath",
            "message": "PYTHONPATH mode does not require installation",
            "env_vars": {
                "MINDSPORE_PACKPATH": remote_root,
                "PYTHONPATH": f"{remote_root}:{remote_root}/build/package:{remote_root}/build/package/build/lib",
            },
        })
        return
    
    find_cmd = f"ls {remote_root}/build/package/mindspore-*.whl 2>/dev/null | head -1"
    find_result = _run_ssh(config, find_cmd, timeout=10)
    
    if not find_result["stdout"].strip():
        _json_output({
            "success": False,
            "message": "No wheel file found in build/package/. Run 'build' first.",
        })
        return
    
    wheel_path = find_result["stdout"].strip()
    wheel_name = wheel_path.split("/")[-1]
    
    uninstall_cmd = _build_remote_command(config, "pip uninstall -y mindspore")
    _run_ssh(config, uninstall_cmd, timeout=60)
    
    install_cmd = _build_remote_command(config, f"pip install {wheel_path}")
    result = _run_ssh(config, install_cmd, timeout=timeout)
    
    verify_cmd = _build_remote_command(config, "pip show mindspore")
    verify_result = _run_ssh(config, verify_cmd, timeout=10)
    
    _json_output({
        "success": result["exit_code"] == 0,
        "method": "pip",
        "wheel_path": wheel_path,
        "wheel_name": wheel_name,
        "elapsed_seconds": result["elapsed_seconds"],
        "verify_output": verify_result["stdout"][:500] if verify_result["stdout"] else "",
        "error": result["stderr"][-300:] if result["stderr"] and result["exit_code"] != 0 else "",
    })


def cmd_run(args):
    """Step 6: Run test command on remote server."""
    config = _load_config(args.config)
    log_dir = _ensure_log_dir(config)
    timeout = _get_timeout(config, "run")
    
    command = args.command
    if not command:
        _json_output({"success": False, "message": "No test command specified"})
        return
    
    full_cmd = _build_remote_command(config, command)
    result = _run_ssh(config, full_cmd, timeout=timeout)
    
    log_file = log_dir / f"run_{_timestamp()}.log"
    log_file.write_text(result["stdout"] + "\n" + result["stderr"], encoding="utf-8")
    
    pytest_summary = _parse_pytest_output(result["stdout"])
    output_lines = (result["stdout"] + "\n" + result["stderr"]).strip().splitlines()
    
    verdict = "pass" if result["exit_code"] == 0 else "fail"
    if result.get("timed_out"):
        verdict = "timeout"
    
    tail_lines = 15
    if verdict in ("fail", "timeout"):
        tail_lines = 30
    
    _json_output({
        "success": result["exit_code"] == 0,
        "command": command,
        "exit_code": result["exit_code"],
        "elapsed_seconds": result["elapsed_seconds"],
        "log_file": str(log_file),
        "pytest_summary": pytest_summary,
        "output_tail": output_lines[-tail_lines:] if output_lines else [],
        "verdict": verdict,
        "timed_out": result.get("timed_out", False),
    })


def cmd_rollback(args):
    """Rollback remote changes."""
    config = _load_config(args.config)
    log_dir = _ensure_log_dir(config)
    remote_root = config["paths"]["remote_root"]
    
    state_file = log_dir / "sync_state.json"
    if not state_file.exists():
        _json_output({"success": True, "message": "No sync state found, remote is already clean"})
        return
    
    sync_state = _load_config(str(state_file))
    
    results = {
        "mindspore_files_restored": [],
        "external_files_deleted": False,
        "errors": [],
    }
    
    is_git = sync_state.get("is_git_repo", False)
    mindspore_files = sync_state.get("mindspore_files", [])
    
    if is_git and mindspore_files:
        checkout_result = _run_ssh(config, f"cd {remote_root} && git checkout -- .", timeout=60)
        if checkout_result["exit_code"] == 0:
            results["mindspore_files_restored"] = mindspore_files
        
        stash_id = sync_state.get("stash_id")
        if stash_id:
            stash_result = _run_ssh(config, f"cd {remote_root} && git stash pop {stash_id}", timeout=30)
            if stash_result["exit_code"] != 0:
                results["errors"].append(f"Stash pop failed: {stash_result['stderr'][:100]}")
    elif not is_git and mindspore_files:
        original_files = sync_state.get("original_files", {})
        for f in mindspore_files:
            remote_path = f"{remote_root}/{f}"
            if f in original_files and original_files[f]:
                restore_cmd = f"cat > {shlex.quote(remote_path)} << 'EOF_ROLLBACK'\n{original_files[f]}\nEOF_ROLLBACK"
                restore_result = _run_ssh(config, restore_cmd, timeout=30)
                if restore_result["exit_code"] == 0:
                    results["mindspore_files_restored"].append(f)
            else:
                _run_ssh(config, f"rm -f {shlex.quote(remote_path)}", timeout=10)
                results["mindspore_files_restored"].append(f)
    
    temp_dir = sync_state.get("temp_dir")
    if temp_dir:
        cleanup_result = _run_ssh(config, f"rm -rf {temp_dir}", timeout=30)
        results["external_files_deleted"] = cleanup_result["exit_code"] == 0
    
    state_file.unlink(missing_ok=True)
    
    _json_output({
        "success": len(results["errors"]) == 0,
        "results": results,
        "message": "Rollback completed" if len(results["errors"]) == 0 else "Rollback completed with errors",
    })


def cmd_deploy(args):
    """One-click deploy and verify (all 6 steps in sequence)."""
    config = _load_config(args.config)
    results = {"steps": {}}
    
    result = _run_ssh(config, "echo OK", timeout=15)
    if result["exit_code"] != 0 or "OK" not in result["stdout"]:
        _json_output({
            "success": False,
            "step": "ssh",
            "message": "SSH connection failed",
            "suggestion": f"Please run: ssh-copy-id {config['remote']['user']}@{config['remote']['host']}",
        })
        return
    results["steps"]["ssh"] = {"success": True}
    
    sync_result = _execute_sync(config, args.files or [])
    results["steps"]["sync"] = sync_result
    if not sync_result.get("success"):
        _json_output({"success": False, "step": "sync", "sync_result": sync_result})
        return
    
    env_cmd = _build_remote_command(config, "python3 -c \"import acl; print(acl.get_soc_name())\" 2>&1 || echo ENV_ERROR")
    env_result = _run_ssh(config, env_cmd, timeout=15)
    if "ENV_ERROR" in env_result["stdout"]:
        _json_output({
            "success": False,
            "step": "env",
            "message": "Environment verification failed",
            "output": env_result["stdout"],
        })
        return
    results["steps"]["env"] = {"success": True, "chip": env_result["stdout"].strip()}
    
    if sync_result.get("summary", {}).get("mindspore_files", 0) > 0:
        build_result = _execute_build(config)
        results["steps"]["build"] = build_result
        if not build_result.get("success"):
            _json_output({"success": False, "step": "build", "build_result": build_result})
            return
        
        install_result = _execute_install(config)
        results["steps"]["install"] = install_result
        if not install_result.get("success"):
            _json_output({"success": False, "step": "install", "install_result": install_result})
            return
    
    run_result = _execute_run(config, args.command)
    results["steps"]["run"] = run_result
    
    if args.auto_rollback and run_result.get("verdict") == "pass":
        _execute_rollback(config)
        results["steps"]["rollback"] = {"success": True}
    
    _json_output({
        "success": run_result.get("success"),
        "step": "complete",
        "test_result": run_result,
        "sync_summary": sync_result.get("summary"),
        "all_steps": results["steps"],
    })


def _execute_sync(config: dict, files: list) -> dict:
    """Execute sync operation and return result dict."""
    log_dir = _ensure_log_dir(config)
    local_root = Path(config["paths"]["local_root"]).resolve()
    remote_root = config["paths"]["remote_root"]
    
    if not files:
        return {"success": True, "message": "No files to sync", "summary": {"mindspore_files": 0, "external_files": 0}}
    
    mindspore_files = []
    external_files = []
    
    for f in files:
        abs_path = Path(f).resolve()
        try:
            rel_path = abs_path.relative_to(local_root)
            mindspore_files.append(str(rel_path).replace("\\", "/"))
        except ValueError:
            external_files.append(str(abs_path).replace("\\", "/"))
    
    sync_state = {
        "timestamp": _timestamp(),
        "mindspore_files": mindspore_files,
        "external_files": external_files,
        "is_git_repo": False,
        "stash_id": None,
        "temp_dir": None,
        "original_files": {},
    }
    
    is_git_result = _run_ssh(config, f"cd {remote_root} && git rev-parse --git-dir 2>/dev/null", timeout=10)
    is_git = is_git_result["exit_code"] == 0
    sync_state["is_git_repo"] = is_git
    
    if is_git and mindspore_files:
        stash_result = _run_ssh(config, f"cd {remote_root} && git stash", timeout=30)
        if stash_result["exit_code"] == 0 and "stash" in stash_result["stdout"].lower():
            import re
            m = re.search(r"stash@\{(\d+)\}", stash_result["stdout"])
            if m:
                sync_state["stash_id"] = m.group(0)
    
    synced = []
    for f in mindspore_files:
        local_path = str(local_root / f)
        remote_path = f"{remote_root}/{f}"
        
        if not is_git:
            check_result = _run_ssh(config, f"test -f {shlex.quote(remote_path)} && echo EXISTS", timeout=10)
            if "EXISTS" in check_result["stdout"]:
                content_result = _run_ssh(config, f"cat {shlex.quote(remote_path)}", timeout=30)
                sync_state["original_files"][f] = content_result["stdout"]
        
        _run_ssh(config, f"mkdir -p $(dirname {shlex.quote(remote_path)})", timeout=10)
        scp_result = _run_scp(config, local_path, remote_path)
        synced.append({"file": f, "type": "mindspore", "success": scp_result["success"]})
    
    if external_files:
        temp_dir = f"{remote_root}/.deploy_temp_{_timestamp()}"
        sync_state["temp_dir"] = temp_dir
        _run_ssh(config, f"mkdir -p {temp_dir}", timeout=10)
        
        for f in external_files:
            remote_path = f"{temp_dir}/{Path(f).name}"
            scp_result = _run_scp(config, f, remote_path)
            synced.append({"file": f, "type": "external", "success": scp_result["success"]})
    
    state_file = log_dir / "sync_state.json"
    _save_json(str(state_file), sync_state)
    
    return {
        "success": all(s["success"] for s in synced),
        "files_synced": synced,
        "sync_state_file": str(state_file),
        "summary": {
            "mindspore_files": len(mindspore_files),
            "external_files": len(external_files),
            "is_git_repo": is_git,
        },
    }


def _execute_build(config: dict) -> dict:
    """Execute build operation and return result dict."""
    log_dir = _ensure_log_dir(config)
    platform = config["build"]["platform"]
    nproc = config["build"].get("nproc", 96)
    timeout = _get_timeout(config, "build")
    
    build_cmd = f"bash build.sh -e {platform} -j{nproc}"
    full_cmd = _build_remote_command(config, build_cmd)
    result = _run_ssh(config, full_cmd, timeout=timeout)
    
    log_file = log_dir / f"build_{_timestamp()}.log"
    log_file.write_text(result["stdout"] + "\n" + result["stderr"], encoding="utf-8")
    
    output_lines = (result["stdout"] + "\n" + result["stderr"]).strip().splitlines()
    
    return {
        "success": result["exit_code"] == 0,
        "platform": platform,
        "nproc": nproc,
        "elapsed_seconds": result["elapsed_seconds"],
        "log_file": str(log_file),
        "output_tail": output_lines[-20:] if output_lines else [],
        "timed_out": result.get("timed_out", False),
    }


def _execute_install(config: dict) -> dict:
    """Execute install operation and return result dict."""
    remote_root = config["paths"]["remote_root"]
    timeout = _get_timeout(config, "install")
    install_method = config.get("install", {}).get("method", "pip")
    
    if install_method == "pythonpath":
        return {
            "success": True,
            "method": "pythonpath",
            "message": "PYTHONPATH mode does not require installation",
        }
    
    find_cmd = f"ls {remote_root}/build/package/mindspore-*.whl 2>/dev/null | head -1"
    find_result = _run_ssh(config, find_cmd, timeout=10)
    
    if not find_result["stdout"].strip():
        return {"success": False, "message": "No wheel file found in build/package/."}
    
    wheel_path = find_result["stdout"].strip()
    
    uninstall_cmd = _build_remote_command(config, "pip uninstall -y mindspore")
    _run_ssh(config, uninstall_cmd, timeout=60)
    
    install_cmd = _build_remote_command(config, f"pip install {wheel_path}")
    result = _run_ssh(config, install_cmd, timeout=timeout)
    
    return {
        "success": result["exit_code"] == 0,
        "method": "pip",
        "wheel_path": wheel_path,
        "elapsed_seconds": result["elapsed_seconds"],
    }


def _execute_run(config: dict, command: str) -> dict:
    """Execute run operation and return result dict."""
    log_dir = _ensure_log_dir(config)
    timeout = _get_timeout(config, "run")
    
    if not command:
        return {"success": False, "message": "No test command specified"}
    
    full_cmd = _build_remote_command(config, command)
    result = _run_ssh(config, full_cmd, timeout=timeout)
    
    log_file = log_dir / f"run_{_timestamp()}.log"
    log_file.write_text(result["stdout"] + "\n" + result["stderr"], encoding="utf-8")
    
    pytest_summary = _parse_pytest_output(result["stdout"])
    output_lines = (result["stdout"] + "\n" + result["stderr"]).strip().splitlines()
    
    verdict = "pass" if result["exit_code"] == 0 else "fail"
    if result.get("timed_out"):
        verdict = "timeout"
    
    tail_lines = 15
    if verdict in ("fail", "timeout"):
        tail_lines = 30
    
    return {
        "success": result["exit_code"] == 0,
        "command": command,
        "exit_code": result["exit_code"],
        "elapsed_seconds": result["elapsed_seconds"],
        "log_file": str(log_file),
        "pytest_summary": pytest_summary,
        "output_tail": output_lines[-tail_lines:] if output_lines else [],
        "verdict": verdict,
        "timed_out": result.get("timed_out", False),
    }


def _execute_rollback(config: dict) -> dict:
    """Execute rollback operation and return result dict."""
    log_dir = _ensure_log_dir(config)
    remote_root = config["paths"]["remote_root"]
    
    state_file = log_dir / "sync_state.json"
    if not state_file.exists():
        return {"success": True, "message": "No sync state found"}
    
    sync_state = _load_config(str(state_file))
    
    is_git = sync_state.get("is_git_repo", False)
    mindspore_files = sync_state.get("mindspore_files", [])
    
    if is_git and mindspore_files:
        _run_ssh(config, f"cd {remote_root} && git checkout -- .", timeout=60)
        stash_id = sync_state.get("stash_id")
        if stash_id:
            _run_ssh(config, f"cd {remote_root} && git stash pop {stash_id}", timeout=30)
    elif not is_git and mindspore_files:
        original_files = sync_state.get("original_files", {})
        for f in mindspore_files:
            remote_path = f"{remote_root}/{f}"
            if f in original_files and original_files[f]:
                restore_cmd = f"cat > {shlex.quote(remote_path)} << 'EOF_ROLLBACK'\n{original_files[f]}\nEOF_ROLLBACK"
                _run_ssh(config, restore_cmd, timeout=30)
            else:
                _run_ssh(config, f"rm -f {shlex.quote(remote_path)}", timeout=10)
    
    temp_dir = sync_state.get("temp_dir")
    if temp_dir:
        _run_ssh(config, f"rm -rf {temp_dir}", timeout=30)
    
    state_file.unlink(missing_ok=True)
    
    return {"success": True, "message": "Rollback completed"}


def main():
    parser = argparse.ArgumentParser(
        description="Remote deploy & verify for MindSpore development"
    )
    sub = parser.add_subparsers(dest="subcmd", help="Sub-commands")
    
    p_ssh = sub.add_parser("check-ssh", help="Step 1: Check SSH connection")
    p_ssh.add_argument("--config", required=True, help="Path to deploy_context.json")
    
    p_env = sub.add_parser("verify-env", help="Step 3: Verify remote environment")
    p_env.add_argument("--config", required=True, help="Path to deploy_context.json")
    
    p_sync = sub.add_parser("sync", help="Step 2: Sync files to remote")
    p_sync.add_argument("--config", required=True, help="Path to deploy_context.json")
    p_sync.add_argument("--files", nargs="*", default=[], help="Files to sync")
    
    p_build = sub.add_parser("build", help="Step 4: Build MindSpore")
    p_build.add_argument("--config", required=True, help="Path to deploy_context.json")
    
    p_install = sub.add_parser("install", help="Step 5: Install MindSpore wheel")
    p_install.add_argument("--config", required=True, help="Path to deploy_context.json")
    
    p_run = sub.add_parser("run", help="Step 6: Run test command")
    p_run.add_argument("--config", required=True, help="Path to deploy_context.json")
    p_run.add_argument("--command", required=True, help="Test command to run")
    
    p_rollback = sub.add_parser("rollback", help="Rollback remote changes")
    p_rollback.add_argument("--config", required=True, help="Path to deploy_context.json")
    
    p_deploy = sub.add_parser("deploy", help="One-click deploy and verify (all steps)")
    p_deploy.add_argument("--config", required=True, help="Path to deploy_context.json")
    p_deploy.add_argument("--files", nargs="*", default=[], help="Files to sync")
    p_deploy.add_argument("--command", required=True, help="Test command to run")
    p_deploy.add_argument("--auto-rollback", action="store_true", help="Auto rollback after success")
    
    args = parser.parse_args()
    
    if not args.subcmd:
        parser.print_help()
        sys.exit(1)
    
    cmd_map = {
        "check-ssh": cmd_check_ssh,
        "verify-env": cmd_verify_env,
        "sync": cmd_sync,
        "build": cmd_build,
        "install": cmd_install,
        "run": cmd_run,
        "rollback": cmd_rollback,
        "deploy": cmd_deploy,
    }
    
    cmd_map[args.subcmd](args)


if __name__ == "__main__":
    main()
