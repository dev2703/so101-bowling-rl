"""Non-polling Colab Pro orchestrator using the `colab` CLI."""

from __future__ import annotations

import argparse
import json
import netrc
import os
import re
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Any

DEFAULT_REPO_URL = "https://github.com/dev2703/so101-bowling-rl.git"
DEFAULT_BRANCH = "main"
DEFAULT_REMOTE_REPO_DIR = "/content/so101-bowling-rl"
DEFAULT_REMOTE_RUN_DIR = "/content/runs/colab_run"
DOWNLOAD_FILES = (
    "status.json",
    "summary.json",
    "preflight_report.json",
    "metrics.jsonl",
    "train.log",
)


def build_colab_prefix(
    colab_bin: str = "colab",
    colab_auth: str | None = None,
    colab_config: str | None = None,
) -> list[str]:
    cmd = [colab_bin]
    if colab_auth:
        cmd.extend(["--auth", colab_auth])
    if colab_config:
        cmd.extend(["--config", colab_config])
    return cmd


def _run_cmd(
    cmd: list[str],
    *,
    dry_run: bool = False,
    check: bool = True,
    capture_output: bool = True,
) -> subprocess.CompletedProcess[str]:
    if dry_run:
        print(f"[dry-run] {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
    return subprocess.run(cmd, check=check, capture_output=capture_output, text=True)


def _find_local_hf_token() -> str | None:
    env_tok = os.environ.get("HF_TOKEN", "").strip()
    if env_tok:
        return env_tok
    for candidate in (
        Path.home() / ".cache" / "huggingface" / "token",
        Path.home() / ".huggingface" / "token",
    ):
        if candidate.is_file():
            tok = candidate.read_text(encoding="utf-8").strip()
            if tok:
                return tok
    return None


def _find_local_wandb_api_key() -> str | None:
    env_key = os.environ.get("WANDB_API_KEY", "").strip()
    if env_key:
        return env_key
    try:
        netrc_path = Path.home() / ".netrc"
        if netrc_path.is_file():
            nrc = netrc.netrc(str(netrc_path))
            auth = nrc.authenticators("api.wandb.ai")
            if auth and auth[2]:
                key = auth[2].strip()
                if key:
                    return key
    except Exception:
        pass
    return None


def _current_git_branch() -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        branch = res.stdout.strip()
        if branch and branch != "HEAD":
            return branch
    except Exception:
        pass
    return DEFAULT_BRANCH


def _ensure_train_args(raw_args: list[str], remote_run_dir: str) -> list[str]:
    cleaned = list(raw_args)
    if cleaned and cleaned[0] == "--":
        cleaned = cleaned[1:]
    if not any(a == "--run-dir" or a.startswith("--run-dir=") for a in cleaned):
        cleaned.extend(["--run-dir", remote_run_dir])
    return cleaned


def _session_in_output(session: str, stdout: str) -> bool:
    if not session or not stdout:
        return False
    pattern = re.compile(rf"(?<![A-Za-z0-9_.-]){re.escape(session)}(?![A-Za-z0-9_.-])")
    return any(pattern.search(line) for line in stdout.splitlines())


def _parse_json_output(stdout: str) -> dict[str, Any] | None:
    stripped = stdout.strip()
    if not stripped:
        return None
    try:
        parsed = json.loads(stripped)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    lines = stripped.splitlines()
    last_line = (lines or [""])[-1]
    try:
        parsed = json.loads(last_line)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    for idx in range(len(lines) - 1, -1, -1):
        if lines[idx].lstrip().startswith("{"):
            try:
                parsed = json.loads("\n".join(lines[idx:]))
                return parsed if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                continue
    return None


def build_bootstrap_script(
    *,
    repo_url: str,
    branch: str,
    remote_repo_dir: str,
    hf_token_remote_tmp: str = "/content/.hf_token_tmp",
    wandb_key_remote_tmp: str = "/content/.wandb_api_key_tmp",
) -> str:
    return f"""import os
import shutil
import subprocess
from pathlib import Path

repo_url = {repo_url!r}
branch = {branch!r}
repo_dir = Path({remote_repo_dir!r})
token_tmp = Path({hf_token_remote_tmp!r})
wandb_tmp = Path({wandb_key_remote_tmp!r})

env = os.environ.copy()
local_bin = str(Path.home() / ".local" / "bin")
env["PATH"] = f"{{local_bin}}:{{env.get('PATH', '')}}"
env["MUJOCO_GL"] = "egl"

if token_tmp.exists():
    hf_dir = Path.home() / ".cache" / "huggingface"
    hf_dir.mkdir(parents=True, exist_ok=True)
    hf_tok = hf_dir / "token"
    shutil.move(str(token_tmp), str(hf_tok))
    os.chmod(hf_tok, 0o600)
    print("Configured Hugging Face token (mode 0600)", flush=True)

if wandb_tmp.exists():
    wb_dir = Path.home() / ".config" / "so101_bowling"
    wb_dir.mkdir(parents=True, exist_ok=True)
    wb_key = wb_dir / "wandb_api_key"
    shutil.move(str(wandb_tmp), str(wb_key))
    os.chmod(wb_key, 0o600)
    print("Configured W&B API key (mode 0600)", flush=True)

if not (repo_dir / ".git").exists():
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--branch", branch, repo_url, str(repo_dir)], check=True, env=env)
else:
    subprocess.run(["git", "fetch", "origin"], cwd=repo_dir, check=True, env=env)
    subprocess.run(["git", "checkout", "-B", branch, f"origin/{{branch}}"], cwd=repo_dir, check=True, env=env)

if shutil.which("uv", path=env["PATH"]) is None:
    subprocess.run("curl -LsSf https://astral.sh/uv/install.sh | sh", shell=True, check=True, env=env)

uv_bin = shutil.which("uv", path=env["PATH"]) or str(Path.home() / ".local" / "bin" / "uv")
venv_python = repo_dir / ".venv" / "bin" / "python"
if not venv_python.exists():
    subprocess.run([uv_bin, "venv", "--python", "3.12", str(repo_dir / ".venv")], cwd=repo_dir, check=True, env=env)

subprocess.run([uv_bin, "sync", "--extra", "hub", "--extra", "dev", "--extra", "tracking"], cwd=repo_dir, check=True, env=env)
subprocess.run(
    [
        str(venv_python),
        "-c",
        "import torch, huggingface_hub, mujoco, wandb; "
        "print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available(), "
        "'hf:', huggingface_hub.__version__, 'mujoco:', mujoco.__version__, 'wandb:', wandb.__version__)",
    ],
    cwd=repo_dir,
    check=True,
    env=env,
)
"""


def build_spawn_script(
    *,
    remote_repo_dir: str,
    remote_run_dir: str,
    train_args: list[str],
    session: str,
    hf_token_remote_tmp: str = "/content/.hf_token_tmp",
    wandb_key_remote_tmp: str = "/content/.wandb_api_key_tmp",
) -> str:
    return f"""import json
import os
import shutil
import subprocess
import time
from pathlib import Path

repo_dir = Path({remote_repo_dir!r})
run_dir = Path({remote_run_dir!r})
run_dir.mkdir(parents=True, exist_ok=True)
token_tmp = Path({hf_token_remote_tmp!r})
wandb_tmp = Path({wandb_key_remote_tmp!r})

if token_tmp.exists():
    hf_dir = Path.home() / ".cache" / "huggingface"
    hf_dir.mkdir(parents=True, exist_ok=True)
    hf_tok = hf_dir / "token"
    shutil.move(str(token_tmp), str(hf_tok))
    os.chmod(hf_tok, 0o600)

wb_key_file = Path.home() / ".config" / "so101_bowling" / "wandb_api_key"
if wandb_tmp.exists():
    wb_key_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(wandb_tmp), str(wb_key_file))
    os.chmod(wb_key_file, 0o600)

venv_python = str(repo_dir / ".venv" / "bin" / "python")
cmd = [venv_python, "scripts/colab_train.py", *{train_args!r}]
log_path = run_dir / "train.log"
pid_path = run_dir / "train.pid"
status_path = run_dir / "status.json"

env = os.environ.copy()
env["MUJOCO_GL"] = "egl"
env["PYTHONUNBUFFERED"] = "1"
if wb_key_file.is_file() and not env.get("WANDB_API_KEY"):
    wb_val = wb_key_file.read_text(encoding="utf-8").strip()
    if wb_val:
        env["WANDB_API_KEY"] = wb_val

with log_path.open("ab") as log_fh:
    proc = subprocess.Popen(
        cmd,
        cwd=str(repo_dir),
        start_new_session=True,
        stdin=subprocess.DEVNULL,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        env=env,
    )

pid_path.write_text(str(proc.pid), encoding="utf-8")
time.sleep(2.0)
rc = proc.poll()
if rc is not None:
    tail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
    raise RuntimeError(f"Detached training process exited immediately with code {{rc}}:\\n{{tail}}")

print(
    json.dumps(
        {{
            "session": {session!r},
            "remote_run_dir": str(run_dir),
            "pid": proc.pid,
            "log_path": str(log_path),
            "status_path": str(status_path),
        }},
        indent=2,
    ),
    flush=True,
)
"""


def build_status_script(*, remote_run_dir: str, tail_lines: int) -> str:
    return f"""import json
import subprocess
from pathlib import Path

run_dir = Path({remote_run_dir!r})
pid_path = run_dir / "train.pid"
status_path = run_dir / "status.json"
summary_path = run_dir / "summary.json"
log_path = run_dir / "train.log"

pid = pid_path.read_text(encoding="utf-8").strip() if pid_path.exists() else None
pid_alive = False
ps_info = ""
if pid:
    res = subprocess.run(["ps", "-p", pid, "-o", "pid=,stat="], capture_output=True, text=True)
    ps_info = res.stdout.strip()
    pid_alive = res.returncode == 0 and bool(ps_info)

status_data = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else None
summary_data = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None
log_tail = []
if log_path.exists():
    log_tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-{int(tail_lines)}:]

print(
    json.dumps(
        {{
            "pid": pid,
            "pid_alive": pid_alive,
            "ps": ps_info,
            "status": status_data,
            "summary": summary_data,
            "log_tail": log_tail,
        }},
        indent=2,
    ),
    flush=True,
)
"""


def cmd_launch(args: argparse.Namespace, colab_prefix: list[str]) -> dict[str, Any]:
    dry_run = bool(args.dry_run)
    session = args.session
    branch = args.branch or _current_git_branch()
    train_args = _ensure_train_args(args.train_args or [], args.remote_run_dir)

    sessions_res = _run_cmd([*colab_prefix, "sessions"], dry_run=dry_run, check=False)
    session_exists = (not dry_run) and _session_in_output(session, sessions_res.stdout)
    if not session_exists:
        new_cmd = [*colab_prefix, "new", "-s", session]
        if args.gpu:
            new_cmd.extend(["--gpu", args.gpu])
        if args.high_mem:
            new_cmd.append("--high-mem")
        _run_cmd(new_cmd, dry_run=dry_run, check=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        if args.forward_hf_token:
            tok = _find_local_hf_token()
            if tok is not None:
                tok_file = tmp_path / "hf_token"
                tok_file.write_text(tok, encoding="utf-8")
                os.chmod(tok_file, 0o600)
                upload_target = "<redacted_hf_token>" if dry_run else str(tok_file)
                _run_cmd(
                    [*colab_prefix, "upload", upload_target, "/content/.hf_token_tmp", "-s", session],
                    dry_run=dry_run,
                    check=True,
                )

        if args.forward_wandb_key:
            wb_key = _find_local_wandb_api_key()
            if wb_key is not None:
                wb_file = tmp_path / "wandb_api_key"
                wb_file.write_text(wb_key, encoding="utf-8")
                os.chmod(wb_file, 0o600)
                upload_target = "<redacted_wandb_key>" if dry_run else str(wb_file)
                _run_cmd(
                    [*colab_prefix, "upload", upload_target, "/content/.wandb_api_key_tmp", "-s", session],
                    dry_run=dry_run,
                    check=True,
                )

        if not args.skip_bootstrap:
            boot_file = tmp_path / "bootstrap.py"
            boot_file.write_text(
                build_bootstrap_script(
                    repo_url=args.repo_url,
                    branch=branch,
                    remote_repo_dir=args.remote_repo_dir,
                ),
                encoding="utf-8",
            )
            _run_cmd(
                [*colab_prefix, "exec", "-s", session, "--timeout", "300", "-f", str(boot_file)],
                dry_run=dry_run,
                check=True,
            )

        spawn_file = tmp_path / "spawn.py"
        spawn_file.write_text(
            build_spawn_script(
                remote_repo_dir=args.remote_repo_dir,
                remote_run_dir=args.remote_run_dir,
                train_args=train_args,
                session=session,
            ),
            encoding="utf-8",
        )
        spawn_res = _run_cmd(
            [*colab_prefix, "exec", "-s", session, "--timeout", "30", "-f", str(spawn_file)],
            dry_run=dry_run,
            check=True,
        )

    if dry_run:
        return {
            "dry_run": True,
            "session": session,
            "branch": branch,
            "remote_repo_dir": args.remote_repo_dir,
            "remote_run_dir": args.remote_run_dir,
            "train_args": train_args,
        }
    if spawn_res.stdout.strip():
        print(spawn_res.stdout.strip(), flush=True)
        parsed = _parse_json_output(spawn_res.stdout)
        if parsed is not None:
            return parsed
    return {"session": session, "remote_run_dir": args.remote_run_dir, "raw_output": spawn_res.stdout}


def cmd_status(args: argparse.Namespace, colab_prefix: list[str]) -> dict[str, Any]:
    dry_run = bool(args.dry_run)
    st_res = _run_cmd([*colab_prefix, "status", "-s", args.session], dry_run=dry_run, check=False)
    if st_res.stdout:
        print(st_res.stdout.strip(), flush=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        status_file = Path(tmpdir) / "status_probe.py"
        status_file.write_text(
            build_status_script(remote_run_dir=args.remote_run_dir, tail_lines=args.tail),
            encoding="utf-8",
        )
        ex_res = _run_cmd(
            [*colab_prefix, "exec", "-s", args.session, "--timeout", "30", "-f", str(status_file)],
            dry_run=dry_run,
            check=False,
        )
    if ex_res.stdout:
        print(ex_res.stdout.strip(), flush=True)
    return {
        "session": args.session,
        "colab_status_rc": st_res.returncode,
        "remote_probe_rc": ex_res.returncode,
    }


def cmd_download(args: argparse.Namespace, colab_prefix: list[str]) -> dict[str, Any]:
    dry_run = bool(args.dry_run)
    local_dir = Path(args.local_dir)
    if not dry_run:
        local_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[str] = []
    skipped: list[str] = []
    for fname in DOWNLOAD_FILES:
        remote_path = f"{args.remote_run_dir.rstrip('/')}/{fname}"
        local_path = local_dir / fname
        res = _run_cmd(
            [*colab_prefix, "download", remote_path, str(local_path), "-s", args.session],
            dry_run=dry_run,
            check=False,
        )
        if res.returncode == 0:
            downloaded.append(str(local_path))
        else:
            skipped.append(remote_path)
    out = {"session": args.session, "downloaded": downloaded, "skipped": skipped}
    print(json.dumps(out, indent=2), flush=True)
    return out


def cmd_stop(args: argparse.Namespace, colab_prefix: list[str]) -> dict[str, Any]:
    res = _run_cmd([*colab_prefix, "stop", "-s", args.session], dry_run=bool(args.dry_run), check=True)
    if res.stdout:
        print(res.stdout.strip(), flush=True)
    return {"session": args.session, "stopped": True}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--colab-auth", type=str, default=None, help="Colab CLI --auth strategy (oauth2|adc).")
    parser.add_argument("--colab-config", type=str, default=None, help="Colab CLI --config path.")
    parser.add_argument("--dry-run", action="store_true", help="Print constructed colab commands without running.")

    sub = parser.add_subparsers(dest="command", required=True)

    p_launch = sub.add_parser("launch", help="Create/reuse Colab session, bootstrap env, and spawn detached training.")
    p_launch.add_argument("-s", "--session", required=True, help="Colab session name.")
    p_launch.add_argument("--gpu", choices=("T4", "L4", "G4", "A100", "H100"), default=None)
    p_launch.add_argument("--high-mem", action="store_true")
    p_launch.add_argument("--repo-url", type=str, default=DEFAULT_REPO_URL)
    p_launch.add_argument("--branch", type=str, default=None)
    p_launch.add_argument("--remote-repo-dir", type=str, default=DEFAULT_REMOTE_REPO_DIR)
    p_launch.add_argument("--remote-run-dir", type=str, default=DEFAULT_REMOTE_RUN_DIR)
    p_launch.add_argument("--forward-hf-token", action=argparse.BooleanOptionalAction, default=True)
    p_launch.add_argument("--forward-wandb-key", action=argparse.BooleanOptionalAction, default=True)
    p_launch.add_argument("--skip-bootstrap", action="store_true")
    p_launch.add_argument(
        "train_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to scripts/colab_train.py (prefix with --).",
    )

    p_status = sub.add_parser("status", help="Inspect Colab session, remote PID, status.json, and train.log tail.")
    p_status.add_argument("-s", "--session", required=True, help="Colab session name.")
    p_status.add_argument("--remote-run-dir", type=str, default=DEFAULT_REMOTE_RUN_DIR)
    p_status.add_argument("--tail", type=int, default=25)

    p_dl = sub.add_parser("download", help="Download status, summary, metrics, and log files from Colab.")
    p_dl.add_argument("-s", "--session", required=True, help="Colab session name.")
    p_dl.add_argument("--remote-run-dir", type=str, default=DEFAULT_REMOTE_RUN_DIR)
    p_dl.add_argument("--local-dir", type=str, default="runs/colab_downloaded")

    p_stop = sub.add_parser("stop", help="Stop and release a Colab session.")
    p_stop.add_argument("-s", "--session", required=True, help="Colab session name.")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    colab_prefix = build_colab_prefix(
        colab_auth=args.colab_auth,
        colab_config=args.colab_config,
    )
    if args.command == "launch":
        cmd_launch(args, colab_prefix)
    elif args.command == "status":
        cmd_status(args, colab_prefix)
    elif args.command == "download":
        cmd_download(args, colab_prefix)
    elif args.command == "stop":
        cmd_stop(args, colab_prefix)


if __name__ == "__main__":
    main()
