"""Run deployment preflight checks for the backend project."""

from __future__ import annotations

import argparse
import http.client
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
HEALTH_PATH = "/api/health"
HEALTH_TIMEOUT_SECONDS = 30
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8000


class PreflightError(RuntimeError):
    """Raised when a preflight check fails."""


def _run(command: Sequence[str], *, timeout: int = 120) -> None:
    print(f"[preflight] running: {' '.join(command)}")
    try:
        subprocess.run(
            command,
            cwd=ROOT_DIR,
            check=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise PreflightError(f"Command timed out: {' '.join(command)}") from exc
    except subprocess.CalledProcessError as exc:
        raise PreflightError(f"Command failed with exit code {exc.returncode}: {' '.join(command)}") from exc


def _run_if_available(command: Sequence[str], *, timeout: int = 120) -> None:
    if not shutil.which(command[0]):
        print(f"[preflight] skipped: {command[0]} not found")
        return
    _run(command, timeout=timeout)


def _run_python_checks() -> None:
    _run(["uv", "run", "python", "-m", "compileall", "-q", "src", "scripts"])
    _run(["uv", "run", "ruff", "check", "."])
    _run(["uv", "run", "ruff", "format", "--check", "."])
    _run(["uv", "run", "scripts/smoke_test.py"], timeout=180)


def _run_frontend_check() -> None:
    package_json = ROOT_DIR / "package.json"
    if not package_json.exists():
        print("[preflight] skipped: no root package.json found")
        return
    _run_if_available(["npm", "run", "build"], timeout=300)


def _wait_for_health(process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + HEALTH_TIMEOUT_SECONDS
    last_error = "health endpoint did not respond"

    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise PreflightError("FastAPI server exited before health check passed")

        connection = None
        try:
            connection = http.client.HTTPConnection(
                SERVER_HOST,
                SERVER_PORT,
                timeout=2,
            )
            connection.request("GET", HEALTH_PATH)
            response = connection.getresponse()
            response.read()
            if response.status == 200:
                print("[preflight] FastAPI health check passed")
                return
            last_error = f"health endpoint returned HTTP {response.status}"
        except OSError as exc:
            last_error = str(exc)
        finally:
            if connection is not None:
                connection.close()

        time.sleep(1)

    raise PreflightError(last_error)


def _run_health_smoke() -> None:
    command = [
        "uv",
        "run",
        "python",
        "-m",
        "uvicorn",
        "src.api.main:app",
        "--host",
        SERVER_HOST,
        "--port",
        str(SERVER_PORT),
    ]
    print(f"[preflight] starting: {' '.join(command)}")
    process = subprocess.Popen(
        command,
        cwd=ROOT_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        _wait_for_health(process)
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def run_preflight(*, skip_server: bool) -> None:
    _run_python_checks()
    _run_frontend_check()
    if skip_server:
        print("[preflight] skipped: FastAPI health check")
    else:
        _run_health_smoke()
    print("SUCCESS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deployment preflight checks.")
    parser.add_argument(
        "--skip-server",
        action="store_true",
        help="Skip starting FastAPI and checking /api/health.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        run_preflight(skip_server=args.skip_server)
    except PreflightError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
