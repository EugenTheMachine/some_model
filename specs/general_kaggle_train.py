#!/usr/bin/env python3
"""Upload a configured notebook as a Kaggle kernel version and monitor its run."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional


# Reconfigure these values when using this launcher for another notebook/model.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "training_kaggle.ipynb"
TOKEN_PATH = PROJECT_ROOT / "kaggle_token.txt"
ADDITIONAL_UPLOAD_FILES: tuple[str, ...] = ()

KAGGLE_USERNAME = "eugenfromkharkov"
KERNEL_SLUG = "instanseg-kaggle-training"
KERNEL_TITLE = "Model Training"
KERNEL_ID = f"{KAGGLE_USERNAME}/{KERNEL_SLUG}"

DATASET_SOURCES = [
    "eugenfromkharkov/livecell-cellseg1-a172",
]
COMPETITION_SOURCES: List[str] = []
KERNEL_SOURCES: List[str] = []

KERNEL_LANGUAGE = "python"
KERNEL_TYPE = "notebook"
KERNEL_IS_PRIVATE = True
ENABLE_GPU = True
ENABLE_TPU = False
ENABLE_INTERNET = True
GPU_ACCELERATOR = "NvidiaTeslaT4"

POLL_INTERVAL_SECONDS = 60
MONITOR_TIMEOUT_SECONDS = 1800
LOG_TAIL_LINES = 100


def load_token(token_path: Path = TOKEN_PATH) -> str:
    """Read the Kaggle API token without displaying or logging it."""
    if not token_path.is_file():
        raise FileNotFoundError(f"Kaggle token file not found at: {token_path.resolve()}")

    token = token_path.read_text(encoding="utf-8").strip()
    if not token:
        raise ValueError(f"Kaggle token file is empty: {token_path.resolve()}")
    return token


def build_environment(token: str) -> Dict[str, str]:
    """Create the Kaggle CLI environment using the configured API token."""
    env = os.environ.copy()
    env["KAGGLE_API_TOKEN"] = token
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def validate_notebook(notebook_path: Path) -> None:
    """Check that the configured file is a readable Jupyter notebook."""
    if not notebook_path.is_file():
        raise FileNotFoundError(f"Notebook not found at: {notebook_path.resolve()}")

    try:
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Notebook is not valid JSON: {notebook_path}") from error

    if notebook.get("nbformat") != 4 or not isinstance(notebook.get("cells"), list):
        raise ValueError(f"Expected a valid nbformat 4 notebook: {notebook_path}")


def build_kernel_metadata(notebook_path: Path = NOTEBOOK_PATH) -> Dict[str, object]:
    """Build the Kaggle kernel metadata for the selected notebook and inputs."""
    return {
        "id": KERNEL_ID,
        "title": KERNEL_TITLE,
        "code_file": notebook_path.name,
        "language": KERNEL_LANGUAGE,
        "kernel_type": KERNEL_TYPE,
        "is_private": KERNEL_IS_PRIVATE,
        "enable_gpu": ENABLE_GPU,
        "enable_tpu": ENABLE_TPU,
        "enable_internet": ENABLE_INTERNET,
        "dataset_sources": list(DATASET_SOURCES),
        "competition_sources": list(COMPETITION_SOURCES),
        "kernel_sources": list(KERNEL_SOURCES),
    }


def stage_kernel(upload_dir: Path, notebook_path: Path = NOTEBOOK_PATH) -> Path:
    """Stage a notebook, optional files, and generated metadata for Kaggle push."""
    validate_notebook(notebook_path)
    upload_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(notebook_path, upload_dir / notebook_path.name)

    for relative_name in ADDITIONAL_UPLOAD_FILES:
        source = (PROJECT_ROOT / relative_name).resolve()
        try:
            relative_path = source.relative_to(PROJECT_ROOT.resolve())
        except ValueError as error:
            raise ValueError(
                f"Additional upload file must be inside the project root: {relative_name}"
            ) from error
        if not source.is_file():
            raise FileNotFoundError(f"Additional upload file not found: {source}")
        destination = upload_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    metadata = build_kernel_metadata(notebook_path)
    metadata_path = upload_dir / "kernel-metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata_path


def run_kaggle_command(
    arguments: List[str],
    env: Dict[str, str],
    *,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """Run a Kaggle CLI command and explicitly report command failures."""
    command = ["kaggle"] + arguments
    result = subprocess.run(
        command,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if check and result.returncode != 0:
        details = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(
            f"Kaggle command failed with exit code {result.returncode}: "
            f"{' '.join(command)}\n{details}"
        )
    return result


def push_notebook(env: Dict[str, str]) -> None:
    """Publish a new version of the configured Kaggle kernel and start it."""
    if ENABLE_GPU and not GPU_ACCELERATOR:
        raise ValueError("GPU_ACCELERATOR must be set when ENABLE_GPU is true.")

    with tempfile.TemporaryDirectory(prefix="kaggle-kernel-upload-") as temp_dir:
        upload_dir = Path(temp_dir)
        metadata_path = stage_kernel(upload_dir)
        print(f"Uploading notebook version for '{KERNEL_ID}'...", flush=True)
        print(f"Using kernel metadata: {metadata_path.name}", flush=True)

        command = ["kernels", "push", "--path", str(upload_dir)]
        if ENABLE_GPU:
            command.extend(["--accelerator", GPU_ACCELERATOR])
            print(f"Requested GPU accelerator: {GPU_ACCELERATOR}", flush=True)

        result = run_kaggle_command(command, env)
        if result.stdout.strip():
            print(result.stdout.strip(), flush=True)


def get_kernel_status(env: Dict[str, str]) -> str:
    """Return Kaggle's current status for the configured kernel."""
    result = run_kaggle_command(
        ["kernels", "status", KERNEL_ID],
        env,
    )
    status = result.stdout.strip()
    if not status:
        raise RuntimeError(f"Kaggle returned an empty status for {KERNEL_ID}.")
    return status


def fetch_and_display_logs(env: Dict[str, str]) -> None:
    """Print the tail of the latest kernel execution log."""
    print("\nFetching latest execution logs...", flush=True)
    result = run_kaggle_command(
        ["kernels", "logs", KERNEL_ID],
        env,
        check=False,
    )
    if result.returncode != 0:
        details = result.stderr.strip() or result.stdout.strip()
        print(f"Could not fetch kernel logs: {details}", file=sys.stderr, flush=True)
        return

    lines = result.stdout.splitlines()
    if not lines:
        print("Kaggle returned no execution logs.", flush=True)
        return

    print("\n--- Execution Logs (Tail) ---", flush=True)
    for line in lines[-LOG_TAIL_LINES:]:
        print(line, flush=True)
    print("-----------------------------\n", flush=True)


def monitor_kernel(
    env: Dict[str, str],
    *,
    poll_interval: int = POLL_INTERVAL_SECONDS,
    timeout: int = MONITOR_TIMEOUT_SECONDS,
) -> bool:
    """Poll kernel status until completion, failure, or timeout."""
    if poll_interval <= 0:
        raise ValueError("poll_interval must be greater than zero.")
    if timeout <= 0:
        raise ValueError("timeout must be greater than zero.")

    started_at = time.monotonic()
    print(f"Monitoring Kaggle kernel '{KERNEL_ID}'...", flush=True)

    while True:
        status = get_kernel_status(env)
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] Status: {status}", flush=True)
        normalized_status = status.upper()

        if "COMPLETE" in normalized_status and "RUNNING" not in normalized_status:
            print("\nKernel execution completed successfully.", flush=True)
            fetch_and_display_logs(env)
            return True

        if any(state in normalized_status for state in ("ERROR", "CANCELLED", "FAILED")):
            print(f"\nKernel execution failed with status: {status}", flush=True)
            fetch_and_display_logs(env)
            return False

        if time.monotonic() - started_at >= timeout:
            print(f"\nKernel monitoring timed out after {timeout} seconds.", flush=True)
            fetch_and_display_logs(env)
            return False

        time.sleep(poll_interval)


def run_and_monitor(
    *,
    push: bool = True,
    poll_interval: int = POLL_INTERVAL_SECONDS,
    timeout: int = MONITOR_TIMEOUT_SECONDS,
) -> bool:
    """Optionally push a notebook version, then monitor its execution."""
    token = load_token()
    env = build_environment(token)

    if push:
        push_notebook(env)
    else:
        print(
            f"Monitoring the latest execution of '{KERNEL_ID}' without uploading.",
            flush=True,
        )

    return monitor_kernel(
        env,
        poll_interval=poll_interval,
        timeout=timeout,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Upload the configured Jupyter notebook as a new Kaggle kernel "
            "version and monitor its execution."
        )
    )
    parser.add_argument(
        "--no-push",
        action="store_false",
        dest="push",
        help="Monitor the latest execution without uploading a notebook version.",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=POLL_INTERVAL_SECONDS,
        help=f"Seconds between status checks (default: {POLL_INTERVAL_SECONDS}).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=MONITOR_TIMEOUT_SECONDS,
        help=f"Maximum monitoring time in seconds (default: {MONITOR_TIMEOUT_SECONDS}).",
    )
    args = parser.parse_args()

    succeeded = run_and_monitor(
        push=args.push,
        poll_interval=args.poll_interval,
        timeout=args.timeout,
    )
    return 0 if succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
