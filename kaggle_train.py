#!/usr/bin/env python3
"""Push the configured Kaggle training notebook and monitor its execution."""

from __future__ import annotations

import argparse
import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
NOTEBOOK_PATH = PROJECT_ROOT / "training_kernel.ipynb"
METADATA_PATH = PROJECT_ROOT / "kernel-metadata.json"
TOKEN_PATH = PROJECT_ROOT / "kaggle_token.txt"
GPU_ACCELERATOR = "NvidiaTeslaT4"
POLL_INTERVAL_SECONDS = 60
MONITOR_TIMEOUT_SECONDS = 1800
KERNEL_STARTUP_TIMEOUT_SECONDS = 300
LOG_TAIL_LINES = 100


def load_token(token_path: Path = TOKEN_PATH) -> str:
    """Read the Kaggle token without displaying or logging it."""
    token = os.environ.get("KAGGLE_API_TOKEN")
    if token:
        return token
    if not token_path.is_file():
        raise FileNotFoundError(
            f"Kaggle API token not found in the environment or {token_path.resolve()}."
        )
    token = token_path.read_text(encoding="utf-8").strip()
    if not token:
        raise ValueError(f"Kaggle API token file is empty: {token_path.resolve()}")
    return token


def build_environment(token: str) -> dict[str, str]:
    """Build the Kaggle CLI environment without exposing the token."""
    environment = os.environ.copy()
    environment["KAGGLE_API_TOKEN"] = token
    environment["PYTHONIOENCODING"] = "utf-8"
    return environment


def validate_notebook(notebook_path: Path) -> dict[str, Any]:
    """Require a readable nbformat 4 notebook with at least one code cell."""
    if not notebook_path.is_file():
        raise FileNotFoundError(f"Notebook not found: {notebook_path.resolve()}")
    try:
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Notebook is not valid JSON: {notebook_path}") from error
    if not isinstance(notebook, dict) or notebook.get("nbformat") != 4:
        raise ValueError(f"Expected a valid nbformat 4 notebook: {notebook_path}")
    cells = notebook.get("cells")
    if not isinstance(cells, list) or not all(
        isinstance(cell, dict) for cell in cells
    ):
        raise ValueError(f"Expected a valid nbformat 4 notebook: {notebook_path}")
    if not any(cell.get("cell_type") == "code" for cell in cells):
        raise ValueError(f"Expected a valid nbformat 4 notebook: {notebook_path}")
    for index, cell in enumerate(cells):
        if cell.get("cell_type") == "code":
            source = cell.get("source", "")
            if isinstance(source, list) and all(
                isinstance(line, str) for line in source
            ):
                code = "".join(source)
            elif isinstance(source, str):
                code = source
            else:
                raise ValueError(f"Invalid source in notebook cell {index + 1}.")
            try:
                ast.parse(code, filename=f"{notebook_path}:cell-{index + 1}")
            except SyntaxError as error:
                raise ValueError(
                    f"Invalid Python in notebook cell {index + 1}: {error}"
                ) from error
    return notebook


def build_kernel_metadata(
    notebook_path: Path = NOTEBOOK_PATH,
    metadata_path: Path = METADATA_PATH,
) -> dict[str, Any]:
    """Load project metadata and point it at the selected notebook."""
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Kernel metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("Kernel metadata must be a JSON object.")
    if (
        not metadata.get("id")
        or not isinstance(metadata.get("dataset_sources"), list)
        or not metadata["dataset_sources"]
        or not all(isinstance(source, str) for source in metadata["dataset_sources"])
    ):
        raise ValueError("Kernel metadata must include an id and dataset_sources.")
    metadata["code_file"] = notebook_path.name
    metadata["kernel_type"] = "notebook"
    metadata["enable_gpu"] = True
    metadata["enable_tpu"] = False
    metadata["enable_internet"] = True
    return metadata


def stage_kernel(
    upload_dir: Path,
    notebook_path: Path = NOTEBOOK_PATH,
    metadata_path: Path = METADATA_PATH,
) -> Path:
    """Stage only the notebook and metadata, never local credentials."""
    validate_notebook(notebook_path)
    upload_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(notebook_path, upload_dir / notebook_path.name)
    staged_metadata = build_kernel_metadata(notebook_path, metadata_path)
    staged_path = upload_dir / "kernel-metadata.json"
    staged_path.write_text(
        json.dumps(staged_metadata, indent=2) + "\n", encoding="utf-8"
    )
    return staged_path


def run_kaggle_command(
    arguments: list[str],
    environment: dict[str, str],
    *,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run Kaggle CLI and redact its credential from any displayed error."""
    command = ["kaggle", *arguments]
    try:
        result = subprocess.run(
            command,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except FileNotFoundError as error:
        raise RuntimeError(
            "Kaggle CLI was not found. Install it with `python -m pip install kaggle`."
        ) from error
    if check and result.returncode:
        details = result.stderr.strip() or result.stdout.strip()
        token = environment.get("KAGGLE_API_TOKEN", "")
        if token:
            details = details.replace(token, "[REDACTED]")
        raise RuntimeError(
            f"Kaggle command failed with exit code {result.returncode}: "
            f"{' '.join(command)}\n{details}"
        )
    return result


def push_notebook(environment: dict[str, str]) -> None:
    """Publish a new notebook version configured for a Tesla T4."""
    metadata = build_kernel_metadata()
    if not metadata.get("enable_gpu") or not GPU_ACCELERATOR:
        raise ValueError("GPU metadata and an accelerator must be configured.")
    with tempfile.TemporaryDirectory(prefix="kaggle-training-upload-") as temp_dir:
        upload_dir = Path(temp_dir)
        staged_path = stage_kernel(upload_dir)
        print(
            f"Uploading {staged_path.parent / NOTEBOOK_PATH.name} "
            f"to kernel {metadata['id']} with {GPU_ACCELERATOR}...",
            flush=True,
        )
        result = run_kaggle_command(
            [
                "kernels",
                "push",
                "--path",
                str(upload_dir),
                "--accelerator",
                GPU_ACCELERATOR,
            ],
            environment,
        )
        if result.stdout.strip():
            print(result.stdout.strip(), flush=True)


def get_kernel_status(environment: dict[str, str], kernel_id: str) -> str | None:
    """Return status, or None when Kaggle has no session for the kernel yet."""
    try:
        result = run_kaggle_command(["kernels", "status", kernel_id], environment)
    except RuntimeError as error:
        details = str(error).lower()
        if "404" in details and "not found" in details:
            return None
        raise
    status = result.stdout.strip()
    if not status:
        raise RuntimeError(f"Kaggle returned an empty status for {kernel_id}.")
    return status


def fetch_and_display_logs(
    environment: dict[str, str], kernel_id: str
) -> None:
    """Print the tail of the latest execution logs."""
    result = run_kaggle_command(
        ["kernels", "logs", kernel_id], environment, check=False
    )
    if result.returncode:
        details = result.stderr.strip() or result.stdout.strip()
        token = environment.get("KAGGLE_API_TOKEN", "")
        if token:
            details = details.replace(token, "[REDACTED]")
        print(f"Could not fetch kernel logs: {details}", file=sys.stderr)
        return
    lines = result.stdout.splitlines()
    if not lines:
        print("Kaggle returned no execution logs.", flush=True)
        return
    print("\n--- Kaggle Execution Logs (tail) ---", flush=True)
    for line in lines[-LOG_TAIL_LINES:]:
        print(line, flush=True)
    print("------------------------------------", flush=True)


def monitor_kernel(
    environment: dict[str, str],
    kernel_id: str,
    *,
    poll_interval: int = POLL_INTERVAL_SECONDS,
    timeout: int = MONITOR_TIMEOUT_SECONDS,
    wait_for_startup: bool = False,
    startup_timeout: int = KERNEL_STARTUP_TIMEOUT_SECONDS,
) -> bool:
    """Wait for kernel completion, failure, or timeout."""
    if poll_interval <= 0 or timeout <= 0 or startup_timeout <= 0:
        raise ValueError("Polling and timeout values must be greater than zero.")
    started_at = time.monotonic()
    while True:
        status = get_kernel_status(environment, kernel_id)
        elapsed = time.monotonic() - started_at
        if status is None:
            if not wait_for_startup:
                print(
                    f"No Kaggle execution session was found for {kernel_id}. "
                    "Check the owner/kernel slug or push a kernel version first.",
                    file=sys.stderr,
                )
                return False
            startup_deadline = min(timeout, startup_timeout)
            if elapsed >= startup_deadline:
                print(
                    "Kaggle has not created an execution session for "
                    f"{kernel_id} within {startup_deadline} seconds. "
                    "Check the kernel slug and the result of `kaggle kernels push`.",
                    file=sys.stderr,
                )
                return False
            print(
                f"[{time.strftime('%H:%M:%S')}] No execution session yet; "
                "waiting for Kaggle to start the pushed version.",
                flush=True,
            )
            time.sleep(min(poll_interval, startup_deadline - elapsed))
            continue
        print(f"[{time.strftime('%H:%M:%S')}] Status: {status}", flush=True)
        normalized_status = status.upper()
        if "COMPLETE" in normalized_status and "RUNNING" not in normalized_status:
            print("Kernel execution completed successfully.", flush=True)
            fetch_and_display_logs(environment, kernel_id)
            return True
        if any(
            state in normalized_status
            for state in ("ERROR", "CANCELLED", "FAILED")
        ):
            print(f"Kernel execution failed: {status}", flush=True)
            fetch_and_display_logs(environment, kernel_id)
            return False
        if time.monotonic() - started_at >= timeout:
            print(f"Kernel monitoring timed out after {timeout} seconds.")
            fetch_and_display_logs(environment, kernel_id)
            return False
        time.sleep(poll_interval)


def dry_run() -> None:
    """Validate and stage the upload without reading credentials or contacting Kaggle."""
    with tempfile.TemporaryDirectory(prefix="kaggle-training-dry-run-") as temp_dir:
        staged_path = stage_kernel(Path(temp_dir))
        metadata = json.loads(staged_path.read_text(encoding="utf-8"))
        print(f"Notebook validated: {NOTEBOOK_PATH}")
        print(f"Kernel: {metadata['id']}")
        print(f"Dataset sources: {', '.join(metadata['dataset_sources'])}")
        print(f"GPU: {GPU_ACCELERATOR}")
        print("Staging passed; no credentials were read and no request was sent.")


def main() -> int:
    """Parse command-line options and run the requested Kaggle workflow."""
    parser = argparse.ArgumentParser(
        description="Push the training notebook to Kaggle and monitor its T4 run."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and stage the notebook without credentials or network calls.",
    )
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="Monitor the latest execution without uploading a new version.",
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
    if args.dry_run:
        dry_run()
        return 0

    metadata = build_kernel_metadata()
    environment = build_environment(load_token())
    if not args.no_push:
        push_notebook(environment)
    else:
        print("Monitoring the latest kernel run without uploading.", flush=True)
    return int(
        not monitor_kernel(
            environment,
            metadata["id"],
            poll_interval=args.poll_interval,
            timeout=args.timeout,
            wait_for_startup=not args.no_push,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
