#!/usr/bin/env python3
"""
watch_runner.py — cross-platform (macOS / Linux / Windows) automatic pipeline
trigger for new GEF drops.

This is a portable equivalent of watch_runner.sh + run_pipeline.sh, for
platforms where macOS's launchd isn't available. Run it and leave it running;
whenever a *.gef file is created (or modified/finishes copying) in the watched
folder, it automatically runs the pipeline (ensemble clustering -> biological
annotation -> AI agent) and sends a notification when done.

Usage:
    python -m asta.watch_runner                  # watch the current directory
    python -m asta.watch_runner --path /some/dir # watch a specific data folder
    python -m asta.watch_runner --once           # process any existing unprocessed
                                              # .gef file(s) once, then exit
                                              # (useful for testing)

State (lock file + processed-file list + logs) is kept OUTSIDE the watched
folder (in ~/.sttx-watch) so that the pipeline writing its own output files
back into the watched folder does not re-trigger a run.

Requires the "watchdog" package (see environment.yml / requirements.txt).
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

# Stages run, in order. (baseline.py is intentionally excluded — it's a
# standalone comparison/benchmark script, not part of the required chain;
# see README.md Step 3.1.)
STAGES = [
    ("ensemble", "asta.ensemble_clustering"),
    ("annotator", "asta.annotator"),
    ("agent", "asta.ai_agent"),
]

STATE_DIR = Path.home() / ".sttx-watch"
LOCK_DIR = STATE_DIR / "run.lock"
PROCESSED_FILE = STATE_DIR / "processed.list"
WATCH_LOG = STATE_DIR / "watch.log"

STABILIZE_POLL_INTERVAL = 2.0  # seconds
STABILIZE_MAX_POLLS = 60       # ~2 minutes max wait for a file to stop growing

# How often to print a "still running" heartbeat while a stage executes.
# Override with $ST_HEARTBEAT_SECS if you want it chattier/quieter.
HEARTBEAT_SECS = int(os.environ.get("ST_HEARTBEAT_SECS", "60"))


def _format_duration(total_seconds: float) -> str:
    total = int(total_seconds)
    m, s = divmod(total, 60)
    return f"{m}m{s:02d}s" if m else f"{s}s"


def log(message: str) -> None:
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(WATCH_LOG, "a") as f:
        f.write(line + "\n")


def load_processed() -> set:
    if not PROCESSED_FILE.exists():
        return set()
    return set(PROCESSED_FILE.read_text().splitlines())


def mark_processed(sig: str) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(PROCESSED_FILE, "a") as f:
        f.write(sig + "\n")


def signature(path: Path) -> str:
    try:
        size = path.stat().st_size
    except OSError:
        size = 0
    return f"{path}:{size}"


class _Lock:
    """Simple cross-platform atomic lock using an exclusive directory create."""

    def __enter__(self):
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        try:
            LOCK_DIR.mkdir()
        except FileExistsError:
            return False
        return True

    def __exit__(self, exc_type, exc, tb):
        try:
            LOCK_DIR.rmdir()
        except OSError:
            pass


def wait_for_stable_size(path: Path) -> int:
    """Poll file size until it stops changing (handles files still being copied in)."""
    prev = -1
    cur = 0
    for _ in range(STABILIZE_MAX_POLLS):
        try:
            cur = path.stat().st_size
        except OSError:
            cur = 0
        if cur == prev and cur != 0:
            break
        prev = cur
        time.sleep(STABILIZE_POLL_INTERVAL)
    return cur


def resolve_python() -> str:
    """Use whichever Python is currently running this script — respects
    an already-active conda/venv environment, since that's how the user
    launched watch_runner.py in the first place."""
    return sys.executable


def run_stage(project_dir: Path, log_dir: Path, run_ts: str, name: str, module: str,
              input_gef: Path, py: str) -> bool:
    log_path = log_dir / f"{run_ts}_{name}.log"
    log(f">>> [{name}] starting: {module}")
    log(f"    input : {input_gef}")
    log(f"    log   : {log_path}")
    start = time.time()
    env = dict(os.environ)
    env["ST_INPUT_GEF"] = str(input_gef)
    env["ST_RUN_ID"] = run_ts
    with open(log_path, "w") as logfile:
        proc = subprocess.Popen(
            [py, "-m", module, "--input", str(input_gef)],
            cwd=str(project_dir),
            env=env,
            stdout=logfile,
            stderr=subprocess.STDOUT,
        )
        # Heartbeat so long silent stretches (clustering/embedding steps can
        # run many minutes with no console output) don't look like the
        # pipeline hung.
        returncode = None
        while returncode is None:
            try:
                returncode = proc.wait(timeout=HEARTBEAT_SECS)
            except subprocess.TimeoutExpired:
                elapsed = time.time() - start
                log(f"    ... [{name}] still running (elapsed: {_format_duration(elapsed)})")
    dur = time.time() - start
    if returncode == 0:
        log(f"<<< [{name}] done in {_format_duration(dur)}")
        return True
    log(f"!!! [{name}] FAILED (exit {returncode}) — see {log_path}")
    try:
        tail = log_path.read_text().splitlines()[-20:]
        for line in tail:
            log(f"    {line}")
    except OSError:
        pass
    return False


def run_pipeline(project_dir: Path, input_gef: Path) -> bool:
    py = resolve_python()
    log_dir = project_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log(f"=== Spatial Transcriptomics pipeline | run {run_ts} | stage=all ===")
    log(f"Using Python: {py}")
    for name, module in STAGES:
        if not run_stage(project_dir, log_dir, run_ts, name, module, input_gef, py):
            return False
    log("=== pipeline complete ===")
    return True


def notify(project_dir: Path, status: str, gef_name: str, duration: float) -> None:
    summary = (
        f"Spatial Transcriptomics pipeline {status} in {duration:.0f}s — "
        f"file: {gef_name}. Outputs in project folder; details: {WATCH_LOG}"
    )
    log(summary)
    try:
        subprocess.run(
            [sys.executable, "-m", "asta.notify", f"ST pipeline: {status}", summary],
            cwd=str(project_dir),
            timeout=60,
        )
    except Exception as e:  # noqa: BLE001
        log(f"[notify] failed to invoke asta.notify: {e}")


def process_gef(path: Path, project_dir: Path) -> None:
    sig = signature(path)
    if sig in load_processed():
        return

    with _Lock() as acquired:
        if not acquired:
            return  # a run is already in progress; silently skip (loop-safety)

        # Re-check after acquiring the lock in case another thread just finished it.
        if sig in load_processed():
            return

        size = wait_for_stable_size(path)
        # Re-derive signature now that the file has (hopefully) finished copying.
        sig = f"{path}:{size}"
        if sig in load_processed():
            return

        log(f"New GEF detected: {path} (size {size}B). Starting pipeline.")
        start = time.time()
        success = run_pipeline(project_dir, path)
        duration = time.time() - start
        mark_processed(sig)
        notify(project_dir, "SUCCESS" if success else "FAILED", path.name, duration)


class GefHandler(FileSystemEventHandler):
    def __init__(self, project_dir: Path):
        self.project_dir = project_dir

    def _maybe_process(self, event):
        if event.is_directory:
            return
        path = Path(event.src_path)
        if path.suffix.lower() != ".gef":
            return
        # Run in a background thread so the watcher stays responsive and
        # multiple rapid events don't block each other; the lock in
        # process_gef() ensures only one pipeline run happens at a time.
        threading.Thread(
            target=process_gef, args=(path, self.project_dir), daemon=True
        ).start()

    def on_created(self, event):
        self._maybe_process(event)

    def on_modified(self, event):
        self._maybe_process(event)


def scan_existing(project_dir: Path) -> None:
    """Catch any .gef file that was already sitting in the folder before the
    watcher started (e.g. the watcher was restarted, or the file was dropped
    while it was down)."""
    for gef in sorted(project_dir.glob("*.gef")):
        threading.Thread(target=process_gef, args=(gef, project_dir), daemon=True).start()


def main():
    parser = argparse.ArgumentParser(
        description="Cross-platform automatic pipeline trigger for new GEF drops "
                    "(portable equivalent of watch_runner.sh, for non-macOS systems)."
    )
    parser.add_argument(
        "--path", default=None,
        help="Data folder to watch (default: the current directory)",
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Process any existing unprocessed .gef file(s) once, then exit "
             "(does not start the continuous watcher). Useful for testing.",
    )
    args = parser.parse_args()

    project_dir = Path(args.path).resolve() if args.path else Path.cwd().resolve()

    env_file = project_dir / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):]
            if "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

    if args.once:
        scan_existing(project_dir)
        # scan_existing spawns background threads; wait for them to finish.
        for t in threading.enumerate():
            if t is not threading.main_thread():
                t.join()
        return

    log(f"Watching {project_dir} for new .gef files (state dir: {STATE_DIR})")
    scan_existing(project_dir)

    observer = Observer()
    observer.schedule(GefHandler(project_dir), str(project_dir), recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    main()
