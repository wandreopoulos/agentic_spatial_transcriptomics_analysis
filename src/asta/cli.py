"""Console-script entry points for ASTA.

Each function below runs the corresponding module exactly as
``python -m asta.<module>`` would, so the module's own
``if __name__ == "__main__":`` block (and its existing argparse CLI)
keeps working completely unchanged.
"""

import os
import runpy
import sys
from pathlib import Path


def _run_bundled_script(name, args=None, extra_env=None):
    """Hand off to one of our own bundled .sh files (shipped as package data,
    next to this file) via bash. Uses os.execvpe, which *replaces* this
    Python process with bash, so the script's exit code and Ctrl-C behave
    exactly as if it had been run directly — nothing waits around it.
    ST_PYTHON tells the script exactly which Python interpreter this package
    is installed under, which matters under launchd/cron where PATH is
    minimal and no conda/venv is active."""
    script = Path(__file__).with_name(name)
    env = dict(os.environ, ST_PYTHON=sys.executable, **(extra_env or {}))
    script_args = sys.argv[1:] if args is None else args
    os.execvpe("bash", ["bash", str(script), *script_args], env)


def run_pipeline():
    _run_bundled_script("run_pipeline.sh")


def watch_launchd():
    _run_bundled_script("watch_runner.sh")


def baseline():
    runpy.run_module("asta.baseline", run_name="__main__", alter_sys=True)


def ensemble():
    runpy.run_module("asta.ensemble_clustering", run_name="__main__", alter_sys=True)


def annotate():
    runpy.run_module("asta.annotator", run_name="__main__", alter_sys=True)


def agent():
    runpy.run_module("asta.ai_agent", run_name="__main__", alter_sys=True)


def notify():
    runpy.run_module("asta.notify", run_name="__main__", alter_sys=True)
