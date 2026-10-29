"""
Smoke tests for the installed asta package.

These exercise the console scripts a real `pip install .` / `conda install`
provides (asta-ensemble, asta-watch, ...), not repo-relative file paths, so
they still pass when nothing from the source tree is on disk — only the
installed package. They don't require the full (multi-hundred-MB) GEF
dataset.

Every subprocess call below goes through _run(), which closes stdin and
points $HOME at a throwaway directory. Without that, a command that doesn't
actually exit on --help (see TestNotifyInstalled and watch_runner.sh's own
--help handling) could block waiting for keyboard input, or write real
watcher state into the developer's own ~/.sttx-watch / ~/Library/Logs.

If a real (ideally small/subsampled) GEF file is available, point
$ST_TEST_GEF at it to additionally run the baseline stage end-to-end and
confirm it produces output. This is skipped by default / in CI unless that
variable is set.

Run with:
    python -m unittest tests/test_smoke.py -v
"""
import importlib
import os
import shutil
import subprocess
import tempfile
import unittest

STAGE_COMMANDS = ["asta-baseline", "asta-ensemble", "asta-annotate", "asta-agent"]
# asta-notify is deliberately excluded: it takes positional subject/body
# args, not flags, so "--help" would be read as the subject and then block
# on stdin for the body. See TestNotifyInstalled.
HELP_COMMANDS = STAGE_COMMANDS + ["asta-run", "asta-watch", "asta-watch-launchd", "asta-saw"]


def _run(args, **kwargs):
    """subprocess.run with stdin closed and $HOME isolated to a throwaway
    directory, so a CLI invocation can never block on keyboard input or
    touch the developer's real ~/.sttx-watch or ~/Library/Logs."""
    kwargs.setdefault("capture_output", True)
    kwargs.setdefault("text", True)
    kwargs.setdefault("timeout", 30)
    kwargs.setdefault("stdin", subprocess.DEVNULL)
    with tempfile.TemporaryDirectory() as fake_home:
        kwargs.setdefault("env", dict(os.environ, HOME=fake_home))
        return subprocess.run(args, **kwargs)


class TestImports(unittest.TestCase):
    """Each module should import cleanly (no errors, no required GEF file at import time)."""

    def test_modules_import(self):
        for module_name in [
            "asta.ovarian_knowledge_base",
            "asta.baseline",
            "asta.annotator",
            "asta.ensemble_clustering",
            "asta.ai_agent",
            "asta.notify",
            "asta.watch_runner",
        ]:
            with self.subTest(module=module_name):
                importlib.import_module(module_name)


class TestCLI(unittest.TestCase):
    """Every console script below should expose a working --help (proves
    it's actually installed and importable end to end, not just present as
    a file on disk)."""

    def test_help_exits_cleanly(self):
        for command in HELP_COMMANDS:
            with self.subTest(command=command):
                result = _run([command, "--help"])
                self.assertEqual(result.returncode, 0, msg=result.stderr)

    def test_stage_commands_advertise_input_output_flags(self):
        for command in STAGE_COMMANDS:
            with self.subTest(command=command):
                result = _run([command, "--help"])
                self.assertIn("--input", result.stdout)
                self.assertIn("--output", result.stdout)


class TestNotifyInstalled(unittest.TestCase):
    """asta-notify has no --help (see HELP_COMMANDS comment above), so just
    confirm it's installed. TestImports.test_modules_import already proves
    asta.notify loads cleanly."""

    def test_asta_notify_is_installed(self):
        self.assertIsNotNone(shutil.which("asta-notify"))


class TestWatchOnceNoOp(unittest.TestCase):
    """asta-watch --once in a folder with no .gef should no-op and exit 0
    without writing anything into that folder — that's what keeps the
    watcher from re-triggering itself on its own output."""

    def test_watch_once_empty_dir_exits_zero_and_untouched(self):
        with tempfile.TemporaryDirectory() as empty_dir:
            result = _run(["asta-watch", "--once", "--path", empty_dir])
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertEqual(os.listdir(empty_dir), [])


@unittest.skipUnless(
    os.environ.get("ST_TEST_GEF"),
    "Set $ST_TEST_GEF to a sample .gef file to run the end-to-end smoke test",
)
class TestEndToEndBaseline(unittest.TestCase):
    """Runs the baseline stage against a real GEF file end-to-end and checks it produces output."""

    def test_baseline_runs_and_produces_output(self):
        gef_path = os.environ["ST_TEST_GEF"]
        with tempfile.TemporaryDirectory() as out_dir:
            result = _run(
                ["asta-baseline", "--input", gef_path, "--output", out_dir],
                timeout=1800,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            produced = os.listdir(out_dir)
            self.assertTrue(produced, "asta-baseline produced no output files")


if __name__ == "__main__":
    unittest.main()
