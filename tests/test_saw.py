"""
Tests for asta-saw. No real SAW, data or network needed: a fake `saw` script
(shebang = this Python, so it can use h5py) is written to a temp dir.

Run with:
    python -m unittest tests/test_saw.py -v
"""
import os
import stat
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

FAKE_SAW = textwrap.dedent('''\
    #!{python}
    import json, os, sys
    import h5py

    argv = sys.argv[1:]
    if argv == ["--version"]:
        print("SAW fake 0.0.1")
        sys.exit(0)
    with open(os.environ["FAKE_SAW_ARGV"], "w") as f:
        json.dump(argv, f)
    if os.environ.get("FAKE_SAW_FAIL"):
        sys.exit(7)
    run_id = next(a.split("=", 1)[1] for a in argv if a.startswith("--id="))
    d = os.path.join(run_id, "outs", "feature_expression")
    os.makedirs(d)

    def make(name, cellbin):
        with h5py.File(os.path.join(d, name), "w") as h:
            if cellbin:
                for ds in ("cell", "cellExp", "gene"):
                    h.create_dataset("cellBin/" + ds, data=[0])
            else:
                h.create_dataset("geneExp/bin1/expression", data=[0])

    make("S1.tissue.gef", False)
    make("S1.adjusted.cellbin.gef", True)
    make("S1.protein.adjusted.cellbin.gef", True)
''')


def _make_gef(path, cellbin):
    import h5py
    with h5py.File(path, "w") as h:
        if cellbin:
            for ds in ("cell", "cellExp", "gene"):
                h.create_dataset("cellBin/" + ds, data=[0])
        else:
            h.create_dataset("geneExp/bin1/expression", data=[0])


class SawTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.work = self.tmp / "work"
        self.work.mkdir()
        self.stub = self.tmp / "saw"
        self.stub.write_text(FAKE_SAW.format(python=sys.executable))
        self.stub.chmod(self.stub.stat().st_mode | stat.S_IXUSR)
        self.argv_file = self.tmp / "argv.json"

    def asta_saw(self, *args, saw=True, **env):
        e = dict(os.environ, HOME=str(self.tmp), FAKE_SAW_ARGV=str(self.argv_file))
        e.pop("SAW_BIN", None)
        if saw:
            e["SAW_BIN"] = str(self.stub)
        else:
            # PATH without any `saw`
            e["PATH"] = os.path.dirname(sys.executable) + os.pathsep + "/usr/bin:/bin"
        e.update(env)
        return subprocess.run(["asta-saw", *args], cwd=self.work, env=e,
                              capture_output=True, text=True, timeout=60,
                              stdin=subprocess.DEVNULL)


class TestCheck(SawTestCase):
    def test_check_with_stub(self):
        r = self.asta_saw("check")
        self.assertEqual(r.returncode, 0, msg=r.stderr)
        self.assertIn("SAW fake 0.0.1", r.stdout)

    def test_check_without_saw(self):
        r = self.asta_saw("check", saw=False)
        self.assertEqual(r.returncode, 1)
        self.assertIn("install", r.stderr.lower())
        self.assertIn("SAW_BIN", r.stderr)

    def test_check_with_bad_saw_bin_names_the_path(self):
        r = self.asta_saw("check", SAW_BIN="/no/such/saw")
        self.assertEqual(r.returncode, 1)
        self.assertIn("$SAW_BIN is set to '/no/such/saw'", r.stderr)

    def test_check_with_bad_saw_flag_names_the_path(self):
        r = self.asta_saw("check", "--saw", "/no/such/saw")
        self.assertEqual(r.returncode, 1)
        self.assertIn("--saw is set to '/no/such/saw'", r.stderr)

    def test_help(self):
        for args in ([], ["--help"], ["count", "--help"]):
            r = self.asta_saw(*args)
            self.assertEqual(r.returncode, 0, msg=r.stderr)
            self.assertIn("not bundled", r.stdout.lower())


class TestCount(SawTestCase):
    def test_passthrough_and_gef_selection(self):
        args = ["--id=run1", "--kit-version=Stereo-seq T FF V1.3", "--sn", "S1"]
        r = self.asta_saw("count", *args)
        self.assertEqual(r.returncode, 0, msg=r.stderr)
        import json
        # saw is invoked as `saw count <args>`; args must be unchanged.
        self.assertEqual(json.loads(self.argv_file.read_text()), ["count", *args])
        self.assertIn("S1.adjusted.cellbin.gef", r.stdout)
        self.assertNotIn("protein.adjusted.cellbin.gef", r.stdout)
        self.assertTrue(list((self.work / "logs").glob("*_saw.log")))

    def test_saw_failure_exits_3(self):
        r = self.asta_saw("count", "--id=run1", FAKE_SAW_FAIL="1")
        self.assertEqual(r.returncode, 3)

    def test_requires_id_or_output(self):
        r = self.asta_saw("count", "--sn", "S1")
        self.assertEqual(r.returncode, 1)

    def test_analyze_is_accepted_and_not_passed_to_saw(self):
        # Don't run the real pipeline: just use a SAW failure so we stop
        # before the hand-off, then check --analyze was stripped from argv.
        r = self.asta_saw("count", "--analyze", "--id=run1", FAKE_SAW_FAIL="1")
        self.assertEqual(r.returncode, 3)
        import json
        self.assertEqual(json.loads(self.argv_file.read_text()), ["count", "--id=run1"])


class TestFindGef(SawTestCase):
    def test_prints_only_path(self):
        d = self.tmp / "res" / "outs" / "feature_expression"
        d.mkdir(parents=True)
        _make_gef(d / "S1.adjusted.cellbin.gef", True)
        _make_gef(d / "S1.protein.adjusted.cellbin.gef", True)
        _make_gef(d / "S1.tissue.gef", False)
        r = self.asta_saw("find-gef", str(self.tmp / "res"))
        self.assertEqual(r.returncode, 0, msg=r.stderr)
        self.assertEqual(r.stdout.strip(), str(d / "S1.adjusted.cellbin.gef"))

    def test_plain_cellbin_without_cellbin_group_exits_2(self):
        d = self.tmp / "res"
        d.mkdir()
        _make_gef(d / "S1.cellbin.gef", False)
        r = self.asta_saw("find-gef", str(d))
        self.assertEqual(r.returncode, 2)
        self.assertIn("segmentation", r.stderr)

    def test_ambiguous_exits_2(self):
        d = self.tmp / "res"
        d.mkdir()
        _make_gef(d / "A.adjusted.cellbin.gef", True)
        _make_gef(d / "B.adjusted.cellbin.gef", True)
        r = self.asta_saw("find-gef", str(d))
        self.assertEqual(r.returncode, 2)
        self.assertIn("ST_INPUT_GEF", r.stderr)


if __name__ == "__main__":
    unittest.main()
