"""asta-saw: find an installed SAW (STOmics' FASTQ -> GEF tool), run it, and
hand the cell-bin GEF it produces to the ASTA pipeline.

SAW is third-party and is NOT bundled or installed by this package (it is a
Linux-x86_64-only ~6.6 GB binary). These commands only locate and drive an
existing install. Nothing here downloads anything.

Exit codes follow run_pipeline.sh: 1 = setup problem, 2 = no usable GEF,
3 = SAW failed.
"""

import datetime
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

# Keep in sync with the README heading "(Optional) Processing raw FASTQ files
# with SAW". Pinned to the release tag: the repo's default branch may not have
# that section yet.
README_URL = (
    "https://github.com/wandreopoulos/agentic_spatial_transcriptomics_analysis"
    "/blob/v1.0.1/README.md#optional-processing-raw-fastq-files-with-saw"
)

# SAW's documented minimums (warnings only; SAW decides for itself).
MIN_CORES = 8
MIN_RAM_GB = 128
MIN_DISK_GB = 1000

CELLBIN_DATASETS = ("cellBin/cell", "cellBin/cellExp", "cellBin/gene")

USAGE = """\
usage: asta-saw check [--saw PATH]
       asta-saw count [--saw PATH] [--analyze] <saw count arguments...>
       asta-saw find-gef DIR

Bridge to STOmics SAW. SAW is NOT bundled with ASTA: install it separately
(Linux x86_64 only) and see the install instructions at:
  {readme}

subcommands:
  check      find SAW (--saw, $SAW_BIN, then PATH), report platform/resources
  count      run `saw count` with your arguments passed through unchanged,
             then locate the cell-bin GEF it produced. With --analyze, also
             run `asta-run all` on it. Needs --id (or --output).
  find-gef   print the path of the RNA cell-bin GEF inside a SAW results dir

Already have a .gef? You don't need SAW; use asta-run directly.
Exit codes: 1 setup problem, 2 no usable GEF, 3 SAW failed.
"""


def _usage():
    return USAGE.format(readme=_readme_location())


def _readme_location():
    prefix_readme = Path(sys.prefix) / "share" / "asta" / "README.md"
    return str(prefix_readme) if prefix_readme.is_file() else README_URL


def _err(*parts):
    print(*parts, file=sys.stderr)


# ---------------------------------------------------------------- locating SAW

def _pull_flag_value(args, flag):
    """Remove `flag VALUE` / `flag=VALUE` from args by hand (argparse's
    parse_known_args would accept abbreviations and may reorder unknowns).
    Returns (value_or_None, remaining_args)."""
    rest, value, i = [], None, 0
    while i < len(args):
        a = args[i]
        if a == flag and i + 1 < len(args):
            value = args[i + 1]
            i += 2
        elif a.startswith(flag + "="):
            value = a.split("=", 1)[1]
            i += 1
        else:
            rest.append(a)
            i += 1
    return value, rest


def _find_saw(explicit=None):
    """--saw PATH, then $SAW_BIN, then `saw` on PATH. Returns a path or None."""
    for cand in (explicit, os.environ.get("SAW_BIN")):
        if cand:
            return cand if os.path.isfile(cand) and os.access(cand, os.X_OK) else None
    return shutil.which("saw")


def _saw_version(saw):
    try:
        r = subprocess.run([saw, "--version"], capture_output=True, text=True,
                           timeout=60, stdin=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0:
        return None
    return (r.stdout or r.stderr).strip()


def _not_found_message(explicit=None):
    # Same precedence as _find_saw: a set --saw / $SAW_BIN that fails is the
    # real problem, so name it instead of implying nothing was set.
    bad = [(label, val) for label, val in
           (("--saw", explicit), ("$SAW_BIN", os.environ.get("SAW_BIN"))) if val]
    hint = (f"  {bad[0][0]} is set to '{bad[0][1]}', which is not an executable file.\n"
            if bad else "")
    return (
        "ERROR: SAW was not found.\n"
        + hint +
        "  Looked at: --saw PATH, $SAW_BIN, then `saw` on PATH.\n"
        f"  Install instructions: {_readme_location()}\n"
        "  If SAW is installed somewhere not on PATH:\n"
        "    export SAW_BIN=/path/to/saw\n"
        "  If you already have a .gef file you don't need SAW; run asta-run."
    )


def _resources():
    """(cores, ram_gb or None, free_disk_gb) -- best effort, stdlib only."""
    cores = os.cpu_count() or 0
    ram = None
    try:
        ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
    except (ValueError, OSError, AttributeError):
        pass
    disk = shutil.disk_usage(os.getcwd()).free / 1024**3
    return cores, ram, disk


def cmd_check(args):
    if any(a in ("-h", "--help") for a in args):
        print(_usage())
        return 0
    explicit, rest = _pull_flag_value(args, "--saw")
    if rest:
        _err(f"ERROR: unexpected arguments: {' '.join(rest)}")
        return 1

    system, machine = platform.system(), platform.machine()
    ok_platform = system == "Linux" and machine in ("x86_64", "AMD64")
    print(f"platform : {system} {machine} "
          + ("(supported by SAW)" if ok_platform
             else "(WARNING: SAW supports Linux x86_64 only)"))
    cores, ram, disk = _resources()
    print(f"cpu cores: {cores} "
          + ("" if cores >= MIN_CORES else f"(WARNING: SAW wants >= {MIN_CORES})"))
    if ram is not None:
        print(f"ram      : {ram:.0f} GB "
              + ("" if ram >= MIN_RAM_GB else f"(WARNING: SAW wants >= {MIN_RAM_GB} GB)"))
    print(f"free disk: {disk:.0f} GB in {os.getcwd()} "
          + ("" if disk >= MIN_DISK_GB else f"(WARNING: SAW wants >= {MIN_DISK_GB} GB)"))

    saw = _find_saw(explicit)
    if not saw:
        _err(_not_found_message(explicit))
        return 1
    version = _saw_version(saw)
    print(f"saw path : {saw}")
    if version is None:
        _err("ERROR: found SAW but `saw --version` failed.")
        return 1
    print(f"saw ver  : {version}")
    return 0


# ----------------------------------------------------------------- finding GEF

def _has_cellbin(path):
    import h5py  # deferred so `--help` stays fast
    try:
        with h5py.File(path, "r") as f:
            return all(name in f for name in CELLBIN_DATASETS)
    except (OSError, ValueError):
        return False


def _find_gef(directory):
    """Returns (gef_path or None, exit_code, message). Prints nothing."""
    d = Path(directory)
    if not d.is_dir():
        return None, 2, f"ERROR: not a directory: {d}"
    search = d / "outs" / "feature_expression"
    if not search.is_dir():
        search = d

    # No protein GEFs (Stereo-CITE), ever.
    all_gefs = sorted(p for p in search.glob("*.gef") if ".protein." not in p.name)
    adjusted = [p for p in all_gefs if p.name.endswith(".adjusted.cellbin.gef")]
    plain = [p for p in all_gefs
             if p.name.endswith(".cellbin.gef") and p not in adjusted]

    for tier in (adjusted, plain):
        valid = [p for p in tier if _has_cellbin(p)]
        if len(valid) == 1:
            return valid[0], 0, ""
        if len(valid) > 1:
            listing = "\n".join(f"  {p}" for p in valid)
            return None, 2, (f"ERROR: several usable cell-bin GEFs in {search}:\n"
                             f"{listing}\n  Pick one with ST_INPUT_GEF=<file>.")

    found = "\n".join(f"  {p}" for p in all_gefs) or "  (none)"
    return None, 2, (
        f"ERROR: no usable cell-bin GEF (with {', '.join(CELLBIN_DATASETS)}) in {search}.\n"
        f"  GEFs found:\n{found}\n"
        "  A cell-bin GEF only exists if cell segmentation ran: pass a stain image\n"
        "  (--image / --image-tar) to `saw count` and don't use --skip-cellbin.\n"
        "  tissue/raw GEFs (square bins) and *.protein.* GEFs are not valid input."
    )


def cmd_find_gef(args):
    if any(a in ("-h", "--help") for a in args):
        print(_usage())
        return 0
    if len(args) != 1:
        _err("usage: asta-saw find-gef DIR")
        return 1
    gef, code, msg = _find_gef(args[0])
    if gef is None:
        _err(msg)
        return code
    print(gef)  # stdout: path only, so $(asta-saw find-gef DIR) works
    return 0


# ----------------------------------------------------------------------- count

def cmd_count(args):
    if any(a in ("-h", "--help") for a in args):
        print(_usage())
        return 0
    explicit, args = _pull_flag_value(args, "--saw")
    analyze = "--analyze" in args
    args = [a for a in args if a != "--analyze"]

    # Read (not remove) --output / --id: they still go to saw unchanged.
    output, _ = _pull_flag_value(args, "--output")
    run_id, _ = _pull_flag_value(args, "--id")
    if output:
        results = Path(output)
    elif run_id:
        results = Path.cwd() / run_id
    else:
        _err("ERROR: pass --id (or --output) so the results folder is known.")
        return 1

    saw = _find_saw(explicit)
    if not saw:
        _err(_not_found_message(explicit))
        return 1

    log_dir = Path.cwd() / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"{datetime.datetime.now():%Y%m%d_%H%M%S}_saw.log"
    print(f"running: {saw} count {' '.join(args)}\nlog    : {log_path}", flush=True)

    with open(log_path, "w") as log:
        proc = subprocess.Popen([saw, "count", *args], stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        for chunk in iter(lambda: proc.stdout.read1(4096), b""):
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
            log.write(chunk.decode(errors="replace"))
        rc = proc.wait()
    if rc != 0:
        _err(f"ERROR: saw count exited with status {rc} (see {log_path})")
        return 3

    gef, code, msg = _find_gef(results)
    if gef is None and output and run_id and (results / run_id).is_dir():
        # --output may be a parent directory that saw puts <id>/ inside.
        gef, code, msg = _find_gef(results / run_id)
    if gef is None:
        _err(msg)
        return code

    print(f"cell-bin GEF: {gef}")
    if not analyze:
        print(f'next: ST_INPUT_GEF="{gef}" asta-run all')
        return 0

    from .cli import _run_bundled_script
    sys.stdout.flush()
    _run_bundled_script("run_pipeline.sh", args=["all"],
                        extra_env={"ST_INPUT_GEF": str(gef)})
    return 0  # not reached: exec replaces this process


# ------------------------------------------------------------------------ main

def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(_usage())
        return 0
    sub, rest = argv[0], argv[1:]
    handlers = {"check": cmd_check, "count": cmd_count, "find-gef": cmd_find_gef}
    if sub not in handlers:
        _err(f"ERROR: unknown subcommand '{sub}'\n\n{_usage()}")
        return 1
    return handlers[sub](rest)


if __name__ == "__main__":
    sys.exit(main())
