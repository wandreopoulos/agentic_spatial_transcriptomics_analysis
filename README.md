# Enhanced Spatial Transcriptomics Analysis Pipeline

---

## Overview

This pipeline processes Stereo-seq spatial transcriptomics data through the following steps:

```
FASTQ Files → SAW Processing → GEF File → Analysis → Results
```

**Spatial Transcriptomics Analysis Framework Components:**
1. **Baseline** - Baseline results
2. **Ensemble Clustering** - Ensemble clustering result
3. **Biological Annotation** - Cell type identification with 248 markers
4. **Knowledge-Based Clustering** - Biology-guided analysis
5. **AI Agent** - Self-directed hypothesis testing

---

## Installation

**Hardware.** On the sample dataset (24,930 cells), `asta-run all` peaked at ~10 GB RAM and took ~23 min on a Linux x86_64 server; `asta-baseline` peaked at ~13 GB. Plan for 16 GB+ RAM.

### 1. Clone This Repository

```bash
git clone <this-repo-url>
cd agentic_spatial_transcriptomics_analysis
```

The code is an installable Python package named `asta`, under `src/asta/`. Installing it (step 4) gives you a set of `asta-*` commands that you can run from **any folder**, so you never need to run scripts out of the repository folder.

### 2. Install Conda (skip if you already have it)

Check first:

```bash
conda --version
```

If that command isn't found, install Miniconda:

- **macOS (Homebrew):** `brew install --cask miniconda`
- **macOS / Linux (official installer, no Homebrew):** https://docs.conda.io/en/latest/miniconda.html
- **Windows:** download the installer from the same page and run it.

After installing, restart your terminal (or run `conda init <your-shell-name>`, e.g. `conda init zsh`, then reopen your terminal) so the `conda` command is available.

### 3. Set Up the Environment (conda, recommended)

The pipeline's dependencies include compiled/native libraries (e.g. `h5py` links against HDF5), so **conda is the recommended way to install**, since it resolves those system-level dependencies automatically:

```bash
conda env create -f environment.yml
conda activate spatial-transcriptomics
```

This creates an isolated environment named `spatial-transcriptomics` with Python 3.11 and every dependency pinned to the exact tested version. **Use `environment.yml` if you want to reproduce the paper's numbers exactly.** The package itself (next step) only asks for minimum versions, so installing it on its own can pick up newer libraries and give slightly different results.

### 4. Install the Package

From the repository folder, with the environment activated:

```bash
pip install .
```

This installs the `asta` package and the `asta-*` commands listed below into your environment. Dependencies are already in place from `environment.yml`.

If you plan to edit the code, use an editable install instead, so changes take effect without reinstalling:

```bash
pip install -e .
```

<details>
<summary>Alternative: pip + venv</summary>

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# OR
.venv\Scripts\activate     # Windows

pip install -r requirements.txt
pip install .
```

Note: this does not manage non-Python system dependencies, so you may need HDF5 installed separately depending on your OS.
</details>

### 5. Verify Installation

```bash
# Check Python version (3.11+ required; environment.yml installs 3.11)
python --version

# Check that the commands are installed (run this from any folder)
asta-run --help
asta-ensemble --help
```

### The `asta-*` commands

| Command | What it does |
|---|---|
| `asta-baseline` | Baseline clustering (optional, for comparison) |
| `asta-ensemble` | Stage 1: ensemble clustering |
| `asta-annotate` | Stage 2: biological annotation |
| `asta-agent` | Stage 3: autonomous AI agent (needs `OPENAI_API_KEY` for the LLM part) |
| `asta-run [all\|ensemble\|annotator\|agent]` | Runs the stages in order, with logging and a notification when done (macOS/Linux) |
| `asta-watch [--path DIR] [--once]` | Watches a folder for new `.gef` files and runs the pipeline on them (macOS, Linux, Windows) |
| `asta-watch-launchd` | macOS-only watcher used by the OpenClaw/launchd workflow (exits with an error on Linux) |
| `asta-notify "subject" "body"` | Email/desktop notification (used by the watchers) |
| `asta-saw check \| count … \| find-gef DIR` | Optional bridge to STOmics SAW, see below |

Every command except `asta-notify` accepts `--help`. (`asta-notify` treats its first argument as the message subject and then waits for the body on standard input, so `asta-notify --help` appears to hang. Press Ctrl-C to cancel.)

### Where your data lives

The commands run on **your data folder**, which is the folder you run them from (or `--path DIR` for `asta-watch`, or `$ST_DATA_DIR` for `asta-run` and `asta-watch-launchd`). It is never the folder the package is installed in. In that folder:

- your `.gef` file is picked up automatically (or pass `--input` to a single stage, or set `ST_INPUT_GEF` for `asta-run`),
- `output/` and `logs/` are created there,
- an optional `.env` file is read from there (for `OPENAI_API_KEY`, `GMAIL_*` and the `ST_*` settings). Never commit this file. `asta-run` and `asta-watch-launchd` load it as a shell script (`source .env`), so values in `.env` override variables already set in your shell. `asta-watch` reads only plain `KEY=VALUE` lines, and there variables already set in your shell take priority over `.env`.
- `asta-watch` keeps its own record of processed files in `~/.sttx-watch/`

```bash
mkdir my_analysis && cd my_analysis
cp /path/to/your.gef .
asta-run all
```

Templates and docs for running the watcher as a background service (`deploy/`, `AI_AGENT_DEPLOY.md`) are in the repository.

### (Optional) Processing raw FASTQ files with SAW

**You only need this if you are starting from raw FASTQ files.** If you already have a cell-bin `.gef` file, skip this section and go to "Pipeline Steps" below.

Raw Stereo-seq reads are turned into a `.gef` file by **SAW**, STOmics' own pipeline. SAW is a separate third-party tool and is **not** installed with this package (it is a ~6.6 GB Linux-only binary). ASTA provides `asta-saw`, which finds an installed SAW, runs it for you, and hands the result to the analysis pipeline.

**Platform and requirements.** SAW runs on **64-bit x86_64 Linux only** (tested by STOmics on CentOS/RedHat 7.8 and Ubuntu 20.04) and needs 8+ CPU cores (24+ recommended), 128 GB+ RAM (256 GB+ recommended) and 1 TB+ free disk. It does not run on macOS. On a Mac, run SAW on a Linux server and copy the resulting `.gef` back to analyze it locally.

**Install SAW.** Download the current release from the [STOmics SAW Download Center](https://stereotoolss-organization.gitbook.io/saw-user-manual-v8.3/download-center), which also lists versions and checksums (they change, so they are not copied here), then unpack it:

```bash
mkdir -p ~/tools && tar -xzf saw-<version>.tar.gz -C ~/tools   # pick any location
find ~/tools/saw-<version> -name saw -type f    # prints the path to the saw executable
export SAW_BIN=/path/printed/above              # or add its folder to your PATH
asta-saw check                                  # finds SAW and prints its version
```

`asta-saw check` looks for SAW in this order: `--saw PATH`, `$SAW_BIN`, then `saw` on your `PATH`. It also compares your machine to SAW's minimum requirements (warnings only) and never downloads anything. The SAW [user manual](https://stereotoolss-organization.gitbook.io/saw-user-manual-v8.3) describes everything below in more detail.

**Chip mask and reference genome.** Each SAW run needs the chip mask for your chip's serial number (`saw support download --chip-mask-sn=<SN> --output=<dir>`, SAW 8.3+) and a reference genome index (a separate ~30 GB download, linked from the Download Center).

**Run SAW.** `asta-saw count` runs `saw count` with your arguments passed through unchanged, so use the [`saw count` options](https://stereotoolss-organization.gitbook.io/saw-user-manual-v8.2/analysis/pipelines/saw-commands) exactly as documented. Always pass `--id`: it names the results folder.

```bash
asta-saw count \
  --id=my_run --sn=<SN> --omics=transcriptomics \
  --kit-version="<kit version>" --sequencing-type="<type>" \
  --chip-mask=<chip-mask-file> \
  --fastqs=<fastq-folder> --reference=<reference-folder> \
  --image=<stain-image>
```

> **A stain image is required to get a cell-bin GEF.** The pipeline needs per-cell data, which only exists if cell segmentation ran. Pass `--image` (or `--image-tar`) and do not use `--skip-cellbin`. The `*.tissue.gef` / `*.raw.gef` files (square bins) and `*.protein.*.gef` files (Stereo-CITE protein) are not valid input.

SAW typically runs for hours. Its output is shown on screen and also saved to `logs/<timestamp>_saw.log`.

**Continue with ASTA.** When SAW finishes, `asta-saw` finds the RNA cell-bin GEF in `<id>/outs/feature_expression/` (preferring `*.adjusted.cellbin.gef`) and prints the next step. Either do it in one go:

```bash
asta-saw count --analyze <same arguments as above>    # runs SAW, then asta-run all on the GEF
```

or in two steps:

```bash
asta-saw find-gef my_run                              # prints only the GEF path
ST_INPUT_GEF="$(asta-saw find-gef my_run)" asta-run all
```

`asta-saw` exit codes: `1` setup problem (e.g. SAW not found), `2` no usable GEF found, `3` SAW itself failed. If several usable GEFs are found, `find-gef` lists them and asks you to pick one with `ST_INPUT_GEF`.

---

## Pipeline Steps

**Input:** a cell-bin `.gef` file in the folder you run from, either produced by SAW (above) or one you already have.

Public sample GEF file for mouse ovarian tissue: https://drive.google.com/file/d/1CsbvW46f0W-zSEl-7YmpzNPjbIINF4qG/view

---

### Run the analysis

Every script accepts `--input` and `--output` flags, so no source-code editing is required to point them at your own GEF file. There's no hardcoded filename to match — if you omit `--input`, each script resolves it in this order:

1. The `$ST_INPUT_GEF` environment variable, if set.
2. Otherwise, if there's exactly one `.gef` file in the current directory, that file is used automatically.
3. Otherwise (zero, or more than one, `.gef` file present with no explicit input given), the script exits with a clear error telling you to pass `--input` explicitly.

So for the common case — one dataset per folder — you usually don't need to pass `--input` at all. Run `asta-<command> --help` (e.g. `asta-ensemble --help`) for the full list of options on any command.

**Output goes to `output/<run timestamp>/<stage>/` by default** (e.g. `output/20260727_140512/ensemble/`) if you don't pass `--output` — so re-running the pipeline never silently overwrites a previous run's results, and everything one run produces lives together under a single timestamped folder. All stages run together via `asta-run` or `asta-watch` share one timestamp, so a full run's results all land under `output/<run timestamp>/ensemble/`, `output/<run timestamp>/annotator/`, and `output/<run timestamp>/agent/`. (`asta-baseline` is a standalone command outside the pipeline chain and still writes to its own `output/baseline/<run timestamp>/`.) Override with `--output some/path` (or `$ST_OUTPUT_DIR`) if you want a fixed location instead.

All commands below assume you are in your data folder, that your GEF file is at `./input_data.gef`, and that you've activated the environment where you installed the package.

**Shortcut:** rather than running each command individually, `asta-run` chains 3.2-3.4 (ensemble → annotation → agent) automatically in the required order. For fully automatic, ongoing analysis — drop a new `.gef` file into the data folder and it runs and notifies you when done, no manual invocation needed — run `asta-watch` (cross-platform: macOS, Linux, Windows) and leave it running; add `--path DIR` to watch a folder other than the current one. See [AI_AGENT_DEPLOY.md](AI_AGENT_DEPLOY.md) for full setup, including running it as a persistent background service. The individual commands below are for running or debugging a single stage on its own.

#### 3.1 Run Baseline Clustering (optional, for comparison)

`asta-baseline` is a standalone reference clustering method, not part of the required pipeline chain — its results are used only to benchmark how much the ensemble method (3.2) improves on a simpler baseline (see the Expected Results table below). It's not required to produce annotation or agent results.

```bash
asta-baseline --input input_data.gef

# Output: output/baseline/<run timestamp>/
# - recursive clustering results (JSON) and summary report
```

**Time:** ~10-15 minutes

#### 3.2 Run Ensemble Clustering

```bash
asta-ensemble --input input_data.gef

# Output: output/<run timestamp>/ensemble/
# - ensemble_results.json
# - clustering_analysis.png
# - comprehensive_report.txt
```

**Time:** 30-60 minutes (depending on cell count and computer). Use `--max-cells` to cap the number of cells analyzed for a faster run.

#### 3.3 Run Knowledge-Based Biological Annotation

```bash
asta-annotate --input input_data.gef

# Output: output/<run timestamp>/annotator/
# - cell_type_assignments.csv
# - marker_detection_report.txt
# - spatial_coherence_analysis.json
```

**Time:** ~20 minutes

#### 3.4 Run the Autonomous AI Agent (agentic analysis)

The AI agent generates and tests its own biological hypotheses using an LLM. This step is **optional** and separate from Steps 3.1-3.3, which run and produce full results with no LLM involved.

**Requirements:** an OpenAI API key **with available quota/billing enabled** — not just any key. If the key has no quota (e.g. a free-tier key with no credits, or a trial that's expired), the agent does not error out or warn loudly: it silently falls back to non-LLM (rule-based) hypothesis generation, and you'll only see the problem by opening the run's log and finding lines like `LLM hypothesis generation failed: ... insufficient_quota`. The pipeline still completes and produces results either way, but you only get the LLM-driven discovery behavior with a funded key. Set it either as an environment variable or a flag:

```bash
# Option A: environment variable
export OPENAI_API_KEY="sk-..."
asta-agent --input input_data.gef

# Option B: command-line flag
asta-agent --input input_data.gef --api-key "sk-..."

# Output: output/<run timestamp>/agent/
# - autonomous_discoveries.json
# - autonomous_discovery_report.txt
# - hypothesis_validation_results.json
```

**Time:** 10-20 minutes. Use `--max-iterations` to control how many hypothesis generate/test cycles the agent runs (default: 5).

For fully automated, ongoing deployment of the AI agent (e.g. auto-run on new datasets), see [AI_AGENT_DEPLOY.md](AI_AGENT_DEPLOY.md).

---

## Project Structure

```
agentic_spatial_transcriptomics_analysis/
├── README.md                          # This file
├── pyproject.toml                     # Package metadata and the asta-* commands
├── environment.yml                    # Conda environment spec (exact pins, recommended)
├── requirements.txt                   # Pinned pip dependencies (fallback)
├── LICENSE
│
├── src/asta/                          # The installable package
│   ├── baseline.py                    # asta-baseline  (Step 3.1)
│   ├── ensemble_clustering.py         # asta-ensemble  (Step 3.2)
│   ├── annotator.py                   # asta-annotate  (Step 3.3)
│   ├── ovarian_knowledge_base.py      # Marker/pathway knowledge base (imported by annotator.py)
│   ├── ai_agent.py                    # asta-agent     (Step 3.4, agentic analysis)
│   ├── run_pipeline.sh                # asta-run: runs 3.2-3.4 in order (macOS/Linux)
│   ├── watch_runner.py                # asta-watch: cross-platform automatic watcher
│   ├── watch_runner.sh                # asta-watch-launchd: macOS/OpenClaw watcher
│   ├── notify.py                      # asta-notify: desktop/email notifications
│   ├── saw.py                         # asta-saw: bridge to an installed SAW
│   └── cli.py                         # Entry points for the commands above
├── deploy/                            # Background-service templates (launchd, systemd)
├── tests/                             # Tests (need no data, network or API key)
│
Your data folder (any folder you run the commands from):
├── input_data.gef                     # Your GEF file (not included; see "Pipeline Steps" above)
├── .env                               # Optional: OPENAI_API_KEY etc. (never commit)
├── logs/                              # Run logs
├── output/                            # All results
│   ├── <run timestamp>/{ensemble,annotator,agent}/
│   └── baseline/<run timestamp>/
```

---

## Dependencies

See `environment.yml` (conda, recommended) or `requirements.txt` (pip) for the exact pinned versions used and tested:

```
h5py==3.14.0
llvmlite==0.43.0
matplotlib==3.9.4
numba==0.60.0
numpy==2.0.2
openai==2.43.0
pandas==2.3.3
pynndescent==0.6.0
scikit-learn==1.6.1
scipy==1.13.1
seaborn==0.13.2
umap-learn==0.5.12
```

Requires Python 3.11+; `environment.yml` installs 3.11. The package itself (`pip install .`) declares only minimum versions of these libraries, so use `environment.yml` when you need to reproduce the published numbers exactly.

---

## Expected Results Summary (number might change slightly for each run)

### Spatial Transcriptomics Analysis Framework Performance:

| Component | Metric | Value | vs. Baseline |
|-----------|--------|-------|--------------|
| **Ensemble** | Silhouette | 0.540 | +56.8% |
| **Ensemble** | Methods | 3 combined | N/A |
| **Ensemble** | Time | 3,094s | - |
| **Annotation** | Success Rate | 88.6% | ∞ (0%→88.6%) |
| **Annotation** | Cell Types | 10 | N/A |
| **Agent** | Hypotheses | 3 | 100% validated |
| **Agent** | Confidence | High | 3/3 discoveries |

---

## Visualizations

### Ensemble Clustering

![Comprehensive Ensemble Clustering Analysis](docs/images/comprehensive_clustering_analysis.png)
*Silhouette scores, cluster counts, execution time, and ensemble improvement across kmeans, DBSCAN, hierarchical, and Gaussian mixture methods.*

![Method Diversity and Quality](docs/images/method_diversity_quality.png)
*Method similarity (ARI) and quality metrics (Silhouette, Calinski-Harabasz, Davies-Bouldin) across clustering methods.*

### Biological Annotation

![Annotation Overview](docs/images/annotation_overview.png)
*Cell type spatial distribution, proportions, assignment quality, confidence scores, and marker gene detection across 24,930 analyzed cells.*

![Cell Type Assignments](docs/images/cell_type_assignments.png)
*Spatial map of assigned cell types across the tissue section, including granulosa cells, oocytes, theca cells, stromal fibroblasts, and immune populations.*

![Confidence Analysis](docs/images/confidence_analysis.png)
*Spatial and distribution view of assignment confidence scores.*

### Anatomical & Spatial Structure

![Anatomical Structures](docs/images/anatomical_structures.png)
*Identified anatomical structures including primordial, primary, secondary, and antral follicles, and corpus luteum structures.*

![Spatial Organization](docs/images/spatial_organization.png)
*Local cell density map and cell type centroids across the tissue.*

---

## AI Agent: How It Thinks (Example Hypotheses)

The AI agent doesn't just cluster cells — it proposes an idea about the biology, tests it statistically, and then explains what the result might mean, in plain language. Two examples of what this looks like in practice:

**Example 1: Cell subpopulations**

- *Hypothesis generated:* "Cells cluster into transcriptionally distinct groups that may represent different cell types or states."
- *What the agent did:* Ran PCA + K-means clustering and checked how well-separated the resulting groups were (silhouette score ≈ 0.40, high confidence).
- *Biological explanation:* This pattern is consistent with the tissue containing several different, specialized cell types — for example granulosa cells, theca cells, and oocytes in an ovarian follicle — each with its own distinct gene activity "fingerprint" rather than one uniform cell population.

**Example 2: Localized anomaly / signaling dysregulation**

- *Hypothesis generated:* "The detected anomaly in the dataset corresponds to a localized dysregulation of a specific signalling pathway that drives aberrant cellular behaviour in a subset of cells."
- *What the agent did:* Flagged a small group of cells whose expression didn't match the surrounding tissue, then ran anomaly detection and characterization on that subset.
- *Biological explanation:* A handful of cells are behaving differently from their neighbors, which could point to a signaling pathway (the chemical messages cells use to coordinate) switching on or off abnormally in that spot — for instance, in a follicle undergoing degeneration or a region responding to local stress.

In short: the **Hypothesis Generator** proposes the "what if," the **Analysis Planner/Executor** tests it with the right statistics, and the **Result Interpreter** translates the numbers back into a biological story a reader can follow.

---

## License

Spatial Transcriptomics Analysis Enhanced Framework - released under the [MIT License](LICENSE). Freely usable, modifiable, and redistributable, including for commercial purposes, provided the copyright notice is retained.

Note: this license covers the analysis scripts in this repository only. SAW (used for FASTQ processing, see "Processing raw FASTQ files with SAW" above) is a separate third-party tool — check the [STOmics/SAW repository](https://github.com/STOmics/SAW) for its license terms.

---

For instructions on how to install an AI agent for automatic running of the pipeline and generating and testing hypotheses, please see [AI_AGENT_DEPLOY.md](AI_AGENT_DEPLOY.md).

## Based On

Maiqi Zhang, *Enhanced Autonomous Spatial Transcriptomics Analysis*, MS Thesis, San Jose State University, 2026. Advisor: Dr. William B. Andreopoulos.

---

This file was written and maintained by Minjun Roe.