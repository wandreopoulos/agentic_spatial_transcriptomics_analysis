# Deploying the Spatial Transcriptomics Pipeline for Automatic, Ongoing Analysis

This guide explains how to set up automated, ongoing analysis with new datasets for Maiqi Zhang's spatial transcriptomics analysis pipeline: drop a new `.gef` file into your **data folder** — any folder you choose, separate from wherever the code itself is installed — and the full pipeline runs on its own.

There are two ways to get this behavior, backed by **two separate watcher implementations** under the hood — here's why, and which one you want. Both are started through installed console commands (`asta-watch`, `asta-watch-launchd`), not by running a script file directly.

### Which watcher should I use?

If you're not sure, use `asta-watch`. It's the simpler, self-contained option and works everywhere.

| | `asta-watch` | `asta-watch-launchd` |
|---|---|---|
| **Platforms** | macOS, Linux, Windows | macOS only |
| **Extra install** | None (just the `watchdog` package, already a dependency) | OpenClaw (Node.js + `npm install -g openclaw`) |
| **How it's triggered** | Runs itself continuously in the foreground/background — `asta-watch`, watches for file changes directly | Triggered by macOS `launchd` (installed by `openclaw onboard --install-daemon`); runs the bundled `run_pipeline.sh` to actually run the stages |
| **On-demand triggering** | Not built in — you'd trigger a run by dropping a file, same as always | Browser chat dashboard (`openclaw dashboard`) and iMessage (text your Mac: "Run the full pipeline") |
| **Notifications** | Desktop notification (macOS only) + email (any OS, via `asta-notify`) | Desktop notification + email, same `asta-notify` |

Both do the same underlying job (detect a new `.gef` file, run the three pipeline stages, avoid duplicate/overlapping runs, notify when done) — `asta-watch-launchd` exists because it's what OpenClaw's automatic daemon is built on, giving you the dashboard/iMessage extras in exchange for macOS + Node.js + OpenClaw's setup. `asta-watch` exists so the core "drop and walk away" behavior doesn't require any of that. You don't need both — pick one based on whether you want the OpenClaw extras.

- **Option A: `asta-watch`** (recommended) — a cross-platform Python watcher included in this package. Works on macOS, Linux, and Windows with no extra tools beyond what's already installed with the package.
- **Option B: OpenClaw** (macOS only) — an external AI agent CLI that additionally provides a browser dashboard and iMessage triggering on top of the same automatic behavior. Requires Node.js and its own onboarding; only available on macOS (it installs `asta-watch-launchd` as a `launchd` daemon under the hood).

Both run the same three pipeline stages in sequence:

1. **Ensemble Clustering** — groups cells by gene expression similarity using K-means, Hierarchical, and GMM combined (~30-60 min)
2. **Biological Annotation** — identifies cell types across 8 ovarian cell types (~20 min)
3. **AI Agent Discovery** — autonomously generates and tests biological hypotheses (~10-20 min)

## Option A: `asta-watch` (Recommended, Cross-Platform)

### Prerequisites

- Python 3.11+ with the package installed — see the main [README](README.md) (`pip install -e .` for development, or `conda install`/`pip install` once published), which makes the `asta-watch` command available on your `PATH` and pulls in `watchdog`, the only extra dependency it needs
- Optional: an OpenAI API key (for the AI agent's LLM-based hypothesis generation — falls back to rule-based hypotheses without one)
- Optional: a Gmail account (for email notifications — desktop notifications and log files work regardless)

### Running It

```bash
conda activate spatial-transcriptomics   # or your venv
cd /path/to/your/data-folder             # or pass --path instead of cd'ing
asta-watch
```

Leave it running. Drop any `.gef` file into that data folder, and the pipeline starts automatically within seconds. You'll see progress logged to the terminal and to `logs/` (created inside the data folder), and get a notification (desktop + email, if `.env` is configured — see below) when it finishes.

The watcher is anti-loop safe — it will not re-run if the same file (by path + size) is dropped twice, or if a run is already in progress. Press Ctrl+C to stop it.

**Test it without waiting for a real run:** `asta-watch --once` processes any unprocessed `.gef` file already sitting in the folder once, then exits — useful for verifying the setup works.

### Keeping It Running in the Background

`asta-watch` needs to stay running to catch new files. Templates for running it as a background service are in `deploy/`:

- **macOS:** `deploy/com.sttx-watch.plist.template` — a `launchd` agent (same mechanism OpenClaw's daemon uses, but pointed at the installed `asta-watch` command directly, no OpenClaw/Node.js required). Copy it to `~/Library/LaunchAgents/`, fill in the placeholder paths (including the full path to `asta-watch` inside your conda env's `bin/`), then `launchctl load`.
- **Linux:** `deploy/sttx-watch.service.template` — a `systemd` user service. Copy it to `~/.config/systemd/user/`, fill in the placeholder paths, then `systemctl --user enable --now`.
- **Windows:** no template provided (systemd/launchd don't exist there) — either leave a terminal open running `asta-watch`, or set up a Task Scheduler task that runs `asta-watch` at logon with "Repeat task" disabled and "If the task is already running" set to "Do not start a new instance" (the script's own file-locking still applies as a second safety net).

### Environment Variables

Optional — create a `.env` file in the data folder for email notifications:

```bash
GMAIL_USER=your@gmail.com
NOTIFY_EMAIL_TO=your@gmail.com
GMAIL_APP_PASSWORD=your-app-password
OPENAI_API_KEY=your-key   # optional, enables LLM-based hypothesis generation
```

To get a Gmail App Password: Google Account → Security → 2-Step Verification → App Passwords.

## Option B: OpenClaw (macOS Only, Adds Dashboard + iMessage)

If you're on macOS and want the extra on-demand triggering channels (browser dashboard, iMessage) on top of the same automatic behavior, OpenClaw is an alternative to `asta-watch`.

### Prerequisites

- macOS (the auto-run watcher uses macOS launchd)
- Python 3.11+ (see the main [README](README.md) for the recommended conda setup via `environment.yml`)
- Node.js 22+
- An Anthropic or OpenAI API key
- A Gmail account (for email notifications)

> **Note:** This pipeline was tested using OpenClaw with **Claude (Anthropic)** as the AI provider, not with a local model runner such as Ollama. We recommend using an Anthropic API key with OpenClaw for the most reliable results.

### Installation

```bash
npm install -g openclaw@latest
openclaw onboard --install-daemon
```

During setup:
- Select your AI provider — **Anthropic (Claude) is what this pipeline was tested with**; other providers (e.g. Ollama) are not guaranteed to work the same way
- Select **ClickClack** as the channel to use the browser dashboard
- Skip optional skills and search providers

Then configure the same `.env` file described in Option A above.

### Running the Pipeline

**Automatic (Drop and Walk Away):** same as Option A — drop a `.gef` file into your data folder, and OpenClaw's `launchd` watcher (running `asta-watch-launchd`, which in turn calls the bundled `run_pipeline.sh`) picks it up within ~6 seconds. Configure OpenClaw's watcher to run `<env>/bin/asta-watch-launchd <DATA_DIR>` rather than a path inside a git clone.

**On Demand from the Dashboard:**

```bash
openclaw dashboard
```

Then type in the chat: `Run the full pipeline`

**On Demand from the Terminal:** if you'd rather not leave a browser tab open, OpenClaw also has a local text-based chat interface right in your terminal — no browser required:

```bash
openclaw chat
```

(`openclaw terminal` and `openclaw tui --local` are equivalent aliases.) Once it opens, type `Run the full pipeline` the same as you would in the dashboard. This talks to the same already-running gateway, so it works as long as OpenClaw is running in the background (e.g. via the `launchd` daemon from `openclaw onboard --install-daemon`) — you don't need to start anything else first.

**On Demand via iMessage:**

```bash
openclaw configure --section channels
```

Select iMessage, then text your Mac from your phone: `Run the full pipeline`

## Using a New Dataset

1. Obtain a `.gef` file output from the Stereo-seq Analysis Workflow (SAW)
2. Drop it into your data folder (the one `asta-watch`/`asta-watch-launchd` is watching)
3. The pipeline runs automatically and outputs results to `output/<run timestamp>/<stage>/` inside that data folder, with all three stages from one run sharing the same timestamp folder:
   - `output/<run timestamp>/ensemble/` — clustering results and plots
   - `output/<run timestamp>/annotator/` — cell type assignments
   - `output/<run timestamp>/agent/` — AI-generated hypotheses and discoveries

No code changes are needed between datasets, and results from different runs never overwrite each other.

## Output Files

| Folder | Key Files |
|--------|-----------|
| `output/<run timestamp>/ensemble/` | `ensemble_results.json`, `clustering_analysis.png`, `comprehensive_report.txt` |
| `output/<run timestamp>/annotator/` | `cell_type_assignments.csv`, `marker_detection_report.txt` |
| `output/<run timestamp>/agent/` | `autonomous_discoveries.json`, `autonomous_discovery_report.txt` |

## Sample Outputs

### Ensemble Clustering Performance
Comparison of silhouette scores, cluster counts, execution times, and improvement over individual methods. The ensemble (blue dashed line) achieves a silhouette score of 0.538 — outperforming all individual methods.

![Comprehensive Clustering Analysis](docs/images/comprehensive_clustering_analysis.png)

### Method Diversity & Quality
Shows how different the three clustering methods are from each other (left) and their quality metrics (right). Low similarity between methods (hierarchical vs k-means = 0.00) confirms the ensemble is combining genuinely diverse perspectives.

![Method Diversity and Quality](docs/images/method_diversity_quality.png)

---

### Biological Annotation: Overview
Summary dashboard showing spatial distribution of all 8 annotated cell types, confidence breakdown (High/Medium/Low/Unassigned), and marker gene detection rates. Out of 24,930 cells, 22,100 (88.6%) were successfully annotated across 8 cell types using 211 of 428 marker genes.

![Annotation Overview](docs/images/annotation_overview.png)

### Biological Annotation: Cell Type Map
Spatial scatter plot of all 24,930 cells colored by cell type across the tissue coordinate space (x: 7000–12000, y: 5000–9000). Shows the physical distribution of Granulosa Cells, Stromal Fibroblasts, Endothelial Cells, Oocytes, Theca Cells, and immune populations across the ovarian section.

![Cell Type Assignments](docs/images/cell_type_assignments.png)

### Biological Annotation: Confidence Analysis
Spatial heatmap of per-cell annotation confidence scores alongside a histogram of the score distribution. Mean confidence of 1.163 indicates reliable assignments; cells with low confidence are flagged for manual review.

![Confidence Analysis](docs/images/confidence_analysis.png)

### Biological Annotation: Anatomical Structures
Spatial plots of the 5 identified anatomical structures within the ovary: Primordial Follicles (3,816 cells), Primary Follicles (3,756), Secondary Follicles (3,756), Antral Follicles (3,583), and Corpus Luteum (4,580).

![Anatomical Structures](docs/images/anatomical_structures.png)

### Biological Annotation: Spatial Organization
Left: local cell density heatmap showing high-density tissue core regions (red) vs. sparse periphery (blue). Right: spatial centroids of each cell type, revealing how cell populations are physically partitioned across the tissue.

![Spatial Organization](docs/images/spatial_organization.png)

## Notes

- The first stage (ensemble clustering) is the most compute-intensive. Expect 30-60 minutes on a standard laptop.
- `ai_agent.py` uses an LLM for hypothesis generation. Set `OPENAI_API_KEY` in `.env` to enable this. It will fall back to rule-based hypotheses without a key.
- Keep `.env` out of version control. It is already listed in `.gitignore`.

## Based On

Maiqi Zhang, *Enhanced Autonomous Spatial Transcriptomics Analysis*, MS Thesis, San Jose State University, 2026. Advisor: Dr. William B. Andreopoulos.