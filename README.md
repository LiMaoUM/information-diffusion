# Information Diffusion: Truth Social vs. Bluesky

Computational social science project comparing structural diffusion of political
content on two ideologically divergent platforms, Truth Social and Bluesky,
during the 2024 U.S. presidential campaign (posts mentioning Biden or Trump,
May to June 2024).

Paper: **"Depth, Breadth, and Bias: Structural Diffusion of Political Content on
Divergent Platforms"**, under revise-and-resubmit at ICWSM 2027 (revision due 2026-09-15)
(`paper/AnonymousSubmission/LaTeX/`).

Key finding: repost (amplification) cascades scale similarly across platforms,
but reply (conversational) cascades diverge, wide and shallow on Truth Social,
narrow and deep on Bluesky. The divergence is explained only when ideological
composition and cross-ideology alignment are jointly modeled. Motif analysis
shows asymmetric left/center/right interaction roles.

## Repository layout

| Path | Contents |
|---|---|
| `paper/` | Paper: anonymous submission LaTeX, restructured draft, camera-ready template, figures, copyright forms |
| `figures/` | All generated manuscript figures (scaling, CCDFs, motifs, regression diagnostics) |
| `src/collection/bsky_api.py`, `bsky_follows.py` | Bluesky data collection (threads, reposts, profiles, follow graph) |
| `src/analysis/cascade_analysis.py` | Core library: `InformationCascadeGraph` builds repost/reply cascade DiGraphs for both platform schemas |
| `src/analysis.ipynb`, `src/analysis/analysis.ipynb` | Cascade construction and metrics (size, depth, breadth, structural virality); the `analysis/` copy also writes `parquet_out/` |
| `src/comparison_reply_combined.ipynb` | Main cross-platform reply-cascade comparison (alignment ratios, topic and ideology distributions) |
| `src/matching.ipynb` | LexRank anchor-post selection; reads from sibling repo `../echo-chamber/` (external dependency) |
| `src/modeling/model_R.R` | Robust regression (lmrob/rlm/GLS) of log breadth on log size by platform |
| `src/story.py` | Marimo notebook, reply-network narrative and KS tests |
| `src/eval_topmod/` | BERTopic evaluation: LLM labeling and inter-coder agreement |
| `src/motif/` | CUDA motif counting (`motif_count_node.cu`, 27 chain + 27 star ideology-typed 3-node motifs, null-model z-scores); build with `src/motif/src/motif.sh` |
| `results/` | Small derived result tables (binned breadth/depth, regression frame, anchor outputs) |
| `data/` (git-ignored) | ~45 GB raw platform data. PII-bearing, never commit or publish |
| `data/interim/` (git-ignored) | Archived unreferenced intermediates (old repost edgelists) |
| `models/` (git-ignored) | Trained BERTopic artifacts |

## Pipeline

1. Collect posts, reposts, threads, follows (`src/collection/bsky_api.py`, Truth Social equivalents).
2. Build cascade graphs and metrics (`cascade_analysis.py`, analysis notebooks).
3. Topic modeling (BERTopic, evaluated in `src/eval_topmod/`) and ideology labeling.
4. Regression modeling in R (`src/modeling/model_R.R`).
5. Motif counting on GPU (`src/motif/`).

## Environment

Python dependencies are pinned in `requirements.txt` (frozen from the analysis
venv). Use `uv` to recreate: `uv venv && uv pip install -r requirements.txt`.
The motif counter needs CUDA (`nvcc`).

## Data availability

Raw data in `data/` contains user-identifiable content and follow graphs and is
excluded from git. A de-identified derived dataset is planned for Zenodo; see
`src/analysis/README.md` (draft, placeholders unfinished).
