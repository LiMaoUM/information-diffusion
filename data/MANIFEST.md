# Data manifest (tracked; everything else in data/ is git-ignored)

Updated 2026-07-18. One line per asset so nobody has to re-derive what a file
is. PII rule: nothing in this directory is ever committed or published.

## Raw platform data (collection: May 30 to Jun 30, 2024 window)

| File | What it is |
|---|---|
| `ts_threads_withReblogs.json` (8.7 GB) | Truth Social posts/threads incl. reblogList (account objects, list is time-ordered, no per-reblog timestamps) |
| `ts_threads.json` | Earlier TS thread crawl without reblog lists |
| `bsky_threads.json`, `bsky_reposts.json`, `bsky_reposts_new.json` (1.6 GB), `bsky_reposts_newReposts.json` | Bluesky threads and repost crawls; `_new` is the analysis version (reposts = profile objects, time-ordered, no per-repost timestamps) |
| `bsky_follows.json` (23 GB) | Bluesky follow crawl, profile objects. Superseded for analysis by `bsky_followings.json` |
| `bsky_followings.json` (1.5 GB) | Bluesky follow map, id-only {did: [followee dids]}. USE THIS |
| `ts_user_following_map.json` (2 GB) | TS follow map {user id: [followee ids]} |
| `bsky_author_info.json` | Bluesky author profiles |
| `topic_corpus/` | Post text corpora for topic modeling |

## Labels and model outputs

| File | What it is |
|---|---|
| `bsky_post_to_label.json` (2.2M), `ts_post_to_label.json` | LLM (Llama-3.1-8B) five-category stance per post text: left / lean left / center / lean right / right |
| `bsky_author_ideology_portion.json`, `ts_author_ideology_portion*.json` | Per-author label proportions (user ideology derives from these via threshold) |
| `*_df_id_topic.csv`, `topic_df.csv`, `custom_topic_df.csv` | BERTopic assignments |
| `*_reply_stats*.csv`, `*_repost_stat*.csv`, `*_combined_stats.csv` | Cascade-level statistics; `_06` / `_06_new2` variants are canonical (0.6 ideology threshold), `_05`..`_09` are threshold sweeps |
| `ts_follower_outliers.csv` | The 8 TS influencer accounts (H1 cutoff) |

## Human validation assets (three DISTINCT things; do not conflate)

| File | What it is |
|---|---|
| `eval_bsky_chen.xlsx`, `eval_bsky_Li.xlsx`, `eval_ts_chen.xlsx`, `eval_ts_Li.xlsx` | TOPIC validation: 224 (bsky) + 257 (ts) cascades, coders Chen and Li assign Primary/Secondary topic labels. Appendix topic kappas (0.53 / 0.67 / 0.67) come from these via `src/eval_topmod/eval.ipynb` |
| `topic_eval_*_sample*.csv` | Topic eval samples incl. llama3-70b LLM labeling comparison |
| `annotation_bsky_ts.xlsx` | USER-level ideology annotation, 295 users, single annotator session 2025-12-14, options Left/Center/Right + undecided, thread-context prompts. Copied from Google Drive (file id 1TsGmieCddZBWgYTv2cdWHU3_wWT3pPZd) on 2026-07-18. Provenance/purpose to confirm with Mao |
| MISSING: 200-reply stance validation | The paper appendix's "two authors, 200 replies, five-category scheme" (kappa 0.78 human-human, 0.64/0.73 model-human). NOT FOUND in repo or Drive as of 2026-07-18. Needed for R&R per-class validation. Location unknown |

## Figure assets

`Chain_*.png`, `Star_*.png`: motif diagram images used to compose motif figures.

## Coder key

"Li" = Mao Li, "chen" = second annotator (Chen). The xlsx files contain coder
identities; do not publish raw.
