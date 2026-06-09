# PROJECT FACTS — Spotify Engagement from Metadata

Living, agent-maintained roll-up of empirical knowledge: the leaderboard,
noise bands, dataset lineage, per-family field guide and hard-won pitfalls.
The campaign reports in experiments/*.md are PRIMARY; this file is their
index. When they disagree, the newest report wins — and this file is stale
and must be updated. Writers: the experiment-runner agent and the
model-definitions experiment workflow reconcile this file as part of every
campaign report.

Last updated: 2026-06-08 after the bootstrap first-run — the metadata-only
baseline trio (baseline-median / ridge / xgboost) on the canonical p1 dataset.
xgboost is first-on-record (MAE 2.424, R² 0.437). No campaign report yet; these
are the seed runs.

## Best on record (leaderboard)

| model | dataset | MAE | RMSE | R² | source |
|---|---|---|---|---|---|
| xgboost (metadata-only; auto-promoted `best-xgboost-20260608-161931-33428`) | ds-20260608-161709-p1-s42 | 2.424 | 7.764 | 0.437 | bootstrap first-run |
| baseline-median (group-median floor) | ds-20260608-161709-p1-s42 | 3.024 | 10.625 | −0.054 | bootstrap first-run |
| ridge (auto_alpha) | ds-20260608-161709-p1-s42 | 3.074 | 10.100 | 0.047 | bootstrap first-run |

Read: xgboost beats the median floor by ~20% MAE (R² 0.437) — intrinsic song
metadata carries real engagement signal. Ridge barely clears the floor (R²
0.047, MAE actually a hair above baseline), so the metadata→engagement signal
is **non-linear / interaction-heavy** — trees exploit it, a linear model does
not. Single split, single seed each; not yet noise-characterized (see below).

## Noise bands & significance thresholds

_Not yet measured. Before trusting any single-run MAE delta on this corpus,
measure σ_train (xgboost internal-seed spread, split fixed) and σ_split
(data-split-seed spread) on ds-20260608-161709-p1-s42 — the noise-band study is
the natural first campaign. Until then, treat MAE differences below ~0.2 as
noise by analogy to similar corpora, and confirm with a multi-split repeat._

## Dataset lineage

| dataset (recipe) | feeds | notes |
|---|---|---|
| ds-20260608-161709-p1-s42 (pca=1, metadata-only: playback + curation groups OFF; nonpositive-engagement excluded) | bootstrap baseline trio (baseline-median / ridge / xgboost) | 12,106 rows (12,256 corpus − 150 engagement≤0); 9,685 train / 2,421 test @ seed 42; 115 cols = 1 PCA embedding PC + 8 metadata numerics/indicators + genre(41) + artist(61) + album_type(5); target engagement=play_count×completion_ratio (log1p). pca_dims=1 is the framework FLOOR (the build always includes ≥1 embedding PC; a zero-embedding matrix is not natively supported — see pitfalls) |

## Known pitfalls per predictor family

Generic priors (replace with measured facts as campaigns land):

- Bounded-by-construction families (trees: leaf averages; rbf kernels:
  revert-to-intercept) cannot blow up through a log-target inverse;
  unbounded ones (linear models, MLPs) can — ship MLPs with
  `clamp_output: true` until proven safe.
- Features usually beat architecture: scan dataset-level axes before deep
  hyperparameter scans.

## Known pitfalls per feature / dataset axis

- **This project predicts engagement from INTRINSIC song metadata only.** The
  canonical build keeps the `playback` group (play_count, completion_ratio,
  skip_rate, shuffle_rate, distinct_sessions) and the `curation` group
  (is_saved, in_playlist_count, on_repeat_count) OFF — the first two playback
  fields literally compose the target, and curation is behaviorally entangled
  with engagement. Turning any of them on is a deliberate sanity-ceiling /
  soft-leakage control build, never the default; if R² jumps toward ~0.8+ when a
  group is enabled, suspect leakage.
- **Content-text embedding tested and REJECTED (2026-06-08).** Built an
  alternative corpus `spotify_tracks_content` — a multilingual sentence
  embedding (paraphrase-multilingual-MiniLM-L12-v2, 384-dim) of each track's
  intrinsic-metadata doc ("title. Artist: X. Album: Y (type, year). Genre: Z.")
  via `pipeline/embed_content.py` — to replace the sibling's behavioral
  co-listening vector with a non-behavioral one. It did NOT help and a pca-dims
  scan got monotonically WORSE: xgboost content+one-hots MAE 2.752 (p16) →
  2.784 (p64) → 2.830 (p128) vs the **2.424 one-hot baseline**. Ablation:
  content embedding ALONE (one-hots off, p128) = MAE 2.964 / R² 0.144, barely
  above the no-feature median floor (R² −0.05) and far below one-hots (R²
  0.437). Verdict: engagement is driven by artist/genre **identity** (which the
  top-60/top-40 one-hots encode losslessly), not by semantic similarity of the
  metadata text; the embedding blurs identity and adds noisy dims trees overfit
  (train-loss 0.50 vs val 0.70). Lesson for this corpus: dense vectors (whether
  behavioral co-listening or content-text) lose to categorical identity. The
  evidence-backed levers are instead (a) RAISE vocab caps so the dominant
  `artist=__other__` tail (currently the #1 feature, 7.1% gain) gets real
  columns, and (b) acoustic features/embeddings (what the song *sounds* like —
  the only axis orthogonal to identity), if obtainable.
- **Acoustic audio features tested and REJECTED (2026-06-08).** Fetched the 11
  Spotify-style audio features (danceability/energy/valence/tempo/acousticness/
  instrumentalness/loudness/speechiness/liveness/key/mode) from **ReccoBeats**
  (free non-Spotify drop-in for the deprecated Spotify endpoint), keyed by
  `track_uri` Spotify ids — see `pipeline/fetch_audio_features.py`. Coverage
  64.8% (7,938/12,256; misses → `af_*_missing` indicators). Written as numeric
  fields into `spotify_tracks_content` payloads. Controlled test (same
  collection + same seed-42 split, pca=1): audio OFF MAE **2.684** / R² 0.298
  vs audio ON MAE **2.775** / R² 0.260 — audio features made it slightly WORSE.
  xgboost used them (7.3% total gain, ~0.5% each) but they diluted the
  artist/genre one-hots. Verdict: acoustic character does not add engagement
  signal for this user. NOTE: cross-collection numbers are NOT comparable
  (different scroll order ⇒ different split) — the original 2.424 baseline was
  on `spotify_tracks`; the honest within-collection control here is 2.684.
- **CONVERGENT FINDING (3 modalities): engagement is predicted by artist/genre
  IDENTITY, not by descriptive song attributes.** Co-listening embedding
  (~neutral), content-text embedding (hurt), acoustic features (hurt) all fail
  to beat plain one-hots. `artist=__other__` is consistently the #1 feature.
  Implication: the metadata-only ceiling (R²≈0.3–0.44, split-dependent) is set
  by identity; the residual is behavioral/contextual (mood, time, discovery)
  that no song metadata captures. The remaining lever is identity RESOLUTION —
  raise vocab caps so the `__other__` tail gets real columns — NOT more feature
  modalities. A noise-band study is still needed to know which deltas here are
  even significant (the 2.684↔2.775 gap may be within xgboost's seed noise).
- The 200-dim co-listening embedding is a BEHAVIORAL vector, not song metadata —
  the payload-based predictors (xgboost, lightgbm, random-forest, ridge, svm)
  are the canonical models here; embedding-based predictors would reintroduce
  behavioral signal.
- **Framework floor: every dataset carries ≥1 PCA embedding column.** The build
  rejects `pca_dims < 1` (server `api.rs`; `pca.rs` requires `k>=1`) and all
  predictors read the combined `features.f32` (embedding block + metadata) —
  there is no native zero-embedding / items.json-only feature path. The
  canonical build therefore uses `pca_dims=1` to reduce the behavioral channel
  to a single principal component. A truly metadata-only matrix would require a
  framework change (allow `pca_dims=0` or an embedding-exclude flag); do NOT
  improvise it in the crates.
  - **MEASURED (bootstrap first-run, do not assume):** that 1 PC holds 13.8% of
    the 200-dim embedding's variance (it's the max-variance direction, so *a
    priori* not negligible), yet xgboost gave `pca_0` only **0.98% of total
    gain, rank 27 of 111 used features** — the model is carried by `artist` and
    `genre_primary` one-hots. So on this split the behavioral PC contributes
    little and the metadata-only thesis holds — but this is one PC at one split.
    A pca-dims scan (1 → 16 → 64 → 128, metadata fixed) is the definitive test:
    flat MAE ⇒ thesis confirmed; a sharp drop ⇒ the embedding is carrying
    behavioral signal and we'd be drifting away from "metadata only."

## Taste-fit reframe for Pathfinder (2026-06-08)

The model's ultimate consumer is the **Pathfinder** A* search, where it supplies
the `W_FIT` term: `cost(u→v) += W_FIT·(1 − fit_norm(v))` — a per-track [0,1]
desirability that biases the playlist path. `pathfinder/score.py` calls
`POST /api/models/<MODEL_NAME>/predict {point_ids}` and min-max normalizes;
it currently log-normalizes because engagement is heavy-tailed.

**Reframe:** model **"enters rotation" = play_count ≥ 2** (binary, 44.5% base
rate) instead of engagement magnitude. On the one-hot feature set, xgboost gets
**AUC 0.765 / acc 0.715** — a genuinely useful taste-fit ranker, vs engagement
regression's R²≈0.3. And the predicted probability is naturally spread
(p10/p50/p90 = 0.16/0.41/0.81), so it's a cleaner Pathfinder `fit_norm` with no
log-hack. This is the better-posed target for the Pathfinder goal: replay
*magnitude* is contextual noise, but the *in/out* decision tracks taste-fit,
which the features capture. (Engagement stays the documented scientific target;
taste-fit is the actionable signal that feeds the path search.)

**Pivot executed (2026-06-08):** domain `[target]` = `rotation` (transform none),
engagement demoted to a documented non-target field; `rotation` (play_count>=2)
written to `spotify_tracks_song_ae` payloads; server repointed to that AE
collection (so Pathfinder's graph = the rich song-character manifold). Rotation
xgboost on AE-collection features: **AUC 0.733, acc 0.690, spread 0.16/0.41/0.80**
(predictions saved; AUC computed offline). Build with
`quality.nonpositive_price = FALSE` (rotation==0 is the negative class, not an
outlier). Artifacts: ds-20260608-204402-p64-s42 / run-20260608-204415-6eeda-xgboost.

**FRAMEWORK LIMITATION SURFACED — binary targets break the metric pipeline.**
The predictors + lensing-core compute MAPE/medAPE = |err/actual|, undefined when
actual==0 (half the rotation labels). The run TRAINS fine (exit 0, good
predictions) but is marked `failed` on the NaN metric, blocking auto-promote →
`/api/models/<name>/predict` → Pathfinder's `score.py`. Native classification
isn't supported. Per the ground rules (crates/predictors shouldn't be hacked),
resolution is a DECISION, not an improvisation:
  (a) minimal upstream fix: make the % metrics zero-safe (skip actual==0) so
      binary runs succeed + promote normally — contribute upstream; OR
  (b) self-contained Pathfinder scorer: load the rotation model.ubj, score corpus
      point_ids, feed `score.py` directly (bypass the metric/promotion path).

## Best-models group curation (best-model-selector agent)

Additive log of judgment-pass overrides applied via `PUT /api/best-models`
(the deterministic top-12-by-MAE recompute can't apply these). Newest first.

_No curation overrides yet — the deterministic top-12-by-MAE recompute stands
until the first campaign warrants a judgment pass._
