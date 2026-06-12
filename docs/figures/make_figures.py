#!/usr/bin/env python3
"""Generate the vector-PDF figures for docs/experiments.tex.

All data is transcribed from the experiment reports in experiments/.
Each figure cites its source report in a comment; one fig_<name>()
function per figure. Run from anywhere: outputs land next to this script.
"""

from pathlib import Path

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

OUT = Path(__file__).resolve().parent

# Consistent style
plt.rcParams.update(
    {
        "figure.figsize": (6.0, 3.5),
        "figure.constrained_layout.use": True,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
    }
)

ACCENT = "#2563eb"  # blue
GOOD = "#16a34a"  # green
BAD = "#dc2626"  # red
MUTED = "#9ca3af"  # gray

# Domain style header — keep axis units/labels going through these instead
# of hardcoding the unit per figure (values mirror [metrics] in domain.toml).
PRIMARY_METRIC = "MAE"
VALUE_UNIT = "engagement"
VALUE_AXIS = f"{PRIMARY_METRIC} ({VALUE_UNIT})"
kfmt = FuncFormatter(lambda v, _: f"{v / 1000:.0f}k")

# Rotation is the active target: a binary [0,1] taste-fit signal scored by
# AUC (classification) — its values are O(1) probabilities, so a plain 3-dp
# formatter reads them rather than the thousands-separator engagement kfmt.
afmt = FuncFormatter(lambda v, _: f"{v:.3f}")


def save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)
    print(f"wrote {name}")


# Figures are added by the report-curator agent, one fig_<name>() per
# figure, each citing its source report.


# Fig 1: the AE-latent standalone signal washes out as the kNN neighborhood
# widens, and never reaches the metadata-only tree
# (2026-06-09-ae-xgboost-blend.md, "The mae_ae(k) curve").
def fig_ae_latent_knn():
    k = [10, 25, 50, 100, 200]
    mae_ae = [0.4259, 0.4403, 0.4510, 0.4609, 0.4702]
    ridge = 0.4623  # ridge-on-latent arm, between k=100 and k=200
    meta_xgb = 0.3879  # metadata-only XGBoost (the branch the blend collapses to)
    ae_only_xgb = 0.4098  # xgboost on AE-latent only (64 cols), on record

    fig, ax = plt.subplots()
    ax.plot(k, mae_ae, "o-", color=ACCENT, label="AE-latent kNN ($mae_{ae}$)")
    ax.axhline(ridge, color=MUTED, ls=":", lw=1.4, label=f"ridge-on-latent ({ridge:.3f})")
    ax.axhline(
        ae_only_xgb, color=MUTED, ls="--", lw=1.2,
        label=f"AE-latent-only xgboost ({ae_only_xgb:.3f})",
    )
    ax.axhline(
        meta_xgb, color=GOOD, ls="-", lw=1.6,
        label=f"metadata-only XGBoost ({meta_xgb:.3f})",
    )
    ax.set_xscale("log")
    ax.set_xticks(k)
    ax.set_xticklabels([str(v) for v in k])
    ax.set_xlabel("kNN neighborhood size $k$")
    ax.set_ylabel("MAE (P(rotation))")
    ax.set_title("The song-AE latent is a dead modality: even its best read\nnever reaches the metadata tree")
    ax.legend(loc="center right", fontsize=7.5)
    save(fig, "ae_latent_knn.pdf")


# Fig 2: on a binary target, low SVR MAE is an epsilon-tube artifact — the
# honest R² says every kernel is worse than the mean
# (2026-06-09-svm-kernel-rotation.md, Results + Finding 3).
def fig_svm_mae_vs_r2():
    labels = ["xgboost\n(champion)", "svm\nrbf", "svm\nlinear", "svm\npoly", "svm\nsigmoid"]
    mae = [0.3879, 0.3847, 0.3901, 0.3814, 0.6240]
    r2 = [0.191, -0.130, -0.198, -0.137, -8.479]
    colors = [GOOD, BAD, BAD, BAD, BAD]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 3.3))
    x = np.arange(len(labels))

    ax1.bar(x, mae, color=colors)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=7)
    ax1.set_ylabel("MAE (P(rotation))")
    ax1.set_title("MAE: SVR looks competitive\n(the $\\epsilon$-tube artifact)")
    ax1.axhline(0.3879, color=MUTED, ls=":", lw=1.0)

    ax2.bar(x, r2, color=colors)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=7)
    ax2.set_ylabel("$R^2$")
    ax2.axhline(0.0, color="black", lw=1.0)
    ax2.set_title("$R^2$: every SVR kernel is\nworse than the mean")
    # sigmoid R^2 = -8.48 would flatten the scale; clip the view and annotate.
    ax2.set_ylim(-0.6, 0.3)
    ax2.annotate(
        "sigmoid $R^2=-8.48$\n(off scale)", xy=(4, -0.55), xytext=(2.4, -0.45),
        fontsize=7, color=BAD, ha="left",
        arrowprops=dict(arrowstyle="->", color=BAD, lw=0.8),
    )
    save(fig, "svm_mae_vs_r2.pdf")


# Fig 3: the honest AUC bake-off — xgboost-classifier beats every dense
# distance/margin family by many noise bands
# (2026-06-09-rotation-classification.md, Results + Findings).
def fig_auc_bakeoff():
    labels = [
        "xgboost-classifier\n(champion)",
        "svc poly",
        "logistic",
        "svc rbf",
        "svc linear",
        "svc sigmoid",
    ]
    auc = [0.7429, 0.6855, 0.6762, 0.6722, 0.6363, 0.6255]
    colors = [GOOD] + [MUTED] * 5
    band = 0.0126  # 2-sigma AUC noise band, measured this campaign

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    y = np.arange(len(labels))[::-1]
    ax.barh(y, auc, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("AUC (P(rotation))")
    ax.set_xlim(0.5, 0.78)
    ax.xaxis.set_major_formatter(afmt)
    ax.axvline(0.5, color="black", lw=1.0, label="chance (0.5)")
    # band envelope around the champion
    ax.axvspan(0.7429 - band, 0.7429, color=GOOD, alpha=0.12)
    for yi, a in zip(y, auc):
        ax.text(a + 0.003, yi, f"{a:.3f}", va="center", fontsize=7.5)
    ax.set_title("Honest AUC: the tree beats every dense family by 4.6--9.3 noise bands")
    ax.legend(loc="lower right", fontsize=7.5)
    save(fig, "auc_bakeoff.pdf")


# Fig 4: the rotation AUC noise band — three split seeds of the champion
# config, with the 2-sigma envelope
# (2026-06-09-rotation-classification.md, decision rule step 2).
def fig_auc_noise_band():
    seeds = ["s42", "s17", "s101"]
    auc = [0.742934, 0.730659, 0.734469]
    mean = 0.736020
    sigma = 0.00628
    band = 0.01257  # 2 sigma

    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    x = np.arange(len(seeds))
    ax.axhspan(mean - band, mean + band, color=ACCENT, alpha=0.12, label="$\\pm 2\\sigma$ band (0.0126)")
    ax.axhline(mean, color=ACCENT, ls="--", lw=1.2, label=f"mean {mean:.4f}")
    ax.plot(x, auc, "o", color=GOOD, markersize=9)
    for xi, a in zip(x, auc):
        ax.text(xi, a + 0.0013, f"{a:.4f}", ha="center", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(seeds)
    ax.set_xlabel("data-split seed")
    ax.set_ylabel("AUC (P(rotation))")
    ax.yaxis.set_major_formatter(afmt)
    ax.set_ylim(mean - 2.2 * band, mean + 2.2 * band)
    ax.set_title("First measured rotation AUC band: $\\sigma=0.0063$, $2\\sigma=0.0126$")
    ax.legend(loc="lower right", fontsize=7.5)
    save(fig, "auc_noise_band.pdf")


# Fig 5: the four-family bake-off on the NEW full-64-d corpus — the family
# ordering survives the corpus change; the two tree families lead, the dense
# net is last (2026-06-12-classifier-bakeoff-p64.md, Phase 1).
def fig_bakeoff_p64():
    labels = [
        "xgboost-classifier\n(winner)",
        "catboost-classifier\n(2nd tree, in-band)",
        "ngboost-classifier",
        "pyramid-mlp-classifier\n(dense net)",
    ]
    auc = [0.7204, 0.7111, 0.6917, 0.6781]
    colors = [GOOD, ACCENT, MUTED, BAD]
    band = 0.0126  # 2-sigma AUC band (old-corpus estimate, approximate here)
    winner = 0.7204

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    y = np.arange(len(labels))[::-1]
    ax.barh(y, auc, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("AUC (P(rotation))")
    ax.set_xlim(0.5, 0.76)
    ax.xaxis.set_major_formatter(afmt)
    ax.axvline(0.5, color="black", lw=1.0, label="chance (0.5)")
    # in-band strip below the winner: catboost falls inside it
    ax.axvspan(winner - band, winner, color=GOOD, alpha=0.12, label="$2\\sigma$ band below winner")
    for yi, a in zip(y, auc):
        ax.text(a + 0.003, yi, f"{a:.3f}", va="center", fontsize=7.5)
    ax.set_title("New full-64-d corpus: two tree families lead, the dense net is last")
    ax.legend(loc="lower right", fontsize=7.5)
    save(fig, "bakeoff_p64.pdf")


# Fig 6: xgboost depth × learning-rate sweep on the new corpus — depth is the
# dominant axis, high lr hurts; the best cell beats the default sub-band
# (2026-06-12-classifier-bakeoff-p64.md, Phase 2).
def fig_xgb_depth_lr_heatmap():
    depths = [4, 6, 8]
    lrs = [0.03, 0.05, 0.1]
    # AUC[depth_idx][lr_idx], transcribed from the Phase-2 sweep table.
    auc = np.array([
        [0.7074, 0.7094, 0.7119],  # depth 4
        [0.7205, 0.7204, 0.7153],  # depth 6
        [0.7261, 0.7196, 0.7173],  # depth 8
    ])
    best = (2, 0)  # depth 8, lr 0.03

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    im = ax.pcolormesh(
        np.arange(len(lrs) + 1), np.arange(len(depths) + 1), auc,
        cmap="viridis", shading="flat",
    )
    ax.set_xticks(np.arange(len(lrs)) + 0.5)
    ax.set_xticklabels([f"{v:g}" for v in lrs])
    ax.set_yticks(np.arange(len(depths)) + 0.5)
    ax.set_yticklabels([str(d) for d in depths])
    ax.set_xlabel("learning rate")
    ax.set_ylabel("max_depth")
    ax.grid(False)
    for i in range(len(depths)):
        for j in range(len(lrs)):
            is_best = (i, j) == best
            ax.text(
                j + 0.5, i + 0.5, f"{auc[i, j]:.4f}",
                ha="center", va="center", fontsize=8,
                color="white" if auc[i, j] < 0.718 else "black",
                fontweight="bold" if is_best else "normal",
            )
            if is_best:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor=BAD, lw=2.2))
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("AUC (P(rotation))")
    cbar.formatter = afmt
    cbar.update_ticks()
    ax.set_title("Depth is the lever, fast lr hurts;\nbest (depth 8, lr 0.03) is only sub-band over default")
    save(fig, "xgb_depth_lr_heatmap.pdf")


# Fig 7: the dense-net diagnostic signature — genuine AUC ranking signal
# coexisting with blown log-loss (calibration ceiling), contrasted with the
# trees. Source: bake-off (xgb/catboost/ngboost/mlp logloss) +
# arch scan (best converged MLP, and the random-baseline log-loss reference)
# (2026-06-12-classifier-bakeoff-p64.md + 2026-06-12-pyramid-mlp-arch-scan.md).
def fig_mlp_calibration_signature():
    labels = [
        "xgboost",
        "catboost",
        "ngboost",
        "pyramid-mlp\n(arch scan best)",
        "pyramid-mlp\n(bake-off)",
    ]
    auc = [0.7204, 0.7111, 0.6917, 0.6792, 0.6781]
    logloss = [0.6075, 0.6132, 0.6293, 2.0584, 2.4085]
    colors = [GOOD, GOOD, MUTED, BAD, BAD]
    rand_logloss = 0.6900  # log-loss of a constant base-rate predictor (~0.445)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 3.4))
    x = np.arange(len(labels))

    ax1.bar(x, auc, color=colors)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=6.5)
    ax1.set_ylim(0.5, 0.74)
    ax1.yaxis.set_major_formatter(afmt)
    ax1.axhline(0.5, color="black", lw=1.0)
    ax1.set_ylabel("AUC (P(rotation))")
    ax1.set_title("AUC: the net DOES rank\n(genuine signal, $\\approx 0.68$)")

    ax2.bar(x, logloss, color=colors)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=6.5)
    ax2.axhline(rand_logloss, color=BAD, ls="--", lw=1.2, label="random baseline ($\\approx 0.69$)")
    ax2.set_ylabel("log-loss (lower better)")
    ax2.set_title("log-loss: but it is confidently WRONG\n(worse than the random baseline)")
    ax2.legend(loc="upper left", fontsize=7)
    save(fig, "mlp_calibration_signature.pdf")


# Fig 8: the pyramid-MLP architecture scan refutes the budget hypothesis —
# every geometry self-stops at 73–100 iters (well below the 800 cap and even
# the old 300 cap), and the whole family lands inside one noise band
# (2026-06-12-pyramid-mlp-arch-scan.md, Results).
def fig_mlp_arch_scan():
    arms = [
        "M1 [512,256]",
        "M0 [256,128,64]",
        "M4 [384,307,246]",
        "M2 [256,128,64,32]",
        "M3 [512,169,56]",
        "M5 [512,307,184,111]",
    ]
    auc = [0.6792, 0.6781, 0.6777, 0.6737, 0.6736, 0.6712]
    iters = [75, 73, 87, 100, 89, 84]
    band = 0.0126
    default = 0.6781  # M0 re-baseline (== the under-trained 300-iter default)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.4))
    y = np.arange(len(arms))[::-1]

    # left: AUC, whole family inside one band of the default
    ax1.barh(y, auc, color=[GOOD] + [MUTED] * 5)
    ax1.set_yticks(y)
    ax1.set_yticklabels(arms, fontsize=7)
    ax1.set_xlim(0.66, 0.685)
    ax1.xaxis.set_major_formatter(afmt)
    ax1.axvspan(default - band, default + band, color=ACCENT, alpha=0.12, label="$\\pm 2\\sigma$ band")
    ax1.axvline(default, color=ACCENT, ls="--", lw=1.0)
    for yi, a in zip(y, auc):
        ax1.text(a - 0.0006, yi, f"{a:.4f}", va="center", ha="right", fontsize=6.5, color="white")
    ax1.set_xlabel("AUC (P(rotation))")
    ax1.set_title("Every geometry is inside one band\n(architecture is exhausted)")
    ax1.legend(loc="lower right", fontsize=6.5)

    # right: stopping iteration vs the caps — budget was never binding
    ax2.barh(y, iters, color=MUTED)
    ax2.set_yticks(y)
    ax2.set_yticklabels([])
    ax2.axvline(300, color=ACCENT, ls=":", lw=1.4, label="old cap (300)")
    ax2.axvline(800, color=BAD, ls="--", lw=1.4, label="raised cap (800)")
    for yi, it in zip(y, iters):
        ax2.text(it + 6, yi, f"{it}", va="center", fontsize=7)
    ax2.set_xlim(0, 850)
    ax2.set_xlabel("iterations to self-stop")
    ax2.set_title("All self-stop at 73--100 iters:\nbudget was never the constraint")
    ax2.legend(loc="lower right", fontsize=6.5)
    save(fig, "mlp_arch_scan.pdf")


# Fig 9: the burn-deep-classifier topology bake-off resolves the open question.
# Left: the three topologies (plain mlp wins, all inside one band, embeddings
# edges wide_deep). Right: the calibration recovery — the Rust/burn net with RAW
# one-hots drops log-loss from the sklearn pyramid's ~2.0 to the trees' ~0.62
# regime (2026-06-12-burn-deep-topology.md, Phase 1 + Findings (i)).
def fig_burn_topology():
    # Left panel: Phase-1 topology AUC (winner mlp, then embeddings, wide_deep).
    topo = ["mlp\n(plain)", "embeddings", "wide_deep"]
    topo_auc = [0.7001, 0.6934, 0.6918]
    band = 0.0126  # 2-sigma AUC band (old-corpus estimate, approximate here)
    winner = 0.7001
    topo_colors = [GOOD, ACCENT, MUTED]

    # Right panel: the calibration recovery — sklearn pyramid vs the burn net,
    # the headline finding (log-loss 2.0 -> 0.62, into the tree regime).
    calib_labels = [
        "sklearn\npyramid-mlp",
        "burn-deep\n(mlp, raw 1-hot)",
        "xgboost\n(tree tier)",
    ]
    calib_logloss = [2.0584, 0.6211, 0.6075]
    calib_colors = [BAD, GOOD, MUTED]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.4))

    # Left: topology comparison, all inside one band of the plain-mlp winner.
    x = np.arange(len(topo))
    ax1.bar(x, topo_auc, color=topo_colors)
    ax1.set_xticks(x)
    ax1.set_xticklabels(topo, fontsize=7.5)
    ax1.set_ylim(0.5, 0.72)
    ax1.yaxis.set_major_formatter(afmt)
    ax1.axhline(0.5, color="black", lw=1.0)
    ax1.axhspan(winner - band, winner, color=GOOD, alpha=0.12, label="$2\\sigma$ band below winner")
    for xi, a in zip(x, topo_auc):
        ax1.text(xi, a + 0.004, f"{a:.4f}", ha="center", fontsize=7.5)
    ax1.set_ylabel("AUC (P(rotation))")
    ax1.set_title("Plain mlp wins; entity embeddings\nstarved at low cardinality")
    ax1.legend(loc="lower center", fontsize=6.5)

    # Right: the calibration fix — the headline.
    xc = np.arange(len(calib_labels))
    ax2.bar(xc, calib_logloss, color=calib_colors)
    ax2.set_xticks(xc)
    ax2.set_xticklabels(calib_labels, fontsize=7)
    rand_logloss = 0.6900  # constant base-rate predictor (~0.445)
    ax2.axhline(rand_logloss, color=MUTED, ls="--", lw=1.0, label="random baseline ($\\approx 0.69$)")
    for xi, ll in zip(xc, calib_logloss):
        ax2.text(xi, ll + 0.04, f"{ll:.3f}", ha="center", fontsize=7.5)
    ax2.annotate(
        "", xy=(1, 0.72), xytext=(0, 2.0),
        arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.6),
    )
    ax2.text(0.5, 1.45, "raw one-hots\nfix calibration", ha="center", fontsize=7, color=ACCENT)
    ax2.set_ylabel("log-loss (lower better)")
    ax2.set_title("The headline: log-loss $2.0\\to 0.62$\n(into the tree regime)")
    ax2.legend(loc="upper right", fontsize=6.5)
    save(fig, "burn_topology.pdf")


# Fig 10: raising identity cardinality flips the topology ranking and lifts the
# ANN into the tree tier — the paired re-run on the HIGH-VOCAB matrix.
# Left: the embeddings-vs-mlp reversal across the two cardinalities (60/40 ->
# 300/120) — embeddings goes from below mlp to above it (a +0.0130 swing).
# Right: the five-family AUC ranking on the high-vocab matrix — the saved
# embeddings net reaches the bottom of the tree tier
# (2026-06-12-high-vocab-topology.md, Phase 0 + Phase 1 + Findings (i),(iii)).
def fig_burn_highvocab():
    # Left panel: the paired emb-vs-mlp comparison across the two matrices.
    # 60/40 matrix (the prior burn-deep run): mlp beats embeddings by +0.0067.
    # high-vocab matrix (this run): embeddings beats mlp by +0.0063 — reversal.
    grp_labels = ["60/40 matrix\n(artist 60, genre 40)", "high-vocab matrix\n(artist 300, genre 120)"]
    emb = [0.6934, 0.7192]
    mlp = [0.7001, 0.7129]

    # Right panel: the five-family AUC ranking on the high-vocab matrix.
    band = 0.0126  # 2-sigma AUC band (old-corpus estimate, approximate here)
    fam_labels = [
        "xgboost\n(in-dataset leader)",
        "catboost",
        "burn-deep emb\n(Phase-2 best)",
        "burn-deep emb\n(saved)",
        "burn-deep mlp",
        "burn-deep wide_deep",
    ]
    fam_auc = [0.7299, 0.7252, 0.7228, 0.7192, 0.7129, 0.7061]
    fam_colors = [GOOD, GOOD, ACCENT, ACCENT, MUTED, MUTED]
    tree_floor = 0.7252  # bottom of the in-dataset tree tier (catboost)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.5))

    # Left: grouped bars, embeddings vs mlp, across the two cardinalities.
    x = np.arange(len(grp_labels))
    w = 0.36
    b_emb = ax1.bar(x - w / 2, emb, w, color=ACCENT, label="embeddings")
    b_mlp = ax1.bar(x + w / 2, mlp, w, color=MUTED, label="mlp")
    ax1.set_xticks(x)
    ax1.set_xticklabels(grp_labels, fontsize=6.8)
    ax1.set_ylim(0.66, 0.73)
    ax1.yaxis.set_major_formatter(afmt)
    ax1.set_ylabel("AUC (P(rotation))")
    for bars, vals in ((b_emb, emb), (b_mlp, mlp)):
        for bar, v in zip(bars, vals):
            ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.0008, f"{v:.4f}",
                     ha="center", fontsize=6.3)
    # arrows annotating who wins in each regime
    ax1.annotate("mlp wins\n($-0.0067$)", xy=(0, 0.704), xytext=(0, 0.722),
                 ha="center", fontsize=6.5, color=BAD,
                 arrowprops=dict(arrowstyle="->", color=BAD, lw=0.8))
    ax1.annotate("embeddings wins\n($+0.0063$)", xy=(1, 0.722), xytext=(1, 0.668),
                 ha="center", fontsize=6.5, color=GOOD,
                 arrowprops=dict(arrowstyle="->", color=GOOD, lw=0.8))
    ax1.set_title("Higher cardinality FLIPS the ranking\n(emb vs mlp, a $+0.0130$ swing)")
    ax1.legend(loc="lower left", fontsize=6.5)

    # Right: five-family ranking on the high-vocab matrix, tree-tier floor shaded.
    y = np.arange(len(fam_labels))[::-1]
    ax2.barh(y, fam_auc, color=fam_colors)
    ax2.set_yticks(y)
    ax2.set_yticklabels(fam_labels, fontsize=6.3)
    ax2.set_xlim(0.69, 0.735)
    ax2.xaxis.set_major_formatter(afmt)
    # tree tier: from the band below catboost up to xgboost
    ax2.axvspan(tree_floor - band, fam_auc[0], color=GOOD, alpha=0.10,
                label="in-dataset tree tier")
    ax2.axvline(tree_floor, color=GOOD, ls="--", lw=1.0)
    for yi, a in zip(y, fam_auc):
        ax2.text(a + 0.0006, yi, f"{a:.4f}", va="center", fontsize=6.3)
    ax2.set_xlabel("AUC (P(rotation))")
    ax2.set_title("The saved embeddings net reaches\nthe bottom of the tree tier")
    ax2.legend(loc="lower right", fontsize=6.5)
    save(fig, "burn_highvocab.pdf")


if __name__ == "__main__":
    fig_ae_latent_knn()
    fig_svm_mae_vs_r2()
    fig_auc_bakeoff()
    fig_auc_noise_band()
    fig_bakeoff_p64()
    fig_xgb_depth_lr_heatmap()
    fig_mlp_calibration_signature()
    fig_mlp_arch_scan()
    fig_burn_topology()
    fig_burn_highvocab()
