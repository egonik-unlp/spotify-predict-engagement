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


if __name__ == "__main__":
    fig_ae_latent_knn()
    fig_svm_mae_vs_r2()
    fig_auc_bakeoff()
    fig_auc_noise_band()
