"""Generate the paper figures from results/*_analysis.json."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT_DIR = os.environ.get("RESULTS_DIR", os.path.join(ROOT, "results"))
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "legend.fontsize": 6,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "axes.linewidth": 0.6, "pdf.fonttype": 42,
})
COL_P, COL_C = "#c0392b", "#1f5f9e"
TASK_TITLE = {"masakhanews": "MasakhaNEWS: topic, 16 langs",
              "afrisenti": "AfriSenti: sentiment, 12 + 2 unseen langs"}
A = "0.1"


def load(task):
    return json.load(open(os.path.join(OUT_DIR, f"{task}_analysis.json")))


def fig_coverage_deferral():
    fig, axes = plt.subplots(2, 2, figsize=(4.3, 2.0), sharex="col")
    for j, task in enumerate(["masakhanews", "afrisenti"]):
        r = load(task)
        langs = list(r["languages"].keys())
        # sort: trained languages by base accuracy (hard -> easy), unseen languages last
        trained = [l for l in langs if not r["languages"][l]["unseen"]]
        unseen = [l for l in langs if r["languages"][l]["unseen"]]
        trained.sort(key=lambda l: r["languages"][l][A]["pooled"]["accuracy_mean"])
        order = trained + unseen
        x = np.arange(len(order))
        for i, metric in enumerate(["coverage", "deferral"]):
            ax = axes[i, j]
            for pol, col, mk, dx in [("pooled", COL_P, "o", -0.17), ("conditional", COL_C, "s", 0.17)]:
                m = np.array([r["languages"][l][A][pol][f"{metric}_mean"] for l in order])
                s = np.array([r["languages"][l][A][pol][f"{metric}_std"] for l in order])
                ax.errorbar(x + dx, m, yerr=s, fmt=mk, ms=3, color=col, ecolor=col, elinewidth=0.6,
                            capsize=1.5, lw=0, label="pooled threshold" if pol == "pooled" else "language-conditional")
            if metric == "coverage":
                ax.axhline(0.90, color="k", ls="--", lw=0.7)
                ax.axhspan(0.0, 0.88, color="#f2dede", alpha=0.6, lw=0)
                ax.set_ylim(0.72, 0.99)
                ax.set_ylabel("coverage on held-out half")
                ax.set_title(TASK_TITLE[task], pad=3)
            else:
                ax.set_ylim(0, 1.0)
                ax.set_ylabel("deferral rate")
                ax.set_xticks(x)
                ax.set_xticklabels([l + ("*" if l in unseen else "") for l in order], rotation=60, ha="right")
            if unseen:
                ax.axvline(len(trained) - 0.5, color="gray", lw=0.6, ls=":")
            ax.grid(axis="y", lw=0.3, alpha=0.5)
            ax.tick_params(length=2)
        axes[0, j].set_xticks(x)
    axes[0, 0].legend(loc="lower right", frameon=False, handletextpad=0.3)
    fig.tight_layout(h_pad=0.4, w_pad=0.8)
    fig.savefig(os.path.join(FIG_DIR, "coverage_deferral.pdf"))
    plt.close(fig)


def fig_calibration_size():
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 1.7))
    sizes = ["20", "50", "100", "200", "all"]
    for j, task in enumerate(["masakhanews", "afrisenti"]):
        r = load(task)
        ax = axes[j]
        means, sds, mins, defs = [], [], [], []
        for n in sizes:
            rows = [d[n] for d in r["ablation"].values() if n in d]
            means.append(np.mean([d["coverage_mean"] for d in rows]))
            sds.append(np.mean([d["coverage_std"] for d in rows]))
            mins.append(np.min([d["coverage_min"] for d in rows]))
            defs.append(np.mean([d["deferral_mean"] for d in rows]))
        x = np.arange(len(sizes))
        ax.errorbar(x, means, yerr=sds, fmt="s-", color=COL_C, ms=3, lw=0.8, capsize=2, elinewidth=0.7,
                    label="mean coverage $\\pm$ seed s.d.")
        ax.plot(x, mins, "v", color="#7f8c8d", ms=3.5, label="worst realized (any language, any seed)")
        ax.axhline(0.90, color="k", ls="--", lw=0.7)
        ax2 = ax.twinx()
        ax2.plot(x, defs, "o:", color=COL_P, ms=2.5, lw=0.7, label="mean deferral")
        ax2.set_ylim(0, 1.0)
        ax2.tick_params(axis="y", colors=COL_P, length=2)
        if j == 1:
            ax2.set_ylabel("deferral rate", color=COL_P)
        ax.set_xticks(x)
        ax.set_xticklabels(["20", "50", "100", "200", "all"])
        ax.set_xlabel("calibration examples per language")
        ax.set_ylim(0.6, 1.0)
        if j == 0:
            ax.set_ylabel("coverage")
        ax.set_title(TASK_TITLE[task].split(":")[0], pad=3)
        ax.grid(axis="y", lw=0.3, alpha=0.5)
        ax.tick_params(length=2)
        if j == 0:
            h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
            handles, labels = h1 + h2, l1 + l2
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=6.5,
               handletextpad=0.3, columnspacing=1.2, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(w_pad=1.2, rect=(0, 0.1, 1, 1))
    fig.savefig(os.path.join(FIG_DIR, "calibration_size.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    fig_coverage_deferral()
    fig_calibration_size()
    print("figures written to", FIG_DIR)
