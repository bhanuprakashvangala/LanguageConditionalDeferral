"""Print the numbers quoted in the paper from results/*_analysis.json and results/*_results.json.

    python src/report.py                 # reads results/
    RESULTS_DIR=some/dir python src/report.py
"""
import json
import os

import numpy as np
from scipy.stats import spearmanr

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT_DIR = os.environ.get("RESULTS_DIR", os.path.join(ROOT, "results"))
A = "0.1"


def load(name):
    with open(os.path.join(OUT_DIR, name)) as f:
        return json.load(f)


def claims():
    """Return an ordered list of (claim, value) pairs, values rounded as in the paper."""
    out = []
    mn, af = load("masakhanews_analysis.json"), load("afrisenti_analysis.json")
    L = lambda r, l, pol, k: r["languages"][l][A][pol][k]  # noqa: E731

    # efficiency and base quality (official test split, no deferral)
    for task, r in [("MasakhaNEWS", load("masakhanews_results.json")), ("AfriSenti", load("afrisenti_results.json"))]:
        e = r["efficiency"]
        trained = [l for l, d in r["languages"].items() if not d["unseen"]]
        out.append((f"{task} training time, min", round((e["feature_time_s"] + e["fit_time_s"]) / 60, 1)))
        out.append((f"{task} model size, MB", round(e["model_size_mb"])))
        out.append((f"{task} mean test accuracy over trained languages, %",
                    round(100 * np.mean([r["languages"][l]["accuracy"] for l in trained]), 1)))
        out.append((f"{task} mean test macro-F1 over trained languages",
                    round(100 * np.mean([r["languages"][l]["macro_f1"] for l in trained]), 1)))

    # pooled vs conditional, 30 re-splits
    for task, r in [("MasakhaNEWS", mn), ("AfriSenti", af)]:
        for pol in ["pooled", "conditional"]:
            o = r["overall"][A][pol]
            out.append((f"{task} {pol} aggregate coverage", f"{o['coverage_mean']:.3f} +/- {o['coverage_std']:.3f}"))
            out.append((f"{task} {pol} aggregate deferral, %", round(100 * o["deferral_mean"], 1)))
        trained = [l for l, d in r["languages"].items() if not d["unseen"]]
        for pol in ["pooled", "conditional"]:
            n_under = sum(L(r, l, pol, "coverage_mean") < 0.88 for l in trained)
            out.append((f"{task} {pol}: trained languages > 2 pts below 0.90", n_under))
        cov_c = [L(r, l, "conditional", "coverage_mean") for l in r["languages"]]
        out.append((f"{task} conditional per-language coverage range", f"{min(cov_c):.3f} to {max(cov_c):.3f}"))
        acc = [L(r, l, "pooled", "accuracy_mean") for l in trained]
        dfr = [L(r, l, "conditional", "deferral_mean") for l in trained]
        out.append((f"{task} Spearman rho(conditional deferral, accuracy), trained languages",
                    round(spearmanr(dfr, acc).correlation, 2)))

    for l in ["som", "tir", "swa", "pcm", "xho", "sna"]:
        out.append((f"MasakhaNEWS pooled coverage {l}",
                    f"{L(mn, l, 'pooled', 'coverage_mean'):.3f} +/- {L(mn, l, 'pooled', 'coverage_std'):.3f}"))
    for l in ["hau", "ibo", "yor", "orm", "tir"]:
        out.append((f"AfriSenti pooled coverage {l}", f"{L(af, l, 'pooled', 'coverage_mean'):.3f}"))
    out.append(("MasakhaNEWS som selective accuracy, pooled %", round(100 * L(mn, "som", "pooled", "selective_accuracy_mean"), 1)))
    out.append(("MasakhaNEWS som selective accuracy, conditional %", round(100 * L(mn, "som", "conditional", "selective_accuracy_mean"), 1)))
    out.append(("MasakhaNEWS som deferral, pooled %", round(100 * L(mn, "som", "pooled", "deferral_mean"), 1)))
    for l in ["som", "tir", "swa", "amh", "yor", "sna", "pcm"]:
        out.append((f"MasakhaNEWS {l} deferral, conditional %", round(100 * L(mn, l, "conditional", "deferral_mean"), 1)))
    for l in ["amh", "tso", "ibo"]:
        out.append((f"AfriSenti {l} deferral, conditional %", round(100 * L(af, l, "conditional", "deferral_mean"), 1)))
    for l in ["orm", "tir"]:
        out.append((f"AfriSenti unseen {l}: auto-labelled share, pooled %", round(100 * (1 - L(af, l, "pooled", "deferral_mean")), 1)))
        out.append((f"AfriSenti unseen {l}: selective accuracy, pooled %", round(100 * L(af, l, "pooled", "selective_accuracy_mean"), 1)))
        out.append((f"AfriSenti unseen {l}: deferral, conditional %", round(100 * L(af, l, "conditional", "deferral_mean"), 1)))

    # calibration-size ablation (conditional policy, alpha = 0.1), averaged over languages
    for task, r in [("MasakhaNEWS", mn), ("AfriSenti", af)]:
        for n in ["20", "50", "100", "200", "all"]:
            rows = [d[n] for d in r["ablation"].values() if n in d]
            out.append((f"{task} ablation n={n}: mean coverage / mean seed s.d. / mean deferral %",
                        f"{np.mean([d['coverage_mean'] for d in rows]):.3f} / "
                        f"{np.mean([d['coverage_std'] for d in rows]):.3f} / "
                        f"{100 * np.mean([d['deferral_mean'] for d in rows]):.1f}"))

    # official split check (calibrate on dev, evaluate on test)
    for task, r in [("MasakhaNEWS", mn), ("AfriSenti", af)]:
        out.append((f"{task} official split: pooled aggregate coverage",
                    round(r["official_split"][A]["overall_pooled"]["coverage"], 3)))
    for l in ["ary", "arq", "pcm"]:
        out.append((f"AfriSenti official split: conditional coverage {l}",
                    round(af["official_split"][A]["per_language"][l]["conditional"]["coverage"], 3)))
    return out


if __name__ == "__main__":
    print(f"numbers from {os.path.abspath(OUT_DIR)}\n")
    for k, v in claims():
        print(f"{k:80s} {v}")
