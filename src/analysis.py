"""Calibration-policy analysis on cached model probabilities (run experiment.py first).

Protocol (per task):
  * For every language, the official dev and test sets are pooled and, for each of N_SEEDS seeds,
    split at random 50/50 into a calibration half and an evaluation half. This makes calibration
    and evaluation data exchangeable by construction, which is what the conformal guarantee assumes.
  * POOLED policy: one LAC threshold from the union of all trained languages' calibration halves.
  * CONDITIONAL policy: one LAC threshold per language from that language's calibration half.
  * A prediction set with exactly one label is auto-decided; anything else is deferred.
  * Calibration-size ablation: the conditional threshold is re-estimated from n in CAL_SIZES
    random calibration examples of the language.
  * Official-split check: calibrate on the official dev split, evaluate on the official test split,
    to expose calibration/evaluation shift in the released benchmark splits.
"""
import json
import os

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT_DIR = os.environ.get("RESULTS_DIR", os.path.join(ROOT, "results"))

N_SEEDS = 30
ALPHAS = [0.05, 0.10]
MAIN_ALPHA = 0.10
CAL_SIZES = [20, 50, 100, 200]
TASK_LANGS = {
    "masakhanews": (["amh", "eng", "fra", "hau", "ibo", "lin", "lug", "orm", "pcm", "run",
                     "sna", "som", "swa", "tir", "xho", "yor"], []),
    "afrisenti": (["amh", "arq", "ary", "hau", "ibo", "kin", "pcm", "por", "swa", "tso", "twi", "yor"],
                  ["orm", "tir"]),
}
LANG_NAMES = {"amh": "Amharic", "eng": "English", "fra": "French", "hau": "Hausa", "ibo": "Igbo",
              "lin": "Lingala", "lug": "Luganda", "orm": "Oromo", "pcm": "Nigerian Pidgin", "run": "Rundi",
              "sna": "chiShona", "som": "Somali", "swa": "Swahili", "tir": "Tigrinya", "xho": "isiXhosa",
              "yor": "Yoruba", "arq": "Algerian Arabic", "ary": "Moroccan Arabic", "kin": "Kinyarwanda",
              "por": "Mozambican Portuguese", "tso": "Xitsonga", "twi": "Twi"}


def conformal_quantile(scores, alpha):
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return np.inf
    return float(np.sort(scores)[k - 1])


def evaluate(P, yidx, qhat):
    in_set = (1.0 - P) <= qhat
    sizes = in_set.sum(axis=1)
    covered = in_set[np.arange(len(yidx)), yidx]
    single = sizes == 1
    correct = P.argmax(axis=1) == yidx
    n_auto = int(single.sum())
    return dict(n=int(len(yidx)), coverage=float(covered.mean()), deferral=float(1 - single.mean()),
                selective_accuracy=float(correct[single].mean()) if n_auto else np.nan,
                accuracy=float(correct.mean()), n_auto=n_auto)


def summarize(list_of_dicts):
    keys = list_of_dicts[0].keys()
    out = {}
    for k in keys:
        vals = np.array([d[k] for d in list_of_dicts], dtype=float)
        out[k + "_mean"] = float(np.nanmean(vals))
        out[k + "_std"] = float(np.nanstd(vals))
        out[k + "_min"] = float(np.nanmin(vals))
    return out


def analyse(task):
    trained, unseen = TASK_LANGS[task]
    langs = trained + unseen
    z = np.load(os.path.join(OUT_DIR, f"{task}_probs.npz"), allow_pickle=True)
    classes = z["classes"]
    P = np.concatenate([z["P_dev"], z["P_test"]])
    y = np.concatenate([z["y_dev"], z["y_test"]])
    lang = np.concatenate([z["lang_dev"], z["lang_test"]])
    is_dev = np.concatenate([np.ones(len(z["y_dev"]), bool), np.zeros(len(z["y_test"]), bool)])
    yidx = np.searchsorted(classes, y)
    S = 1.0 - P[np.arange(len(y)), yidx]
    n_train = {l: int((z["lang_train"] == l).sum()) for l in langs}

    res = {"task": task, "n_seeds": N_SEEDS, "languages": {l: {"name": LANG_NAMES[l], "n_train": n_train[l],
                                                                  "n_pool": int((lang == l).sum()),
                                                                  "unseen": l in unseen} for l in langs},
           "overall": {}, "ablation": {}, "official_split": {}}

    # ------------------------------------------------ random re-splits, both policies, both alphas
    per_lang = {a: {l: {"pooled": [], "conditional": []} for l in langs} for a in ALPHAS}
    overall = {a: {"pooled": [], "conditional": []} for a in ALPHAS}
    abl = {l: {n: [] for n in CAL_SIZES + ["all"]} for l in langs}
    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed)
        cal_mask = np.zeros(len(y), bool)
        for l in langs:
            idx = np.where(lang == l)[0]
            rng.shuffle(idx)
            cal_mask[idx[: len(idx) // 2]] = True
        ev_mask = ~cal_mask
        for a in ALPHAS:
            q_pool = conformal_quantile(S[cal_mask & np.isin(lang, trained)], a)
            ov_p, ov_c = [], []
            for l in langs:
                m_cal = cal_mask & (lang == l)
                m_ev = ev_mask & (lang == l)
                q_l = conformal_quantile(S[m_cal], a)
                rp = evaluate(P[m_ev], yidx[m_ev], q_pool)
                rc = evaluate(P[m_ev], yidx[m_ev], q_l)
                per_lang[a][l]["pooled"].append(rp)
                per_lang[a][l]["conditional"].append(rc)
                if l in trained:
                    ov_p.append(rp); ov_c.append(rc)
                if a == MAIN_ALPHA:
                    cal_idx = np.where(m_cal)[0]
                    for n in CAL_SIZES:
                        if len(cal_idx) >= n:
                            sub = rng.choice(cal_idx, size=n, replace=False)
                            abl[l][n].append(evaluate(P[m_ev], yidx[m_ev], conformal_quantile(S[sub], a)))
                    abl[l]["all"].append(rc)
            # population-level (trained languages) numbers, weighted by evaluation-half sizes
            for name, rows in [("pooled", ov_p), ("conditional", ov_c)]:
                n_tot = sum(r["n"] for r in rows)
                overall[a][name].append(dict(
                    coverage=sum(r["coverage"] * r["n"] for r in rows) / n_tot,
                    deferral=sum(r["deferral"] * r["n"] for r in rows) / n_tot,
                    worst_language_coverage=min(r["coverage"] for r in rows),
                    n_under=sum(r["coverage"] < 1 - a - 0.02 for r in rows),
                    selective_accuracy=sum(r["selective_accuracy"] * r["n_auto"] for r in rows if r["n_auto"]) /
                                        max(1, sum(r["n_auto"] for r in rows)),
                    accuracy=sum(r["accuracy"] * r["n"] for r in rows) / n_tot))

    for a in ALPHAS:
        res["overall"][str(a)] = {k: summarize(v) for k, v in overall[a].items()}
        for l in langs:
            res["languages"][l][str(a)] = {k: summarize(v) for k, v in per_lang[a][l].items()}
    for l in langs:
        res["ablation"][l] = {str(n): summarize(v) for n, v in abl[l].items() if v}

    # ------------------------------------------------ official split check (dev -> test)
    for a in ALPHAS:
        q_pool = conformal_quantile(S[is_dev & np.isin(lang, trained)], a)
        rows = {}
        for l in langs:
            m_cal = is_dev & (lang == l); m_ev = (~is_dev) & (lang == l)
            rows[l] = {"pooled": evaluate(P[m_ev], yidx[m_ev], q_pool),
                       "conditional": evaluate(P[m_ev], yidx[m_ev], conformal_quantile(S[m_cal], a))}
        m_tr = (~is_dev) & np.isin(lang, trained)
        res["official_split"][str(a)] = {"per_language": rows,
                                         "overall_pooled": evaluate(P[m_tr], yidx[m_tr], q_pool)}
    with open(os.path.join(OUT_DIR, f"{task}_analysis.json"), "w") as f:
        json.dump(res, f, indent=1)
    return res


def print_report(res):
    a = str(MAIN_ALPHA)
    t = res["task"]
    print(f"\n===== {t}  (alpha={MAIN_ALPHA}, {res['n_seeds']} random re-splits) =====")
    print(f"{'lang':5s} {'ntr':>5s} {'acc':>6s} | {'covP':>6s}±{'sd':4s} {'defP':>6s} {'saccP':>6s} | {'covC':>6s}±{'sd':4s} {'defC':>6s} {'saccC':>6s}")
    for l, d in res["languages"].items():
        p = d[a]["pooled"]; c = d[a]["conditional"]
        print(f"{l:5s} {d['n_train']:5d} {p['accuracy_mean']:6.3f} | {p['coverage_mean']:6.3f}±{p['coverage_std']:4.3f} {p['deferral_mean']:6.3f} {p['selective_accuracy_mean']:6.3f} | "
              f"{c['coverage_mean']:6.3f}±{c['coverage_std']:4.3f} {c['deferral_mean']:6.3f} {c['selective_accuracy_mean']:6.3f}")
    for pol in ["pooled", "conditional"]:
        o = res["overall"][a][pol]
        print(f"overall[{pol}]: cov={o['coverage_mean']:.3f}±{o['coverage_std']:.3f} worst-lang cov={o['worst_language_coverage_mean']:.3f} "
              f"n_under(>2pp)={o['n_under_mean']:.1f} defer={o['deferral_mean']:.3f} sel.acc={o['selective_accuracy_mean']:.3f} acc={o['accuracy_mean']:.3f}")
    print("official split (dev->test), pooled overall coverage:", round(res["official_split"][a]["overall_pooled"]["coverage"], 3))
    print("official split per-language conditional coverage:",
          {l: round(v["conditional"]["coverage"], 3) for l, v in res["official_split"][a]["per_language"].items()})
    print("ablation (conditional, alpha=0.1): mean coverage over languages, and min over languages of the seed-min")
    for n in [str(x) for x in CAL_SIZES] + ["all"]:
        covs = [d[n]["coverage_mean"] for d in res["ablation"].values() if n in d]
        mins = [d[n]["coverage_min"] for d in res["ablation"].values() if n in d]
        stds = [d[n]["coverage_std"] for d in res["ablation"].values() if n in d]
        defs = [d[n]["deferral_mean"] for d in res["ablation"].values() if n in d]
        print(f"  n_cal={n:>4s}: langs={len(covs):2d} mean cov={np.mean(covs):.3f} mean sd={np.mean(stds):.3f} worst seed-min={np.min(mins):.3f} mean defer={np.mean(defs):.3f}")


if __name__ == "__main__":
    all_res = {}
    for task in TASK_LANGS:
        all_res[task] = analyse(task)
        print_report(all_res[task])
