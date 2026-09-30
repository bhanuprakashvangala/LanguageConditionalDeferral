"""Language-conditional selective prediction for low-resource African text classification.

One CPU-trainable multilingual model per task (TF-IDF char+word n-grams -> multinomial
logistic regression), wrapped with split conformal prediction (LAC score, Sadinle et al. 2019).
Singleton prediction sets are auto-decided; non-singleton sets are deferred.

Compares POOLED calibration (one global threshold estimated on the union of all languages'
dev sets) against LANGUAGE-CONDITIONAL (Mondrian) calibration (one threshold per language).

Outputs: results/<task>_results.json, <task>_efficiency.json, <task>_probs.npz (used by analysis.py)
         and results/summary.json
Run:     python src/experiment.py   (after python data/download.py)
"""
import csv
import json
import os
import pickle
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(ROOT, "data"))
OUT_DIR = os.environ.get("RESULTS_DIR", os.path.join(ROOT, "results"))
os.makedirs(OUT_DIR, exist_ok=True)

ALPHAS = [0.05, 0.10, 0.20]
MAIN_ALPHA = 0.10
CAL_SIZES = [20, 50, 100, 200]
N_SEEDS = 30
MAX_CHARS_NEWS = 1000  # headline + lead paragraph: what a cheap on-device classifier would see

TASKS = {
    "masakhanews": {
        "train_langs": ["amh", "eng", "fra", "hau", "ibo", "lin", "lug", "orm", "pcm", "run",
                        "sna", "som", "swa", "tir", "xho", "yor"],
        "unseen_langs": [],
        "text_cols": ("headline", "text"),
        "label_col": "category",
        "char_max_features": 300_000,
        "word_max_features": 150_000,
        "C": 20.0,
    },
    "afrisenti": {
        "train_langs": ["amh", "arq", "ary", "hau", "ibo", "kin", "pcm", "por", "swa", "tso",
                        "twi", "yor"],
        "unseen_langs": ["orm", "tir"],  # dev/test only in AfriSenti-SemEval (zero-shot track)
        "text_cols": ("tweet",),
        "label_col": "label",
        "char_max_features": 300_000,
        "word_max_features": 150_000,
        "C": 5.0,
    },
}


@dataclass
class Split:
    texts: list
    labels: np.ndarray
    langs: np.ndarray


def read_split(task, lang, split_name):
    cfg = TASKS[task]
    path = os.path.join(DATA_DIR, task, lang, f"{split_name}.tsv")
    if not os.path.exists(path) or os.path.getsize(path) < 20:
        return None
    df = pd.read_csv(path, sep="\t", quoting=csv.QUOTE_MINIMAL, dtype=str, keep_default_na=False)
    if task == "masakhanews":
        text = (df["headline"].fillna("") + " " + df["text"].fillna("")).str.slice(0, MAX_CHARS_NEWS)
    else:
        text = df["tweet"].fillna("")
    labels = df[cfg["label_col"]].str.strip().values
    return text.tolist(), labels


def load_task(task):
    cfg = TASKS[task]
    out = {}
    for split_name in ["train", "dev", "test"]:
        texts, labels, langs = [], [], []
        for lang in cfg["train_langs"] + cfg["unseen_langs"]:
            r = read_split(task, lang, split_name)
            if r is None:
                continue
            t, l = r
            texts += t
            labels.append(l)
            langs += [lang] * len(t)
        out[split_name] = Split(texts, np.concatenate(labels), np.array(langs))
    return out


def build_model(cfg, train: Split):
    t0 = time.time()
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True,
                               max_features=cfg["char_max_features"], dtype=np.float32)
    word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                               max_features=cfg["word_max_features"], dtype=np.float32,
                               token_pattern=r"(?u)\b\w+\b")
    X = hstack([char_vec.fit_transform(train.texts), word_vec.fit_transform(train.texts)]).tocsr()
    t_feat = time.time() - t0
    clf = LogisticRegression(C=cfg["C"], max_iter=2000)
    t1 = time.time()
    clf.fit(X, train.labels)
    t_fit = time.time() - t1
    model = {"char_vec": char_vec, "word_vec": word_vec, "clf": clf}
    size_bytes = len(pickle.dumps(model))
    return model, {"feature_time_s": t_feat, "fit_time_s": t_fit, "n_features": int(X.shape[1]),
                   "n_train": int(X.shape[0]), "model_size_mb": size_bytes / 1e6}


def predict_proba(model, texts):
    X = hstack([model["char_vec"].transform(texts), model["word_vec"].transform(texts)]).tocsr()
    return model["clf"].predict_proba(X)


# ---------------------------------------------------------------- conformal machinery
def lac_scores(probs, labels, classes):
    """Nonconformity s(x, y) = 1 - p_hat(y | x) for the true label."""
    idx = np.searchsorted(classes, labels)
    return 1.0 - probs[np.arange(len(labels)), idx]


def conformal_quantile(scores, alpha):
    """Finite-sample corrected (1 - alpha) quantile; returns inf when n is too small."""
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return np.inf
    return float(np.sort(scores)[k - 1])


def evaluate_sets(probs, labels, classes, qhat):
    """Prediction set C(x) = {y : 1 - p_hat(y|x) <= qhat}. Singleton => auto-decide, else defer."""
    in_set = (1.0 - probs) <= qhat  # (n, K) boolean
    set_sizes = in_set.sum(axis=1)
    idx = np.searchsorted(classes, labels)
    covered = in_set[np.arange(len(labels)), idx]
    singleton = set_sizes == 1
    argmax = probs.argmax(axis=1)
    correct = argmax == idx
    n = len(labels)
    n_single = int(singleton.sum())
    return {
        "n": int(n),
        "coverage": float(covered.mean()),
        "deferral": float(1.0 - singleton.mean()),
        "selective_accuracy": float(correct[singleton].mean()) if n_single > 0 else None,
        "selective_error": float(1.0 - correct[singleton].mean()) if n_single > 0 else None,
        "n_auto": n_single,
        "avg_set_size": float(set_sizes.mean()),
        "empty_rate": float((set_sizes == 0).mean()),
    }


def run_task(task):
    cfg = TASKS[task]
    print(f"\n=== {task} ===", flush=True)
    data = load_task(task)
    train, dev, test = data["train"], data["dev"], data["test"]
    print({k: len(v.texts) for k, v in data.items()}, flush=True)

    model, eff = build_model(cfg, train)
    classes = model["clf"].classes_
    print("efficiency:", eff, flush=True)

    t0 = time.time()
    P_dev = predict_proba(model, dev.texts)
    P_test = predict_proba(model, test.texts)
    eff["inference_docs_per_s"] = (len(dev.texts) + len(test.texts)) / (time.time() - t0)

    np.savez_compressed(os.path.join(OUT_DIR, f"{task}_probs.npz"), P_dev=P_dev, P_test=P_test,
                        y_dev=dev.labels, y_test=test.labels, lang_dev=dev.langs, lang_test=test.langs,
                        lang_train=train.langs, classes=classes)
    with open(os.path.join(OUT_DIR, f"{task}_efficiency.json"), "w") as f:
        json.dump(eff, f, indent=1)

    S_dev = lac_scores(P_dev, dev.labels, classes)
    all_langs = cfg["train_langs"] + cfg["unseen_langs"]
    dev_mask = {l: dev.langs == l for l in all_langs}
    test_mask = {l: test.langs == l for l in all_langs}
    pooled_dev = np.isin(dev.langs, cfg["train_langs"])  # unseen languages never enter the pooled pool

    res = {"task": task, "efficiency": eff, "classes": classes.tolist(), "alphas": ALPHAS,
           "main_alpha": MAIN_ALPHA, "languages": {}, "pooled_quantile": {}, "overall": {}}

    # base accuracy / F1 per language (argmax, no deferral)
    idx_test = np.searchsorted(classes, test.labels)
    pred_test = P_test.argmax(axis=1)
    for l in all_langs:
        m = test_mask[l]
        res["languages"][l] = {
            "n_train": int((train.langs == l).sum()),
            "n_dev": int(dev_mask[l].sum()),
            "n_test": int(m.sum()),
            "accuracy": float((pred_test[m] == idx_test[m]).mean()),
            "macro_f1": float(f1_score(idx_test[m], pred_test[m], average="macro")),
            "unseen": l in cfg["unseen_langs"],
            "by_alpha": {},
        }

    for alpha in ALPHAS:
        q_pooled = conformal_quantile(S_dev[pooled_dev], alpha)
        res["pooled_quantile"][str(alpha)] = q_pooled
        # overall (marginal) coverage over the pooled test distribution of trained languages
        m_all = np.isin(test.langs, cfg["train_langs"])
        res["overall"][str(alpha)] = {
            "pooled": evaluate_sets(P_test[m_all], test.labels[m_all], classes, q_pooled)}
        # language-conditional evaluation of the overall population
        mond_rows = []
        for l in all_langs:
            q_l = conformal_quantile(S_dev[dev_mask[l]], alpha)
            m = test_mask[l]
            r_pooled = evaluate_sets(P_test[m], test.labels[m], classes, q_pooled)
            r_mond = evaluate_sets(P_test[m], test.labels[m], classes, q_l)
            res["languages"][l]["by_alpha"][str(alpha)] = {
                "pooled": r_pooled, "conditional": r_mond, "q_conditional": q_l}
            if l in cfg["train_langs"]:
                mond_rows.append((m, q_l))
        # marginal coverage of the conditional policy over the pooled trained-language test set
        cov, defer, n_tot = 0.0, 0.0, 0
        for m, q_l in mond_rows:
            r = evaluate_sets(P_test[m], test.labels[m], classes, q_l)
            cov += r["coverage"] * r["n"]
            defer += r["deferral"] * r["n"]
            n_tot += r["n"]
        res["overall"][str(alpha)]["conditional"] = {"coverage": cov / n_tot, "deferral": defer / n_tot, "n": n_tot}

    # calibration-size ablation for the conditional policy (main alpha)
    rng = np.random.default_rng(0)
    abl = {}
    for l in all_langs:
        d_idx = np.where(dev_mask[l])[0]
        m = test_mask[l]
        abl[l] = {}
        for n_cal in CAL_SIZES:
            if len(d_idx) < n_cal:
                continue
            covs, defs = [], []
            for _ in range(N_SEEDS):
                sub = rng.choice(d_idx, size=n_cal, replace=False)
                q = conformal_quantile(S_dev[sub], MAIN_ALPHA)
                r = evaluate_sets(P_test[m], test.labels[m], classes, q)
                covs.append(r["coverage"]); defs.append(r["deferral"])
            abl[l][str(n_cal)] = {"coverage_mean": float(np.mean(covs)), "coverage_std": float(np.std(covs)),
                                  "coverage_min": float(np.min(covs)),
                                  "deferral_mean": float(np.mean(defs)), "deferral_std": float(np.std(defs))}
    res["calibration_size_ablation"] = abl

    with open(os.path.join(OUT_DIR, f"{task}_results.json"), "w") as f:
        json.dump(res, f, indent=1)
    return res


if __name__ == "__main__":
    summary = {}
    for task in TASKS:
        r = run_task(task)
        summary[task] = {"efficiency": r["efficiency"], "overall": r["overall"], "pooled_quantile": r["pooled_quantile"]}
        a = str(MAIN_ALPHA)
        print(f"\n{task}: alpha={MAIN_ALPHA}  pooled q={r['pooled_quantile'][a]:.4f}")
        print(f"{'lang':5s} {'ntr':>6s} {'acc':>6s} | {'cov_P':>6s} {'def_P':>6s} {'sacc_P':>6s} | {'cov_C':>6s} {'def_C':>6s} {'sacc_C':>6s}")
        for l, d in r["languages"].items():
            p = d["by_alpha"][a]["pooled"]; c = d["by_alpha"][a]["conditional"]
            sp = p['selective_accuracy']; sc = c['selective_accuracy']
            print(f"{l:5s} {d['n_train']:6d} {d['accuracy']:6.3f} | {p['coverage']:6.3f} {p['deferral']:6.3f} {(sp if sp is not None else float('nan')):6.3f} | {c['coverage']:6.3f} {c['deferral']:6.3f} {(sc if sc is not None else float('nan')):6.3f}")
        print("overall:", json.dumps(r["overall"][a]))
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
