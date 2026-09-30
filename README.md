# One Threshold Does Not Fit All Languages

Code and results for **One Threshold Does Not Fit All Languages: Language-Conditional Deferral for Reliable and Efficient Low-Resource Text Classification**
Bhanu Prakash Vangala, Navya Vangala.
Oral, Global South in AI Workshop at NeurIPS 2026. [PDF](https://bhanuprakashvangala.github.io/files/papers/one-threshold.pdf)

A multilingual classifier with a human fallback usually gets one conformal threshold, estimated on validation data
pooled across languages. On MasakhaNEWS (16 African languages) and AfriSenti (12 languages plus 2 unseen), that
pooled threshold meets the 90% coverage target on average but not in the hardest languages. Estimating one threshold
per language fixes this without retraining, and shows how much human review each language really needs.

The model is a TF-IDF (character and word n-grams) + logistic regression classifier that trains on one CPU core.

## Layout

```
data/       download script for the MasakhaNEWS and AfriSenti TSVs (data is not included)
src/        experiment.py (train, cache probabilities), analysis.py (30 re-splits, both policies,
            ablation, official-split check), make_figures.py, report.py (prints the paper's numbers)
scripts/    reproduce.py, the single entry point
results/    the JSON outputs behind every number in the paper, and the figures
```

## Setup

```bash
git clone https://github.com/bhanuprakashvangala/LanguageConditionalDeferral.git
cd LanguageConditionalDeferral
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Reproduce

From the saved results (a few seconds, no data needed):

```bash
python scripts/reproduce.py
```

This redraws `results/figures/coverage_deferral.pdf` (Figure 2) and `results/figures/calibration_size.pdf`
(calibration-size ablation) from `results/*_analysis.json`, and prints the numbers quoted in the paper.

From scratch (downloads about 110 MB, then about 10 minutes on a laptop CPU):

```bash
python scripts/reproduce.py --full
```

This runs `data/download.py`, `src/experiment.py`, `src/analysis.py`, `src/make_figures.py` and `src/report.py`,
writing everything to `results/rerun/` so the saved results are left untouched.

## Results

Coverage target 0.90 (alpha = 0.1), mean over 30 random 50/50 calibration/evaluation splits of each language's
pooled dev+test data.

| | MasakhaNEWS | AfriSenti |
|---|---|---|
| Aggregate coverage, pooled threshold | 0.902 +/- 0.006 | 0.901 +/- 0.003 |
| Trained languages more than 2 points below target, pooled / per-language | 3 / 0 | 7 / 0 |
| Per-language coverage range, per-language threshold | 0.891 to 0.910 | 0.900 to 0.910 |
| Overall deferral, pooled / per-language | 6.6% / 10.1% | 59.4% / 58.4% |
| Training time on one CPU core / model size | 3.4 min / 40 MB | 1.3 min / 25 MB |

| Language | Pooled coverage | Per-language deferral |
|---|---|---|
| Somali (news) | 0.775 | 42.9% |
| Tigrinya (news) | 0.837 | 26.4% |
| Swahili (news) | 0.868 | 16.5% |
| Nigerian Pidgin (news) | 0.964 | under 8% |
| Tigrinya (tweets, unseen) | 0.775 | 95.0% |
| Oromo (tweets, unseen) | 0.812 | 96.8% |
| Amharic (tweets) | | 84.7% |
| Xitsonga (tweets) | | 85.1% |

`python src/report.py` prints these and the other numbers in the text (selective accuracy, Spearman correlations,
the ablation and the official-split check).

## Notes

- The saved `*_analysis.json` files are the outputs of the original run; `src/report.py` on them gives the paper's
  numbers exactly.
- A full re-run retrains the classifiers, so its numbers can move slightly with library versions. With Python 3.12
  and scikit-learn 1.8.0 it is repeatable run to run, and per-language coverage stays within 0.6 points and deferral
  or selective accuracy within 2 points of the saved results. Training times depend on the machine.
- `experiment.py` also writes `*_results.json` (calibration on the official dev split, evaluation on the official
  test split, alpha in {0.05, 0.1, 0.2}); the paper's base accuracy and macro-F1 come from this file. The paper's
  main numbers come from `analysis.py`.
- The per-language policy needs the language of each input; the paper uses the gold language label.

## Citation

```bibtex
@inproceedings{vangala2026onethreshold,
  title     = {One Threshold Does Not Fit All Languages: Language-Conditional Deferral for Reliable and Efficient Low-Resource Text Classification},
  author    = {Vangala, Bhanu Prakash and Vangala, Navya},
  booktitle = {Global South in AI Workshop at NeurIPS 2026},
  year      = {2026}
}
```

## License

Code: MIT. The datasets keep their own licenses; see `data/README.md`.
