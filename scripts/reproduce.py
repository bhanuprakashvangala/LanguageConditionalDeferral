"""Regenerate the paper's figures and numbers.

    python scripts/reproduce.py           # from the saved results in results/ (seconds)
    python scripts/reproduce.py --full    # download data, retrain, re-run the analysis (about 10 minutes on CPU)

The full run writes to results/rerun/ so the saved paper results are never overwritten.
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def run(script, results_dir):
    env = dict(os.environ, RESULTS_DIR=results_dir)
    print(f"\n$ python {script}", flush=True)
    subprocess.run([sys.executable, os.path.join(ROOT, script)], check=True, env=env, cwd=ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="download data and retrain from scratch")
    args = ap.parse_args()
    if args.full:
        out = os.path.join(ROOT, "results", "rerun")
        os.makedirs(out, exist_ok=True)
        run("data/download.py", out)
        run("src/experiment.py", out)
        run("src/analysis.py", out)
    else:
        out = os.path.join(ROOT, "results")
    run("src/make_figures.py", out)
    run("src/report.py", out)


if __name__ == "__main__":
    main()
