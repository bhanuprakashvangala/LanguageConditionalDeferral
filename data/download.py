"""Download the MasakhaNEWS and AfriSenti TSV splits used in the paper.

Files are fetched from the official GitHub repositories at pinned commits and written to
data/<task>/<lang>/{train,dev,test}.tsv, which is the layout src/experiment.py expects.

    python data/download.py
"""
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))

SOURCES = {
    # https://github.com/masakhane-io/masakhane-news
    "masakhanews": {
        "url": "https://raw.githubusercontent.com/masakhane-io/masakhane-news/"
               "819be052e79e0a600e0ed4a7356594681cb5c4f3/data/{lang}/{split}.tsv",
        "langs": ["amh", "eng", "fra", "hau", "ibo", "lin", "lug", "orm", "pcm", "run",
                  "sna", "som", "swa", "tir", "xho", "yor"],
    },
    # https://github.com/afrisenti-semeval/afrisent-semeval-2023
    "afrisenti": {
        "url": "https://raw.githubusercontent.com/afrisenti-semeval/afrisent-semeval-2023/"
               "5aec3cfcf87b59f885e98641f72e52182776429d/data/{lang}/{split}.tsv",
        "langs": ["amh", "arq", "ary", "hau", "ibo", "kin", "orm", "pcm", "por", "swa",
                  "tir", "tso", "twi", "yor"],
    },
}
SPLITS = ["train", "dev", "test"]


def main():
    for task, src in SOURCES.items():
        for lang in src["langs"]:
            out_dir = os.path.join(HERE, task, lang)
            os.makedirs(out_dir, exist_ok=True)
            for split in SPLITS:
                dest = os.path.join(out_dir, f"{split}.tsv")
                if os.path.exists(dest):
                    continue
                url = src["url"].format(lang=lang, split=split)
                try:
                    urllib.request.urlretrieve(url, dest)
                except Exception as e:  # noqa: BLE001
                    sys.exit(f"failed to download {url}: {e}")
            print(f"{task}/{lang} ok", flush=True)
    print("data written to", HERE)


if __name__ == "__main__":
    main()
