# Data

Both benchmarks are public and are not copied into this repository. Download them with

```bash
python data/download.py
```

This fetches the TSV splits from the official GitHub repositories at pinned commits and writes
`data/<task>/<lang>/{train,dev,test}.tsv` (about 110 MB):

| Task | Source | Commit | Languages |
|---|---|---|---|
| MasakhaNEWS (news topic, 7 classes) | [masakhane-io/masakhane-news](https://github.com/masakhane-io/masakhane-news) | `819be05` | amh eng fra hau ibo lin lug orm pcm run sna som swa tir xho yor |
| AfriSenti-SemEval 2023 (tweet sentiment, 3 classes) | [afrisenti-semeval/afrisent-semeval-2023](https://github.com/afrisenti-semeval/afrisent-semeval-2023) | `5aec3cf` | amh arq ary hau ibo kin pcm por swa tso twi yor, plus orm and tir (dev/test only) |

The same data is mirrored on Hugging Face (`masakhane/masakhanews`, `shmuhammad/AfriSenti-twitter-sentiment`),
but the paper's results were produced from the GitHub TSVs above, so the download script uses those.

Please cite the dataset papers (Adelani et al., 2023; Muhammad et al., 2023) and follow their licenses.
