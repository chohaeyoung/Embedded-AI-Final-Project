"""Download a subset of ILSVRC/imagenet-1k (parquet shards) from Hugging Face.

The dataset is gated: accept the terms at
https://huggingface.co/datasets/ILSVRC/imagenet-1k and log in first (`hf auth login`).

Usage:
    python scripts/download_imagenet.py                  # 2 train shards + all 14 validation shards
    python scripts/download_imagenet.py --train-shards 4 --val-shards 2
"""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "ILSVRC/imagenet-1k"
NUM_TRAIN = 294
NUM_VAL = 14


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-shards", type=int, default=2)
    parser.add_argument("--val-shards", type=int, default=NUM_VAL)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "data" / "imagenet-1k")
    args = parser.parse_args()

    patterns = ["classes.py", "README.md"]
    patterns += [f"data/train-{i:05d}-of-{NUM_TRAIN:05d}.parquet" for i in range(args.train_shards)]
    patterns += [f"data/validation-{i:05d}-of-{NUM_VAL:05d}.parquet" for i in range(args.val_shards)]

    path = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        allow_patterns=patterns,
        local_dir=args.out,
    )
    print(f"Downloaded to {path}")
    for f in sorted(Path(path, "data").glob("*.parquet")):
        print(f"  {f.name}  {f.stat().st_size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
