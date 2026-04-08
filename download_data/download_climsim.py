#!/usr/bin/env python3
"""
Download script for ClimSim low-resolution real geography dataset from HuggingFace.

Dataset: https://huggingface.co/datasets/LEAP/ClimSim_low-res
Size: ~744 GB
Samples: 100 million
Resolution: 11.5° x 11.5° (384 grid columns)

Uses huggingface_hub.snapshot_download to fetch raw NetCDF (.nc) files directly.
"""

import os
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download


DATASET_NAME = "LEAP/ClimSim_low-res"


def download_climsim_dataset(
    local_dir: str,
    split: str = "train",
    num_proc: int = 4,
):
    """
    Download ClimSim NetCDF files from HuggingFace using snapshot_download.

    Args:
        local_dir: Directory to save the downloaded files.
        split: Dataset split to download ("train", "val", or "test").
        num_proc: Number of parallel download workers.
    """
    os.makedirs(local_dir, exist_ok=True)

    print(f"Downloading ClimSim dataset: {DATASET_NAME}")
    print(f"Split: {split}")
    print(f"Destination: {local_dir}")
    print(f"Workers: {num_proc}")

    # Only download files that belong to the requested split folder
    allow_patterns = [f"{split}/*"]

    try:
        path = snapshot_download(
            repo_id=DATASET_NAME,
            repo_type="dataset",
            local_dir=local_dir,
            allow_patterns=allow_patterns,
            max_workers=num_proc,
        )
        print(f"\n✓ Download complete. Files saved to: {path}")

    except Exception as e:
        print(f"\n✗ Error downloading dataset: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Download ClimSim low-resolution dataset from HuggingFace"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        required=True,
        help="Directory to save downloaded NetCDF files",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Dataset split to download (default: train)",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=4,
        help="Number of parallel download workers (default: 4)",
    )

    args = parser.parse_args()
    local_dir = str(Path(args.cache_dir).expanduser())

    download_climsim_dataset(
        local_dir=local_dir,
        split=args.split,
        num_proc=args.num_proc,
    )


if __name__ == "__main__":
    main()
