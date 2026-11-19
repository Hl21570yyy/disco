import argparse
import json
from pathlib import Path

import numpy as np
import torch

from IntraModality.embeddings import load_modalities


ROOT = Path(__file__).resolve().parent


def parse_args(description="Train Disco and export the region embeddings."):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--config-dir", type=Path, default=ROOT / "config")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def train(data_dir, config_dir, output_dir, device):
    from DisentanglePositive.main import fit

    model_config = json.loads((config_dir / "model.json").read_text())
    poi_ids = np.load(data_dir / "poi_h3_order.npy", allow_pickle=False).astype(str)
    region_ids = sorted(key.strip().lower() for key in poi_ids)
    functional, visual = load_modalities(data_dir, region_ids, model_config)
    representation_dir = output_dir / "DisentanglePositive"
    representation_dir.mkdir(parents=True, exist_ok=True)
    print(f"DisentanglePositive device: {device}", flush=True)
    fit(functional, visual, region_ids, model_config, device, representation_dir)


def main():
    args = parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    train(args.data_dir, args.config_dir, args.output_dir, device)


if __name__ == "__main__":
    main()
