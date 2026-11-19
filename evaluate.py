import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from torch.utils.data import DataLoader

from DownstreamTask.data import RegionDataset, load_gdp, load_split
from DownstreamTask.gdp import METHODS, build_head, configure_device, evaluate, train_gdp, write_results


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CONFIG_DIR = ROOT / "config"
OUTPUT_DIR = ROOT / "results"
CHECKPOINT = ROOT / "checkpoint"


def evaluate_run(device):
    config = json.loads((CONFIG_DIR / "downstream.json").read_text())
    labels = load_gdp(DATA_DIR / config["label_file"])
    region_ids = np.load(CHECKPOINT / "h3_order.npy", allow_pickle=False).tolist()
    split = load_split(DATA_DIR / config["split_file"], config["split_seed"], region_ids)
    embeddings = tuple(np.load(CHECKPOINT / f"{name}_embeddings.npy", allow_pickle=False)
                       for name in ("shared", "functional", "visual"))

    rows = []
    with TemporaryDirectory(prefix="disco-eval-") as work_dir:
        heads = Path(work_dir) / "heads"
        mean, std = train_gdp(embeddings, region_ids, labels, split, config, device, heads)
        test_ids = split[2]
        targets = {key: (labels[key] - mean) / (std + 1e-6) for key in test_ids}
        dataset = RegionDataset(region_ids, embeddings, targets, test_ids)
        loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)

        for seed in config["seeds"]:
            model = build_head(embeddings[0].shape[1], config, device)
            model.load_state_dict(torch.load(heads / f"full_{seed}.pt",
                                             map_location=device, weights_only=True))
            metrics = evaluate(model, loader, device, mean, std)
            rows.append({"method": METHODS["full"], "seed": seed, **metrics})
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_results(OUTPUT_DIR, rows)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the Singapore GDP predictor from the shipped checkpoint."
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    evaluate_run(configure_device(args.device))


if __name__ == "__main__":
    main()
