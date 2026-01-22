#!/usr/bin/env python3
"""
5-fold cross validation for GTOmegaDTAModel.
Example:
  python train_5fold_multiGeo.py \
    --config checkpoints/best_three/assets/configs/kiba_cold_drug.yaml \
    --dataset KIBA \
    --csv_file data/KIBA/KIBA.csv \
    --protein_conformer_dir data/KIBA/results \
    --colabfold_features_dir data/KIBA/colabfold_single_repr \
    --save_dir checkpoints/5fold_gtomega
"""
from __future__ import annotations

import argparse
import copy
import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parent
CODE_DIR = ROOT / "checkpoints" / "best_three" / "code"
if CODE_DIR.exists():
    sys.path.insert(0, str(CODE_DIR))
else:
    sys.path.insert(0, str(ROOT))

from data.dataset import MultiConformerDTADataset, collate_batch  # noqa: E402
from models.dta_model_gtomega import GTOmegaDTAModel  # noqa: E402
from training.trainer import DTATrainer  # noqa: E402
from utils.metrics import compute_metrics  # noqa: E402


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_path(value: str | None, config_path: Path) -> str | None:
    if not value:
        return value
    path = Path(value)
    if path.is_absolute():
        return str(path)
    if value.startswith("assets/"):
        # configs located under checkpoints/best_three/assets/configs
        base = config_path.parents[2]
        return str(base / path)
    return str(Path.cwd() / path)


def resolve_config_paths(config: dict, config_path: Path) -> dict:
    cfg = copy.deepcopy(config)

    data_cfg = cfg.setdefault("data", {})
    if "cache_dir" in data_cfg:
        data_cfg["cache_dir"] = resolve_path(data_cfg.get("cache_dir"), config_path)

    protein_cfg = cfg.setdefault("graph_construction", {}).setdefault("protein", {})
    if "contact_map_dir" in protein_cfg:
        protein_cfg["contact_map_dir"] = resolve_path(
            protein_cfg.get("contact_map_dir"), config_path
        )

    return cfg


def maybe_extract_kiba_csv(csv_path: Path) -> None:
    if csv_path.exists():
        return
    archive = csv_path.parent / "KIBA.7z"
    if not archive.exists():
        raise FileNotFoundError(f"Missing CSV and archive: {csv_path}")
    logging.info("Extracting %s -> %s", archive, csv_path)
    cmd = ["7z", "e", str(archive), f"-o{csv_path.parent}", "-y"]
    subprocess.run(cmd, check=True)


def evaluate(model, loader, device, label_mean=None, label_std=None, return_preds=False):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            protein_graphs = batch["protein_conformers"]
            protein_conformers = [g.to(device) for g in protein_graphs]
            drug_graph = batch["drug_graphs"].to(device)
            labels = batch["affinities"].to(device)

            mask = batch.get("protein_conformer_mask")
            scores = batch.get("protein_conformer_scores")
            if mask is not None:
                mask = mask[:, :len(protein_conformers)].to(device)
            if scores is not None:
                scores = scores[:, :len(protein_conformers)].to(device)

            outputs = model(
                protein_conformers,
                drug_graph,
                protein_conformer_mask=mask,
                protein_conformer_scores=scores,
            )
            preds.append(outputs["affinity"].cpu())
            targets.append(labels.cpu())

    if not preds:
        metrics = compute_metrics([], [])
        if return_preds:
            return metrics, np.array([]), np.array([])
        return metrics
    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()
    if label_mean is not None and label_std is not None:
        preds = preds * label_std + label_mean
        targets = targets * label_std + label_mean
    metrics = compute_metrics(targets, preds)
    if return_preds:
        return metrics, preds, targets
    return metrics


def build_model(config: dict) -> torch.nn.Module:
    return GTOmegaDTAModel(config)


def main() -> None:
    parser = argparse.ArgumentParser(description="5-Fold CV for GTOmegaDTAModel")
    parser.add_argument("--config", required=True, help="Config YAML path")
    parser.add_argument("--dataset", choices=["Davis", "KIBA"], default="Davis")
    parser.add_argument("--csv_file", default="", help="CSV file path (defaults by dataset)")
    parser.add_argument("--protein_conformer_dir", required=True)
    parser.add_argument("--colabfold_features_dir", default="")
    parser.add_argument("--pair_repr_dir", default="")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--k_folds", type=int, default=5)
    parser.add_argument("--val_ratio", type=float, default=0.125)
    parser.add_argument("--start_fold", type=int, default=1)
    parser.add_argument("--end_fold", type=int, default=None)
    parser.add_argument("--save_dir", default="checkpoints/5fold_gtomega")
    args = parser.parse_args()

    config_path = Path(args.config)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    config = resolve_config_paths(config, config_path)

    set_seed(config["general"]["seed"])

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    csv_file = args.csv_file
    if not csv_file:
        if args.dataset == "Davis":
            csv_file = "data/Davis/davis.csv"
        else:
            csv_file = "data/KIBA/KIBA.csv"
    csv_path = Path(csv_file)
    if args.dataset == "KIBA":
        maybe_extract_kiba_csv(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    full_data = pd.read_csv(csv_path)
    logger.info("Loaded %d samples from %s", len(full_data), csv_path)

    if full_data["Y"].max() > 100:
        logger.info("Converting labels from nM to pKd")
        full_data["Y"] = -np.log10(full_data["Y"].astype(float) / 1e9)
        logger.info(
            "Transformed label range: [%.4f, %.4f]",
            full_data["Y"].min(),
            full_data["Y"].max(),
        )

    kf = KFold(
        n_splits=args.k_folds,
        shuffle=True,
        random_state=config["general"]["seed"],
    )

    save_root = Path(args.save_dir) / args.dataset
    save_root.mkdir(parents=True, exist_ok=True)

    results = []

    for fold, (train_full_idx, test_idx) in enumerate(kf.split(full_data), start=1):
        if fold < args.start_fold:
            logger.info("Skipping fold %d (before start_fold=%d)", fold, args.start_fold)
            continue
        if args.end_fold is not None and fold > args.end_fold:
            logger.info("Stopping at fold %d (end_fold=%d)", fold, args.end_fold)
            break
        logger.info("=" * 60)
        logger.info("Fold %d / %d", fold, args.k_folds)
        logger.info("=" * 60)

        test_data = full_data.iloc[test_idx]
        train_full_data = full_data.iloc[train_full_idx]
        train_data, val_data = train_test_split(
            train_full_data,
            test_size=args.val_ratio,
            random_state=config["general"]["seed"],
        )

        fold_dir = save_root / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        train_csv = fold_dir / "train.csv"
        val_csv = fold_dir / "valid.csv"
        test_csv = fold_dir / "test.csv"
        train_data.to_csv(train_csv, index=False)
        val_data.to_csv(val_csv, index=False)
        test_data.to_csv(test_csv, index=False)

        label_mean = None
        label_std = None
        if config.get("data", {}).get("label_normalization", False):
            label_mean = train_data["Y"].mean()
            label_std = train_data["Y"].std() + 1e-8

        cache_dir = config.get("data", {}).get("cache_dir")
        use_cache = config.get("data", {}).get("use_cache", True)

        train_dataset = MultiConformerDTADataset(
            csv_file=str(train_csv),
            protein_conformer_dir=args.protein_conformer_dir,
            config=config,
            cache_dir=cache_dir,
            use_cache=use_cache,
            label_mean=label_mean,
            label_std=label_std,
            pair_repr_dir=args.pair_repr_dir or None,
            colabfold_features_dir=args.colabfold_features_dir or None,
        )
        val_dataset = MultiConformerDTADataset(
            csv_file=str(val_csv),
            protein_conformer_dir=args.protein_conformer_dir,
            config=config,
            cache_dir=cache_dir,
            use_cache=use_cache,
            label_mean=label_mean,
            label_std=label_std,
            pair_repr_dir=args.pair_repr_dir or None,
            colabfold_features_dir=args.colabfold_features_dir or None,
        )
        test_dataset = MultiConformerDTADataset(
            csv_file=str(test_csv),
            protein_conformer_dir=args.protein_conformer_dir,
            config=config,
            cache_dir=cache_dir,
            use_cache=use_cache,
            label_mean=label_mean,
            label_std=label_std,
            pair_repr_dir=args.pair_repr_dir or None,
            colabfold_features_dir=args.colabfold_features_dir or None,
        )

        num_workers = config["data"]["num_workers"]
        pin_memory = config["data"]["pin_memory"]
        common_loader_kwargs = {
            "batch_size": config["training"]["batch_size"],
            "collate_fn": collate_batch,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
        }

        train_loader = DataLoader(train_dataset, shuffle=True, **common_loader_kwargs)
        val_loader = DataLoader(val_dataset, shuffle=False, **common_loader_kwargs)
        test_loader = DataLoader(test_dataset, shuffle=False, **common_loader_kwargs)

        model = build_model(config).to(device)
        trainer = DTATrainer(model, config, device=str(device))
        trainer.label_mean = label_mean
        trainer.label_std = label_std

        trainer.train_model(
            train_loader,
            val_loader,
            num_epochs=config["training"]["num_epochs"],
            save_dir=str(fold_dir),
        )

        checkpoint = torch.load(trainer.best_model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        metrics, preds, targets = evaluate(
            model,
            test_loader,
            device,
            label_mean,
            label_std,
            return_preds=True,
        )

        metrics_file = fold_dir / "test_metrics.json"
        with open(metrics_file, "w") as f:
            import json

            json.dump(
                {
                    "fold": fold,
                    "best_epoch": checkpoint.get("epoch", 0),
                    "best_val_loss": checkpoint.get("val_metrics", {}).get("val_loss", 0),
                    "metrics": metrics,
                    "predictions": preds.tolist(),
                    "targets": targets.tolist(),
                },
                f,
                indent=2,
            )

        logger.info(
            "Fold %d metrics | RMSE %.4f | MAE %.4f | Pearson %.4f | Spearman %.4f | CI %.4f | Rm2 %.4f",
            fold,
            metrics["rmse"],
            metrics["mae"],
            metrics["pearson"],
            metrics["spearman"],
            metrics["ci"],
            metrics["rm2"],
        )

        results.append(metrics)
        torch.cuda.empty_cache()

    summary = {}
    for key in ["mse", "rmse", "mae", "pearson", "spearman", "ci", "rm2"]:
        values = [item[key] for item in results]
        summary[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }

    summary_file = save_root / "summary.json"
    with open(summary_file, "w") as f:
        import json

        json.dump(summary, f, indent=2)

    logger.info("5-fold summary saved to %s", summary_file)


if __name__ == "__main__":
    main()
