"""
Standalone VAE reconstruction inference script.
Loads a VAE checkpoint, reconstructs motions from the test set,
computes reconstruction metrics (MPJPE, PAMPJPE, ACCEL, DTW),
and saves GT/Pred joint .npy files to the specified output directory.

Usage:
    python infer_vae.py \
        --checkpoint /path/to/stage1.ckpt \
        --output_dir outputs/vae_reconstruction \
        --cfg configs/config_h3d_stage1.yaml \
        --device 0 \
        --batch_size 32 \
        --split train_part
"""

import argparse
import json
import os
import logging
import time
import glob
import numpy as np
import torch
from collections import OrderedDict
from os.path import join as pjoin
from tqdm import tqdm
from omegaconf import OmegaConf
from rich import get_console
from rich.table import Table

from mGPT.config import get_module_config, instantiate_from_config
from mGPT.data.build_data import build_data
from mGPT.models.build_model import build_model
from mGPT.metrics.mr import MRMetrics


def parse_infer_args():
    parser = argparse.ArgumentParser(description="VAE Reconstruction Inference")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to VAE checkpoint (.ckpt)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save GT/Pred npy files and metrics")
    _repo_root = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--cfg", type=str,
                        default=os.path.join(_repo_root, "configs", "config_h3d_stage1.yaml"),
                        help="Path to experiment config yaml")
    parser.add_argument("--cfg_assets", type=str,
                        default=os.path.join(_repo_root, "configs", "assets.yaml"),
                        help="Path to assets config yaml")
    parser.add_argument("--device", type=int, default=0,
                        help="GPU device id")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for inference")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split to evaluate (test/val)")
    return parser.parse_args()


def load_config(args):
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    cfg_assets = OmegaConf.load(args.cfg_assets)
    cfg_base = OmegaConf.load(pjoin(cfg_assets.CONFIG_FOLDER, "default.yaml"))
    cfg_exp = OmegaConf.merge(cfg_base, OmegaConf.load(args.cfg))
    if not cfg_exp.FULL_CONFIG:
        cfg_exp = get_module_config(cfg_exp, cfg_assets.CONFIG_FOLDER)
    cfg = OmegaConf.merge(cfg_exp, cfg_assets)

    cfg.TRAIN.STAGE = "vae"
    cfg.DEVICE = [args.device]
    cfg.TEST.BATCH_SIZE = args.batch_size
    cfg.TEST.SPLIT = args.split
    cfg.DEBUG = False
    cfg.TRAIN.RESUME = ""
    cfg.TIME = time.strftime("%Y-%m-%d-%H-%M-%S")
    return cfg


def setup_logger(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger("infer_vae")
    logger.setLevel(logging.INFO)
    fmt = "%(asctime)-15s %(message)s"
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(fmt))
    logger.addHandler(console)
    fh = logging.FileHandler(pjoin(output_dir, "infer_vae.log"), "w")
    fh.setFormatter(logging.Formatter(fmt))
    logger.addHandler(fh)
    return logger


def load_vae_weights(model, ckpt_path, logger):
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    vae_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("vae."):
            vae_dict[k[len("vae."):]] = v
    if not vae_dict:
        raise RuntimeError(
            f"No VAE weights (keys starting with 'vae.') found in {ckpt_path}. "
            f"Available top keys: {list(state_dict.keys())[:10]}")
    model.vae.load_state_dict(vae_dict, strict=True)
    logger.info(f"VAE loaded from {ckpt_path} (strict=True, {len(vae_dict)} tensors)")


def print_metrics_table(metrics, logger):
    table = Table(title="VAE Reconstruction Metrics")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")
    for key, value in metrics.items():
        table.add_row(key, f"{value:.6f}")
    get_console().print(table, justify="center")
    logger.info(f"Metrics: {metrics}")


def main():
    args = parse_infer_args()

    # Output directories
    gt_dir = pjoin(args.output_dir, "GT")
    pred_dir = pjoin(args.output_dir, "Pred")
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    logger = setup_logger(args.output_dir)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Split: {args.split}, Device: cuda:{args.device}, Batch size: {args.batch_size}")

    # Config & data
    cfg = load_config(args)

    # Need FOLDER_EXP for model build
    cfg.FOLDER_EXP = args.output_dir

    datamodule = build_data(cfg, phase="test")
    logger.info(f"Dataset initialized (nfeats={datamodule.nfeats}, njoints={datamodule.njoints})")

    # Build model (stage='vae' so LLM is skipped)
    model = build_model(cfg, datamodule)
    logger.info("Model built (VAE only)")

    # Load VAE checkpoint
    load_vae_weights(model, args.checkpoint, logger)

    # Move to device
    device = torch.device(f"cuda:{args.device}")
    model.vae.to(device)
    model.vae.eval()

    # Metrics
    mr_metrics = MRMetrics(
        njoints=datamodule.njoints,
        jointstype="humanml3d",
        force_in_meter=True,
        dist_sync_on_step=False,
    )

    # Dataloader
    datamodule.setup(stage="test")
    test_loader = datamodule.test_dataloader()
    logger.info(f"Test set: {len(test_loader.dataset)} samples, {len(test_loader)} batches")

    sample_idx = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="VAE Inference"):
            feats_ref = batch["motion"].to(device)
            lengths = batch["length"]

            feats_rst = torch.zeros_like(feats_ref)
            for i in range(len(feats_ref)):
                if lengths[i] == 0:
                    continue
                feats_pred, _, _, _ = model.vae(feats_ref[i:i + 1, :lengths[i]])
                feats_rst[i:i + 1, :feats_pred.shape[1], :] = feats_pred

            # Convert features to joints (on CPU to avoid GPU memory pressure)
            joints_ref = datamodule.feats2joints(feats_ref.cpu())
            joints_rst = datamodule.feats2joints(feats_rst.cpu())

            # Update metrics
            mr_metrics.update(joints_rst, joints_ref, lengths)

            # Save per-sample joint npy files
            for i in range(len(feats_ref)):
                seq_len = lengths[i]
                gt_np = joints_ref[i, :seq_len].numpy()
                pred_np = joints_rst[i, :seq_len].numpy()
                np.save(pjoin(gt_dir, f"{sample_idx:05d}.npy"), gt_np)
                np.save(pjoin(pred_dir, f"{sample_idx:05d}.npy"), pred_np)
                sample_idx += 1

    # Compute final metrics
    metrics_raw = mr_metrics.compute(sanity_flag=False)
    metrics = {k: v.item() for k, v in metrics_raw.items()}

    # Display
    print_metrics_table(metrics, logger)

    # Save metrics to JSON
    metrics_path = pjoin(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    logger.info("=" * 60)
    logger.info(f"GT joints saved to:   {gt_dir}  ({sample_idx} files)")
    logger.info(f"Pred joints saved to: {pred_dir}  ({sample_idx} files)")
    logger.info(f"Metrics saved to:     {metrics_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
