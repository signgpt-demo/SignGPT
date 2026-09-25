import os
import argparse
import numpy as np
import pytorch_lightning as pl
import torch
from pathlib import Path
from tqdm import tqdm

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.insert(0, _REPO_ROOT)

from mGPT.config import parse_args
from mGPT.data.build_data import build_data
from mGPT.models.build_model import build_model
from mGPT.utils.load_checkpoint import load_pretrained_vae

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae_ckpt", type=str, required=True,
                        help="Path to the VQ-VAE checkpoint (.ckpt)")
    parser.add_argument("--output_name", type=str, default="signgpt_motion_tokens",
                        help="Output folder name under data_root (overrides DATASET.CODE_PATH)")
    args, unknown = parser.parse_known_args()

    sys.argv = [sys.argv[0]] + unknown
    cfg = parse_args(phase="test")
    cfg.TRAIN.STAGE = "token"
    cfg.TRAIN.BATCH_SIZE = 1
    if args.vae_ckpt:
        cfg.TRAIN.PRETRAINED_VAE = args.vae_ckpt

    if args.output_name:
        cfg.DATASET.CODE_PATH = args.output_name

    pl.seed_everything(cfg.SEED_VALUE)

    if cfg.ACCELERATOR == "gpu":
        os.environ["PYTHONWARNINGS"] = "ignore"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    datasets = build_data(cfg, phase='token')
    print("datasets module initialized")
    output_dir = os.path.join(datasets.hparams.data_root, cfg.DATASET.CODE_PATH)
    os.makedirs(output_dir, exist_ok=True)

    model = build_model(cfg, datasets)
    if hasattr(model, "motion_vae"):
        model.vae = model.motion_vae
    print("model loaded")

    load_pretrained_vae(cfg, model)

    device = torch.device('cuda')
    if cfg.ACCELERATOR == "gpu":
        model = model.to(device)

    model.eval()

    with torch.no_grad():
        for batch in tqdm(datasets.train_dataloader(), desc='motion tokenize'):
            name = batch['text']
            pose = batch['motion']
            pose = pose.to(device).float()

            if pose.shape[1] == 0:
                continue

            code_dict, _ = model.vae.encode(pose)
            target_tensor = code_dict
            target = target_tensor.to('cpu').numpy()

            target_path = os.path.join(output_dir, name[0] + '.npy')
            Path(target_path).parent.mkdir(parents=True, exist_ok=True)
            np.save(target_path, target)

    print(f'Motion tokenization done, the motion tokens are saved to {output_dir}')


if __name__ == "__main__":
    main()
