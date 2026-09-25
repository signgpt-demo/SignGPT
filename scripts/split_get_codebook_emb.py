import os
import argparse
import pytorch_lightning as pl
import torch

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
    parser.add_argument("--save_name", type=str, default="signgpt_codebook.pt",
                        help="Filename for saved codebook params")
    args, unknown = parser.parse_known_args()

    sys.argv = [sys.argv[0]] + unknown
    cfg = parse_args(phase="test")
    cfg.TRAIN.STAGE = "vae"
    cfg.TRAIN.BATCH_SIZE = 1
    cfg.TRAIN.PRETRAINED_VAE = args.vae_ckpt

    pl.seed_everything(cfg.SEED_VALUE)

    if cfg.ACCELERATOR == "gpu":
        os.environ["PYTHONWARNINGS"] = "ignore"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    datasets = build_data(cfg, phase='test')
    model = build_model(cfg, datasets)
    if hasattr(model, "motion_vae"):
        model.vae = model.motion_vae

    load_pretrained_vae(cfg, model)

    if cfg.ACCELERATOR == "gpu":
        model = model.to('cuda')

    model.eval()

    codebook_dir = os.path.join(datasets.hparams.data_root, "codebook_params")
    os.makedirs(codebook_dir, exist_ok=True)
    codebook_save_path = os.path.join(codebook_dir, args.save_name)

    body_codebook = model.vae.body_quantizer.codebook.detach().cpu()
    left_hand_codebook = model.vae.left_hand_quantizer.codebook.detach().cpu()
    right_hand_codebook = model.vae.right_hand_quantizer.codebook.detach().cpu()

    params_to_save = {
        'body_codebook': body_codebook,
        'left_hand_codebook': left_hand_codebook,
        'right_hand_codebook': right_hand_codebook,
    }

    try:
        body_proj_up_params = {k: v.cpu() for k, v in model.vae.body_proj_up.state_dict().items()}
        lh_proj_up_params = {k: v.cpu() for k, v in model.vae.lh_proj_up.state_dict().items()}
        rh_proj_up_params = {k: v.cpu() for k, v in model.vae.rh_proj_up.state_dict().items()}
        params_to_save['body_proj_up'] = body_proj_up_params
        params_to_save['lh_proj_up'] = lh_proj_up_params
        params_to_save['rh_proj_up'] = rh_proj_up_params
        print("Projection layers (body_proj_up, lh_proj_up, rh_proj_up) included.")
    except AttributeError:
        print("No proj_up layers found in VAE, saving codebook only.")

    torch.save(params_to_save, codebook_save_path)

    print(f"\nSaved to -> {codebook_save_path}")
    for key, val in params_to_save.items():
        if isinstance(val, torch.Tensor):
            print(f"  - '{key}': {val.shape}")
        elif isinstance(val, dict):
            print(f"  - '{key}': state_dict with {len(val)} params")


if __name__ == "__main__":
    main()
