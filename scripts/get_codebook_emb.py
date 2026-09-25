import os
import pytorch_lightning as pl
import torch

# Add repo root to sys.path so `mGPT` package is importable when running as `python -m scripts.xxx`
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.insert(0, _REPO_ROOT)

from mGPT.config import parse_args
from mGPT.data.build_data import build_data
from mGPT.models.build_model import build_model
from mGPT.utils.load_checkpoint import load_pretrained_vae

def main():
    # parse options
    cfg = parse_args(phase="test")
    cfg.TRAIN.STAGE = "vae"
    cfg.TRAIN.BATCH_SIZE = 1

    # set seed
    pl.seed_everything(cfg.SEED_VALUE)

    # gpu setting
    if cfg.ACCELERATOR == "gpu":
        os.environ["PYTHONWARNINGS"] = "ignore"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # create dataset and model
    datasets = build_data(cfg, phase='test')
    model = build_model(cfg, datasets)
    if hasattr(model, "motion_vae"):
        model.vae = model.motion_vae

    # load pretrained VAE
    load_pretrained_vae(cfg, model)

    if cfg.ACCELERATOR == "gpu":
        model = model.to('cuda')

    # save codebook
    codebook_dir = os.path.join(datasets.hparams.data_root, "codebook_params")
    os.makedirs(codebook_dir, exist_ok=True)

    model_name = os.path.basename(cfg.TRAIN.PRETRAINED_VAE).replace('.ckpt', '').replace('.pth', '')
    codebook_save_path = os.path.join(codebook_dir, f"codebook_emb_params.pt")

    codebook = model.vae.quantizer.codebook.detach().cpu()
    torch.save(codebook, codebook_save_path)

    print(f"Codebook saved: {codebook.shape} -> {codebook_save_path}")

if __name__ == "__main__":
    main()