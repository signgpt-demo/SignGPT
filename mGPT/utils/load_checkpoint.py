import torch
from collections import OrderedDict


def load_pretrained(cfg, model, logger=None, phase="train"):
    """Load a full model checkpoint (VAE + LLM + losses)."""
    ckpt_path = cfg.TRAIN.PRETRAINED if phase == "train" else cfg.TEST.CHECKPOINTS

    if logger:
        logger.info(f"Loading full model from {ckpt_path}")

    state_dict = torch.load(ckpt_path, weights_only=False, map_location="cpu")["state_dict"]
    model.load_state_dict(state_dict, strict=True)

    if logger:
        logger.info("Full model loaded successfully")
    return model


def load_pretrained_vae(cfg, model, logger=None):
    """
    Strictly load VAE weights from a Stage 1 checkpoint.
    Supports both Stage 1 checkpoints (VAE-only) and legacy full-model checkpoints.
    """
    ckpt_path = cfg.TRAIN.PRETRAINED_VAE
    if logger:
        logger.info(f"Loading VAE weights from {ckpt_path}")

    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)['state_dict']

    vae_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("vae."):
            vae_dict[k[len("vae."):]] = v

    if not vae_dict:
        raise RuntimeError(
            f"No VAE weights (keys starting with 'vae.') found in {ckpt_path}. "
            f"Available keys: {list(state_dict.keys())[:10]}...")

    model.vae.load_state_dict(vae_dict, strict=True)

    if logger:
        logger.info(f"VAE loaded successfully (strict=True, {len(vae_dict)} parameter tensors)")
    return model
