import os
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from mGPT.callback import build_callbacks
from mGPT.config import parse_args, instantiate_from_config
from mGPT.data.build_data import build_data
from mGPT.models.build_model import build_model
from mGPT.utils.logger import create_logger
from mGPT.utils.load_checkpoint import load_pretrained, load_pretrained_vae
import tempfile
from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint

# Temp dir for Lightning / tensor caching — override via $MGPT_TMP_DIR, else ./tmp relative to repo
_REPO_ROOT = Path(__file__).resolve().parent
TMP_DIR = os.environ.get("MGPT_TMP_DIR", str(_REPO_ROOT / ".tmp"))
Path(TMP_DIR).mkdir(exist_ok=True)

os.environ.update({
    'TMPDIR': TMP_DIR,
    'TMP': TMP_DIR,
    'TEMP': TMP_DIR,
    'PYTORCH_KERNEL_CACHE_PATH': TMP_DIR,
})

tempfile.tempdir = TMP_DIR

def main():
    cfg = parse_args(phase="train")
    logger = create_logger(cfg, phase="train")
    logger.info(OmegaConf.to_yaml(cfg))

    pl.seed_everything(cfg.SEED_VALUE)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    pl_loggers = []
    for logger_name in cfg.LOGGER.TYPE:
        logger_name = logger_name.upper()
        if logger_name == "WANDB" and not cfg.LOGGER.WANDB.params.project:
            continue
        pl_loggers.append(instantiate_from_config(cfg.LOGGER[logger_name]))

    callbacks = build_callbacks(cfg, logger=logger, phase='train')
    callbacks_to_remove = []
    for i, callback in enumerate(callbacks):
        if isinstance(callback, ModelCheckpoint):
            for j in range(i):
                if (isinstance(callbacks[j], ModelCheckpoint) and
                    callbacks[j].monitor == callback.monitor and
                    callbacks[j].mode == callback.mode and
                    callbacks[j].every_n_epochs == callback.every_n_epochs):
                    callbacks_to_remove.append(i)
                    break
    for i in reversed(callbacks_to_remove):
        callbacks.pop(i)
    logger.info("Callbacks initialized")

    datamodule = build_data(cfg)
    logger.info("datasets module {} initialized".format(
        cfg.DATASET.target.split('.')[-2]))

    model = build_model(cfg, datamodule)
    logger.info("model {} loaded".format(cfg.model.target))

    trainer = pl.Trainer(
        default_root_dir=cfg.FOLDER_EXP,
        max_epochs=cfg.TRAIN.END_EPOCH,
        logger=pl_loggers,
        callbacks=callbacks,
        check_val_every_n_epoch=cfg.LOGGER.VAL_EVERY_STEPS,
        accelerator=cfg.ACCELERATOR,
        devices=cfg.DEVICE,
        num_nodes=cfg.NUM_NODES,
        strategy="ddp_find_unused_parameters_false"
        if len(cfg.DEVICE) > 1 else 'auto',
        benchmark=False,
        deterministic=False,
    )
    logger.info("Trainer initialized")

    is_vae_stage = (cfg.TRAIN.STAGE == 'vae')
    is_resume = bool(cfg.TRAIN.RESUME)

    if is_vae_stage:
        if is_resume:
            logger.info(f"[VAE] Resuming training from {cfg.TRAIN.PRETRAINED}")
        elif cfg.TRAIN.PRETRAINED_VAE:
            load_pretrained_vae(cfg, model, logger)
            logger.info(f"[VAE] Fine-tuning from pretrained VAE: {cfg.TRAIN.PRETRAINED_VAE}")
        else:
            logger.info("[VAE] Training from scratch (only VAE weights will be saved)")
    else:
        if is_resume:
            logger.info(f"[LLM] Resuming training from {cfg.TRAIN.PRETRAINED}")
        else:
            if not cfg.TRAIN.PRETRAINED_VAE:
                raise ValueError(
                    "TRAIN.PRETRAINED_VAE must be set for LLM training! "
                    "Provide a Stage 1 (VAE) checkpoint path.")
            load_pretrained_vae(cfg, model, logger)

            if cfg.TRAIN.PRETRAINED:
                load_pretrained(cfg, model, logger)

    if is_resume:
        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.TRAIN.PRETRAINED)
    else:
        trainer.fit(model, datamodule=datamodule)

    logger.info(f"The outputs of this experiment are stored in {cfg.FOLDER_EXP}")
    logger.info("Training ends!")


if __name__ == "__main__":
    main()
