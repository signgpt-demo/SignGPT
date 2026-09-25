"""Run file-based SignGPT generation or translation inference."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from mGPT.config import parse_args
from mGPT.data.build_data import build_data
from mGPT.models.build_model import build_model
from mGPT.utils.logger import create_logger


def load_examples(path: str, model) -> list[dict]:
    examples = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            text, separator, motion_path = line.partition("#")
            motion = None
            length = 0
            if separator:
                motion_file = Path(motion_path.strip()).expanduser()
                if not motion_file.is_file():
                    raise FileNotFoundError(
                        f"Motion file on line {line_number} does not exist: {motion_file}"
                    )
                features = torch.as_tensor(
                    np.load(motion_file), device=model.device, dtype=torch.float32
                )
                motion = model.datamodule.normalize(features).unsqueeze(0)
                length = features.shape[0]
            examples.append({"text": text.strip(), "motion": motion, "length": length})
    return examples


def main() -> None:
    cfg = parse_args(phase="demo")
    if not cfg.DEMO.EXAMPLE:
        raise ValueError("Provide an input file with --example.")
    if not cfg.TEST.CHECKPOINTS:
        raise ValueError("Set TEST.CHECKPOINTS in the selected configuration.")

    cfg.FOLDER = cfg.TEST.FOLDER
    logger = create_logger(cfg, phase="test")
    logger.info(OmegaConf.to_yaml(cfg))
    pl.seed_everything(cfg.SEED_VALUE)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    device = torch.device(
        "cuda:0" if cfg.ACCELERATOR == "gpu" and torch.cuda.is_available() else "cpu"
    )
    datamodule = build_data(cfg)
    model = build_model(cfg, datamodule)
    state = torch.load(cfg.TEST.CHECKPOINTS, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"])
    model.to(device).eval()

    output_dir = Path(cfg.TEST.FOLDER) / cfg.NAME / cfg.model.params.task
    output_dir.mkdir(parents=True, exist_ok=True)
    face_dir = output_dir / "face"
    face_dir.mkdir(exist_ok=True)

    output_texts = []
    examples = load_examples(cfg.DEMO.EXAMPLE, model)
    for index, example in enumerate(tqdm(examples, desc="Running inference")):
        batch = {
            "length": [example["length"]],
            "init_texts": [example["text"]],
            "motion_feat": example["motion"],
        }
        with torch.inference_mode():
            outputs = model(batch, task=cfg.model.params.task)

        texts = outputs.get("texts") or [""]
        output_texts.append(texts[0])
        joints = outputs.get("joints")
        features = outputs.get("feats")
        lengths = outputs.get("length") or []
        length = lengths[0] if lengths else None

        if joints is not None:
            motion = joints[0][:length].detach().cpu().numpy() if length else joints[0].detach().cpu().numpy()
            np.save(output_dir / f"{index}_out.npy", motion)
        if features is not None and features.shape[1] > 0:
            end = length or features.shape[1]
            np.save(
                face_dir / f"{index}_face.npy",
                features[0, :end, -10:].detach().cpu().numpy(),
            )
        (output_dir / f"{index}_in.txt").write_text(
            example["text"], encoding="utf-8"
        )

    (output_dir / "out.txt").write_text(
        "\n".join(output_texts) + "\n", encoding="utf-8"
    )
    logger.info("Saved %d outputs to %s", len(examples), output_dir)


if __name__ == "__main__":
    main()
