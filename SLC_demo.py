"""Run the one-turn motion-to-text-to-text-to-motion SignGPT pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import pipeline

from mGPT.config import get_module_config
from mGPT.data.build_data import build_data
from mGPT.models.build_model import build_model


SYSTEM_PROMPT = (
    "Reply in everyday conversational English with exactly one short sentence "
    "of no more than 30 words. Answer the user's question directly."
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", default=str(root / "configs/config_h3d_stage2.yaml"))
    parser.add_argument("--cfg_assets", default=str(root / "configs/assets.yaml"))
    parser.add_argument("--checkpoint", required=True, help="SignGPT checkpoint")
    parser.add_argument(
        "--mediator",
        default=os.environ.get("SIGNGPT_INSTRUCT_LLM_PATH"),
        help="Instruction-tuned LLM directory or Hugging Face model ID",
    )
    parser.add_argument(
        "--inputs", required=True, help="Text file containing one .npy path per line"
    )
    parser.add_argument("--output_dir", default="outputs/sign_conversation")
    parser.add_argument("--device", default=None, help="For example: cuda:0 or cpu")
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def load_config(config_path: str, assets_path: str):
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    assets = OmegaConf.load(assets_path)
    base = OmegaConf.load(Path(assets.CONFIG_FOLDER) / "default.yaml")
    experiment = OmegaConf.merge(base, OmegaConf.load(config_path))
    if not experiment.FULL_CONFIG:
        experiment = get_module_config(experiment, assets.CONFIG_FOLDER)
    return OmegaConf.merge(experiment, assets)


def read_motion_paths(path: str | Path) -> list[Path]:
    with Path(path).open(encoding="utf-8") as handle:
        paths = [Path(line.strip()).expanduser() for line in handle if line.strip()]
    missing = [str(item) for item in paths if not item.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing motion files: {missing[:5]}")
    return paths


def build_mediator(model_name_or_path: str, device: torch.device):
    kwargs = {"torch_dtype": torch.bfloat16 if device.type == "cuda" else torch.float32}
    if device.type == "cuda":
        kwargs["device_map"] = "auto"
    else:
        kwargs["device"] = -1
    return pipeline("text-generation", model=model_name_or_path, **kwargs)


def load_signgpt(cfg, checkpoint: str, device: torch.device):
    datamodule = build_data(cfg)
    model = build_model(cfg, datamodule)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"])
    model.to(device).eval()
    return datamodule, model


def mediator_reply(generator, question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    output = generator(
        messages,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        top_k=30,
        max_new_tokens=50,
        pad_token_id=generator.tokenizer.eos_token_id,
    )[0]["generated_text"]
    if isinstance(output, list):
        return output[-1]["content"].strip()
    return str(output).strip()


@torch.inference_mode()
def run_sample(model, datamodule, generator, motion_path: Path, device: torch.device):
    features = torch.as_tensor(
        np.load(motion_path), dtype=torch.float32, device=device
    )
    normalized = datamodule.normalize(features)
    motion_tokens, _ = model.vae.encode(normalized.unsqueeze(0))

    translated = model.slc_forward(
        {
            "motion_tokens": motion_tokens,
            "motion_lengths": [motion_tokens.shape[1]],
            "task_prompts": ["<Motion_Placeholder>\n"],
        },
        task="m2t",
    )["texts"][0]

    reply = mediator_reply(generator, translated)
    generated = model.slc_forward(
        {
            "texts": [reply],
            "task_prompts": ["<Caption_Placeholder>\n"],
        },
        task="t2m",
    )
    length = int(generated["length"][0])
    response_motion = generated["joints"][0][:length].detach().cpu().numpy()
    input_motion = model.feats2joints(normalized).detach().cpu().numpy()
    return translated, reply, input_motion, response_motion


def main() -> None:
    args = parse_args()
    if not args.mediator:
        raise ValueError(
            "Provide --mediator or set SIGNGPT_INSTRUCT_LLM_PATH."
        )

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    torch.manual_seed(args.seed)
    device = torch.device(
        args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    cfg = load_config(args.cfg, args.cfg_assets)
    datamodule, model = load_signgpt(cfg, args.checkpoint, device)
    generator = build_mediator(args.mediator, device)

    output_dir = Path(args.output_dir)
    input_dir = output_dir / "input_motions"
    response_dir = output_dir / "response_motions"
    metadata_dir = output_dir / "metadata"
    for directory in (input_dir, response_dir, metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)

    for motion_path in tqdm(read_motion_paths(args.inputs), desc="Generating replies"):
        translated, reply, input_motion, response_motion = run_sample(
            model, datamodule, generator, motion_path, device
        )
        stem = motion_path.stem
        np.save(input_dir / f"{stem}.npy", input_motion)
        np.save(response_dir / f"{stem}.npy", response_motion)
        metadata = {
            "source": str(motion_path),
            "translated_question": translated,
            "mediator_reply": reply,
            "response_frames": len(response_motion),
        }
        (metadata_dir / f"{stem}.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    logging.info("Saved outputs to %s", output_dir)


if __name__ == "__main__":
    main()
