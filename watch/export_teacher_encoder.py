import argparse
import os

import torch

from watch.vae.model import GraphSequenceVAE


def parse_args():
    parser = argparse.ArgumentParser(description="Export frozen teacher encoder block from a watch VAE checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to watch_vae_best.pt or watch_vae_last.pt")
    parser.add_argument("--output", type=str, default="", help="Output path for teacher encoder checkpoint")
    parser.add_argument(
        "--scope",
        type=str,
        default="backbone",
        choices=["backbone", "full_encoder"],
        help="Which part of encoder to export.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu")

    if "model_config" not in ckpt or "model_state" not in ckpt:
        raise ValueError("Checkpoint must contain model_config and model_state")

    model = GraphSequenceVAE.from_config(ckpt["model_config"])
    model.load_state_dict(ckpt["model_state"], strict=True)

    if args.output:
        out_path = args.output
    else:
        base_dir = os.path.dirname(args.checkpoint) or "."
        out_path = os.path.join(base_dir, "teacher_encoder_export.pt")

    teacher_ckpt = {
        "source_checkpoint": args.checkpoint,
        "epoch": ckpt.get("epoch"),
        "run_name": ckpt.get("run_name"),
        "run_output_dir": ckpt.get("run_output_dir"),
        "teacher_scope": args.scope,
        "teacher_prefixes": model.get_teacher_prefixes(scope=args.scope),
        "model_config": model.get_config(),
        "encoder_state": model.get_teacher_state_dict(scope=args.scope),
        "tensorizer_config": ckpt.get("tensorizer_config"),
        "val_metrics": ckpt.get("val_metrics"),
        "args": ckpt.get("args"),
    }

    torch.save(teacher_ckpt, out_path)
    print("Saved teacher encoder to:", out_path)
    print("Teacher scope:", args.scope)


if __name__ == "__main__":
    main()
