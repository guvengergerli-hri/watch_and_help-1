import argparse
import os
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from watch.vae.dataset import WatchVAEDataset, collate_watch_vae
from watch.vae.model import GraphSequenceVAE
from watch.vae.tensorizer import WatchGraphTensorizer
from watch.vae_joint_train import (
    compute_val_belief_profile,
    load_goal_predicates,
    print_device_summary,
    resolve_device,
    save_and_log_belief_profile,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build belief_profile artifacts for a trained joint Watch VAE checkpoint."
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to watch_vae_joint_{best,last}.pt")
    parser.add_argument("--metadata", type=str, default=None, help="Override metadata path")
    parser.add_argument("--data-json", type=str, default=None, help="Override dataset JSON path")
    parser.add_argument("--split-key", type=str, default=None, help="Override split key")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-demos", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--min-seq-len", type=int, default=2)
    parser.add_argument("--stable-slots", action="store_true", default=None)
    parser.add_argument("--no-stable-slots", action="store_false", dest="stable_slots")
    parser.set_defaults(stable_slots=None)
    parser.add_argument("--belief-profile-bins", type=int, default=21)
    parser.add_argument("--epoch", type=int, default=None, help="Epoch number to use in output filenames")
    parser.add_argument("--file-prefix", type=str, default="val_belief_profile")
    parser.add_argument("--output-dir", type=str, default=None, help="Run dir to write belief_profile/ under")
    parser.add_argument("--disable-plot", action="store_true", default=False)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def infer_data_paths(
    args: argparse.Namespace,
    checkpoint: Dict[str, object],
) -> Dict[str, str]:
    ckpt_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    if not isinstance(ckpt_args, dict):
        ckpt_args = {}

    metadata = args.metadata or ckpt_args.get("metadata")
    data_json = args.data_json or ckpt_args.get("val_json") or ckpt_args.get("train_json")
    split_key = args.split_key or ckpt_args.get("val_split_key") or ckpt_args.get("train_split_key", "test_data")

    missing = [
        key
        for key, value in {
            "metadata": metadata,
            "data_json": data_json,
            "split_key": split_key,
        }.items()
        if value is None
    ]
    if len(missing) > 0:
        raise ValueError(
            "Could not infer required dataset inputs from checkpoint args. Missing: {}. "
            "Provide them explicitly.".format(", ".join(missing))
        )
    return {
        "metadata": str(metadata),
        "data_json": str(data_json),
        "split_key": str(split_key),
    }


def _load_tensorizer(
    checkpoint: Dict[str, object],
    metadata_path: str,
) -> WatchGraphTensorizer:
    tensorizer_config = checkpoint.get("tensorizer_config")
    if isinstance(tensorizer_config, dict):
        return WatchGraphTensorizer.from_config(tensorizer_config)
    return WatchGraphTensorizer(metadata_path=metadata_path, max_nodes=None)


def _load_goal_to_col(
    checkpoint: Dict[str, object],
    metadata_path: str,
) -> Dict[str, int]:
    goal_to_col = checkpoint.get("goal_to_col")
    if isinstance(goal_to_col, dict) and len(goal_to_col) > 0:
        return {str(k): int(v) for k, v in goal_to_col.items()}
    _, inferred = load_goal_predicates(metadata_path)
    return inferred


def main() -> None:
    args = parse_args()
    if args.belief_profile_bins < 2:
        raise ValueError("--belief-profile-bins must be >= 2")

    set_seed(args.seed)
    device = resolve_device(args.device)
    if device.type == "cuda":
        if device.index is not None:
            torch.cuda.set_device(device.index)
        torch.backends.cudnn.benchmark = True
    print_device_summary(device)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must be a dict payload")
    model_config = checkpoint.get("model_config")
    model_state = checkpoint.get("model_state")
    if not isinstance(model_config, dict) or not isinstance(model_state, dict):
        raise KeyError("Checkpoint missing model_config/model_state required for belief profiling")

    inferred = infer_data_paths(args=args, checkpoint=checkpoint)
    metadata_path = inferred["metadata"]
    data_json = inferred["data_json"]
    split_key = inferred["split_key"]

    ckpt_args = checkpoint.get("args", {}) if isinstance(checkpoint.get("args"), dict) else {}
    stable_slots = args.stable_slots
    if stable_slots is None:
        stable_slots = bool(ckpt_args.get("stable_slots", True))
    use_actions = bool(model_config.get("use_actions", ckpt_args.get("use_actions", False)))

    tensorizer = _load_tensorizer(checkpoint=checkpoint, metadata_path=metadata_path)
    goal_to_col = _load_goal_to_col(checkpoint=checkpoint, metadata_path=metadata_path)

    dataset = WatchVAEDataset(
        data_path=data_json,
        split_key=split_key,
        tensorizer=tensorizer,
        max_seq_len=args.max_seq_len,
        min_seq_len=args.min_seq_len,
        use_actions=use_actions,
        max_demos=args.max_demos,
        stable_slots=stable_slots,
    )
    loader_common = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "collate_fn": collate_watch_vae,
        "drop_last": False,
        "shuffle": False,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        loader_common["persistent_workers"] = True
    loader = DataLoader(dataset, **loader_common)

    print("checkpoint:", args.checkpoint)
    print("data_json:", data_json)
    print("split_key:", split_key)
    print("dataset demos:", len(dataset))
    print("stable slots:", stable_slots)
    print("use actions:", use_actions)
    print("goal predicates:", len(goal_to_col))

    model = GraphSequenceVAE(**model_config).to(device)
    model.load_state_dict(model_state, strict=True)
    model.eval()

    profile = compute_val_belief_profile(
        model=model,
        loader=loader,
        goal_to_col=goal_to_col,
        device=device,
        num_bins=args.belief_profile_bins,
    )

    epoch = int(checkpoint.get("epoch", 0)) if args.epoch is None else int(args.epoch)
    run_output_dir = args.output_dir
    if run_output_dir is None:
        run_output_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    paths = save_and_log_belief_profile(
        profile=profile,
        epoch=epoch,
        run_output_dir=run_output_dir,
        writer=None,
        make_plot=(not args.disable_plot),
        file_prefix=args.file_prefix,
    )
    print("saved belief profile json:", paths.get("json_path"))
    if "png_path" in paths:
        print("saved belief profile plot:", paths.get("png_path"))


if __name__ == "__main__":
    main()
