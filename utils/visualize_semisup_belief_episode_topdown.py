#!/usr/bin/env python3
import argparse
import json
import os
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

# Headless-safe matplotlib backend before importing plotting utils.
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt

if __package__:
    from .visualize_dataset_episode_topdown import (
        _safe_name,
        action_text_and_ids,
        detect_char_id,
        extract_goal_class_names,
        goal_ids_for_frame,
        has_valid_bbox,
        load_split,
        normalize_graph_frame,
        run_ffmpeg,
        select_episode,
        visible_ids_for_frame,
    )
    from .utils_plot import plot_graph_2d
else:
    repo_root = str(Path(__file__).resolve().parents[1])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from utils.visualize_dataset_episode_topdown import (
        _safe_name,
        action_text_and_ids,
        detect_char_id,
        extract_goal_class_names,
        goal_ids_for_frame,
        has_valid_bbox,
        load_split,
        normalize_graph_frame,
        run_ffmpeg,
        select_episode,
        visible_ids_for_frame,
    )
    from utils.utils_plot import plot_graph_2d

from watch.vae.semisup_model import GraphSequenceSemiSupVAE
from watch.vae.model import GraphSequenceVAE
from watch.vae.tensorizer import WatchGraphTensorizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a dataset episode as top-down frames with semisup/joint VAE y-belief overlays and MP4 export."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a watch_semisup_vae_{best,last}.pt or watch_vae_joint_{best,last}.pt checkpoint",
    )
    parser.add_argument(
        "--data-json",
        type=str,
        default=None,
        help="Override dataset JSON. If omitted, inferred from checkpoint args (val_json/train_json).",
    )
    parser.add_argument(
        "--split-key",
        type=str,
        default=None,
        help="Override split key. If omitted, inferred from checkpoint args (val_split_key/train_split_key).",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default=None,
        help="Optional metadata.json path (only used as fallback if checkpoint lacks goal predicate names).",
    )
    parser.add_argument("--episode-index", type=int, default=0, help="Episode index within the split.")
    parser.add_argument("--episode-name", type=str, default=None, help="Exact demo name (overrides --episode-index).")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory. Defaults to outputs/dataset_topdown_belief_viz/<ckpt>/<data>/<episode>/",
    )
    parser.add_argument("--device", type=str, default=None, help="Torch device (e.g., cuda, cuda:0, cpu).")
    parser.add_argument("--fps", type=int, default=5, help="Video frame rate.")
    parser.add_argument("--dpi", type=int, default=120, help="PNG DPI for saved frames.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional cap on rendered/inferred steps.")
    parser.add_argument("--top-k-text", type=int, default=5, help="Top-K y predicates shown in per-frame text overlay.")
    parser.add_argument(
        "--plot-k",
        type=int,
        default=6,
        help="Number of predicate curves to show in the right-side belief timeline panel.",
    )
    parser.add_argument(
        "--no-belief-panel",
        action="store_true",
        default=False,
        help="Disable the right-side timestep belief graph panel; keep text belief stats only.",
    )
    parser.add_argument(
        "--stable-slots",
        action="store_true",
        default=True,
        help="Use stable object-id slot mapping across timesteps (matches training default).",
    )
    parser.add_argument("--no-stable-slots", action="store_false", dest="stable_slots")
    parser.add_argument("--skip-video", action="store_true", default=False, help="Render frames only, skip ffmpeg MP4.")
    parser.add_argument("--overwrite", action="store_true", default=False, help="Allow writing into existing output dir.")
    return parser.parse_args()


def _resolve_device(device_arg: Optional[str]) -> torch.device:
    if device_arg is not None and str(device_arg).strip():
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _infer_data_inputs_from_checkpoint(
    checkpoint: Dict[str, Any],
    data_json: Optional[str],
    split_key: Optional[str],
) -> Tuple[str, str]:
    ckpt_args = checkpoint.get("args", {})
    if not isinstance(ckpt_args, dict):
        ckpt_args = {}

    resolved_data_json = data_json or ckpt_args.get("val_json") or ckpt_args.get("train_json")
    resolved_split_key = split_key or ckpt_args.get("val_split_key") or ckpt_args.get("train_split_key")
    if resolved_data_json is None or resolved_split_key is None:
        raise ValueError(
            "Could not infer data-json/split-key from checkpoint args. Pass --data-json and --split-key explicitly."
        )
    return str(resolved_data_json), str(resolved_split_key)


def _load_goal_vocab(
    checkpoint: Dict[str, Any],
    metadata_path: Optional[str],
) -> Tuple[List[str], Dict[str, int]]:
    names = checkpoint.get("goal_predicate_names")
    mapping = checkpoint.get("goal_to_col")

    if isinstance(names, list) and len(names) > 0:
        names = [str(x) for x in names]
        if isinstance(mapping, dict) and len(mapping) > 0:
            goal_to_col = {str(k): int(v) for k, v in mapping.items()}
        else:
            goal_to_col = {name: idx for idx, name in enumerate(names)}
        return names, goal_to_col

    if metadata_path is None:
        ckpt_args = checkpoint.get("args", {})
        if isinstance(ckpt_args, dict):
            metadata_path = ckpt_args.get("metadata")

    if metadata_path is None:
        raise ValueError("Checkpoint lacks goal_predicate_names; pass --metadata for fallback vocab loading.")

    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    raw = metadata.get("goal_predicates", {})
    if not isinstance(raw, dict) or len(raw) == 0:
        raise KeyError("metadata missing goal_predicates")

    max_idx = max(int(v) for v in raw.values())
    names_by_idx = ["<PAD>"] * (max_idx + 1)
    for name, idx in raw.items():
        idx_i = int(idx)
        if 0 <= idx_i < len(names_by_idx):
            names_by_idx[idx_i] = str(name)
    names_out = [names_by_idx[i] for i in range(1, len(names_by_idx)) if names_by_idx[i] != "<PAD>"]
    return names_out, {name: i for i, name in enumerate(names_out)}


def _load_semisup_model_and_tensorizer(
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[Dict[str, Any], torch.nn.Module, WatchGraphTensorizer, str]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint is not a dict: {}".format(type(checkpoint).__name__))
    if "model_config" not in checkpoint or "model_state" not in checkpoint:
        raise KeyError("Checkpoint must contain model_config and model_state")
    if "tensorizer_config" not in checkpoint:
        raise KeyError("Checkpoint must contain tensorizer_config")

    model_config = checkpoint["model_config"]
    if not isinstance(model_config, dict):
        raise TypeError("checkpoint['model_config'] must be a dict")
    model_state = checkpoint["model_state"]
    if not isinstance(model_state, dict):
        raise TypeError("checkpoint['model_state'] must be a state dict")

    state_keys = set([str(k) for k in model_state.keys()])
    is_semisup = "goal_head.weight" in state_keys
    is_joint = ("belief_weight" in state_keys) or bool(model_config.get("enable_predicate_head", False))

    if is_semisup:
        model_type = "semisup"
        model = GraphSequenceSemiSupVAE.from_config(model_config).to(device)
    elif is_joint:
        model_type = "joint"
        model = GraphSequenceVAE.from_config(model_config).to(device)
        if not bool(getattr(model, "enable_predicate_head", False)):
            raise ValueError("Joint VAE checkpoint does not have predicate head enabled; cannot visualize beliefs.")
    else:
        raise ValueError(
            "Unsupported checkpoint for belief visualization. Expected semisup (goal_head.*) or joint (belief_*)."
        )

    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()

    tensorizer = WatchGraphTensorizer.from_config(checkpoint["tensorizer_config"])
    return checkpoint, model, tensorizer, model_type


def _infer_y_probs_seq(model: torch.nn.Module, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    if hasattr(model, "infer_y_probs_seq"):
        out = model.infer_y_probs_seq(batch)  # type: ignore[attr-defined]
        if isinstance(out, dict) and "y_probs_seq" in out:
            return out["y_probs_seq"]
        raise TypeError("model.infer_y_probs_seq(batch) must return dict with key 'y_probs_seq'")

    if hasattr(model, "predicate_logits_from_mu_sequence") and hasattr(model, "encode_sequence"):
        action_ids = batch.get("action_ids")
        _, mu, _ = model.encode_sequence(  # type: ignore[attr-defined]
            class_ids=batch["class_ids"],
            coords=batch["coords"],
            states=batch["states"],
            node_mask=batch["node_mask"],
            lengths=batch["lengths"],
            action_ids=action_ids,
        )
        logits_seq = model.predicate_logits_from_mu_sequence(mu)  # type: ignore[attr-defined]
        return torch.sigmoid(logits_seq)

    raise TypeError("Model does not expose semisup or joint belief inference APIs")


def _encode_single_demo_batch(
    demo: Dict[str, Any],
    tensorizer: WatchGraphTensorizer,
    use_actions: bool,
    stable_slots: bool,
    max_steps: Optional[int],
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], int]:
    graphs = demo.get("graphs", [])
    actions = demo.get("valid_action_with_walk", [])
    if not isinstance(graphs, list) or not isinstance(actions, list):
        raise TypeError("Episode graphs/actions are not lists.")
    if len(graphs) == 0:
        raise ValueError("Selected episode has no graph frames.")
    if len(actions) == 0 and use_actions:
        raise ValueError("Selected episode has no actions but checkpoint expects actions.")

    seq_len = len(graphs)
    if use_actions:
        seq_len = min(seq_len, len(actions))
    if max_steps is not None:
        seq_len = min(seq_len, int(max_steps))
    if seq_len <= 0:
        raise ValueError("No timesteps available after applying max-steps / action alignment.")

    slot_map: Optional[Dict[int, int]] = None
    if stable_slots:
        slot_map = tensorizer.build_stable_slot_map(graphs[:seq_len])

    class_seq: List[np.ndarray] = []
    coords_seq: List[np.ndarray] = []
    states_seq: List[np.ndarray] = []
    mask_seq: List[np.ndarray] = []
    action_seq: List[int] = []

    for t in range(seq_len):
        graph_t = graphs[t]
        if stable_slots and slot_map is not None:
            frame = tensorizer.encode_nodes_with_slot_map(graph_t, slot_map=slot_map, allow_new_ids=False)
        else:
            frame = tensorizer.encode_nodes(graph_t)

        class_seq.append(frame["class_objects"])
        coords_seq.append(frame["object_coords"])
        states_seq.append(frame["states_objects"])
        mask_seq.append(frame["mask_object"])
        if use_actions:
            action_seq.append(tensorizer.action_index(actions[t]))

    batch: Dict[str, torch.Tensor] = {
        "class_ids": torch.from_numpy(np.stack(class_seq, axis=0)).long().unsqueeze(0).to(device),
        "coords": torch.from_numpy(np.stack(coords_seq, axis=0)).float().unsqueeze(0).to(device),
        "states": torch.from_numpy(np.stack(states_seq, axis=0)).float().unsqueeze(0).to(device),
        "node_mask": torch.from_numpy(np.stack(mask_seq, axis=0)).float().unsqueeze(0).to(device),
        "lengths": torch.tensor([seq_len], dtype=torch.long, device=device),
    }
    if use_actions:
        batch["action_ids"] = torch.tensor(action_seq, dtype=torch.long, device=device).unsqueeze(0)
    return batch, seq_len


def _bernoulli_entropy_mean_per_step(y_probs_seq: np.ndarray) -> np.ndarray:
    eps = 1e-6
    p = np.clip(y_probs_seq, eps, 1.0 - eps)
    ent = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))
    return ent.mean(axis=1)


def _choose_tracked_predicates(
    y_probs_seq: np.ndarray,
    gt_indices: Sequence[int],
    plot_k: int,
) -> List[int]:
    k = max(1, int(plot_k))
    tracked: List[int] = []
    seen = set()

    for idx in gt_indices:
        idx_i = int(idx)
        if idx_i < 0 or idx_i >= y_probs_seq.shape[1] or idx_i in seen:
            continue
        tracked.append(idx_i)
        seen.add(idx_i)
        if len(tracked) >= k:
            return tracked

    peak = y_probs_seq.max(axis=0)
    drift = y_probs_seq.max(axis=0) - y_probs_seq.min(axis=0)
    score = peak + 0.5 * drift
    order = np.argsort(score)[::-1]
    for idx in order.tolist():
        if idx in seen:
            continue
        tracked.append(int(idx))
        seen.add(int(idx))
        if len(tracked) >= k:
            break
    return tracked


def _short_label(text: str, max_len: int = 28) -> str:
    s = str(text)
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def _step_topk(y_probs_t: np.ndarray, predicate_names: Sequence[str], top_k: int) -> List[Dict[str, Any]]:
    k = max(1, min(int(top_k), len(predicate_names)))
    order = np.argsort(y_probs_t)[::-1][:k]
    return [
        {
            "predicate_index": int(i),
            "predicate": str(predicate_names[int(i)]),
            "prob": float(y_probs_t[int(i)]),
        }
        for i in order
    ]


def _step_gt_probs(y_probs_t: np.ndarray, predicate_names: Sequence[str], gt_indices: Sequence[int]) -> List[Dict[str, Any]]:
    items = []
    for idx in gt_indices:
        idx_i = int(idx)
        if 0 <= idx_i < len(predicate_names):
            items.append(
                {
                    "predicate_index": idx_i,
                    "predicate": str(predicate_names[idx_i]),
                    "prob": float(y_probs_t[idx_i]),
                }
            )
    items.sort(key=lambda d: d["prob"], reverse=True)
    return items


def _render_frame_with_belief_overlay(
    *,
    frame_path: Path,
    graph: Dict[str, Any],
    char_id: int,
    visible_ids: Sequence[int],
    action_ids: Sequence[int],
    goal_ids: Sequence[int],
    step_idx: int,
    total_steps: int,
    action_text: str,
    episode_name: str,
    task_name: str,
    y_probs_seq: np.ndarray,
    predicate_names: Sequence[str],
    tracked_pred_indices: Sequence[int],
    gt_pred_indices: Sequence[int],
    entropy_mean_per_step: np.ndarray,
    top_k_text: int,
    dpi: int,
    show_belief_panel: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    fig = plot_graph_2d(
        graph=graph,
        char_id=char_id,
        visible_ids=list(visible_ids),
        action_ids=list(action_ids),
        goal_ids=list(goal_ids),
    )

    # Reserve room for overlay text (top) and optional belief panel (right).
    if show_belief_panel:
        fig.subplots_adjust(top=0.83, right=0.63)
    else:
        fig.subplots_adjust(top=0.83, right=0.98)

    y_t = y_probs_seq[step_idx]
    topk = _step_topk(y_t, predicate_names, top_k_text)
    gt_probs = _step_gt_probs(y_t, predicate_names, gt_pred_indices)

    top_lines = ["Top y @ step:"]
    for item in topk:
        top_lines.append("  {:>5.3f} {}".format(item["prob"], item["predicate"]))

    gt_lines = ["GT goal predicates (y):"]
    if len(gt_probs) == 0:
        gt_lines.append("  <none matched y vocab>")
    else:
        show_n = min(len(gt_probs), max(3, min(int(top_k_text), 8)))
        for item in gt_probs[:show_n]:
            gt_lines.append("  {:>5.3f} {}".format(item["prob"], item["predicate"]))
        if len(gt_probs) > show_n:
            gt_lines.append("  ... +{} more".format(len(gt_probs) - show_n))

    wrapped_action = textwrap.fill(action_text if action_text else "<none>", width=72)
    header = "{} | task: {}".format(episode_name, task_name)
    step_line = "Step: {}/{} | mean entropy(y): {:.3f}".format(
        step_idx, max(total_steps - 1, 0), float(entropy_mean_per_step[step_idx])
    )
    action_line = "Alice valid action: {}".format(wrapped_action)
    overlay_text = "{}\n{}\n{}\n{}\n{}".format(
        header,
        step_line,
        action_line,
        "\n".join(top_lines),
        "\n".join(gt_lines),
    )

    fig.text(
        0.01,
        0.985,
        overlay_text,
        ha="left",
        va="top",
        fontsize=9,
        family="monospace",
        bbox={"facecolor": "white", "alpha": 0.94, "edgecolor": "black", "pad": 4},
    )

    if show_belief_panel:
        # Belief timeline panel on the right.
        ax_bel = fig.add_axes([0.66, 0.20, 0.31, 0.60])
        t_axis = np.arange(total_steps)
        gt_set = set([int(i) for i in gt_pred_indices])
        for idx in tracked_pred_indices:
            idx_i = int(idx)
            if idx_i < 0 or idx_i >= len(predicate_names):
                continue
            y_curve = y_probs_seq[:, idx_i]
            is_gt = idx_i in gt_set
            ax_bel.plot(
                t_axis,
                y_curve,
                linewidth=2.0 if is_gt else 1.4,
                linestyle="-" if is_gt else "--",
                alpha=0.95 if is_gt else 0.85,
                label=_short_label(predicate_names[idx_i]),
            )
            ax_bel.scatter([step_idx], [y_curve[step_idx]], s=14, zorder=4)

        ax_bel.axvline(step_idx, color="black", linestyle=":", linewidth=1.2, alpha=0.8)
        ax_bel.set_xlim(0, max(total_steps - 1, 1))
        ax_bel.set_ylim(-0.02, 1.02)
        ax_bel.set_xlabel("Timestep")
        ax_bel.set_ylabel("p(y=1)")
        ax_bel.set_title("Belief y (selected predicates)", fontsize=9)
        ax_bel.grid(alpha=0.25)
        if len(tracked_pred_indices) > 0:
            ax_bel.legend(fontsize=6, loc="upper left", framealpha=0.9)

    fig.savefig(str(frame_path), dpi=dpi)
    plt.close(fig)
    return topk, gt_probs


def main() -> None:
    args = parse_args()
    device = _resolve_device(args.device)

    checkpoint_path = Path(args.checkpoint)
    checkpoint, model, tensorizer, model_type = _load_semisup_model_and_tensorizer(str(checkpoint_path), device=device)
    predicate_names, goal_to_col = _load_goal_vocab(checkpoint, args.metadata)

    data_json_str, split_key = _infer_data_inputs_from_checkpoint(checkpoint, args.data_json, args.split_key)
    data_json = Path(data_json_str)
    split_key, demos = load_split(data_json, split_key)
    selected_index, demo = select_episode(demos, args.episode_index, args.episode_name)

    batch, inferred_steps = _encode_single_demo_batch(
        demo=demo,
        tensorizer=tensorizer,
        use_actions=bool(getattr(model, "use_actions", False)),
        stable_slots=bool(args.stable_slots),
        max_steps=args.max_steps,
        device=device,
    )

    with torch.no_grad():
        y_probs_seq = _infer_y_probs_seq(model, batch)[0].detach().cpu().numpy()  # [T, K]

    total_steps = int(inferred_steps)
    if y_probs_seq.shape[0] != total_steps:
        total_steps = min(total_steps, int(y_probs_seq.shape[0]))
        y_probs_seq = y_probs_seq[:total_steps]

    episode_name = str(demo.get("name", "episode_{}".format(selected_index)))
    task_name = str(demo.get("task_name", "unknown"))
    goal_list = [str(g) for g in demo.get("goal", [])]
    gt_pred_names = sorted(set([g for g in goal_list if g in goal_to_col]))
    gt_pred_indices = [int(goal_to_col[g]) for g in gt_pred_names]
    tracked_pred_indices = _choose_tracked_predicates(y_probs_seq, gt_pred_indices, plot_k=int(args.plot_k))
    entropy_mean_per_step = _bernoulli_entropy_mean_per_step(y_probs_seq)
    show_belief_panel = not bool(args.no_belief_panel)

    if args.out_dir is None:
        out_dir = (
            Path("outputs")
            / "dataset_topdown_belief_viz"
            / _safe_name(checkpoint_path.stem)
            / data_json.stem
            / _safe_name(episode_name)
        )
    else:
        out_dir = Path(args.out_dir)

    frames_dir = out_dir / "frames"
    if out_dir.exists() and not args.overwrite:
        raise FileExistsError("Output directory exists: {} (pass --overwrite to reuse)".format(out_dir))
    frames_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for stale_frame in frames_dir.glob("frame_*.png"):
            try:
                stale_frame.unlink()
            except OSError:
                pass

    raw_graphs = demo.get("graphs", [])
    raw_actions = demo.get("valid_action_with_walk", [])
    raw_total_steps = min(len(raw_graphs), len(raw_actions)) if bool(getattr(model, "use_actions", False)) else len(raw_graphs)
    total_steps = min(total_steps, raw_total_steps)
    y_probs_seq = y_probs_seq[:total_steps]
    entropy_mean_per_step = entropy_mean_per_step[:total_steps]

    goal_class_names = extract_goal_class_names(goal_list)
    steps_manifest: List[Dict[str, Any]] = []

    for step_idx in range(total_steps):
        graph = normalize_graph_frame(raw_graphs[step_idx])
        nodes = graph["nodes"]

        char_id = detect_char_id(nodes)
        visible_ids = visible_ids_for_frame(nodes)
        action_text, parsed_action_ids = action_text_and_ids(raw_actions[step_idx] if step_idx < len(raw_actions) else None)
        visible_set = set(visible_ids)
        action_ids = [obj_id for obj_id in parsed_action_ids if obj_id in visible_set]
        goal_ids = [obj_id for obj_id in goal_ids_for_frame(nodes, goal_class_names) if obj_id in visible_set]

        frame_file = "frame_{:04d}.png".format(step_idx)
        frame_path = frames_dir / frame_file

        topk, gt_probs = _render_frame_with_belief_overlay(
            frame_path=frame_path,
            graph=graph,
            char_id=char_id,
            visible_ids=visible_ids,
            action_ids=action_ids,
            goal_ids=goal_ids,
            step_idx=step_idx,
            total_steps=total_steps,
            action_text=action_text,
            episode_name=episode_name,
            task_name=task_name,
            y_probs_seq=y_probs_seq,
            predicate_names=predicate_names,
            tracked_pred_indices=tracked_pred_indices,
            gt_pred_indices=gt_pred_indices,
            entropy_mean_per_step=entropy_mean_per_step,
            top_k_text=int(args.top_k_text),
            dpi=int(args.dpi),
            show_belief_panel=show_belief_panel,
        )

        steps_manifest.append(
            {
                "step_idx": step_idx,
                "frame_file": str(Path("frames") / frame_file),
                "action_text": action_text,
                "parsed_action_object_ids": parsed_action_ids,
                "highlighted_action_object_ids": action_ids,
                "goal_object_ids": goal_ids,
                "char_id": char_id,
                "y_entropy_mean": float(entropy_mean_per_step[step_idx]),
                "topk_y": topk,
                "gt_goal_y_probs": gt_probs,
            }
        )

    belief_seq_json = {
        "predicate_names": [str(x) for x in predicate_names],
        "y_probs_seq": y_probs_seq.tolist(),
        "y_entropy_mean_per_step": entropy_mean_per_step.tolist(),
        "tracked_predicate_indices": [int(i) for i in tracked_pred_indices],
        "tracked_predicate_names": [str(predicate_names[int(i)]) for i in tracked_pred_indices],
        "gt_goal_predicate_indices": [int(i) for i in gt_pred_indices],
        "gt_goal_predicate_names": gt_pred_names,
    }

    belief_summary_json = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "model_type": model_type,
        "device": str(device),
        "model_use_actions": bool(getattr(model, "use_actions", False)),
        "num_goal_predicates": len(predicate_names),
        "data_json": str(data_json),
        "split_key": split_key,
        "episode_index": selected_index,
        "episode_name": episode_name,
        "task_name": task_name,
        "goal_list_raw": goal_list,
        "gt_goal_predicate_names_in_vocab": gt_pred_names,
        "num_rendered_steps": total_steps,
        "tracked_predicate_names": [str(predicate_names[int(i)]) for i in tracked_pred_indices],
        "topk_text": int(args.top_k_text),
        "plot_k": int(args.plot_k),
        "belief_panel_enabled": bool(show_belief_panel),
        "stable_slots": bool(args.stable_slots),
        "fps": int(args.fps),
        "entropy_mean_over_steps": float(np.mean(entropy_mean_per_step)) if total_steps > 0 else float("nan"),
        "entropy_std_over_steps": float(np.std(entropy_mean_per_step)) if total_steps > 0 else float("nan"),
    }

    with open(out_dir / "episode_meta.json", "w") as f:
        json.dump(
            {
                "data_json": str(data_json),
                "split_key": split_key,
                "episode_index": selected_index,
                "episode_name": episode_name,
                "task_name": task_name,
                "goal": goal_list,
                "num_rendered_steps": total_steps,
                "fps": int(args.fps),
            },
            f,
            indent=2,
        )
    with open(out_dir / "steps_with_belief.json", "w") as f:
        json.dump(steps_manifest, f, indent=2)
    with open(out_dir / "belief_seq.json", "w") as f:
        json.dump(belief_seq_json, f, indent=2)
    with open(out_dir / "belief_summary.json", "w") as f:
        json.dump(belief_summary_json, f, indent=2)

    video_path = out_dir / "episode_topdown_belief.mp4"
    if not args.skip_video:
        run_ffmpeg(frames_dir=frames_dir, out_path=video_path, fps=int(args.fps))

    print("Rendered {} frames to {}".format(total_steps, frames_dir))
    if not args.skip_video and video_path.exists():
        print("Video written to {}".format(video_path))
    print("Belief summary: {}".format(out_dir / "belief_summary.json"))
    print("Belief sequence: {}".format(out_dir / "belief_seq.json"))
    print("Per-step manifest: {}".format(out_dir / "steps_with_belief.json"))


if __name__ == "__main__":
    main()
