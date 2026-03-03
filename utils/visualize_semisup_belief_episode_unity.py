#!/usr/bin/env python3
import argparse
import copy
import json
import os
import sys
import textwrap
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

# Headless-safe backend before importing helper modules that use matplotlib.
os.environ.setdefault("MPLBACKEND", "Agg")

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception as e:  # pragma: no cover - import guard
    raise RuntimeError("Pillow is required for Unity frame overlays: {}".format(repr(e)))

if __package__:
    from .visualize_dataset_episode_topdown import (
        _safe_name,
        action_text_and_ids,
        detect_char_id,
        extract_goal_class_names,
        goal_ids_for_frame,
        load_split,
        normalize_graph_frame,
        run_ffmpeg,
        select_episode,
        visible_ids_for_frame,
    )
    from .visualize_semisup_belief_episode_topdown import (
        _bernoulli_entropy_mean_per_step,
        _choose_tracked_predicates,
        _encode_single_demo_batch,
        _infer_y_probs_seq,
        _infer_data_inputs_from_checkpoint,
        _load_goal_vocab,
        _load_semisup_model_and_tensorizer,
        _resolve_device,
        _step_gt_probs,
        _step_topk,
    )
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
        load_split,
        normalize_graph_frame,
        run_ffmpeg,
        select_episode,
        visible_ids_for_frame,
    )
    from utils.visualize_semisup_belief_episode_topdown import (
        _bernoulli_entropy_mean_per_step,
        _choose_tracked_predicates,
        _encode_single_demo_batch,
        _infer_y_probs_seq,
        _infer_data_inputs_from_checkpoint,
        _load_goal_vocab,
        _load_semisup_model_and_tensorizer,
        _resolve_device,
        _step_gt_probs,
        _step_topk,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a dataset episode in Unity and render per-step snapshots with semisup/joint VAE belief overlays."
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
        help="Optional metadata.json path (fallback if checkpoint lacks goal predicate names).",
    )
    parser.add_argument("--episode-index", type=int, default=0, help="Episode index within the split.")
    parser.add_argument("--episode-name", type=str, default=None, help="Exact demo name (overrides --episode-index).")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory. Defaults to outputs/dataset_unity_belief_viz/<ckpt>/<data>/<episode>/",
    )
    parser.add_argument("--device", type=str, default=None, help="Torch device (e.g., cuda, cuda:0, cpu).")
    parser.add_argument("--fps", type=int, default=5, help="Video frame rate.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional cap on rendered/inferred steps.")
    parser.add_argument("--top-k-text", type=int, default=5, help="Top-K y predicates shown in the frame overlay.")
    parser.add_argument(
        "--plot-k",
        type=int,
        default=6,
        help="Used for tracked predicate selection metadata (timeline panel not drawn in this MVP).",
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

    # Unity connection / launch.
    parser.add_argument(
        "--unity-executable",
        type=str,
        default=None,
        help="Path to Unity executable. If omitted, connect to an already-running simulator on --base-port + --port-id.",
    )
    parser.add_argument("--base-port", type=int, default=8080, help="Base Unity port (same convention as repo wrappers).")
    parser.add_argument("--port-id", type=int, default=0, help="Port offset; actual port is base-port + port-id.")
    parser.add_argument("--x-display", type=str, default=None, help="X display for headless Linux rendering, if needed.")
    parser.add_argument("--no-graphics", action="store_true", default=False, help="Launch Unity with no graphics.")
    parser.add_argument("--timeout-wait", type=int, default=60, help="Unity HTTP request timeout in seconds.")
    parser.add_argument("--docker-enabled", action="store_true", default=False, help="Pass docker_enabled to UnityCommunication.")

    # Unity replay/capture behavior.
    parser.add_argument(
        "--capture-phase",
        type=str,
        default="before",
        choices=["before", "after"],
        help="Capture snapshot before executing action[t] (aligns with top-down belief overlays) or after.",
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=640,
        help="Unity snapshot image width.",
    )
    parser.add_argument(
        "--image-height",
        type=int,
        default=480,
        help="Unity snapshot image height.",
    )
    parser.add_argument(
        "--camera-index-within-char",
        type=int,
        default=1,
        help="Character camera slot index (0=top, 1=front in repo wrappers).",
    )
    parser.add_argument(
        "--character-resource",
        type=str,
        default="Chars/Female1",
        help="Unity character prefab/resource for Alice.",
    )
    parser.add_argument(
        "--initial-room",
        type=str,
        default="",
        help="Optional room for initial character spawn (kitchen/bedroom/livingroom/bathroom). If empty, Unity chooses.",
    )
    parser.add_argument(
        "--align-char-to-dataset-start",
        action="store_true",
        default=True,
        help="Move Alice near the dataset step-0 character position before replay (improves alignment).",
    )
    parser.add_argument("--no-align-char-to-dataset-start", action="store_false", dest="align_char_to_dataset_start")
    parser.add_argument(
        "--processing-time-limit",
        type=int,
        default=20,
        help="Unity script processing time limit (seconds) per action.",
    )
    parser.add_argument(
        "--time-scale",
        type=float,
        default=10.0,
        help="Unity action execution timescale (higher is faster).",
    )
    parser.add_argument(
        "--stop-on-action-failure",
        action="store_true",
        default=False,
        help="Abort replay if Unity reports an action execution failure.",
    )
    return parser.parse_args()


def _log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print("[unity-belief {}] {}".format(ts, msg), flush=True)


def _import_comm_unity():
    # This repo expects a sibling checkout at ../virtualhome.
    repo_root = Path(__file__).resolve().parents[1]
    sim_path = repo_root.parent / "virtualhome" / "simulation"
    if sim_path.exists():
        sim_path_str = str(sim_path)
        if sim_path_str not in sys.path:
            sys.path.insert(0, sim_path_str)

    try:
        import collections
        import collections.abc

        if not hasattr(collections, "Iterable"):
            collections.Iterable = collections.abc.Iterable  # type: ignore[attr-defined]
    except Exception:
        pass

    try:
        from unity_simulator import comm_unity  # type: ignore
    except Exception as e:
        raise ImportError(
            "Failed to import VirtualHome UnityCommunication. Expected ../virtualhome/simulation on disk. Error: {}".format(
                repr(e)
            )
        )
    return comm_unity


def _install_comm_startup_debug_hooks(comm_unity: Any) -> None:
    # Patch only once.
    if getattr(comm_unity.UnityCommunication, "_watch_debug_check_connection_installed", False):
        return

    def _debug_check_connection(self):  # type: ignore[no-untyped-def]
        _log("Unity check_connection attempt (timeout_wait={}s)".format(getattr(self, "timeout_wait", "unknown")))
        request = {"id": str(time.time()), "action": "idle"}
        try:
            # Use repeat=False so timeout_wait is honored and hangs are easier to diagnose.
            response = self.post_command(request, repeat=False)
        except Exception as e:
            _log("Unity check_connection failed: {}".format(repr(e)))
            raise
        ok = bool(response.get("success", False)) if isinstance(response, dict) else False
        _log("Unity check_connection response: success={}".format(ok))
        return ok

    comm_unity.UnityCommunication.check_connection = _debug_check_connection  # type: ignore[assignment]
    comm_unity.UnityCommunication._watch_debug_check_connection_installed = True


def _separate_new_ids_graph(graph: Dict[str, Any], max_id: int) -> Dict[str, Any]:
    new_graph = copy.deepcopy(graph)
    for node in new_graph.get("nodes", []):
        try:
            if int(node.get("id", -1)) > max_id:
                node["id"] = int(node["id"]) - max_id + 1000
        except Exception:
            continue
    for edge in new_graph.get("edges", []):
        try:
            if int(edge.get("from_id", -1)) > max_id:
                edge["from_id"] = int(edge["from_id"]) - max_id + 1000
            if int(edge.get("to_id", -1)) > max_id:
                edge["to_id"] = int(edge["to_id"]) - max_id + 1000
        except Exception:
            continue
    return new_graph


def _single_agent_script_list(action_text: str) -> List[str]:
    action_clean = str(action_text).strip()
    if not action_clean:
        return [""]
    return ["<char0> {}".format(action_clean)]


def _dataset_char_node(frame_nodes: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    try:
        char_id = detect_char_id(frame_nodes)
    except Exception:
        char_id = None
    if char_id is not None:
        for node in frame_nodes:
            try:
                if int(node.get("id", -1)) == int(char_id):
                    return node
            except Exception:
                continue
    for node in frame_nodes:
        if "character" in str(node.get("class_name", "")).lower():
            return node
    return None


def _bbox_center(node: Optional[Dict[str, Any]]) -> Optional[List[float]]:
    if not isinstance(node, dict):
        return None
    bbox = node.get("bounding_box")
    if not isinstance(bbox, dict):
        return None
    center = bbox.get("center")
    if not isinstance(center, (list, tuple)) or len(center) != 3:
        return None
    try:
        return [float(center[0]), float(center[1]), float(center[2])]
    except Exception:
        return None


def _bbox_bottom_position(node: Optional[Dict[str, Any]]) -> Optional[List[float]]:
    if not isinstance(node, dict):
        return None
    bbox = node.get("bounding_box")
    if not isinstance(bbox, dict):
        return None
    center = bbox.get("center")
    size = bbox.get("size")
    if not isinstance(center, (list, tuple)) or len(center) != 3:
        return None
    try:
        x = float(center[0])
        y = float(center[1])
        z = float(center[2])
        if isinstance(size, (list, tuple)) and len(size) == 3:
            y = y - 0.5 * float(size[1])
        return [x, y, z]
    except Exception:
        return None


def _unity_char_node(graph: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    nodes = graph.get("nodes", [])
    if not isinstance(nodes, list):
        return None
    for node in nodes:
        if "character" in str(node.get("class_name", "")).lower():
            return node
    for node in nodes:
        try:
            if int(node.get("id", -1)) == 1:
                return node
        except Exception:
            continue
    return None


def _get_unity_visible_ids(comm: Any, camera_index: int) -> Tuple[Optional[List[int]], Optional[str]]:
    try:
        ok, msg = comm.get_visible_objects(int(camera_index))
    except Exception as e:
        return None, repr(e)
    if not ok:
        return None, "Unity get_visible_objects returned success=False"
    if not isinstance(msg, dict):
        return None, "Unexpected visible object payload type: {}".format(type(msg).__name__)
    ids: List[int] = []
    for key in msg.keys():
        try:
            ids.append(int(key))
        except Exception:
            continue
    ids.sort()
    return ids, None


def _capture_unity_state(
    comm: Any,
    camera_index: int,
    image_width: int,
    image_height: int,
) -> Dict[str, Any]:
    graph_ok, graph = comm.environment_graph()
    if not graph_ok or not isinstance(graph, dict):
        raise RuntimeError("Unity environment_graph failed")

    char_node = _unity_char_node(graph)
    char_center = _bbox_center(char_node)
    char_id = None
    if isinstance(char_node, dict):
        try:
            char_id = int(char_node.get("id", -1))
        except Exception:
            char_id = None

    unity_visible_ids, visible_err = _get_unity_visible_ids(comm, camera_index=int(camera_index))

    img_ok, images = comm.camera_image(
        [int(camera_index)],
        mode="normal",
        image_width=int(image_width),
        image_height=int(image_height),
    )
    if not img_ok:
        raise RuntimeError("Unity camera_image failed for camera {}".format(camera_index))
    if not isinstance(images, list) or len(images) == 0:
        raise RuntimeError("Unity camera_image returned no images")

    img = images[0]
    if not isinstance(img, np.ndarray):
        raise TypeError("Unity camera_image returned non-array image: {}".format(type(img).__name__))
    if img.ndim == 2:
        img_rgb = np.stack([img] * 3, axis=-1)
    elif img.ndim == 3 and img.shape[2] >= 3:
        # VirtualHome/comm_unity decodes with OpenCV, so images are BGR.
        img_rgb = img[:, :, :3][:, :, ::-1]
    else:
        raise ValueError("Unsupported Unity image shape: {}".format(tuple(img.shape)))

    return {
        "unity_graph": graph,
        "unity_char_id": char_id,
        "unity_char_center": char_center,
        "unity_visible_ids": unity_visible_ids,
        "unity_visible_ids_error": visible_err,
        "image_rgb": np.ascontiguousarray(img_rgb),
    }


def _float_list_or_none(x: Optional[Sequence[float]]) -> Optional[List[float]]:
    if x is None:
        return None
    try:
        return [float(v) for v in x]
    except Exception:
        return None


def _l2_or_none(a: Optional[Sequence[float]], b: Optional[Sequence[float]]) -> Optional[float]:
    if a is None or b is None:
        return None
    if len(a) != len(b):
        return None
    try:
        arr_a = np.asarray(a, dtype=np.float32)
        arr_b = np.asarray(b, dtype=np.float32)
        return float(np.linalg.norm(arr_a - arr_b))
    except Exception:
        return None


def _set_overlap_metrics(a_ids: Optional[Sequence[int]], b_ids: Optional[Sequence[int]]) -> Dict[str, Any]:
    if a_ids is None or b_ids is None:
        return {
            "visible_intersection_count": None,
            "visible_union_count": None,
            "visible_iou": None,
        }
    a = set([int(x) for x in a_ids])
    b = set([int(x) for x in b_ids])
    inter = len(a & b)
    union = len(a | b)
    iou = None if union <= 0 else float(inter) / float(union)
    return {
        "visible_intersection_count": inter,
        "visible_union_count": union,
        "visible_iou": iou,
    }


def _safe_message_text(msg: Any, max_len: int = 800) -> str:
    try:
        text = json.dumps(msg, sort_keys=True)
    except Exception:
        text = repr(msg)
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def _build_overlay_text(
    *,
    episode_name: str,
    task_name: str,
    step_idx: int,
    total_steps: int,
    action_text: str,
    capture_phase: str,
    y_entropy_mean: float,
    topk: Sequence[Dict[str, Any]],
    gt_probs: Sequence[Dict[str, Any]],
    action_exec_success: Optional[bool],
    char_pos_l2: Optional[float],
    visible_iou: Optional[float],
    visible_ids_error: Optional[str],
) -> str:
    wrapped_action = textwrap.fill(action_text if action_text else "<none>", width=84)
    header = "{} | task: {}".format(episode_name, task_name)
    status_str = "unknown"
    if action_exec_success is True:
        status_str = "ok"
    elif action_exec_success is False:
        status_str = "failed"
    diag_line = "phase={} | action_exec={} | char_l2={} | visible_iou={}".format(
        capture_phase,
        status_str,
        "{:.3f}".format(char_pos_l2) if char_pos_l2 is not None else "n/a",
        "{:.3f}".format(visible_iou) if visible_iou is not None else "n/a",
    )
    step_line = "Step: {}/{} | mean entropy(y): {:.3f}".format(step_idx, max(total_steps - 1, 0), float(y_entropy_mean))
    action_line = "Alice action: {}".format(wrapped_action)

    top_lines = ["Top y @ step:"]
    for item in topk:
        top_lines.append("  {:>5.3f} {}".format(float(item["prob"]), str(item["predicate"])))

    gt_lines = ["GT goal predicates (y):"]
    if len(gt_probs) == 0:
        gt_lines.append("  <none matched y vocab>")
    else:
        show_n = min(len(gt_probs), 6)
        for item in gt_probs[:show_n]:
            gt_lines.append("  {:>5.3f} {}".format(float(item["prob"]), str(item["predicate"])))
        if len(gt_probs) > show_n:
            gt_lines.append("  ... +{} more".format(len(gt_probs) - show_n))

    lines = [header, step_line, diag_line, action_line] + top_lines + gt_lines
    if visible_ids_error:
        lines.append("visible_ids warning: {}".format(visible_ids_error))
    return "\n".join(lines)


def _draw_text_overlay_and_save(
    image_rgb: np.ndarray,
    frame_path: Path,
    overlay_text: str,
    action_exec_success: Optional[bool],
) -> None:
    img = Image.fromarray(image_rgb.astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(img, mode="RGBA")

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    pad = 8
    try:
        text_bbox = draw.multiline_textbbox((pad, pad), overlay_text, font=font, spacing=2)
    except Exception:
        # Pillow<8 fallback.
        lines = overlay_text.splitlines() or [""]
        max_w = 0
        line_h = 12
        for line in lines:
            try:
                bbox = draw.textbbox((0, 0), line, font=font)
                w = max(0, int(bbox[2] - bbox[0]))
                h = max(0, int(bbox[3] - bbox[1]))
            except Exception:
                w, h = draw.textsize(line, font=font)
            max_w = max(max_w, int(w))
            line_h = max(line_h, int(h))
        total_h = len(lines) * line_h + max(0, len(lines) - 1) * 2
        text_bbox = (pad, pad, pad + max_w, pad + total_h)
    bg_color = (0, 0, 0, 180)
    outline = (255, 255, 255, 180)
    if action_exec_success is False:
        outline = (255, 80, 80, 220)
    rect_xy = (
        max(0, text_bbox[0] - pad),
        max(0, text_bbox[1] - pad),
        min(img.width - 1, text_bbox[2] + pad),
        min(img.height - 1, text_bbox[3] + pad),
    )
    try:
        draw.rounded_rectangle(
            rect_xy,
            radius=6,
            fill=bg_color,
            outline=outline,
            width=2,
        )
    except Exception:
        draw.rectangle(rect_xy, fill=bg_color, outline=outline, width=2)
    draw.multiline_text((pad, pad), overlay_text, fill=(255, 255, 255, 255), font=font, spacing=2)
    img.save(str(frame_path))


def _reset_unity_scene_for_demo(
    comm: Any,
    demo: Dict[str, Any],
    character_resource: str,
    initial_room: str,
    align_char_to_dataset_start: bool,
    dataset_step0_char_move_pos: Optional[List[float]],
) -> Dict[str, Any]:
    env_id = demo.get("env_id")
    if env_id is None:
        raise KeyError("Dataset episode missing env_id; Unity replay requires env_id and init_graph.")
    init_graph = demo.get("init_graph")
    if not isinstance(init_graph, dict):
        raise KeyError("Dataset episode missing init_graph dict; Unity replay requires env_id + init_graph.")

    _log("Unity reset(scene={})".format(env_id))
    ok = comm.reset(int(env_id))
    if not ok:
        raise RuntimeError("Unity reset failed for env_id {}".format(env_id))

    _log("Unity environment_graph() after reset")
    base_ok, base_graph = comm.environment_graph()
    if not base_ok or not isinstance(base_graph, dict):
        raise RuntimeError("Unity environment_graph failed after reset")
    base_nodes = base_graph.get("nodes", [])
    if not isinstance(base_nodes, list) or len(base_nodes) == 0:
        raise RuntimeError("Unity base graph has no nodes after reset")
    max_id = max([int(node.get("id", 0)) for node in base_nodes if isinstance(node, dict)])

    expanded_graph = _separate_new_ids_graph(init_graph, max_id=max_id)
    _log("Unity expand_scene(init_graph)")
    expand_ok, expand_msg = comm.expand_scene(expanded_graph)
    if not expand_ok:
        raise RuntimeError("Unity expand_scene failed: {}".format(_safe_message_text(expand_msg)))

    _log("Unity camera_count() before add_character")
    cam_ok, static_camera_count = comm.camera_count()
    if not cam_ok:
        raise RuntimeError("Unity camera_count failed before adding character")
    static_camera_count = int(static_camera_count)

    _log(
        "Unity add_character(resource={}, initial_room={})".format(
            str(character_resource), str(initial_room) if str(initial_room) else "<random>"
        )
    )
    add_ok = comm.add_character(character_resource=str(character_resource), initial_room=str(initial_room))
    if not add_ok:
        raise RuntimeError("Unity add_character failed")

    moved_to_dataset = False
    move_error = None
    if align_char_to_dataset_start and dataset_step0_char_move_pos is not None:
        try:
            _log("Unity move_character(0, dataset step-0 position)")
            moved_to_dataset = bool(comm.move_character(0, dataset_step0_char_move_pos))
            if not moved_to_dataset:
                move_error = "move_character returned False"
        except Exception as e:
            move_error = repr(e)

    char_cam_names = None
    try:
        names_ok, names_msg = comm.character_cameras()
        if names_ok:
            char_cam_names = names_msg
    except Exception:
        char_cam_names = None

    return {
        "env_id": int(env_id),
        "static_camera_count": static_camera_count,
        "expanded_init_graph_max_base_id": int(max_id),
        "moved_char_to_dataset_start": bool(moved_to_dataset),
        "char_move_error": move_error,
        "character_cameras": char_cam_names,
    }


def main() -> None:
    args = parse_args()
    _log("Starting Unity belief visualization")
    device = _resolve_device(args.device)

    _log("Loading checkpoint/model: {}".format(args.checkpoint))
    checkpoint_path = Path(args.checkpoint)
    checkpoint, model, tensorizer, model_type = _load_semisup_model_and_tensorizer(str(checkpoint_path), device=device)
    predicate_names, goal_to_col = _load_goal_vocab(checkpoint, args.metadata)

    _log("Loading dataset split metadata")
    data_json_str, split_key = _infer_data_inputs_from_checkpoint(checkpoint, args.data_json, args.split_key)
    data_json = Path(data_json_str)
    split_key, demos = load_split(data_json, split_key)
    selected_index, demo = select_episode(demos, args.episode_index, args.episode_name)

    _log("Running belief inference for selected episode")
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

    raw_graphs = demo.get("graphs", [])
    raw_actions = demo.get("valid_action_with_walk", [])
    if not isinstance(raw_graphs, list) or not isinstance(raw_actions, list):
        raise TypeError("Episode graphs/actions are not lists.")
    raw_total_steps = min(len(raw_graphs), len(raw_actions)) if bool(getattr(model, "use_actions", False)) else len(raw_graphs)
    total_steps = min(total_steps, raw_total_steps)
    if total_steps <= 0:
        raise ValueError("No steps available after alignment.")
    y_probs_seq = y_probs_seq[:total_steps]
    entropy_mean_per_step = entropy_mean_per_step[:total_steps]

    goal_class_names = extract_goal_class_names(goal_list)
    _log(
        "Episode selected: name={} task={} steps(target={})".format(
            episode_name, task_name, total_steps
        )
    )

    if args.out_dir is None:
        out_dir = (
            Path("outputs")
            / "dataset_unity_belief_viz"
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

    # Infer a better initial character placement from dataset step 0.
    step0_graph = normalize_graph_frame(raw_graphs[0])
    step0_nodes = step0_graph["nodes"]
    step0_char_node = _dataset_char_node(step0_nodes)
    dataset_step0_char_center = _bbox_center(step0_char_node)
    dataset_step0_char_move_pos = _bbox_bottom_position(step0_char_node)

    _log("Importing VirtualHome UnityCommunication")
    comm_unity = _import_comm_unity()
    _install_comm_startup_debug_hooks(comm_unity)
    unity_port = int(args.base_port) + int(args.port_id)
    _log(
        "Creating UnityCommunication(port={}, executable={})".format(
            unity_port, args.unity_executable if args.unity_executable else "<attach>"
        )
    )
    t_comm_start = time.time()
    comm = comm_unity.UnityCommunication(
        port=str(unity_port),
        file_name=args.unity_executable,
        x_display=args.x_display,
        no_graphics=bool(args.no_graphics),
        timeout_wait=int(args.timeout_wait),
        docker_enabled=bool(args.docker_enabled),
    )
    _log("UnityCommunication initialized in {:.1f}s".format(time.time() - t_comm_start))

    steps_manifest: List[Dict[str, Any]] = []
    unity_setup_info: Dict[str, Any] = {}
    replay_errors: List[str] = []
    action_failures = 0

    try:
        try:
            _log("Explicit Unity check_connection()")
            comm.check_connection()
            _log("Unity connection confirmed")
        except Exception as e:
            hint = ""
            if not bool(args.no_graphics):
                hint = (
                    " Hint: Unity failed before opening the HTTP port. On headless machines, pass --no-graphics "
                    "(or set NO_GRAPHICS=1 in the shell wrapper)."
                )
            raise RuntimeError(
                "Could not connect to Unity on port {}. If not already running, pass --unity-executable. Error: {}{}".format(
                    unity_port, repr(e), hint
                )
            )

        _log("Bootstrapping Unity scene for dataset episode")
        unity_setup_info = _reset_unity_scene_for_demo(
            comm=comm,
            demo=demo,
            character_resource=str(args.character_resource),
            initial_room=str(args.initial_room),
            align_char_to_dataset_start=bool(args.align_char_to_dataset_start),
            dataset_step0_char_move_pos=dataset_step0_char_move_pos,
        )
        camera_index = int(unity_setup_info["static_camera_count"]) + int(args.camera_index_within_char)
        _log(
            "Unity scene ready (camera_index={} = static {} + char_slot {})".format(
                camera_index,
                unity_setup_info["static_camera_count"],
                int(args.camera_index_within_char),
            )
        )
        _log("Starting replay loop over {} steps (capture_phase={})".format(total_steps, args.capture_phase))

        for step_idx in range(total_steps):
            dataset_graph = normalize_graph_frame(raw_graphs[step_idx])
            dataset_nodes = dataset_graph["nodes"]
            dataset_visible_ids = visible_ids_for_frame(dataset_nodes)
            dataset_char_node = _dataset_char_node(dataset_nodes)
            dataset_char_center = _bbox_center(dataset_char_node)

            action_text, parsed_action_ids = action_text_and_ids(raw_actions[step_idx] if step_idx < len(raw_actions) else None)
            dataset_visible_set = set(dataset_visible_ids)
            action_ids = [obj_id for obj_id in parsed_action_ids if obj_id in dataset_visible_set]
            goal_ids = [obj_id for obj_id in goal_ids_for_frame(dataset_nodes, goal_class_names) if obj_id in dataset_visible_set]

            y_t = y_probs_seq[step_idx]
            topk = _step_topk(y_t, predicate_names, int(args.top_k_text))
            gt_probs = _step_gt_probs(y_t, predicate_names, gt_pred_indices)

            action_exec_success: Optional[bool] = None
            action_exec_message: Any = None
            unity_script_lines: List[str] = []

            capture: Dict[str, Any]
            _log(
                "Step {}/{}: {}".format(
                    step_idx + 1,
                    total_steps,
                    (action_text if action_text else "<none>"),
                )
            )
            if args.capture_phase == "before":
                capture = _capture_unity_state(
                    comm=comm,
                    camera_index=camera_index,
                    image_width=int(args.image_width),
                    image_height=int(args.image_height),
                )

                unity_script_lines = _single_agent_script_list(action_text)
                if len(unity_script_lines) > 0 and str(unity_script_lines[0]).strip():
                    action_exec_success, action_exec_message = comm.render_script(
                        unity_script_lines,
                        recording=False,
                        processing_time_limit=int(args.processing_time_limit),
                        time_scale=float(args.time_scale),
                        skip_animation=True,
                    )
                else:
                    action_exec_success, action_exec_message = True, {"skipped": "empty_action"}
            else:
                unity_script_lines = _single_agent_script_list(action_text)
                if len(unity_script_lines) > 0 and str(unity_script_lines[0]).strip():
                    action_exec_success, action_exec_message = comm.render_script(
                        unity_script_lines,
                        recording=False,
                        processing_time_limit=int(args.processing_time_limit),
                        time_scale=float(args.time_scale),
                        skip_animation=True,
                    )
                else:
                    action_exec_success, action_exec_message = True, {"skipped": "empty_action"}

                capture = _capture_unity_state(
                    comm=comm,
                    camera_index=camera_index,
                    image_width=int(args.image_width),
                    image_height=int(args.image_height),
                )

            if action_exec_success is False:
                action_failures += 1
                _log("Step {} action execution failed".format(step_idx))
            else:
                _log("Step {} action execution success={}".format(step_idx, action_exec_success))

            unity_visible_ids = capture.get("unity_visible_ids")
            visible_overlap = _set_overlap_metrics(unity_visible_ids, dataset_visible_ids)
            char_pos_l2 = _l2_or_none(capture.get("unity_char_center"), dataset_char_center)

            overlay_text = _build_overlay_text(
                episode_name=episode_name,
                task_name=task_name,
                step_idx=step_idx,
                total_steps=total_steps,
                action_text=action_text,
                capture_phase=str(args.capture_phase),
                y_entropy_mean=float(entropy_mean_per_step[step_idx]),
                topk=topk,
                gt_probs=gt_probs,
                action_exec_success=action_exec_success,
                char_pos_l2=char_pos_l2,
                visible_iou=visible_overlap["visible_iou"],
                visible_ids_error=capture.get("unity_visible_ids_error"),
            )

            frame_file = "frame_{:04d}.png".format(step_idx)
            frame_path = frames_dir / frame_file
            _draw_text_overlay_and_save(
                image_rgb=capture["image_rgb"],
                frame_path=frame_path,
                overlay_text=overlay_text,
                action_exec_success=action_exec_success,
            )

            steps_manifest.append(
                {
                    "step_idx": step_idx,
                    "frame_file": str(Path("frames") / frame_file),
                    "capture_phase": str(args.capture_phase),
                    "action_text": action_text,
                    "unity_script_lines": unity_script_lines,
                    "action_exec_success": None if action_exec_success is None else bool(action_exec_success),
                    "action_exec_message": _safe_message_text(action_exec_message),
                    "parsed_action_object_ids": parsed_action_ids,
                    "highlighted_action_object_ids": action_ids,
                    "goal_object_ids": goal_ids,
                    "dataset_char_id": None if dataset_char_node is None else int(dataset_char_node.get("id", -1)),
                    "dataset_char_center": _float_list_or_none(dataset_char_center),
                    "unity_char_id": capture.get("unity_char_id"),
                    "unity_char_center": _float_list_or_none(capture.get("unity_char_center")),
                    "char_pos_l2": char_pos_l2,
                    "dataset_visible_ids_count": int(len(dataset_visible_ids)),
                    "unity_visible_ids_count": None if unity_visible_ids is None else int(len(unity_visible_ids)),
                    "unity_visible_ids_error": capture.get("unity_visible_ids_error"),
                    "visible_intersection_count": visible_overlap["visible_intersection_count"],
                    "visible_union_count": visible_overlap["visible_union_count"],
                    "visible_iou": visible_overlap["visible_iou"],
                    "y_entropy_mean": float(entropy_mean_per_step[step_idx]),
                    "topk_y": topk,
                    "gt_goal_y_probs": gt_probs,
                }
            )

            if bool(args.stop_on_action_failure) and action_exec_success is False:
                replay_errors.append("Stopped on action failure at step {}".format(step_idx))
                _log("Stopping early on action failure at step {}".format(step_idx))
                break

    finally:
        try:
            _log("Closing UnityCommunication")
            comm.close()
        except Exception:
            pass

    rendered_steps = len(steps_manifest)
    if rendered_steps == 0:
        raise RuntimeError("No frames were rendered.")

    # Trim belief arrays to the actual rendered length if replay was stopped early.
    y_probs_seq = y_probs_seq[:rendered_steps]
    entropy_mean_per_step = entropy_mean_per_step[:rendered_steps]

    char_l2_vals = [s["char_pos_l2"] for s in steps_manifest if isinstance(s.get("char_pos_l2"), (float, int))]
    visible_iou_vals = [s["visible_iou"] for s in steps_manifest if isinstance(s.get("visible_iou"), (float, int))]

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
        "num_target_steps": int(total_steps),
        "num_rendered_steps": int(rendered_steps),
        "tracked_predicate_names": [str(predicate_names[int(i)]) for i in tracked_pred_indices],
        "topk_text": int(args.top_k_text),
        "plot_k": int(args.plot_k),
        "stable_slots": bool(args.stable_slots),
        "fps": int(args.fps),
        "entropy_mean_over_steps": float(np.mean(entropy_mean_per_step)) if rendered_steps > 0 else float("nan"),
        "entropy_std_over_steps": float(np.std(entropy_mean_per_step)) if rendered_steps > 0 else float("nan"),
        "unity_executable": args.unity_executable,
        "unity_port": int(unity_port),
        "unity_capture_phase": str(args.capture_phase),
        "unity_image_width": int(args.image_width),
        "unity_image_height": int(args.image_height),
        "unity_camera_index_within_char": int(args.camera_index_within_char),
        "unity_character_resource": str(args.character_resource),
        "unity_initial_room": str(args.initial_room),
        "unity_align_char_to_dataset_start": bool(args.align_char_to_dataset_start),
        "unity_setup": unity_setup_info,
        "replay_action_failures": int(action_failures),
        "replay_errors": replay_errors,
        "char_pos_l2_mean": float(np.mean(char_l2_vals)) if len(char_l2_vals) > 0 else None,
        "char_pos_l2_std": float(np.std(char_l2_vals)) if len(char_l2_vals) > 0 else None,
        "visible_iou_mean": float(np.mean(visible_iou_vals)) if len(visible_iou_vals) > 0 else None,
        "visible_iou_std": float(np.std(visible_iou_vals)) if len(visible_iou_vals) > 0 else None,
    }

    episode_meta = {
        "data_json": str(data_json),
        "split_key": split_key,
        "episode_index": selected_index,
        "episode_name": episode_name,
        "task_name": task_name,
        "goal": goal_list,
        "env_id": demo.get("env_id"),
        "has_init_graph": isinstance(demo.get("init_graph"), dict),
        "num_dataset_graph_frames": len(raw_graphs),
        "num_dataset_actions": len(raw_actions),
        "num_rendered_steps": int(rendered_steps),
        "fps": int(args.fps),
        "unity_capture_phase": str(args.capture_phase),
        "unity_setup": unity_setup_info,
        "dataset_step0_char_center": dataset_step0_char_center,
        "dataset_step0_char_move_pos": dataset_step0_char_move_pos,
    }

    with open(out_dir / "episode_meta.json", "w") as f:
        json.dump(episode_meta, f, indent=2)
    with open(out_dir / "steps_with_belief.json", "w") as f:
        json.dump(steps_manifest, f, indent=2)
    with open(out_dir / "belief_seq.json", "w") as f:
        json.dump(belief_seq_json, f, indent=2)
    with open(out_dir / "belief_summary.json", "w") as f:
        json.dump(belief_summary_json, f, indent=2)

    video_path = out_dir / "episode_unity_belief.mp4"
    if not args.skip_video:
        try:
            _log("Exporting MP4 via ffmpeg")
            run_ffmpeg(frames_dir=frames_dir, out_path=video_path, fps=int(args.fps))
        except FileNotFoundError as e:
            print("ffmpeg not found; frames were rendered but video export was skipped: {}".format(repr(e)), file=sys.stderr)
        except Exception:
            raise

    _log("Finished")
    print("Rendered {} Unity frames to {}".format(rendered_steps, frames_dir))
    if not args.skip_video and video_path.exists():
        print("Video written to {}".format(video_path))
    print("Belief summary: {}".format(out_dir / "belief_summary.json"))
    print("Belief sequence: {}".format(out_dir / "belief_seq.json"))
    print("Per-step manifest: {}".format(out_dir / "steps_with_belief.json"))


if __name__ == "__main__":
    main()
