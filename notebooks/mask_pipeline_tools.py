from __future__ import annotations

import glob
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

# from sam3.visualization_utils import (
#     prepare_masks_for_visualization,
#     visualize_formatted_frame_output,
# )


def cleanup_process_group():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        try:
            torch.distributed.destroy_process_group()
            print("[cleanup] distributed process group destroyed")
        except Exception as e:
            print(f"[warn] destroy_process_group() failed: {e}")


def cleanup_resources(predictor_obj=None, session_id_value=None):
    if predictor_obj is not None and session_id_value is not None:
        try:
            _ = predictor_obj.handle_request(
                request=dict(
                    type="close_session",
                    session_id=session_id_value,
                )
            )
            print(f"[cleanup] session closed: {session_id_value}")
        except Exception as e:
            print(f"[warn] close_session failed: {e}")

    if predictor_obj is not None:
        try:
            predictor_obj.shutdown()
            print("[cleanup] predictor shutdown finished")
        except Exception as e:
            print(f"[warn] predictor.shutdown() failed: {e}")

    cleanup_process_group()


def load_video_frames_for_visualization(video_path):
    # 仅用于可视化，不参与模型推理输入。
    if isinstance(video_path, str) and video_path.endswith(".mp4"):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        return frames

    if isinstance(video_path, str) and os.path.isdir(video_path):
        frame_names = sorted(
            glob.glob(os.path.join(video_path, "*.jpg"))
            + glob.glob(os.path.join(video_path, "*.jpeg"))
            + glob.glob(os.path.join(video_path, "*.png"))
        )
        if not frame_names:
            raise ValueError(f"视频目录为空或无可识别帧文件: {video_path}")
        frames = []
        for fp in frame_names:
            img = cv2.imread(fp)
            if img is None:
                raise ValueError(f"读取帧失败: {fp}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img)
        return frames

    raise ValueError(f"不支持的 VIDEO_PATH: {video_path}")


def get_frame_size(video_frames):
    if not video_frames:
        raise ValueError("video_frames 为空")
    frame0 = video_frames[0]
    if isinstance(frame0, torch.Tensor):
        frame0 = frame0.detach().cpu().numpy()
    if frame0.ndim == 4:
        frame0 = frame0[0]
    h, w = frame0.shape[:2]
    return int(w), int(h)


def abs_to_rel_points(points_abs, img_w, img_h):
    return [[float(x) / img_w, float(y) / img_h] for x, y in points_abs]


def validate_prompt_entry(entry, total_frames, img_w, img_h, tag="prompt"):
    required = ["frame_index", "obj_id", "points", "labels", "coord_type"]
    for k in required:
        if k not in entry:
            raise ValueError(f"[{tag}] 缺少字段: {k}; entry={entry}")

    frame_index = entry["frame_index"]
    obj_id = entry["obj_id"]
    points = entry["points"]
    labels = entry["labels"]
    coord_type = entry["coord_type"]

    if frame_index is None or not isinstance(frame_index, int):
        raise ValueError(f"[{tag}] frame_index 必须为 int，当前: {frame_index}")
    if frame_index < 0 or frame_index >= total_frames:
        raise ValueError(f"[{tag}] frame_index 越界: {frame_index}, 合法范围 [0, {total_frames - 1}]")

    if obj_id is None or not isinstance(obj_id, int):
        raise ValueError(f"[{tag}] obj_id 不能为空且必须为 int，当前: {obj_id}")

    if coord_type not in {"abs", "rel"}:
        raise ValueError(f"[{tag}] coord_type 仅支持 abs/rel，当前: {coord_type}")

    if not isinstance(points, (list, tuple)) or len(points) == 0:
        raise ValueError(f"[{tag}] points 不能为空")
    if not isinstance(labels, (list, tuple)) or len(labels) == 0:
        raise ValueError(f"[{tag}] labels 不能为空")
    if len(points) != len(labels):
        raise ValueError(f"[{tag}] points/labels 长度不一致: {len(points)} vs {len(labels)}")

    for i, (p, lb) in enumerate(zip(points, labels)):
        if not isinstance(p, (list, tuple)) or len(p) != 2:
            raise ValueError(f"[{tag}] 第{i}个点格式错误，应为 [x, y]，当前: {p}")
        x, y = float(p[0]), float(p[1])

        if coord_type == "abs":
            if not (0 <= x < img_w and 0 <= y < img_h):
                raise ValueError(
                    f"[{tag}] 第{i}个 abs 点越界: ({x}, {y}), 图像尺寸=({img_w}, {img_h})"
                )
        else:
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise ValueError(f"[{tag}] 第{i}个 rel 点越界: ({x}, {y}), 应在 [0,1]")

        if int(lb) not in {0, 1}:
            raise ValueError(f"[{tag}] 第{i}个 label 非法: {lb}, 仅支持 0/1")


def normalize_prompt_entry(entry, img_w, img_h):
    points = [[float(p[0]), float(p[1])] for p in entry["points"]]
    labels = [int(v) for v in entry["labels"]]

    if entry["coord_type"] == "abs":
        points_rel = abs_to_rel_points(points, img_w, img_h)
    else:
        points_rel = points

    return dict(
        frame_index=int(entry["frame_index"]),
        obj_id=int(entry["obj_id"]),
        points_rel=points_rel,
        labels=labels,
    )


def validate_and_normalize_prompt_list(
    prompt_list,
    total_frames,
    img_w,
    img_h,
    tag,
    allow_empty=False,
):
    if not isinstance(prompt_list, (list, tuple)):
        raise ValueError(f"[{tag}] 必须为 list")
    if len(prompt_list) == 0 and not allow_empty:
        raise ValueError(f"[{tag}] 不能为空")

    normalized = []
    for idx, entry in enumerate(prompt_list):
        if not isinstance(entry, dict):
            raise ValueError(f"[{tag}] 第{idx}项必须为 dict")
        validate_prompt_entry(entry, total_frames, img_w, img_h, tag=f"{tag}[{idx}]")
        normalized.append(normalize_prompt_entry(entry, img_w, img_h))
    return normalized


def validate_obj_id_constraints(
    arm_cable_obj_id,
    gripper_left_obj_id,
    gripper_right_obj_id,
    arm_cable_prompts,
    gripper_left_prompts,
    gripper_right_prompts,
    arm_cable_obj_id_2=None,
):
    obj_items = [
        ("arm_cable_obj_id", arm_cable_obj_id),
        ("gripper_left_obj_id", gripper_left_obj_id),
        ("gripper_right_obj_id", gripper_right_obj_id),
    ]
    if arm_cable_obj_id_2 is not None:
        obj_items.append(("arm_cable_obj_id_2", int(arm_cable_obj_id_2)))

    for name, obj_id in obj_items:
        if obj_id is None or not isinstance(obj_id, int):
            raise ValueError(f"{name} 不能为空且必须为 int，当前: {obj_id}")

    all_obj_ids = [obj_id for _, obj_id in obj_items]
    if len(set(all_obj_ids)) != len(all_obj_ids):
        raise ValueError(
            "对象 ID 冲突：所有已配置 obj_id 必须互不重复，当前="
            f"{dict(obj_items)}"
        )

    def _check_prompt_obj_ids(prompt_list, expected_obj_id, prompt_tag):
        for i, p in enumerate(prompt_list):
            if p["obj_id"] != expected_obj_id:
                raise ValueError(
                    f"{prompt_tag}[{i}].obj_id={p['obj_id']} 与期望 obj_id={expected_obj_id} 不一致"
                )

    _check_prompt_obj_ids(arm_cable_prompts, arm_cable_obj_id, "ARM_CABLE_INITIAL_PROMPTS")
    _check_prompt_obj_ids(gripper_left_prompts, gripper_left_obj_id, "GRIPPER_LEFT_KEYFRAME_PROMPTS")
    _check_prompt_obj_ids(gripper_right_prompts, gripper_right_obj_id, "GRIPPER_RIGHT_KEYFRAME_PROMPTS")


def _median(values):
    if not values:
        return None
    sorted_vals = sorted(float(v) for v in values)
    n = len(sorted_vals)
    mid = n // 2
    if n % 2 == 1:
        return float(sorted_vals[mid])
    return float((sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0)


def summarize_normalized_prompt_spatial_x(prompt_list, img_w, side_name=""):
    img_w = max(float(img_w), 1.0)
    x_all = []
    x_pos = []

    for p in prompt_list:
        if not isinstance(p, dict):
            continue
        points_rel = p.get("points_rel", [])
        labels = p.get("labels", [])
        if not isinstance(points_rel, list):
            continue

        for i, point in enumerate(points_rel):
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                continue

            x_rel = float(point[0])
            x_abs = x_rel * img_w
            x_all.append(float(x_abs))

            label_value = None
            if isinstance(labels, list) and i < len(labels):
                try:
                    label_value = int(labels[i])
                except Exception:
                    label_value = None
            if label_value == 1:
                x_pos.append(float(x_abs))

    basis_x = list(x_pos) if len(x_pos) > 0 else list(x_all)
    basis_name = "positive" if len(x_pos) > 0 else "all"

    if len(basis_x) == 0:
        return {
            "side_name": str(side_name),
            "entries": int(len(prompt_list)),
            "all_points": 0,
            "pos_points": 0,
            "basis": basis_name,
            "basis_points": 0,
            "x_mean": None,
            "x_median": None,
            "left_ratio": None,
            "right_ratio": None,
            "dominant_side": "unknown",
        }

    half_x = img_w / 2.0
    left_count = sum(1 for x in basis_x if float(x) < half_x)
    right_count = sum(1 for x in basis_x if float(x) >= half_x)
    total = len(basis_x)

    return {
        "side_name": str(side_name),
        "entries": int(len(prompt_list)),
        "all_points": int(len(x_all)),
        "pos_points": int(len(x_pos)),
        "basis": basis_name,
        "basis_points": int(total),
        "x_mean": float(sum(basis_x) / total),
        "x_median": _median(basis_x),
        "left_ratio": float(left_count / total),
        "right_ratio": float(right_count / total),
        "dominant_side": "unknown",
    }


def diagnose_gripper_prompt_side_consistency(
    left_prompts,
    right_prompts,
    img_w,
    *,
    stage_name="stage B",
    dominance_ratio=0.6,
):
    left_stats = summarize_normalized_prompt_spatial_x(
        prompt_list=left_prompts,
        img_w=img_w,
        side_name="left",
    )
    right_stats = summarize_normalized_prompt_spatial_x(
        prompt_list=right_prompts,
        img_w=img_w,
        side_name="right",
    )

    def _attach_dominant(stats):
        left_ratio = stats.get("left_ratio")
        right_ratio = stats.get("right_ratio")
        dominant = "unknown"
        if left_ratio is not None and right_ratio is not None:
            if float(left_ratio) >= float(dominance_ratio):
                dominant = "left"
            elif float(right_ratio) >= float(dominance_ratio):
                dominant = "right"
        stats["dominant_side"] = dominant
        return stats

    left_stats = _attach_dominant(dict(left_stats))
    right_stats = _attach_dominant(dict(right_stats))

    print(
        f"[{stage_name}][diag][spatial][left] entries={left_stats['entries']} "
        f"all_points={left_stats['all_points']} pos_points={left_stats['pos_points']} "
        f"basis={left_stats['basis']} basis_points={left_stats['basis_points']} "
        f"x_mean={left_stats['x_mean']} x_median={left_stats['x_median']} "
        f"left_ratio={left_stats['left_ratio']} right_ratio={left_stats['right_ratio']} "
        f"dominant={left_stats['dominant_side']}"
    )
    print(
        f"[{stage_name}][diag][spatial][right] entries={right_stats['entries']} "
        f"all_points={right_stats['all_points']} pos_points={right_stats['pos_points']} "
        f"basis={right_stats['basis']} basis_points={right_stats['basis_points']} "
        f"x_mean={right_stats['x_mean']} x_median={right_stats['x_median']} "
        f"left_ratio={right_stats['left_ratio']} right_ratio={right_stats['right_ratio']} "
        f"dominant={right_stats['dominant_side']}"
    )

    conflict_type = "none"
    warning = ""
    if left_stats["basis_points"] > 0 and right_stats["basis_points"] > 0:
        if left_stats["dominant_side"] == "right" and right_stats["dominant_side"] == "left":
            conflict_type = "likely_swapped"
            warning = (
                "检测到左右标签与空间主侧明显反向：left 点主要在右半区且 right 点主要在左半区。"
            )
    elif left_stats["basis_points"] > 0 and right_stats["basis_points"] == 0:
        if left_stats["dominant_side"] == "right":
            conflict_type = "left_points_mainly_right"
            warning = "仅 left 有点，且主要位于右半区；请人工确认 left/right 标注语义。"
    elif right_stats["basis_points"] > 0 and left_stats["basis_points"] == 0:
        if right_stats["dominant_side"] == "left":
            conflict_type = "right_points_mainly_left"
            warning = "仅 right 有点，且主要位于左半区；请人工确认 left/right 标注语义。"

    if conflict_type != "none":
        print(f"[{stage_name}][diag][spatial][warn] conflict_type={conflict_type} | {warning}")
    else:
        print(f"[{stage_name}][diag][spatial] conflict_type=none")

    return {
        "left": left_stats,
        "right": right_stats,
        "dominance_ratio": float(dominance_ratio),
        "conflict_type": conflict_type,
        "warning": warning,
    }


def propagate_in_video(predictor_obj, session_id_value, propagation_direction="forward"):
    outputs_per_frame = {}
    for response in predictor_obj.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id_value,
            propagation_direction=propagation_direction,
        )
    ):
        outputs_per_frame[response["frame_index"]] = response["outputs"]
    return outputs_per_frame


def propagate_bidirectional_and_merge(predictor_obj, session_id_value, stage_name=""):
    outputs_forward = propagate_in_video(
        predictor_obj=predictor_obj,
        session_id_value=session_id_value,
        propagation_direction="forward",
    )
    outputs_backward = propagate_in_video(
        predictor_obj=predictor_obj,
        session_id_value=session_id_value,
        propagation_direction="backward",
    )

    merged_outputs = {}
    merged_outputs.update(outputs_forward)
    merged_outputs.update(outputs_backward)

    stage_prefix = f"[{stage_name}] " if stage_name else ""
    print(
        f"{stage_prefix}propagation summary | forward_frames={len(outputs_forward)} "
        f"backward_frames={len(outputs_backward)} merged_frames={len(merged_outputs)}"
    )
    return merged_outputs


def add_point_prompt(predictor_obj, session_id_value, prompt, stage_name=""):
    points_tensor = torch.tensor(prompt["points_rel"], dtype=torch.float32)
    labels_tensor = torch.tensor(prompt["labels"], dtype=torch.int32)

    _ = predictor_obj.handle_request(
        request=dict(
            type="add_prompt",
            session_id=session_id_value,
            frame_index=prompt["frame_index"],
            points=points_tensor,
            point_labels=labels_tensor,
            obj_id=prompt["obj_id"],
        )
    )
    print(
        f"[{stage_name}] add_prompt done | frame={prompt['frame_index']} obj_id={prompt['obj_id']} points={len(prompt['points_rel'])}"
    )


def apply_prompt_list(predictor_obj, session_id_value, prompt_list, stage_name=""):
    for p in prompt_list:
        add_point_prompt(predictor_obj, session_id_value, p, stage_name=stage_name)


def add_text_prompt(predictor_obj, session_id_value, frame_index, text_prompt, stage_name=""):
    if not isinstance(text_prompt, str) or len(text_prompt.strip()) == 0:
        raise ValueError("text_prompt 不能为空，请设置 ARM_BOOTSTRAP_TEXT_PROMPT")

    _ = predictor_obj.handle_request(
        request=dict(
            type="add_prompt",
            session_id=session_id_value,
            frame_index=int(frame_index),
            text=text_prompt.strip(),
        )
    )
    print(
        f"[{stage_name}] add_text_prompt done | frame={int(frame_index)} text={text_prompt.strip()!r}"
    )


def resolve_stage_a_bootstrap_configs(stage_side_configs, total_frames):
    if not isinstance(stage_side_configs, (list, tuple)) or len(stage_side_configs) == 0:
        raise ValueError("stage_side_configs 必须是非空 list")

    resolved_configs = []

    for idx, cfg in enumerate(stage_side_configs):
        side_name = str(cfg.get("side_name", f"side-{idx}"))
        prompt_list = cfg.get("prompt_list", [])
        bootstrap_text_prompt = cfg.get("bootstrap_text_prompt", "")
        fallback_text_prompt = cfg.get("fallback_text_prompt", "")
        bootstrap_frame_index = cfg.get("bootstrap_frame_index", None)

        has_points = len(prompt_list) > 0
        user_text = bootstrap_text_prompt.strip() if isinstance(bootstrap_text_prompt, str) else ""
        fallback_text = fallback_text_prompt.strip() if isinstance(fallback_text_prompt, str) else ""

        if user_text:
            resolved_text = user_text
            text_source = "configured"
        elif has_points and fallback_text:
            resolved_text = fallback_text
            text_source = "fallback"
            print(
                f"[stage A][bootstrap][{side_name}] 文本提示为空，自动使用 fallback: {resolved_text!r}"
            )
        elif has_points:
            raise ValueError(
                f"[{side_name}] 检测到 arm points，但 bootstrap 文本为空且无 fallback。"
                "请设置对应侧的 *_BOOTSTRAP_TEXT_PROMPT 或 *_BOOTSTRAP_FALLBACK_TEXT_PROMPT。"
            )
        else:
            continue

        if bootstrap_frame_index is None:
            resolved_frame = int(prompt_list[0]["frame_index"]) if has_points else 0
        else:
            if not isinstance(bootstrap_frame_index, int):
                raise ValueError(
                    f"[{side_name}] bootstrap_frame_index 必须为 int 或 None，当前: {bootstrap_frame_index}"
                )
            if bootstrap_frame_index < 0 or bootstrap_frame_index >= total_frames:
                raise ValueError(
                    f"[{side_name}] bootstrap_frame_index 越界: {bootstrap_frame_index}, "
                    f"合法范围 [0, {total_frames - 1}]"
                )
            resolved_frame = int(bootstrap_frame_index)

        resolved_configs.append(
            dict(
                side_name=side_name,
                text_prompt=resolved_text,
                text_source=text_source,
                frame_index=resolved_frame,
            )
        )

    if len(resolved_configs) == 0:
        raise ValueError(
            "阶段A缺少可用提示：左右机械臂均未提供 bootstrap 文本，也没有可触发 fallback 的 points。"
        )

    return resolved_configs


def visualize_outputs(outputs_per_frame, video_frames, stride=60, max_plots=8, title="SAM3 outputs"):
    outputs_for_vis = prepare_masks_for_visualization(outputs_per_frame)
    frame_indices = list(range(0, len(video_frames), stride))[:max_plots]

    if not frame_indices:
        frame_indices = [0]

    plt.close("all")
    for frame_idx in frame_indices:
        visualize_formatted_frame_output(
            frame_idx,
            video_frames,
            outputs_list=[outputs_for_vis],
            titles=[title],
            figsize=(6, 4),
        )


def resolve_export_obj_ids(export_mode, arm_obj_ids, gripper_obj_ids, custom_obj_ids=None):
    custom_obj_ids = custom_obj_ids or []
    arm_obj_ids = sorted(set(int(x) for x in arm_obj_ids))
    gripper_obj_ids = sorted(set(int(x) for x in gripper_obj_ids))

    if export_mode == "arm-only":
        return arm_obj_ids
    if export_mode == "gripper-only":
        return gripper_obj_ids
    if export_mode == "union":
        return sorted(set(arm_obj_ids + gripper_obj_ids))
    if export_mode == "custom":
        if len(custom_obj_ids) == 0:
            raise ValueError("EXPORT_MODE=custom 时，EXPORT_CUSTOM_OBJ_IDS 不能为空")
        return sorted(set(int(x) for x in custom_obj_ids))

    raise ValueError(f"不支持的 EXPORT_MODE: {export_mode}")


def _to_binary_mask(mask):
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    if mask.ndim > 2:
        mask = np.squeeze(mask)
    return (mask > 0).astype(np.uint8) * 255


def iter_object_masks_from_frame_output(frame_outputs):
    """
    统一解析单帧输出结构，兼容两种格式：
    1) 原始 predictor 输出：{"out_obj_ids": ..., "out_binary_masks": ...}
    2) 已处理输出：{obj_id: binary_mask}

    Yields:
        (obj_id_int, mask)
    """
    if not isinstance(frame_outputs, dict):
        return

    # 优先解析 SAM3 原始输出结构（按 out_obj_ids 与 out_binary_masks 对齐）
    if "out_obj_ids" in frame_outputs and "out_binary_masks" in frame_outputs:
        out_obj_ids = frame_outputs.get("out_obj_ids", [])
        out_binary_masks = frame_outputs.get("out_binary_masks", [])

        try:
            obj_id_list = out_obj_ids.tolist() if hasattr(out_obj_ids, "tolist") else list(out_obj_ids)
        except Exception:
            obj_id_list = []

        try:
            mask_list = list(out_binary_masks)
        except Exception:
            mask_list = []

        pair_count = min(len(obj_id_list), len(mask_list))
        for i in range(pair_count):
            try:
                obj_id_int = int(obj_id_list[i])
            except (TypeError, ValueError):
                continue
            yield obj_id_int, mask_list[i]
        return

    # 回退：将 dict 键视为对象 ID（跳过元数据键）
    for obj_id, mask in frame_outputs.items():
        if isinstance(obj_id, bool):
            continue
        try:
            obj_id_int = int(obj_id)
        except (TypeError, ValueError):
            continue
        yield obj_id_int, mask


def sample_obj_ids_from_outputs(outputs_per_frame, max_frames=5, max_obj_ids=12):
    """返回用于调试的 obj_id 样例：[(frame_idx, [obj_id,...]), ...]"""
    samples = []
    frame_indices = sorted(int(k) for k in outputs_per_frame.keys())[: max(int(max_frames), 0)]

    for frame_idx in frame_indices:
        frame_outputs = outputs_per_frame.get(frame_idx, {})
        obj_ids = []
        for obj_id_int, _ in iter_object_masks_from_frame_output(frame_outputs):
            obj_ids.append(int(obj_id_int))
            if len(obj_ids) >= max(int(max_obj_ids), 1):
                break
        samples.append((int(frame_idx), sorted(set(obj_ids))))

    return samples


def collect_present_obj_ids(outputs_per_frame, target_obj_ids=None):
    """
    收集 outputs 中实际出现的对象 ID。
    若 target_obj_ids 不为空，则只返回其子集。
    """
    target_set = None if target_obj_ids is None else set(int(x) for x in target_obj_ids)
    present = set()

    for frame_outputs in outputs_per_frame.values():
        for obj_id_int, _ in iter_object_masks_from_frame_output(frame_outputs):
            if target_set is None or int(obj_id_int) in target_set:
                present.add(int(obj_id_int))

    return sorted(present)


def save_masks_for_propainter(
    outputs_per_frame,
    video_frames,
    output_dir,
    target_obj_ids,
    dilate_radius=8,
):
    os.makedirs(output_dir, exist_ok=True)

    img_w, img_h = get_frame_size(video_frames)
    num_frames = len(video_frames)

    kernel = None
    if dilate_radius > 0:
        kernel_size = 2 * int(dilate_radius) + 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        print(f"[export] dilation enabled: radius={dilate_radius}, kernel={kernel_size}x{kernel_size}")

    print(f"[export] size={img_w}x{img_h}, frames={num_frames}, target_obj_ids={target_obj_ids}")

    saved_paths = []
    target_set = set(target_obj_ids)

    for frame_idx in range(num_frames):
        combined_mask = np.zeros((img_h, img_w), dtype=np.uint8)

        frame_outputs = outputs_per_frame.get(frame_idx, {})
        for obj_id_int, mask in iter_object_masks_from_frame_output(frame_outputs):
            if int(obj_id_int) not in target_set:
                continue

            binary = _to_binary_mask(mask)
            combined_mask = np.maximum(combined_mask, binary)

        if kernel is not None and np.any(combined_mask):
            combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)

        out_fp = os.path.join(output_dir, f"{frame_idx:05d}.png")
        Image.fromarray(combined_mask, mode="L").save(out_fp)
        saved_paths.append(out_fp)

        if frame_idx % 50 == 0:
            print(f"[export] frame {frame_idx:05d}/{num_frames - 1:05d} done")

    print(f"[export] finished, saved {len(saved_paths)} masks to: {output_dir}")
    return saved_paths


def save_arm_only_masks_for_propainter(
    outputs_per_frame,
    video_frames,
    output_dir,
    arm_obj_ids,
    gripper_obj_ids,
    dilate_radius=8,
    log_every=50,
):
    arm_obj_ids = sorted(set(int(x) for x in arm_obj_ids))
    gripper_obj_ids = sorted(set(int(x) for x in gripper_obj_ids))

    if len(arm_obj_ids) == 0:
        raise ValueError("[export] arm_obj_ids 不能为空")
    if len(gripper_obj_ids) == 0:
        raise ValueError("[export] gripper_obj_ids 不能为空")

    os.makedirs(output_dir, exist_ok=True)

    img_w, img_h = get_frame_size(video_frames)
    num_frames = len(video_frames)

    kernel = None
    if dilate_radius > 0:
        kernel_size = 2 * int(dilate_radius) + 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        print(
            f"[export] dilation enabled (arm/gripper separately): radius={dilate_radius}, "
            f"kernel={kernel_size}x{kernel_size}"
        )

    print(
        f"[export] arm-only subtraction enabled | size={img_w}x{img_h}, frames={num_frames}, "
        f"arm_obj_ids={arm_obj_ids}, gripper_obj_ids={gripper_obj_ids}"
    )

    saved_paths = []
    arm_set = set(arm_obj_ids)
    gripper_set = set(gripper_obj_ids)
    total_gripper_pixels = 0

    for frame_idx in range(num_frames):
        arm_union = np.zeros((img_h, img_w), dtype=np.uint8)
        gripper_union = np.zeros((img_h, img_w), dtype=np.uint8)

        frame_outputs = outputs_per_frame.get(frame_idx, {})
        for obj_id_int, mask in iter_object_masks_from_frame_output(frame_outputs):
            obj_id_int = int(obj_id_int)
            if obj_id_int not in arm_set and obj_id_int not in gripper_set:
                continue

            binary = _to_binary_mask(mask)
            if obj_id_int in arm_set:
                arm_union = np.maximum(arm_union, binary)
            if obj_id_int in gripper_set:
                gripper_union = np.maximum(gripper_union, binary)

        if kernel is not None:
            if np.any(arm_union):
                arm_union = cv2.dilate(arm_union, kernel, iterations=1)
            if np.any(gripper_union):
                gripper_union = cv2.dilate(gripper_union, kernel, iterations=1)

        arm_only = np.where((arm_union > 0) & (gripper_union == 0), 255, 0).astype(np.uint8)

        arm_pixels = int(np.count_nonzero(arm_union))
        gripper_pixels = int(np.count_nonzero(gripper_union))
        arm_only_pixels = int(np.count_nonzero(arm_only))
        total_gripper_pixels += gripper_pixels

        if frame_idx % max(int(log_every), 1) == 0 or frame_idx == num_frames - 1:
            print(
                f"[export][frame {frame_idx:05d}] arm_pixels={arm_pixels} "
                f"gripper_pixels={gripper_pixels} arm_only_pixels={arm_only_pixels}"
            )

        out_fp = os.path.join(output_dir, f"{frame_idx:05d}.png")
        Image.fromarray(arm_only, mode="L").save(out_fp)
        saved_paths.append(out_fp)

    if total_gripper_pixels <= 0:
        raise ValueError(
            "[export] Stage B gripper 掩码总像素为 0，视为 gripper 提示无效；"
            "已阻断导出以避免错误 arm-only 结果。"
        )

    print(f"[export] finished (arm-only subtraction), saved {len(saved_paths)} masks to: {output_dir}")
    return saved_paths
