import os
import yaml
import torch
import argparse
import numpy as np
import open3d as o3d
import torchvision.transforms as T

from copy import deepcopy
from easydict import EasyDict as edict

from utils.training import set_seed
from utils.ensemble import EnsembleBuffer
# from remote_eval import WebsocketClientPolicy
from eval_agent import SingleArmAgent, DualArmAgent
from dataset.data_utils import resize_image, ImageProcessor
from dataset.projector import SingleArmProjector, DualArmProjector

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

import cv2

# test_color = "/data/haoxiang/realdata_rise2_ready/train/task_0014_user_0020_scene_0001_cfg_0001/cam_104122063550/color/1765000020748.png"
# test_depth = "/data/haoxiang/realdata_rise2_ready/train/task_0014_user_0020_scene_0001_cfg_0001/cam_104122063550/depth/1765000020748.png"


test_color = "/data/haoxiang/data/airexo2/task_0013/train/scene_0001/cam_105422061350/color/1737546126606.png"
test_depth = "/data/haoxiang/data/airexo2/task_0013/train/scene_0001/cam_105422061350/depth/1737546126606.png"

fake_intrinsics = np.array([
    [912.4466 ,   0.     , 633.4127 ],
    [  0.     , 911.4704 , 364.21265],
    [  0.     ,   0.     ,   1.     ]
])

fake_depth_scale = 1000.0

test_low_dim = "/data/haoxiang/data/airexo2/task_0013/train/scene_0001/lowdim/1737546126606.npy"
# lowdim npy 期望是 dict，包含 robot_left/right 和 gripper_left/right。


default_args = edict({
    "type": "local",
    "calib_rise2": "calib_rise2/",
    "calib_airexo": "calib_airexo/",
    "config": "config/dual_teleop_dino.yaml",
    "ckpt": "logs/collect_toys",
    "host": "127.0.0.1",
    "port": 8000
})


def _build_mask_aware_cfg(config):
    # 统一推理期 mask-aware 配置并补齐默认值
    default_cfg = {
        "enabled": False,
        "enable_3d_filter": True,
        "enable_2d_reweight": True,
        "mask_threshold": 0,
        "mask_white_is_untrusted": True,
        "r_min": 1e-3,
        "interp_eps": 1e-6,
        "interp_tiny": 1e-6,
        "infer_allow_none": True,
        "infer_none_policy": "no_mask_fallback",
        "empty_cloud_policy": "warn_and_skip_filter",
        "urdf": None,
        "sam2": {},
    }
    raw_cfg = getattr(config, "mask_aware", {})
    raw_cfg = dict(raw_cfg) if raw_cfg is not None else {}

    merged_cfg = deepcopy(default_cfg)
    for key in default_cfg:
        if key in raw_cfg and raw_cfg[key] is not None:
            merged_cfg[key] = raw_cfg[key]

    valid_none_policy = {"no_mask_fallback", "fail_fast"}
    valid_empty_cloud_policy = {"warn_and_skip_filter", "fail_fast"}

    try:
        merged_cfg["enabled"] = bool(merged_cfg["enabled"])
        merged_cfg["enable_3d_filter"] = bool(merged_cfg["enable_3d_filter"])
        merged_cfg["enable_2d_reweight"] = bool(merged_cfg["enable_2d_reweight"])
        merged_cfg["mask_threshold"] = float(merged_cfg["mask_threshold"])
        merged_cfg["mask_white_is_untrusted"] = bool(merged_cfg["mask_white_is_untrusted"])
        merged_cfg["r_min"] = float(merged_cfg["r_min"])
        merged_cfg["interp_eps"] = float(merged_cfg["interp_eps"])
        merged_cfg["interp_tiny"] = float(merged_cfg["interp_tiny"])
        merged_cfg["infer_allow_none"] = bool(merged_cfg["infer_allow_none"])
        merged_cfg["infer_none_policy"] = str(merged_cfg["infer_none_policy"])
        merged_cfg["empty_cloud_policy"] = str(merged_cfg["empty_cloud_policy"])
        merged_cfg["sam2"] = dict(merged_cfg["sam2"]) if merged_cfg["sam2"] is not None else {}
    except Exception as exc:
        print(f"[mask-aware] invalid config, fallback to disabled: {exc}")
        merged_cfg = deepcopy(default_cfg)

    if merged_cfg["infer_none_policy"] not in valid_none_policy:
        print("[mask-aware] invalid infer_none_policy, use no_mask_fallback")
        merged_cfg["infer_none_policy"] = "no_mask_fallback"

    if merged_cfg["empty_cloud_policy"] not in valid_empty_cloud_policy:
        print("[mask-aware] invalid empty_cloud_policy, use warn_and_skip_filter")
        merged_cfg["empty_cloud_policy"] = "warn_and_skip_filter"

    return edict(merged_cfg)


def _log_mask_aware_summary(mask_cfg):
    # 启动时打印一次配置摘要用于排查
    print(
        "[mask-aware] enabled={} 3d_filter={} 2d_reweight={} threshold={} "
        "none_policy={} empty_cloud_policy={}".format(
            mask_cfg.enabled,
            mask_cfg.enable_3d_filter,
            mask_cfg.enable_2d_reweight,
            mask_cfg.mask_threshold,
            mask_cfg.infer_none_policy,
            mask_cfg.empty_cloud_policy,
        )
    )


def _log_mask_fallback(step, reason, action):
    # 异常回退统一日志输出
    print(f"[mask-aware] step={step} reason={reason} action={action}")


def _build_sam2_cfg(mask_aware_cfg):
    # 统一 SAM2 在线分割配置并补齐默认值
    default_cfg = {
        "enabled": True,
        "config_file": "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "ckpt_path": "checkpoints/sam2.1_hiera_base_plus.pt",
        "device": "cuda_if_available",
        "auto_every_n_steps": 15,
        "pred_iou_thresh": 0.80,
        "stability_score_thresh": 0.92,
        "points_per_side": 24,
        "min_area_ratio": 0.003,
        "max_area_ratio": 0.6,
        "track_iou_min": 0.01,
        "max_track_failures": 3,
        "prompt_num_pos_points": 12,
        "prompt_num_neg_points": 16,
        "prompt_dilate_radius": 15,
        "multimask_output": False,
    }

    raw_cfg = getattr(mask_aware_cfg, "sam2", {})
    raw_cfg = dict(raw_cfg) if raw_cfg is not None else {}

    merged_cfg = deepcopy(default_cfg)
    for key in default_cfg:
        if key in raw_cfg and raw_cfg[key] is not None:
            merged_cfg[key] = raw_cfg[key]

    try:
        merged_cfg["enabled"] = bool(merged_cfg["enabled"])
        merged_cfg["config_file"] = str(merged_cfg["config_file"])
        merged_cfg["ckpt_path"] = str(merged_cfg["ckpt_path"])
        merged_cfg["device"] = str(merged_cfg["device"])
        merged_cfg["auto_every_n_steps"] = int(merged_cfg["auto_every_n_steps"])
        merged_cfg["pred_iou_thresh"] = float(merged_cfg["pred_iou_thresh"])
        merged_cfg["stability_score_thresh"] = float(merged_cfg["stability_score_thresh"])
        merged_cfg["points_per_side"] = int(merged_cfg["points_per_side"])
        merged_cfg["min_area_ratio"] = float(merged_cfg["min_area_ratio"])
        merged_cfg["max_area_ratio"] = float(merged_cfg["max_area_ratio"])
        merged_cfg["track_iou_min"] = float(merged_cfg["track_iou_min"])
        merged_cfg["max_track_failures"] = int(merged_cfg["max_track_failures"])
        merged_cfg["prompt_num_pos_points"] = int(merged_cfg["prompt_num_pos_points"])
        merged_cfg["prompt_num_neg_points"] = int(merged_cfg["prompt_num_neg_points"])
        merged_cfg["prompt_dilate_radius"] = int(merged_cfg["prompt_dilate_radius"])
        merged_cfg["multimask_output"] = bool(merged_cfg["multimask_output"])
    except Exception as exc:
        print(f"[mask-aware/sam2] invalid config, fallback to disabled: {exc}")
        merged_cfg = deepcopy(default_cfg)
        merged_cfg["enabled"] = False

    return edict(merged_cfg)


def _log_sam2_summary(sam2_cfg):
    print(
        "[mask-aware/sam2] enabled={} cfg={} ckpt={} device={} auto_every={} "
        "area=[{:.4f}, {:.4f}] track_iou_min={} max_track_failures={}".format(
            sam2_cfg.enabled,
            sam2_cfg.config_file,
            sam2_cfg.ckpt_path,
            sam2_cfg.device,
            sam2_cfg.auto_every_n_steps,
            sam2_cfg.min_area_ratio,
            sam2_cfg.max_area_ratio,
            sam2_cfg.track_iou_min,
            sam2_cfg.max_track_failures,
        )
    )


# ── 模块级 SAM2 runtime 引用，由 evaluate() 在启动时初始化 ──
_sam2_runtime = None


def _resolve_sam2_device(device_str):
    device_str = str(device_str).lower()
    if device_str in ["cuda_if_available", "auto", "cuda"]:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if device_str == "cuda":
            raise RuntimeError("SAM2 device is set to cuda but CUDA is not available")
        return torch.device("cpu")
    if device_str == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError("SAM2 device is set to mps but MPS is not available")
    if device_str == "cpu":
        return torch.device("cpu")
    raise ValueError(f"unsupported sam2 device: {device_str}")


def _init_sam2_runtime(sam2_cfg):
    device = _resolve_sam2_device(sam2_cfg.device)
    sam_model = build_sam2(
        config_file=sam2_cfg.config_file,
        ckpt_path=sam2_cfg.ckpt_path,
        device=device,
        mode="eval",
    )
    img_predictor = SAM2ImagePredictor(sam_model)
    auto_mask_generator = SAM2AutomaticMaskGenerator(
        model=sam_model,
        points_per_side=sam2_cfg.points_per_side,
        pred_iou_thresh=sam2_cfg.pred_iou_thresh,
        stability_score_thresh=sam2_cfg.stability_score_thresh,
        output_mode="binary_mask",
        multimask_output=sam2_cfg.multimask_output,
    )
    return {
        "enabled": bool(sam2_cfg.enabled),
        "cfg": sam2_cfg,
        "device": device,
        "sam_model": sam_model,
        "img_predictor": img_predictor,
        "auto_mask_generator": auto_mask_generator,
        "last_mask": None,
        "last_bbox": None,
        "last_step": None,
        "last_auto_step": None,
        "track_fail_count": 0,
        "last_fail_reason": None,
    }


def _extract_step_from_meta(meta):
    if not isinstance(meta, dict):
        return None
    step = meta.get("step", None)
    if step is None:
        return None
    try:
        return int(step)
    except Exception:
        return None


def _to_mask_bool(mask):
    if isinstance(mask, torch.Tensor):
        mask_np = mask.detach().cpu().numpy()
    else:
        mask_np = np.asarray(mask)

    if mask_np.ndim == 3:
        if mask_np.shape[0] == 1:
            mask_np = mask_np[0]
        elif mask_np.shape[-1] == 1:
            mask_np = mask_np[..., 0]
        else:
            raise ValueError(f"unsupported mask shape: {mask_np.shape}")
    elif mask_np.ndim != 2:
        raise ValueError(f"unsupported mask shape: {mask_np.shape}")

    if mask_np.dtype == np.bool_:
        return mask_np
    return mask_np > 0


def _mask_area_ratio(mask_bool):
    h, w = mask_bool.shape[:2]
    denom = max(1, int(h) * int(w))
    return float(mask_bool.sum()) / float(denom)


def _mask_iou(mask_a, mask_b):
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def _mask_to_bbox_xyxy(mask_bool):
    ys, xs = np.where(mask_bool)
    if xs.size == 0 or ys.size == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def _sample_prompt_points(mask_bool, num_pos, num_neg, dilate_radius=15):
    fg_yx = np.column_stack(np.where(mask_bool))
    if fg_yx.shape[0] == 0:
        return None, None

    pos_n = min(int(num_pos), fg_yx.shape[0])
    pos_idx = np.random.choice(fg_yx.shape[0], size=pos_n, replace=False)
    pos_yx = fg_yx[pos_idx]

    mask_u8 = mask_bool.astype(np.uint8)
    if int(dilate_radius) > 0:
        ks = 2 * int(dilate_radius) + 1
        kernel = np.ones((ks, ks), np.uint8)
        dilated = cv2.dilate(mask_u8, kernel, iterations=1).astype(np.bool_)
        neg_region = np.logical_and(dilated, np.logical_not(mask_bool))
    else:
        neg_region = np.logical_not(mask_bool)

    neg_yx = np.column_stack(np.where(neg_region))
    if neg_yx.shape[0] == 0:
        neg_yx = np.column_stack(np.where(np.logical_not(mask_bool)))

    if neg_yx.shape[0] > 0 and int(num_neg) > 0:
        neg_n = min(int(num_neg), neg_yx.shape[0])
        neg_idx = np.random.choice(neg_yx.shape[0], size=neg_n, replace=False)
        neg_yx = neg_yx[neg_idx]
    else:
        neg_yx = np.zeros((0, 2), dtype=np.int64)

    # yx -> xy
    pos_xy = pos_yx[:, [1, 0]].astype(np.float32)
    neg_xy = neg_yx[:, [1, 0]].astype(np.float32)

    point_coords = np.concatenate([pos_xy, neg_xy], axis=0)
    point_labels = np.concatenate(
        [
            np.ones((pos_xy.shape[0],), dtype=np.int32),
            np.zeros((neg_xy.shape[0],), dtype=np.int32),
        ],
        axis=0,
    )

    return point_coords, point_labels


def _choose_best_auto_mask(candidates, prev_mask, sam2_cfg):
    best_score = -1e18
    best_mask = None

    for ann in candidates:
        seg = ann.get("segmentation", None)
        if seg is None:
            continue
        try:
            mask_bool = _to_mask_bool(seg)
        except Exception:
            continue

        area_ratio = _mask_area_ratio(mask_bool)
        if area_ratio < sam2_cfg.min_area_ratio or area_ratio > sam2_cfg.max_area_ratio:
            continue

        pred_iou = float(ann.get("predicted_iou", 0.0))
        stability = float(ann.get("stability_score", 0.0))
        temporal_iou = _mask_iou(mask_bool, prev_mask) if prev_mask is not None else 0.0

        score = pred_iou + 0.5 * stability + 1.5 * temporal_iou
        if score > best_score:
            best_score = score
            best_mask = mask_bool

    return best_mask


def _select_best_prompt_mask(masks, iou_predictions, prev_mask, sam2_cfg):
    if masks is None:
        return None

    masks_np = np.asarray(masks)
    if masks_np.ndim == 2:
        masks_np = masks_np[None, ...]

    ious_np = np.asarray(iou_predictions) if iou_predictions is not None else None

    best_score = -1e18
    best_mask = None
    for idx in range(masks_np.shape[0]):
        mask_bool = _to_mask_bool(masks_np[idx])
        area_ratio = _mask_area_ratio(mask_bool)
        if area_ratio < sam2_cfg.min_area_ratio or area_ratio > sam2_cfg.max_area_ratio:
            continue

        pred_iou = float(ious_np[idx]) if ious_np is not None and idx < ious_np.shape[0] else 0.0
        temporal_iou = _mask_iou(mask_bool, prev_mask) if prev_mask is not None else 0.0
        if prev_mask is not None and temporal_iou < sam2_cfg.track_iou_min:
            continue

        score = pred_iou + 1.5 * temporal_iou
        if score > best_score:
            best_score = score
            best_mask = mask_bool

    return best_mask


def _should_run_auto_stage(runtime, step):
    cfg = runtime["cfg"]
    if runtime.get("last_mask", None) is None:
        return True

    if runtime.get("track_fail_count", 0) >= cfg.max_track_failures:
        return True

    if cfg.auto_every_n_steps > 0 and step is not None:
        last_auto_step = runtime.get("last_auto_step", None)
        if last_auto_step is None:
            return True
        if step - last_auto_step >= cfg.auto_every_n_steps:
            return True

    return False


def infer_mask(color, depth, proprio, meta, agent = None):
    """
    使用 SAM2 在线分割当前 RGB 帧，返回像素级 mask。

    Returns
    -------
    mask : np.ndarray (H, W) uint8  —— 255 = 机械臂像素，0 = 背景
    None  —— SAM2 未初始化或当前帧分割失败
    """
    del depth, proprio, agent  # keep interface stable with previous implementation

    global _sam2_runtime
    if _sam2_runtime is None or not bool(_sam2_runtime.get("enabled", False)):
        return None

    cfg = _sam2_runtime["cfg"]
    _sam2_runtime["last_fail_reason"] = None

    color_np = np.asarray(color)
    if color_np.ndim != 3 or color_np.shape[2] != 3:
        _sam2_runtime["last_fail_reason"] = "sam2_invalid_color"
        return None
    if color_np.dtype != np.uint8:
        color_np = np.clip(color_np, 0, 255).astype(np.uint8)

    step = _extract_step_from_meta(meta)
    prev_mask = _sam2_runtime.get("last_mask", None)

    run_auto = _should_run_auto_stage(_sam2_runtime, step)

    # Stage B: prompt-based tracking on current frame
    if (not run_auto) and prev_mask is not None:
        try:
            point_coords, point_labels = _sample_prompt_points(
                prev_mask,
                cfg.prompt_num_pos_points,
                cfg.prompt_num_neg_points,
                cfg.prompt_dilate_radius,
            )
            box = _sam2_runtime.get("last_bbox", None)
            _sam2_runtime["img_predictor"].set_image(color_np)
            masks, ious, _ = _sam2_runtime["img_predictor"].predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
                multimask_output=cfg.multimask_output,
                return_logits=False,
                normalize_coords=True,
            )
            track_mask = _select_best_prompt_mask(masks, ious, prev_mask, cfg)
            if track_mask is not None:
                _sam2_runtime["last_mask"] = track_mask
                _sam2_runtime["last_bbox"] = _mask_to_bbox_xyxy(track_mask)
                _sam2_runtime["last_step"] = step
                _sam2_runtime["track_fail_count"] = 0
                return (track_mask.astype(np.uint8) * 255)

            _sam2_runtime["track_fail_count"] += 1
            _sam2_runtime["last_fail_reason"] = "sam2_track_fail"
        except Exception:
            _sam2_runtime["track_fail_count"] += 1
            _sam2_runtime["last_fail_reason"] = "sam2_track_exception"

    # Stage A: periodic re-detection
    try:
        candidates = _sam2_runtime["auto_mask_generator"].generate(color_np)
        auto_mask = _choose_best_auto_mask(candidates, prev_mask, cfg)
    except Exception:
        _sam2_runtime["last_fail_reason"] = "sam2_auto_exception"
        return None

    if auto_mask is None:
        _sam2_runtime["last_fail_reason"] = "sam2_auto_fail"
        return None

    _sam2_runtime["last_mask"] = auto_mask
    _sam2_runtime["last_bbox"] = _mask_to_bbox_xyxy(auto_mask)
    _sam2_runtime["last_step"] = step
    _sam2_runtime["last_auto_step"] = step
    _sam2_runtime["track_fail_count"] = 0
    return (auto_mask.astype(np.uint8) * 255)


def _to_numpy_mask(mask):
    # 将输入 mask 统一转换为 numpy
    if isinstance(mask, torch.Tensor):
        return mask.detach().cpu().numpy()
    return np.asarray(mask)


def _normalize_mask01(mask, depth_shape, mask_cfg):
    # 将输入 mask 统一到 [0, 1] 浮点二值图，1 表示不可信（需要屏蔽）
    mask_np = _to_numpy_mask(mask)

    if mask_np.ndim == 3:
        if mask_np.shape[0] == 1:
            mask_np = mask_np[0]
        elif mask_np.shape[-1] == 1:
            mask_np = mask_np[..., 0]
        else:
            raise ValueError(f"unsupported mask shape: {mask_np.shape}")
    elif mask_np.ndim != 2:
        raise ValueError(f"unsupported mask shape: {mask_np.shape}")

    target_h, target_w = int(depth_shape[0]), int(depth_shape[1])
    if mask_np.shape[0] != target_h or mask_np.shape[1] != target_w:
        print(
            "[mask-aware/3donly] mask size mismatch, resize with nearest: "
            f"mask={mask_np.shape}, depth=({target_h}, {target_w})"
        )
        mask_np = cv2.resize(
            mask_np.astype(np.float32),
            (target_w, target_h),
            interpolation = cv2.INTER_NEAREST,
        )

    threshold = float(mask_cfg.mask_threshold)
    if bool(mask_cfg.mask_white_is_untrusted):
        mask01 = (mask_np > threshold).astype(np.float32)
    else:
        mask01 = (mask_np <= threshold).astype(np.float32)

    # hard-coded dilation for quick testing
    dilate_radius = 10
    if dilate_radius > 0 and np.any(mask01 > 0):
        ks = 2 * int(dilate_radius) + 1
        kernel = np.ones((ks, ks), np.uint8)
        mask01 = cv2.dilate(
            (mask01 > 0).astype(np.uint8),
            kernel,
            iterations = 1,
        ).astype(np.float32)

    return mask01


def _safe_infer_mask(color, depth, proprio, meta, mask_cfg, agent = None):
    # 执行 infer_mask 并把异常统一转换为回退信号
    try:
        raw_mask = infer_mask(color, depth, proprio, meta, agent = agent)
    except Exception:
        return None, "infer_exception", None

    if raw_mask is None:
        return None, "infer_none", None

    try:
        mask01 = _normalize_mask01(raw_mask, depth.shape[:2], mask_cfg)
    except Exception:
        return None, "mask_invalid", raw_mask

    return mask01, None, raw_mask



def _save_mask_visualization(colors, mask01, step, config):
    if mask01 is None:
        return

    vis_save_dir = getattr(config.deploy, "vis_save_dir", ".")
    if vis_save_dir is None or len(str(vis_save_dir).strip()) == 0:
        vis_save_dir = "."
    os.makedirs(vis_save_dir, exist_ok = True)

    vis_save_prefix = getattr(config.deploy, "vis_save_prefix", "vis_debug")
    if vis_save_prefix is None or len(str(vis_save_prefix).strip()) == 0:
        vis_save_prefix = "vis_debug"
    vis_save_prefix = str(vis_save_prefix)

    mask_np = np.asarray(mask01)
    if mask_np.ndim == 3:
        mask_np = mask_np[..., 0]
    mask_u8 = ((mask_np > 0).astype(np.uint8) * 255)

    overlay = np.asarray(colors, dtype = np.uint8).copy()
    mask_bool = mask_u8 > 0
    if np.any(mask_bool):
        overlay_f32 = overlay.astype(np.float32)
        overlay_f32[mask_bool] = 0.6 * overlay_f32[mask_bool] + 0.4 * np.array([255.0, 0.0, 0.0], dtype = np.float32)
        overlay = np.clip(overlay_f32, 0, 255).astype(np.uint8)

    mask_path = os.path.join(vis_save_dir, "{}_step_{:06d}_mask.png".format(vis_save_prefix, step))
    overlay_path = os.path.join(vis_save_dir, "{}_step_{:06d}_mask_overlay.png".format(vis_save_prefix, step))
    cv2.imwrite(mask_path, mask_u8)
    cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print("[vis] saved mask: {}".format(mask_path))
    print("[vis] saved mask overlay: {}".format(overlay_path))


def _build_image_mask_weight(mask01, image_processor):
    # 根据二维 mask 构建图像 patch 级可信度权重
    try:
        mask_tensor = torch.from_numpy(mask01[np.newaxis].astype(np.float32))
        mask_tensor = resize_image(
            mask_tensor,
            image_processor.img_size,
            interpolation = T.InterpolationMode.NEAREST,
        )
        mask_ratio = image_processor.image_coord_pooling(mask_tensor)
        image_mask_weight = (1.0 - mask_ratio).clamp(0.0, 1.0).to(torch.float32)
    except Exception:
        return None

    return image_mask_weight


def load_test_obs(color_path, depth_path):
    # 1. 加载彩色图并转为 RGB (OpenCV 默认读入是 BGR)
    color_image = cv2.imread(color_path)
    if color_image is None:
        raise ValueError(f"无法加载图片: {color_path}")
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB).astype(np.uint8)

    # 2. 加载深度图
    # 注意：必须使用 cv2.IMREAD_UNCHANGED 才能保留 16bit 深度信息
    depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_image is None:
        raise ValueError(f"无法加载深度图: {depth_path}")
    
    # 确保是 uint16。如果你的 PNG 是 8bit 的，需要根据量化比例转回 uint16 (通常单位是毫米)
    depth_image = depth_image.astype(np.uint16)

    return color_image, depth_image


def create_point_cloud(colors, depths, intrinsics, config, depth_scale = 1000.0, rescale_factor = 1):
    """
    color, depth => point cloud
    """
    if rescale_factor != 1:
        H, W = depths.shape
        h, w = int(H * rescale_factor), int(W * rescale_factor)
        colors = colors.transpose([2, 0, 1]).astype(np.float32)
        colors = torch.from_numpy(colors)
        colors = np.ascontiguousarray(resize_image(colors, [h, w]).numpy().transpose([1, 2, 0]))
        depths = depths.astype(np.float32)
        depths = torch.from_numpy(depths[np.newaxis])
        depths = resize_image(depths, [h,w], interpolation = T.InterpolationMode.NEAREST)[0]
        depths = depths.numpy()

    # generate point cloud
    h, w = depths.shape
    fx, fy = intrinsics[0, 0] * rescale_factor, intrinsics[1, 1] * rescale_factor
    cx, cy = intrinsics[0, 2] * rescale_factor, intrinsics[1, 2] * rescale_factor
    colors = o3d.geometry.Image(colors.astype(np.uint8))
    depths = o3d.geometry.Image(depths.astype(np.float32))
    camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
        width = w, height = h, fx = fx, fy = fy, cx = cx, cy = cy
    )
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        colors, depths, depth_scale, convert_rgb_to_intensity = False
    )
    cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, camera_intrinsics)
    # crop point cloud
    bbox3d = o3d.geometry.AxisAlignedBoundingBox(config.deploy.workspace.min, config.deploy.workspace.max)
    cloud = cloud.crop(bbox3d)
    # downsample
    cloud = cloud.voxel_down_sample(config.data.voxel_size)
    return cloud


def create_input(colors, depths, cam_intrinsics, config, depth_scale = 1000.0, rescale_factor = 1):
    """
    colors, depths => coords, points
    """
    # create point cloud
    cloud = create_point_cloud(
        colors, 
        depths, 
        cam_intrinsics, 
        config,
        depth_scale = depth_scale,
        rescale_factor = rescale_factor,
    )

    # convert to sparse tensor
    points = np.asarray(cloud.points)
    coords = np.ascontiguousarray(points / config.data.voxel_size, dtype = np.int32)

    return coords, points, cloud
    

def create_batch(coords, points):
    """
    coords, points => batch coords, batch feats
    """
    import MinkowskiEngine as ME
    input_coords = [coords]
    input_feats = [points.astype(np.float32)]
    coords_batch, feats_batch = ME.utils.sparse_collate(input_coords, input_feats)
    return coords_batch, feats_batch


def process_state(state, config, to_control = True):
    if config.robot_type == "single":
        if to_control:
            state[..., 0: 3] = (state[..., 0: 3] + 1) / 2.0 * (config.data.normalization.trans_max - config.data.normalization.trans_min) + config.data.normalization.trans_min
            state[..., 9] = (state[..., 9] + 1) / 2.0 * config.data.normalization.max_gripper_width
        else:
            state[..., 0: 3] = (state[..., 0: 3] - config.data.normalization.trans_min) / (config.data.normalization.trans_max - config.data.normalization.trans_min) * 2.0 - 1
            state[..., 9] = state[..., 9] / config.data.normalization.max_gripper_width * 2.0 - 1
    else:
        if to_control:
            state[..., 0: 3] = (state[..., 0: 3] + 1) / 2.0 * (config.data.normalization.trans_max - config.data.normalization.trans_min) + config.data.normalization.trans_min
            state[..., 10: 13] = (state[..., 10: 13] + 1) / 2.0 * (config.data.normalization.trans_max - config.data.normalization.trans_min) + config.data.normalization.trans_min
            state[..., 9] = (state[..., 9] + 1) / 2.0 * config.data.normalization.max_gripper_width
            state[..., 19] = (state[..., 19] + 1) / 2.0 * config.data.normalization.max_gripper_width
        else:
            state[..., 0: 3] = (state[..., 0: 3] - config.data.normalization.trans_min) / (config.data.normalization.trans_max - config.data.normalization.trans_min) * 2.0 - 1
            state[..., 10: 13] = (state[..., 10: 13] - config.data.normalization.trans_min) / (config.data.normalization.trans_max - config.data.normalization.trans_min) * 2.0 - 1
            state[..., 9] = state[..., 9] / config.data.normalization.max_gripper_width * 2.0 - 1
            state[..., 19] = state[..., 19] / config.data.normalization.max_gripper_width * 2.0 - 1

    return state



def evaluate(args_override):
    # load default arguments
    args = deepcopy(default_args)
    for key, value in args_override.items():
        args[key] = value

    # load config
    with open(args.config, "r") as f:
        config = edict(yaml.load(f, Loader = yaml.FullLoader))
    config.data.normalization.trans_min = np.asarray(config.data.normalization.trans_min)
    config.data.normalization.trans_max = np.asarray(config.data.normalization.trans_max)
    config.mask_aware = _build_mask_aware_cfg(config)
    config.mask_aware.sam2 = _build_sam2_cfg(config.mask_aware)

    # set seed
    set_seed(config.deploy.seed)

    # load policy for local inference
    if args.type == "local":
        from policy import RISE2
        # set up device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # load policy
        print("Loading policy ...")
        policy = RISE2(
            num_action = config.data.num_action,
            obs_feature_dim = config.model.obs_feature_dim,
            cloud_enc_dim = config.model.cloud_enc_dim,
            image_enc_dim = config.model.image_enc_dim,
            action_dim = 10 if config.robot_type == "single" else 20,
            hidden_dim = config.model.hidden_dim,
            nheads = config.model.nheads,
            num_attn_layers = config.model.num_attn_layers,
            dim_feedforward = config.model.dim_feedforward,
            dropout = config.model.dropout,
            image_enc = config.model.image_enc,
            interp_fn_mode = config.model.interp_fn_mode,
            image_enc_finetune = config.model.image_enc_finetune,
            image_enc_dtype = config.model.image_enc_dtype
        ).to(device)

        # load checkpoint
        assert args.ckpt is not None, "Please provide the checkpoint to evaluate."
        policy.load_state_dict(torch.load(args.ckpt, map_location = device), strict = False)
        print("Checkpoint {} loaded.".format(args.ckpt))

        # set evaluation
        policy.eval()

    else:
        # connect to remote inference service
        print("Connecting to remote server ...")
        policy = WebsocketClientPolicy(host = args.host, port = args.port)

    # projector
    Projector = SingleArmProjector if config.robot_type == "single" else DualArmProjector
    projector = Projector(args.calib_rise2, config.deploy.agent.camera_serial)

    # image processor
    image_enc = config.model.image_enc
    if image_enc == "resnet18":
        img_size = config.data.aligner.img_size_resnet
        img_coord_size = config.data.aligner.img_coord_size_resnet
    elif image_enc.startswith("dinov2"):
        img_size = config.data.aligner.img_size_dinov2
        img_coord_size = config.data.aligner.img_coord_size_dinov2
    elif image_enc.startswith("dinov3"):
        img_size = config.data.aligner.img_size_dinov3
        img_coord_size = config.data.aligner.img_coord_size_dinov3
    else:
        raise ValueError(f"Unknown image encoder: {image_enc}")
    
    image_processor = ImageProcessor(
        img_size = img_size,
        img_coord_size = img_coord_size,
        voxel_size = config.data.voxel_size,
        img_mean = config.data.normalization.img_mean,
        img_std = config.data.normalization.img_std
    )

    # evaluation
    Agent = SingleArmAgent if config.robot_type == "single" else DualArmAgent
    agent = Agent(**config.deploy.agent)

    # ensemble buffer
    ensemble_buffer = EnsembleBuffer(mode = config.deploy.ensemble_mode)

    # 输出 mask-aware 配置摘要
    _log_mask_aware_summary(config.mask_aware)
    _log_sam2_summary(config.mask_aware.sam2)

    # 初始化 SAM2 runtime（仅本地 mask-aware 模式）
    global _sam2_runtime
    _sam2_runtime = None
    if config.mask_aware.enabled and args.type == "local" and config.mask_aware.sam2.enabled:
        try:
            _sam2_runtime = _init_sam2_runtime(config.mask_aware.sam2)
            print("[mask-aware/sam2] runtime initialized")
        except Exception as _e:
            print(f"[mask-aware/sam2] runtime init failed, mask disabled: {_e}")
            _sam2_runtime = None

    # 记录异常回退统计
    mask_stats = {
        "infer_none": 0,
        "infer_exception": 0,
        "mask_invalid": 0,
        "empty_cloud_skip": 0,
        "reweight_fallback": 0,
        "points_nonfinite": 0,
        "weight_nonfinite": 0,
        "sam2_auto_fail": 0,
        "sam2_auto_exception": 0,
        "sam2_track_fail": 0,
        "sam2_track_exception": 0,
        "sam2_invalid_color": 0,
    }

    # evaluation rollout
    print("Ready for rollout. Press Enter to continue...")
    input()
    
    with torch.inference_mode():
        for t in range(config.deploy.max_steps):
            if t % config.deploy.num_inference_steps == 0:
                # pre-process inputs
                colors, depths = agent.get_global_observation()

                # 本地推理启用 mask-aware 分支（SAM2 runtime 成功初始化后才进入）
                mask_enabled = bool(
                    config.mask_aware.enabled
                    and args.type == "local"
                    and (_sam2_runtime is not None)
                )
                mask01, mask_reason, raw_mask = None, None, None
                if mask_enabled:
                    mask01, mask_reason, raw_mask = _safe_infer_mask(
                        color = colors,
                        depth = depths,
                        proprio = None,
                        meta = {"step": t, "mode": args.type},
                        mask_cfg = config.mask_aware,
                        agent = agent,
                    )
                    if getattr(config.deploy, "vis", False) and mask01 is not None:
                        _save_mask_visualization(colors, mask01, t, config)
                    if mask01 is None:
                        reason = mask_reason or "unknown_infer_failure"
                        if reason in mask_stats:
                            mask_stats[reason] += 1

                        if _sam2_runtime is not None:
                            runtime_reason = _sam2_runtime.get("last_fail_reason", None)
                            if runtime_reason in mask_stats:
                                mask_stats[runtime_reason] += 1

                        if config.mask_aware.infer_none_policy == "fail_fast":
                            _log_mask_fallback(t, reason, "fail_fast")
                            raise RuntimeError(f"mask unavailable with fail_fast, reason={reason}")
                        _log_mask_fallback(t, reason, "no_mask_fallback")

                # 根据 mask 生成点云深度输入
                depths_for_cloud = depths
                if mask_enabled and config.mask_aware.enable_3d_filter and mask01 is not None:
                    depths_for_cloud = depths.copy()
                    depths_for_cloud[mask01 > 0.5] = 0

                # create cloud inputs
                create_input_kwargs = dict(
                    cam_intrinsics = agent.intrinsics,
                    config = config,
                    depth_scale = agent.camera.depth_scale,
                    rescale_factor = 1.0,
                )
                coords, points, cloud = create_input(
                    colors,
                    depths_for_cloud,
                    **create_input_kwargs,
                )

                # 点云出现非法数值时优先回退到原始深度重建
                if points.size > 0 and (not np.isfinite(points).all()):
                    if config.mask_aware.empty_cloud_policy == "fail_fast":
                        _log_mask_fallback(t, "points_nonfinite", "fail_fast")
                        raise RuntimeError("non-finite points after cloud build")
                    mask_stats["points_nonfinite"] += 1
                    _log_mask_fallback(t, "points_nonfinite", "rebuild_from_original_depth")
                    coords, points, cloud = create_input(
                        colors,
                        depths,
                        **create_input_kwargs,
                    )

                # 过滤后空点云回退
                if (
                    mask_enabled
                    and config.mask_aware.enable_3d_filter
                    and mask01 is not None
                    and points.shape[0] == 0
                ):
                    if config.mask_aware.empty_cloud_policy == "fail_fast":
                        _log_mask_fallback(t, "empty_cloud_after_3d_filter", "fail_fast")
                        raise RuntimeError("empty cloud after 3d filter")
                    mask_stats["empty_cloud_skip"] += 1
                    _log_mask_fallback(t, "empty_cloud_after_3d_filter", "skip_filter_rebuild")
                    coords, points, cloud = create_input(
                        colors,
                        depths,
                        **create_input_kwargs,
                    )

                # create image inputs
                image_coords = image_processor.get_image_coordinates(depths, agent.intrinsics, agent.camera.depth_scale)
                colors, image_coords = image_processor.preprocess_images(colors, image_coords)

                # 根据 mask 构建 patch 级可信度权重
                image_mask_weight = None
                if mask_enabled and config.mask_aware.enable_2d_reweight and mask01 is not None:
                    image_mask_weight = _build_image_mask_weight(mask01, image_processor)
                    if image_mask_weight is None:
                        mask_stats["reweight_fallback"] += 1
                        _log_mask_fallback(t, "reweight_build_failed", "disable_2d_reweight")
                    elif not torch.isfinite(image_mask_weight).all():
                        mask_stats["weight_nonfinite"] += 1
                        _log_mask_fallback(t, "reweight_nonfinite", "disable_2d_reweight")
                        image_mask_weight = None

                # predict action
                if args.type == "local":
                    import MinkowskiEngine as ME
                    coords_batch, feats_batch = create_batch(coords, points)
                    coords_batch, feats_batch = coords_batch.to(device), feats_batch.to(device)
                    cloud_data = ME.SparseTensor(feats_batch, coords_batch)

                    colors = colors.unsqueeze(0).to(device)
                    image_coords = image_coords.unsqueeze(0).to(device)
                    if image_mask_weight is not None:
                        image_mask_weight = image_mask_weight.unsqueeze(0).to(device)

                    # predict
                    pred_raw_action = policy(
                        cloud_data,
                        colors,
                        image_coords,
                        image_mask_weight = image_mask_weight,
                        actions = None,
                    ).squeeze(0).cpu().numpy()

                else:
                    obs_dict = {
                        "coords": coords,
                        "points": points,
                        "colors": colors.numpy(),
                        "image_coords": image_coords.numpy()
                    }

                    pred_raw_action = deepcopy(policy.infer(obs_dict)["actions"])

                # unnormalize predicted actions
                action = process_state(pred_raw_action, config, to_control = True)

                # visualization
                if config.deploy.vis:
                    tcp_vis_list = []
                    for raw_tcp in action:
                        tcp_vis = o3d.geometry.TriangleMesh.create_sphere(0.01).translate(raw_tcp[:3])
                        tcp_vis_list.append(tcp_vis)
                        if config.robot_type == "dual":
                            tcp_vis_r = o3d.geometry.TriangleMesh.create_sphere(0.01).translate(raw_tcp[10:13])
                            tcp_vis_list.append(tcp_vis_r)
                    o3d.visualization.draw_geometries([cloud, *tcp_vis_list])
                    input("press enter")
                
                # project action to base coordinate
                if config.robot_type == "single":
                    action_tcp = projector.project_tcp_to_base_coord(action[..., :9], rotation_rep = "rotation_6d")
                    action = np.concatenate([action_tcp, action[..., 9:10]], axis = -1)
                else:
                    action_left_tcp = projector.project_tcp_to_base_coord(action[..., :9], "left", rotation_rep = "rotation_6d")
                    action_right_tcp = projector.project_tcp_to_base_coord(action[..., 10:19], "right", rotation_rep = "rotation_6d")
                    action = np.concatenate([action_left_tcp, action[..., 9:10], action_right_tcp, action[..., 19:20]], axis = -1)
                
                # add to ensemble buffer
                ensemble_buffer.add_action(action, t)
            
            # get step action from ensemble buffer
            step_action = ensemble_buffer.get_action()
            # 这个是 config.deploy.num_inference_steps 这么多次循环完成后
            # 根据 ensemble_buffer 存的一串动作加权平均得到的
            
            if step_action is None:   # no action in the buffer => no movement.
                continue
            
            agent.action(step_action, rotation_rep = "rotation_6d")
            print(f"execute {step_action}")
            # input("enter")

    print(
        "[mask-aware] summary infer_none={} infer_exception={} mask_invalid={} "
        "empty_cloud_skip={} reweight_fallback={} points_nonfinite={} weight_nonfinite={} "
        "sam2_auto_fail={} sam2_auto_exception={} sam2_track_fail={} sam2_track_exception={} sam2_invalid_color={}".format(
            mask_stats["infer_none"],
            mask_stats["infer_exception"],
            mask_stats["mask_invalid"],
            mask_stats["empty_cloud_skip"],
            mask_stats["reweight_fallback"],
            mask_stats["points_nonfinite"],
            mask_stats["weight_nonfinite"],
            mask_stats["sam2_auto_fail"],
            mask_stats["sam2_auto_exception"],
            mask_stats["sam2_track_fail"],
            mask_stats["sam2_track_exception"],
            mask_stats["sam2_invalid_color"],
        )
    )

    agent.stop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', action = 'store', type = str, help = 'evaluation type, choices: ["local", "remote"].', required = True, choices = ["local", "remote"])
    parser.add_argument('--calib_airexo', action = 'store', type = str, help = 'airexo calibration path', required = True)
    parser.add_argument('--calib_rise2', action = 'store', type = str, help = 'rise2 calibration path', required = True)
    parser.add_argument('--config', action = 'store', type = str, help = 'data and model config during training and deployment', required = True)
    parser.add_argument('--ckpt', action = 'store', type = str, help = 'checkpoint path', required = False, default = None)
    parser.add_argument('--host', action = 'store', type = str, help = 'server host address', required = False, default = "127.0.0.1")
    parser.add_argument('--port', action = 'store', type = int, help = 'server port', required = False, default = 8000)

    evaluate(vars(parser.parse_args()))