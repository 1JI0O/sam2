# 微调 SAM2 方案（机械臂分割）

## Context

用户有50个机械臂操作场景（scene_0001 ~ scene_0050），每场景约300帧RGB图片，目标是微调 SAM2.1 Base Plus 模型，使其能准确分割 **arm（含gripper区域）** 和 **gripper** 两个独立对象。硬件环境：4x A100 40GB。

SAM2 官方仓库已内置完整训练框架（`training/`），支持 DAVIS/MOSE 风格数据集（`PNGRawDataset`），因此方案核心是：将原始数据转换为该格式，再配置 YAML 启动训练。

---

## 数据现状

```
RGB图片: /data/haoxiang/data/airexo2/task_0013/train/scene_XXXX/cam_105422061350/color/

Mask:    /data/haoxiang/data/airexo2_processed/task_0013/
         ├── scene_XXXX/              ← dilated arm（无gripper）
         ├── scene_XXXX_ckpt_arm/     ← arm mask（含gripper，无膨胀）← Object 1
         └── scene_XXXX_ckpt_gripper/ ← gripper mask（无膨胀）       ← Object 2
```

选用 `_ckpt_arm` 和 `_ckpt_gripper`（未膨胀，边界干净），合并为 palettised PNG：`0=背景，1=arm，2=gripper`。

---

## 目标数据格式（DAVIS-style VOS）

`PNGRawDataset` 要求：
- `JPEGImages/{scene_name}/*.jpg`（文件名为纯数字，如 `00000.jpg`）
- `Annotations/{scene_name}/*.png`（palettised PNG，像素值 = 物体ID）

```
/data/haoxiang/data/airexo2_processed/vos_finetune/
├── JPEGImages/
│   ├── scene_0001/
│   │   ├── 00000.jpg
│   │   ├── 00001.jpg
│   │   └── ...
│   └── scene_0002/
│       └── ...
├── Annotations/
│   ├── scene_0001/
│   │   ├── 00000.png   ← 像素值: 0=bg, 1=arm, 2=gripper
│   │   ├── 00001.png
│   │   └── ...
│   └── scene_0002/
│       └── ...
├── train_list.txt      ← scene_0001 ~ scene_0040
└── val_list.txt        ← scene_0041 ~ scene_0050
```

---

## 步骤一：数据转换脚本

**新建文件**：`scripts/convert_to_vos_format.py`

```python
import os
import glob
import shutil
import numpy as np
from PIL import Image

RAW_IMG_ROOT = "/data/haoxiang/data/airexo2/task_0013/train"
MASK_ROOT    = "/data/haoxiang/data/airexo2_processed/task_0013"
OUT_ROOT     = "/data/haoxiang/data/airexo2_processed/vos_finetune"
CAM_DIR      = "cam_105422061350/color"

# DAVIS 调色板
PALETTE = [0] * 768
PALETTE[0:3] = [0,   0,   0]    # 0: 背景
PALETTE[3:6] = [128, 0,   0]    # 1: arm  (红)
PALETTE[6:9] = [0,   128, 0]    # 2: gripper (绿)

for i in range(1, 51):
    scene = f"scene_{i:04d}"
    print(f"Processing {scene} ...")

    # ---------- 1. 图片 ----------
    img_src = os.path.join(RAW_IMG_ROOT, scene, CAM_DIR)
    img_dst = os.path.join(OUT_ROOT, "JPEGImages", scene)
    os.makedirs(img_dst, exist_ok=True)

    img_files = sorted(
        glob.glob(f"{img_src}/*.jpg") + glob.glob(f"{img_src}/*.png")
    )
    for idx, src in enumerate(img_files):
        dst = os.path.join(img_dst, f"{idx:05d}.jpg")
        if src.endswith(".png"):
            Image.open(src).convert("RGB").save(dst, "JPEG", quality=95)
        else:
            shutil.copy(src, dst)

    # ---------- 2. Mask ----------
    arm_dir     = os.path.join(MASK_ROOT, f"{scene}_ckpt_arm")
    gripper_dir = os.path.join(MASK_ROOT, f"{scene}_ckpt_gripper")
    ann_dst = os.path.join(OUT_ROOT, "Annotations", scene)
    os.makedirs(ann_dst, exist_ok=True)

    arm_files     = sorted(glob.glob(f"{arm_dir}/*.png"))
    gripper_files = sorted(glob.glob(f"{gripper_dir}/*.png"))

    # 以较少的帧数为准（防止不对齐）
    n_frames = min(len(img_files), len(arm_files), len(gripper_files))
    if n_frames < len(img_files):
        print(f"  WARNING: {scene} img={len(img_files)}, arm={len(arm_files)}, "
              f"gripper={len(gripper_files)}, using first {n_frames} frames")

    for idx in range(n_frames):
        arm_mask     = np.array(Image.open(arm_files[idx]).convert("L")) > 0
        gripper_mask = np.array(Image.open(gripper_files[idx]).convert("L")) > 0

        palette_arr = np.zeros(arm_mask.shape, dtype=np.uint8)
        palette_arr[arm_mask]     = 1  # arm
        palette_arr[gripper_mask] = 2  # gripper（覆盖重叠区）

        out_img = Image.fromarray(palette_arr, mode="P")
        out_img.putpalette(PALETTE)
        out_img.save(os.path.join(ann_dst, f"{idx:05d}.png"))

# ---------- 3. 生成 train/val 列表 ----------
with open(os.path.join(OUT_ROOT, "train_list.txt"), "w") as f:
    f.write("\n".join([f"scene_{i:04d}" for i in range(1, 41)]))

with open(os.path.join(OUT_ROOT, "val_list.txt"), "w") as f:
    f.write("\n".join([f"scene_{i:04d}" for i in range(41, 51)]))

print("Done!")
```

**验证转换结果**：
```bash
python -c "
from PIL import Image
img = Image.open('/data/.../vos_finetune/Annotations/scene_0001/00000.png')
print('unique pixel values:', set(img.getdata()))  # 应包含 0,1,2
"
```

---

## 步骤二：新建训练配置

**新建文件**：`sam2/configs/sam2.1_training/sam2.1_hiera_b+_robot_finetune.yaml`

基于 `sam2/configs/sam2.1_training/sam2.1_hiera_b+_MOSE_finetune.yaml`，主要改动：

| 参数 | MOSE原值 | 机械臂新值 | 原因 |
|------|---------|-----------|------|
| `img_folder` | null | `/data/.../vos_finetune/JPEGImages` | 指向新数据 |
| `gt_folder` | null | `/data/.../vos_finetune/Annotations` | 指向新数据 |
| `file_list_txt` | MOSE_sample_train_list.txt | `/data/.../vos_finetune/train_list.txt` | 40个训练场景 |
| `max_num_objects` | 3 | **2** | arm + gripper |
| `num_frames` | 8 | **6** | 40GB显存，适当减少 |
| `train_batch_size` | 1 | **1** | 每GPU 1个视频片段 |
| `num_epochs` | 40 | **30** | 数据集小 |
| `multiplier` | 2 | **8** | 50场景少，多重复采样 |
| `base_lr` | 5e-6 | **3e-6** | 小数据集稍低 LR |
| `vision_lr` | 3e-6 | **2e-6** | 同上 |
| `gpus_per_node` | 8 | **4** | 4卡 A100 |
| `use_act_ckpt_iterative_pt_sampling` | false | **true** | 节省40GB显存 |
| `checkpoint_path` | sam2.1_hiera_base_plus.pt | sam2.1_hiera_base_plus.pt | 保持不变 |

完整 YAML（在 MOSE 基础上修改的关键部分）：

```yaml
# @package _global_

scratch:
  resolution: 1024
  train_batch_size: 1
  num_train_workers: 8
  num_frames: 6               # 改: 8→6，节省40GB显存
  max_num_objects: 2          # 改: 3→2，arm+gripper
  base_lr: 3.0e-6             # 改: 5e-6→3e-6
  vision_lr: 2.0e-6           # 改: 3e-6→2e-6
  phases_per_epoch: 1
  num_epochs: 30              # 改: 40→30

dataset:
  img_folder: /data/haoxiang/data/airexo2_processed/vos_finetune/JPEGImages
  gt_folder:  /data/haoxiang/data/airexo2_processed/vos_finetune/Annotations
  file_list_txt: /data/haoxiang/data/airexo2_processed/vos_finetune/train_list.txt
  multiplier: 8               # 改: 2→8，扩大小数据集采样

# 其余 vos transforms、model architecture、loss、optim 配置与 MOSE 相同
# 仅在 model 段内增加：
#   use_act_ckpt_iterative_pt_sampling: true

launcher:
  num_nodes: 1
  gpus_per_node: 4            # 改: 8→4
  experiment_log_dir: null
```

---

## 步骤三：启动训练

```bash
cd /path/to/sam2_repo

# 首次需要安装
pip install -e ".[dev]"

# 启动微调（4卡）
python training/train.py \
    -c sam2/configs/sam2.1_training/sam2.1_hiera_b+_robot_finetune.yaml \
    --use-cluster 0 \
    --num-gpus 4
```

监控训练：
```bash
tensorboard --logdir sam2_logs/sam2.1_hiera_b+_robot_finetune/tensorboard
```

Checkpoint 保存在：`sam2_logs/sam2.1_hiera_b+_robot_finetune/checkpoints/`

---

## 步骤四：推理与评估

**推理**（给定第一帧的 prompt，追踪后续帧）：

```python
from sam2.build_sam import build_sam2_video_predictor
import torch

predictor = build_sam2_video_predictor(
    config_file="sam2/configs/sam2.1/sam2.1_hiera_b+.yaml",
    ckpt_path="sam2_logs/sam2.1_hiera_b+_robot_finetune/checkpoints/checkpoint.pt"
)

with torch.inference_mode():
    state = predictor.init_state(video_path="path/to/scene_0041/JPEGImages")
    # 给第一帧添加点 prompt
    predictor.add_new_points_or_box(
        inference_state=state, frame_idx=0, obj_id=1,
        points=[[x_arm, y_arm]], labels=[1]   # arm
    )
    predictor.add_new_points_or_box(
        inference_state=state, frame_idx=0, obj_id=2,
        points=[[x_gripper, y_gripper]], labels=[1]  # gripper
    )
    for frame_idx, obj_ids, masks in predictor.propagate_in_video(state):
        # masks shape: [n_objects, H, W]
        pass
```

**批量评估**：使用 `tools/vos_inference.py` + `sav_dataset/sav_evaluator.py` 计算 J&F。

---

## 关键文件路径

| 用途 | 路径 |
|------|------|
| 训练入口 | `training/train.py` |
| Trainer 核心 | `training/trainer.py` |
| 数据集加载 | `training/dataset/vos_raw_dataset.py`（PNGRawDataset） |
| Mask加载 | `training/dataset/vos_segment_loader.py`（PalettisedPNGSegmentLoader） |
| 参考配置（MOSE） | `sam2/configs/sam2.1_training/sam2.1_hiera_b+_MOSE_finetune.yaml` |
| 预训练权重 | `checkpoints/sam2.1_hiera_base_plus.pt` |
| **新建：转换脚本** | `scripts/convert_to_vos_format.py` |
| **新建：训练配置** | `sam2/configs/sam2.1_training/sam2.1_hiera_b+_robot_finetune.yaml` |

---

## 风险与应对

| 风险 | 解决方案 |
|------|---------|
| 40GB OOM | 进一步降 `num_frames: 4`；或开启 `compile_image_encoder: False`（默认已关） |
| arm/gripper mask 帧数不对齐 | 转换脚本中取三者最小帧数，并打印 WARNING |
| 数据量少（50场景）导致过拟合 | 增大 `multiplier`，关注 val loss，早停 |
| 推理时需要初始 prompt | SAM2 是 prompted model，推理时给第一帧点/框即可，之后自动传播 |
