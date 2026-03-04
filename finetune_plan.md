# 微调 SAM2 方案（机械臂分割）

## Context

用户有50个机械臂操作场景（scene_0001 ~ scene_0050），每场景约300帧RGB图片，目标是微调 SAM2.1 Base Plus 模型，使其能准确分割 **arm（含gripper区域）** 和 **gripper** 两个独立对象。硬件环境：4x A100 40GB。

SAM2 官方仓库已内置完整训练框架（`training/`），支持 DAVIS/MOSE 风格数据集（`PNGRawDataset`），因此方案核心是：将原始数据转换为该格式，再配置 YAML 启动训练。

---

## 策略对比：单模型 vs 两模型

### 结论：推荐单模型

一次微调，采用 MultiplePNG（`is_palette: false`）格式同时监督 arm 与 gripper，训练出一个能同时追踪两种对象的模型。

| 维度 | 单模型（推荐） | 两模型（各训一次） |
|------|--------------|-----------------|
| 推理效率 | 一次前向传播同时输出 arm + gripper mask | 需运行两次，延迟翻倍 |
| 训练成本 | 训一次，一个 checkpoint | 训两次，时间翻倍 |
| 空间关系学习 | ✅ 模型同时看到两个对象，学到 gripper 始终在 arm 末端的约束，有助于遮挡帧的传播 | ❌ 各自独立，无法利用 arm-gripper 的空间先验 |
| 数据利用率 | ✅ 同一帧同时监督两个对象 | 每次训练只用一半标注 |
| SAM2 设计契合度 | ✅ SAM2 的 multi-object tracking 就是为此设计的 | 退化为单对象，未充分利用模型能力 |
| 边界一致性 | ✅ arm 和 gripper 的边界由同一模型决定，不会重叠或矛盾 | ❌ 两次独立推理的 mask 可能在 arm-gripper 交界处产生重叠/间隙 |

**为什么单模型更好**：arm 和 gripper 属于同一刚体系统，空间关系高度固定（gripper 永远在 arm 末端）。联合训练让模型学到这种约束，对视频帧间传播和遮挡处理都有帮助。两模型的理论优势（单任务专注）仅在两对象互相干扰训练时成立，这里完全不适用。

### 标签细节说明（重要修正）

原始标注关系：
- `_ckpt_arm` = **完整 arm**（包含 gripper 区域）
- `_ckpt_gripper` = gripper，是 arm 的子集，两者在 gripper 区域**天然重叠**

**palettised PNG 方案（当前，有问题）**：每个像素只能有一个标签，gripper 覆盖重叠区后，arm 标签变成 "arm-minus-gripper"，语义失真。

**修正方案：使用 `MultiplePNGSegmentLoader`（`is_palette: False`）**，每个对象独立存二值 PNG：
- arm（obj 0 文件夹） = 直接用 `_ckpt_arm`，完整 arm 包含 gripper 区域
- gripper（obj 1 文件夹） = 直接用 `_ckpt_gripper`，仅 gripper

两者在 gripper 区域重叠，但 SAM2 对每个对象独立监督 loss，重叠完全合法，模型学到的就是：arm = 完整臂，gripper = 末端夹爪（是 arm 的子集）。

---

## 数据现状

```
RGB图片: /data/haoxiang/data/airexo2/task_0013/train/scene_XXXX/cam_105422061350/color/

Mask:    /data/haoxiang/data/airexo2_processed/task_0013/
         ├── scene_XXXX/              ← dilated arm（无gripper）
         ├── scene_XXXX_ckpt_arm/     ← arm mask（含gripper，无膨胀）← Object 1
         └── scene_XXXX_ckpt_gripper/ ← gripper mask（无膨胀）       ← Object 2
```

选用 `_ckpt_arm` 和 `_ckpt_gripper`（未膨胀，边界干净）。
Mask 为黑白图（**黑色=背景 0，白色=mask区域 255**），保持为两个独立二值 PNG（arm 与 gripper），不做 palettised 合并。

---

## 目标数据格式（MultiplePNG per object）

`PNGRawDataset`（`is_palette=False`）+ `MultiplePNGSegmentLoader` 要求：
- `JPEGImages/{scene_name}/*.jpg`（文件名为纯数字）
- `Annotations/{scene_name}/{obj_folder}/{frame_id:05d}.png`（二值 PNG，白=mask，黑=背景）
  - `obj_folder` 为纯数字，loader 读取时 `obj_id = int(folder_name) + 1`

```
/data/haoxiang/data/airexo2_processed/vos_finetune/
├── JPEGImages/
│   ├── scene_0001/
│   │   ├── 00000.jpg
│   │   └── ...
│   └── scene_0002/...
├── Annotations/
│   ├── scene_0001/
│   │   ├── 0/              ← obj_id=1（arm，完整 _ckpt_arm，含 gripper 区域）
│   │   │   ├── 00000.png
│   │   │   └── ...
│   │   └── 1/              ← obj_id=2（gripper，_ckpt_gripper）
│   │       ├── 00000.png
│   │       └── ...
│   └── scene_0002/...
├── train_list.txt      ← scene_0001 ~ scene_0040
└── val_list.txt        ← scene_0041 ~ scene_0050
```

arm 和 gripper 在 gripper 区域**重叠**，但各自独立监督，语义正确：arm = 完整臂，gripper = 末端夹爪。

---

## 步骤一：数据转换脚本

**新建文件**：`scripts/convert_to_vos_format.py`

原始图片（PNG）和 mask（二值 PNG）均无需转换，全程用**软链接**零拷贝复用原始数据。

图片源文件为 `.png`，但软链接命名为 `00000.jpg`——`PNGRawDataset` 靠 glob `*.jpg` 发现文件，PIL `Image.open()` 靠文件头魔数判断格式，不看扩展名，因此 `.png` 内容用 `.jpg` 名字软链接完全可行。

```python
import os
import glob

RAW_IMG_ROOT = "/data/haoxiang/data/airexo2/task_0013/train"
MASK_ROOT    = "/data/haoxiang/data/airexo2_processed/task_0013"
OUT_ROOT     = "/data/haoxiang/data/airexo2_processed/vos_finetune"
CAM_DIR      = "cam_105422061350/color"

for i in range(1, 51):
    scene = f"scene_{i:04d}"
    print(f"Processing {scene} ...")

    # ---------- 1. 图片软链接（.png 源 → .jpg 名） ----------
    img_src = os.path.join(RAW_IMG_ROOT, scene, CAM_DIR)
    img_dst = os.path.join(OUT_ROOT, "JPEGImages", scene)
    os.makedirs(img_dst, exist_ok=True)

    img_files = sorted(glob.glob(f"{img_src}/*.png"))
    for idx, src in enumerate(img_files):
        lnk = os.path.join(img_dst, f"{idx:05d}.jpg")  # glob *.jpg 能找到
        if not os.path.lexists(lnk):
            os.symlink(os.path.abspath(src), lnk)

    # ---------- 2. Mask 软链接（每个对象独立子文件夹）----------
    arm_dir     = os.path.join(MASK_ROOT, f"{scene}_ckpt_arm")
    gripper_dir = os.path.join(MASK_ROOT, f"{scene}_ckpt_gripper")

    # folder "0" → obj_id=1 (arm), folder "1" → obj_id=2 (gripper)
    arm_dst     = os.path.join(OUT_ROOT, "Annotations", scene, "0")
    gripper_dst = os.path.join(OUT_ROOT, "Annotations", scene, "1")
    os.makedirs(arm_dst, exist_ok=True)
    os.makedirs(gripper_dst, exist_ok=True)

    arm_files     = sorted(glob.glob(f"{arm_dir}/*.png"))
    gripper_files = sorted(glob.glob(f"{gripper_dir}/*.png"))

    n_frames = min(len(img_files), len(arm_files), len(gripper_files))
    if n_frames < len(img_files):
        print(f"  WARNING: {scene} img={len(img_files)}, arm={len(arm_files)}, "
              f"gripper={len(gripper_files)}, using first {n_frames} frames")

    for idx in range(n_frames):
        arm_lnk = os.path.join(arm_dst, f"{idx:05d}.png")
        grp_lnk = os.path.join(gripper_dst, f"{idx:05d}.png")
        if not os.path.lexists(arm_lnk):
            os.symlink(os.path.abspath(arm_files[idx]), arm_lnk)
        if not os.path.lexists(grp_lnk):
            os.symlink(os.path.abspath(gripper_files[idx]), grp_lnk)

# ---------- 3. 生成 train/val 列表 ----------
with open(os.path.join(OUT_ROOT, "train_list.txt"), "w") as f:
    f.write("\n".join([f"scene_{i:04d}" for i in range(1, 41)]))
with open(os.path.join(OUT_ROOT, "val_list.txt"), "w") as f:
    f.write("\n".join([f"scene_{i:04d}" for i in range(41, 51)]))

print("Done!")
```

**验证软链接和 mask 格式**：

```bash
python -c "
import numpy as np
from PIL import Image
# 验证图片可读（PIL 能正确识别 PNG 内容）
img = Image.open('/data/.../JPEGImages/scene_0001/00000.jpg')
print('img size:', img.size, 'mode:', img.mode)
# 验证 mask 像素值和重叠区
arm = np.array(Image.open('/data/.../Annotations/scene_0001/0/00000.png'))
grp = np.array(Image.open('/data/.../Annotations/scene_0001/1/00000.png'))
print('arm unique:', set(arm.flatten()))      # 应为 {0, 255}
print('gripper unique:', set(grp.flatten()))  # 应为 {0, 255}
print('overlap pixels:', ((arm > 0) & (grp > 0)).sum())  # 应 > 0
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
| `is_palette` | true（默认） | **false** | 使用 MultiplePNGSegmentLoader，支持重叠 mask |
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
# 在 model 段内增加：
#   use_act_ckpt_iterative_pt_sampling: true
# 在 video_dataset 段内增加：
#   is_palette: false

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
| Mask加载 | `training/dataset/vos_segment_loader.py`（MultiplePNGSegmentLoader） |
| 参考配置（MOSE） | `sam2/configs/sam2.1_training/sam2.1_hiera_b+_MOSE_finetune.yaml` |
| 预训练权重 | `checkpoints/sam2.1_hiera_base_plus.pt` |
| **新建：转换脚本** | `scripts/convert_to_vos_format.py` |
| **新建：训练配置** | `sam2/configs/sam2.1_training/sam2.1_hiera_b+_robot_finetune.yaml` |

---

## 风险与应对

| 风险 | 解决方案 |
|------|---------|
| 40GB OOM | 进一步降 `num_frames: 4`；或开启 `compile_image_encoder: False`（默认已关） |
| arm/gripper mask 帧数不一致（低概率） | 当前数据理论上一一对齐；转换脚本保留 `min(...)` 与 WARNING 作为保险 |
| 数据量少（50场景）导致过拟合 | 增大 `multiplier`，关注 val loss，早停 |
| 推理时需要初始 prompt | SAM2 是 prompted model，推理时给第一帧点/框即可，之后自动传播 |
