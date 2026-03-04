"""
Convert robotic arm dataset to VOS format for SAM2 fine-tuning.

Uses symbolic links (zero disk copy). Source images are PNG but symlinked
with .jpg extensions so PNGRawDataset's glob("*.jpg") finds them.
PIL.Image.open() detects format from file header magic bytes, not extension.

Output structure:
  vos_finetune/
  ├── JPEGImages/{scene}/00000.jpg -> (symlink to .png source)
  ├── Annotations/{scene}/0/00000.png -> (symlink to _ckpt_arm mask)   # obj_id=1
  │                       1/00000.png -> (symlink to _ckpt_gripper mask) # obj_id=2
  ├── train_list.txt
  └── val_list.txt

MultiplePNGSegmentLoader: obj_id = int(folder_name) + 1
  folder "0" -> obj_id=1 (full arm, including gripper region)
  folder "1" -> obj_id=2 (gripper only)
Arm and gripper masks overlap in the gripper region; SAM2 supervises each
object independently so this is valid and semantically correct.
"""

import os
import glob

RAW_IMG_ROOT = "/data/haoxiang/data/airexo2/task_0013/train"
MASK_ROOT    = "/data/haoxiang/data/airexo2_processed/task_0013"
OUT_ROOT     = "/data/haoxiang/data/airexo2_processed/vos_finetune"
CAM_DIR      = "cam_105422061350/color"

for i in range(1, 51):
    scene = f"scene_{i:04d}"
    print(f"Processing {scene} ...")

    # ---------- 1. Image symlinks (.png source -> .jpg named symlink) ----------
    img_src = os.path.join(RAW_IMG_ROOT, scene, CAM_DIR)
    img_dst = os.path.join(OUT_ROOT, "JPEGImages", scene)
    os.makedirs(img_dst, exist_ok=True)

    img_files = sorted(glob.glob(os.path.join(img_src, "*.png")))
    if not img_files:
        print(f"  WARNING: no PNG images found in {img_src}, skipping")
        continue
    for idx, src in enumerate(img_files):
        lnk = os.path.join(img_dst, f"{idx:05d}.jpg")
        if not os.path.lexists(lnk):
            os.symlink(os.path.abspath(src), lnk)

    # ---------- 2. Mask symlinks (per-object subfolder) ----------
    arm_dir     = os.path.join(MASK_ROOT, f"{scene}_ckpt_arm")
    gripper_dir = os.path.join(MASK_ROOT, f"{scene}_ckpt_gripper")

    # folder "0" -> obj_id=1 (arm), folder "1" -> obj_id=2 (gripper)
    arm_dst     = os.path.join(OUT_ROOT, "Annotations", scene, "0")
    gripper_dst = os.path.join(OUT_ROOT, "Annotations", scene, "1")
    os.makedirs(arm_dst, exist_ok=True)
    os.makedirs(gripper_dst, exist_ok=True)

    arm_files     = sorted(glob.glob(os.path.join(arm_dir, "*.png")))
    gripper_files = sorted(glob.glob(os.path.join(gripper_dir, "*.png")))

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

# ---------- 3. Train / val split lists ----------
with open(os.path.join(OUT_ROOT, "train_list.txt"), "w") as f:
    f.write("\n".join([f"scene_{i:04d}" for i in range(1, 41)]))
with open(os.path.join(OUT_ROOT, "val_list.txt"), "w") as f:
    f.write("\n".join([f"scene_{i:04d}" for i in range(41, 51)]))

print("Done!")
print(f"\nVerification command:")
print(f"python -c \"")
print(f"import numpy as np; from PIL import Image")
print(f"img = Image.open('{OUT_ROOT}/JPEGImages/scene_0001/00000.jpg')")
print(f"print('img size:', img.size, 'mode:', img.mode)")
print(f"arm = np.array(Image.open('{OUT_ROOT}/Annotations/scene_0001/0/00000.png'))")
print(f"grp = np.array(Image.open('{OUT_ROOT}/Annotations/scene_0001/1/00000.png'))")
print(f"print('arm unique:', set(arm.flatten()))      # expect {{0, 255}}")
print(f"print('gripper unique:', set(grp.flatten()))  # expect {{0, 255}}")
print(f"print('overlap pixels:', ((arm > 0) & (grp > 0)).sum())  # expect > 0")
print(f"\"")
