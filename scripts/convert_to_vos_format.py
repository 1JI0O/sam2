"""
Convert raw robotic RGB frames into SAM2-friendly video frame folders.

This script creates symbolic links only (zero disk copy):
  {RAW_IMG_ROOT}/{scene}/{CAM_DIR}/*.png
      -> {OUT_ROOT}/JPEGImages/{scene}/{frame_idx:05d}.jpg

Note:
- Source files remain PNG; symlink names use .jpg so downstream loaders
  that glob "*.jpg" can find them.
- PIL.Image.open() infers true format from file header bytes, not extension.
- Mask conversion is intentionally disabled in this RGB-only version.
"""

import os
import glob

RAW_IMG_ROOT = "/data/haoxiang/data/airexo2/task_0012/train"
# OUT_ROOT     = "/data/haoxiang/data/airexo2_processed/sam2_finetune"
OUT_ROOT     = "/data/haoxiang/data/airexo2/task_0012/sam2_symlinks"
CAM_DIR      = "cam_105422061350/color"

for i in range(1, 53): # 这里范围要手动改
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

    # # ---------- 2. Mask symlinks (per-object subfolder) ----------
    # arm_dir     = os.path.join(MASK_ROOT, f"{scene}_ckpt_arm")
    # gripper_dir = os.path.join(MASK_ROOT, f"{scene}_ckpt_gripper")

    # # folder "0" -> obj_id=1 (arm), folder "1" -> obj_id=2 (gripper)
    # arm_dst     = os.path.join(OUT_ROOT, "Annotations", scene, "0")
    # gripper_dst = os.path.join(OUT_ROOT, "Annotations", scene, "1")
    # os.makedirs(arm_dst, exist_ok=True)
    # os.makedirs(gripper_dst, exist_ok=True)

    # arm_files     = sorted(glob.glob(os.path.join(arm_dir, "*.png")))
    # gripper_files = sorted(glob.glob(os.path.join(gripper_dir, "*.png")))

    # n_frames = min(len(img_files), len(arm_files), len(gripper_files))
    # if n_frames < len(img_files):
    #     print(f"  WARNING: {scene} img={len(img_files)}, arm={len(arm_files)}, "
    #           f"gripper={len(gripper_files)}, using first {n_frames} frames")

    # for idx in range(n_frames):
    #     arm_lnk = os.path.join(arm_dst, f"{idx:05d}.png")
    #     grp_lnk = os.path.join(gripper_dst, f"{idx:05d}.png")
    #     if not os.path.lexists(arm_lnk):
    #         os.symlink(os.path.abspath(arm_files[idx]), arm_lnk)
    #     if not os.path.lexists(grp_lnk):
    #         os.symlink(os.path.abspath(gripper_files[idx]), grp_lnk)

# ---------- 3. Train / val split lists ----------
# with open(os.path.join(OUT_ROOT, "train_list.txt"), "w") as f:
#     f.write("\n".join([f"scene_{i:04d}" for i in range(1, 41)]))
# with open(os.path.join(OUT_ROOT, "val_list.txt"), "w") as f:
#     f.write("\n".join([f"scene_{i:04d}" for i in range(41, 51)]))

print("Done!")
print("\nVerification command:")
print('python -c "')
print("import glob; from PIL import Image")
print(f"img = Image.open('{OUT_ROOT}/JPEGImages/scene_0001/00000.jpg')")
print("print('img size:', img.size, 'mode:', img.mode)")
print(
    f"print('scene_0001 frame count:', "
    f"len(glob.glob('{OUT_ROOT}/JPEGImages/scene_0001/*.jpg')))"
)
print('"')
