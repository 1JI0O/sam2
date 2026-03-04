我的原始图片是很多机械臂工作的图片，机械臂整体是个arm，其末端是gripper

原始图片和mask图片相同scene保证数量一致

原始图片有50个scene
比如scene_0001都在 /data/haoxiang/data/airexo2/task_0013/train/scene_0001/cam_105422061350/color
scene_0002都在 /data/haoxiang/data/airexo2/task_0013/train/scene_0002/cam_105422061350/color
以此类推

每个scene下面有大概300张rgb图片

mask都在 /data/haoxiang/data/airexo2_processed/tasks_0013
(base) haoxiang@aiadm-desktop:/data/haoxiang/data/airexo2_processed/task_0013$ ls
scene_0001               scene_0006_ckpt_gripper  scene_0012_ckpt_arm      scene_0018               scene_0023_ckpt_gripper  scene_0029_ckpt_arm      scene_0035               scene_0040_ckpt_gripper  scene_0046_ckpt_arm
scene_0001_ckpt_arm      scene_0007               scene_0012_ckpt_gripper  scene_0018_ckpt_arm      scene_0024               scene_0029_ckpt_gripper  scene_0035_ckpt_arm      scene_0041               scene_0046_ckpt_gripper
scene_0001_ckpt_gripper  scene_0007_ckpt_arm      scene_0013               scene_0018_ckpt_gripper  scene_0024_ckpt_arm      scene_0030               scene_0035_ckpt_gripper  scene_0041_ckpt_arm      scene_0047
scene_0002               scene_0007_ckpt_gripper  scene_0013_ckpt_arm      scene_0019               scene_0024_ckpt_gripper  scene_0030_ckpt_arm      scene_0036               scene_0041_ckpt_gripper  scene_0047_ckpt_arm
scene_0002_ckpt_arm      scene_0008               scene_0013_ckpt_gripper  scene_0019_ckpt_arm      scene_0025               scene_0030_ckpt_gripper  scene_0036_ckpt_arm      scene_0042               scene_0047_ckpt_gripper
scene_0002_ckpt_gripper  scene_0008_ckpt_arm      scene_0014               scene_0019_ckpt_gripper  scene_0025_ckpt_arm      scene_0031               scene_0036_ckpt_gripper  scene_0042_ckpt_arm      scene_0048
scene_0003               scene_0008_ckpt_gripper  scene_0014_ckpt_arm      scene_0020               scene_0025_ckpt_gripper  scene_0031_ckpt_arm      scene_0037               scene_0042_ckpt_gripper  scene_0048_ckpt_arm
scene_0003_ckpt_arm      scene_0009               scene_0014_ckpt_gripper  scene_0020_ckpt_arm      scene_0026               scene_0031_ckpt_gripper  scene_0037_ckpt_arm      scene_0043               scene_0048_ckpt_gripper
scene_0003_ckpt_gripper  scene_0009_ckpt_arm      scene_0015               scene_0020_ckpt_gripper  scene_0026_ckpt_arm      scene_0032               scene_0037_ckpt_gripper  scene_0043_ckpt_arm      scene_0049
scene_0004               scene_0009_ckpt_gripper  scene_0015_ckpt_arm      scene_0021               scene_0026_ckpt_gripper  scene_0032_ckpt_arm      scene_0038               scene_0043_ckpt_gripper  scene_0049_ckpt_arm
scene_0004_ckpt_arm      scene_0010               scene_0015_ckpt_gripper  scene_0021_ckpt_arm      scene_0027               scene_0032_ckpt_gripper  scene_0038_ckpt_arm      scene_0044               scene_0049_ckpt_gripper
scene_0004_ckpt_gripper  scene_0010_ckpt_arm      scene_0016               scene_0021_ckpt_gripper  scene_0027_ckpt_arm      scene_0033               scene_0038_ckpt_gripper  scene_0044_ckpt_arm      scene_0050
scene_0005               scene_0010_ckpt_gripper  scene_0016_ckpt_arm      scene_0022               scene_0027_ckpt_gripper  scene_0033_ckpt_arm      scene_0039               scene_0044_ckpt_gripper  scene_0050_ckpt_arm
scene_0005_ckpt_arm      scene_0011               scene_0016_ckpt_gripper  scene_0022_ckpt_arm      scene_0028               scene_0033_ckpt_gripper  scene_0039_ckpt_arm      scene_0045               scene_0050_ckpt_gripper
scene_0005_ckpt_gripper  scene_0011_ckpt_arm      scene_0017               scene_0022_ckpt_gripper  scene_0028_ckpt_arm      scene_0034               scene_0039_ckpt_gripper  scene_0045_ckpt_arm
scene_0006               scene_0011_ckpt_gripper  scene_0017_ckpt_arm      scene_0023               scene_0028_ckpt_gripper  scene_0034_ckpt_arm      scene_0040               scene_0045_ckpt_gripper
scene_0006_ckpt_arm      scene_0012               scene_0017_ckpt_gripper  scene_0023_ckpt_arm      scene_0029               scene_0034_ckpt_gripper  scene_0040_ckpt_arm      scene_0046

scene_0001 内是膨胀处理后，去掉gripper部分的arm的mask
scene_0001_ckpt_arm 是没有膨胀处理，包含gripper部分的arm的mask
scene_0001_ckpt_gripper 是没有膨胀处理，gripper部分的mask