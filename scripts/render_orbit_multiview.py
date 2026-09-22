#!/usr/bin/env python3
"""渲染"环绕机位"多视图数据集（带真值位姿）用于验证 COLMAP SfM/MVS。

为什么需要它：
    test_video/g1_wave.mp4 是**固定机位**素材（只有机械臂在动，背景完全静止），
    零相机基线使 SfM 在数学上不可解，COLMAP 报 "No good initial image pair found"
    属于正确行为。因此另建同源（同一 MuJoCo G1 场景）的移动机位数据集，
    并同步导出真值位姿，用于量化验证重建精度。

关键实现：
    每个视角渲染前都 mj_resetDataKeyframe → 同一时刻的快照，
    保证"相机在动、场景刚体不动"，这是 SfM 成立的前提。
"""
from __future__ import annotations

import argparse
import json
import os
import math
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

DEFAULT_SCENE = os.path.join(
    os.environ.get("MUJOCO_MENAGERIE", ""), "unitree_g1", "scene.xml"
)


def rotation_to_quaternion_wxyz(R: np.ndarray) -> list[float]:
    """旋转矩阵 → 四元数 (w, x, y, z)，与 COLMAP 的 qvec 约定一致。"""
    trace = float(np.trace(R))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.asarray(q, dtype=float)
    return list(q / np.linalg.norm(q))


def main() -> None:
    ap = argparse.ArgumentParser(description="渲染环绕机位多视图数据集（含真值位姿）")
    ap.add_argument("--model", default=DEFAULT_SCENE, help="MuJoCo 场景 XML")
    ap.add_argument("--out", required=True, help="输出目录（生成 images/ 与 gt_poses.json）")
    ap.add_argument("--views", type=int, default=36, help="视角数量（环绕一周）")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--lookat", default="0.05,0.0,1.05")
    ap.add_argument("--distance", type=float, default=2.7)
    ap.add_argument("--elevation", type=float, default=-6.0)
    ap.add_argument("--azimuth-start", type=float, default=0.0)
    ap.add_argument("--azimuth-sweep", type=float, default=360.0,
                    help="整段视频/数据集覆盖的方位角范围（度）；360=环绕一周")
    ap.add_argument("--arm-joints",
                    default="right_shoulder_pitch_joint,right_shoulder_roll_joint,right_elbow_joint",
                    help="按正弦摆动的关节名（逗号分隔）；传空字符串则保持静止")
    ap.add_argument("--arm-amplitude", type=float, default=0.30, help="关节摆动幅度（弧度）")
    ap.add_argument("--arm-period", type=float, default=0.0,
                    help="关节摆动周期（帧）；<=0 表示按视角索引分频，整体做一次完整摆动")
    args = ap.parse_args()

    out = Path(args.out)
    images_dir = out / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(model)
    # 离屏帧缓冲需不小于渲染尺寸，否则 MuJoCo 直接报错
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), args.width)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), args.height)

    renderer = mujoco.Renderer(model, args.height, args.width)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = np.asarray([float(v) for v in args.lookat.split(",")], dtype=float)
    camera.distance = args.distance
    camera.elevation = args.elevation

    # 关节名 → qpos 下标（用于逐帧正弦摆动；每个视角都在 keyframe 上重置，机器人不会走动）
    arm_slots: list[tuple[int, float]] = []
    if args.arm_joints:
        for idx, jname in enumerate(args.arm_joints.split(",")):
            jname = jname.strip()
            if not jname:
                continue
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid < 0:
                print(f"[render] 跳过不存在的关节：{jname}")
                continue
            phase = idx * math.pi / 3.0
            arm_slots.append((int(model.jnt_qposadr[jid]), phase))

    fovy_deg = float(model.vis.global_.fovy)
    fy = (args.height / 2.0) / math.tan(math.radians(fovy_deg) / 2.0)
    fx = fy  # 方形像素

    poses: list[dict] = []
    for i in range(args.views):
        camera.azimuth = args.azimuth_start + args.azimuth_sweep * i / max(args.views - 1, 1)

        # 每个视角都回到同一时刻 → 场景完全刚体、只有相机在动
        mujoco.mj_resetDataKeyframe(model, data, 0)
        if arm_slots:
            period = args.arm_period if args.arm_period > 0 else float(args.views)
            angle = args.arm_amplitude * math.sin(2.0 * math.pi * i / period)
            for qpos_adr, phase in arm_slots:
                data.qpos[qpos_adr] += angle * math.cos(phase)
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=camera)
        frame = renderer.render()

        name = f"view_{i:03d}.png"
        Image.fromarray(frame).save(images_dir / name)

        # 读取渲染实际使用的相机位姿（MuJoCo 已按 GL 约定算好）
        cam = renderer.scene.camera[0]
        pos = np.asarray(cam.pos, dtype=float)
        forward = np.asarray(cam.forward, dtype=float)
        up = np.asarray(cam.up, dtype=float)
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)

        # MuJoCo/GL（x 右, y 上, z 后）→ COLMAP（x 右, y 下, z 前）
        R = np.stack([right, -up, forward], axis=0)  # world→cam
        t = -R @ pos
        poses.append(
            {
                "image": name,
                "center": pos.tolist(),
                "qvec": rotation_to_quaternion_wxyz(R),
                "tvec": t.tolist(),
                "R": R.reshape(-1).tolist(),
            }
        )

    meta = {
        "scene": args.model,
        "views": args.views,
        "width": args.width,
        "height": args.height,
        "fovy_deg": fovy_deg,
        "fx": fx,
        "fy": fy,
        "note": "每个视角均在 keyframe 重置后渲染，场景刚体、仅相机运动；真值来自 MjvScene 相机",
    }
    (out / "gt_poses.json").write_text(
        json.dumps({"meta": meta, "poses": poses}, indent=2), encoding="utf-8"
    )
    print(f"[render] 完成：{args.views} 个视角 → {images_dir}")
    print(f"[render] 真值位姿：{out / 'gt_poses.json'}（fx={fx:.1f}, local={args.distance}m）")


if __name__ == "__main__":
    main()
