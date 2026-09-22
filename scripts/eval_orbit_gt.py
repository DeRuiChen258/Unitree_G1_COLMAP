#!/usr/bin/env python3
"""把 COLMAP 重建结果与 MuJoCo 真值位姿做相似变换对齐并统计误差。

输出：
    * 注册相机数 / 全部相机数
    * 估计尺度 s 与相机中心 RMSE（绝对 + 相对轨迹半径百分比）
    * 相机朝向平均/中位角误差
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def read_colmap_images_txt(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """读取 COLMAP TXT 模型的 images.txt：返回 {图像名: (R world→cam, 相机中心)}"""
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 10:
            continue
        q = np.array([float(v) for v in parts[1:5]])
        t = np.array([float(v) for v in parts[5:8]])
        name = parts[9]
        R = quat_to_rot(q / np.linalg.norm(q))
        center = -R.T @ t
        out[name] = (R, center)
        i += 1  # 跳过对应 points2D 行
    return out


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """求解 dst ≈ s R src + t（带尺度 Kabsch）。"""
    mu_src, mu_dst = src.mean(0), dst.mean(0)
    src_c, dst_c = src - mu_src, dst - mu_dst
    cov = dst_c.T @ src_c / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    var_src = (src_c**2).sum() / len(src)
    s = float(np.trace(np.diag(D) @ S) / var_src)
    t = mu_dst - s * R @ mu_src
    return s, R, t


def rot_angle_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    cos = (np.trace(R_a.T @ R_b) - 1.0) / 2.0
    return math.degrees(math.acos(float(np.clip(cos, -1.0, 1.0))))


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    """3x3 旋转 → 单位四元数 (w, x, y, z)。"""
    trace = float(np.trace(R))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    else:
        i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        if i == 0:
            s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            q = np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
        elif i == 1:
            s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            q = np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s])
        else:
            s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            q = np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s])
    return q / np.linalg.norm(q)


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def mean_rotation(rots: list[np.ndarray]) -> np.ndarray:
    """一组旋转的四元数平均（外积矩阵主特征向量），用于提取系统性常值偏移。"""
    Q = np.stack([rot_to_quat(R) for R in rots])
    # 统一半球，避免 q 与 -q 抵消
    Q = np.where((Q @ Q[0])[:, None] < 0, -Q, Q)
    M = Q.T @ Q
    w, v = np.linalg.eigh(M)
    return quat_to_rotmat(v[:, -1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--colmap-txt", required=True, help="COLMAP TXT 模型目录（含 images.txt）")
    ap.add_argument("--gt", required=True, help="真值位姿 JSON（render_orbit_multiview.py 产出）")
    ap.add_argument("--selftest", action="store_true", help="自检：对真值施加已知相似变换后应恢复 ~0 误差")
    args = ap.parse_args()

    gt_raw = json.loads(Path(args.gt).read_text(encoding="utf-8"))
    gt = {p["image"]: (np.asarray(p["R"], dtype=float).reshape(3, 3), np.asarray(p["center"], dtype=float))
          for p in gt_raw["poses"]}
    est = read_colmap_images_txt(Path(args.colmap_txt) / "images.txt")

    names = [n for n in est if n in gt]
    if len(names) < 3:
        raise SystemExit(f"[eval] 匹配到的相机太少：est={len(est)} gt={len(gt)} 交集={len(names)}")

    C_est = np.stack([est[n][1] for n in names])
    C_gt = np.stack([gt[n][1] for n in names])
    R_est = [est[n][0] for n in names]
    R_gt = [gt[n][0] for n in names]

    if args.selftest:
        # 施加已知相似变换 C' = s R C + t，旋转按 R' = R R^T 变换；
        # 对齐器应能完全恢复该变换（残差 ~0）。
        R_sim = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        s_sim, t_sim = 3.7, np.array([5.0, -2.0, 1.0])
        C_est = (s_sim * (R_sim @ C_est.T)).T + t_sim
        R_est = [R @ R_sim.T for R in R_est]
        print("[selftest] 已对估计施加已知 3.7x 旋转+平移，对齐残差应接近 0")

    s, R_a, t_a = umeyama(C_est, C_gt)
    C_aligned = (s * (R_a @ C_est.T)).T + t_a
    err = np.linalg.norm(C_aligned - C_gt, axis=1)
    radius = float(np.linalg.norm(C_gt - C_gt.mean(0), axis=1).mean())

    deltas = [R_gt[i] @ (R_est[i] @ R_a.T).T for i in range(len(names))]
    ang = np.asarray([rot_angle_deg(np.eye(3), D) for D in deltas])
    # 去掉系统性常值旋转偏移后的相对朝向一致性（更能反映重建本身的几何质量）
    R_bias = mean_rotation(deltas)
    ang_rel = np.asarray([rot_angle_deg(np.eye(3), R_bias.T @ D) for D in deltas])

    print(f"注册相机         : {len(est)} / 真值 {len(gt)}（交集 {len(names)}）")
    print(f"估计尺度 s       : {s:.4f}")
    print(f"相机中心 RMSE    : {err.mean():.4f} m  (中位 {np.median(err):.4f}, 最大 {err.max():.4f})")
    print(f"轨迹半径(真值)   : {radius:.4f} m  → 相对 RMSE {100.0 * err.mean() / radius:.3f}%")
    print(f"相机朝向绝对误差 : 平均 {ang.mean():.4f}°, 中位 {np.median(ang):.4f}°, 最大 {ang.max():.4f}°")
    print(f"  └ 系统常值偏移 : 四元数平均后 {rot_angle_deg(np.eye(3), R_bias):.4f}°（真值提取约定引起的固定偏差）")
    print(f"相机朝向相对误差 : 平均 {ang_rel.mean():.4f}°, 中位 {np.median(ang_rel):.4f}°, 最大 {ang_rel.max():.4f}°（剔除常值偏移）")
    ok = len(names) == len(gt) and 100.0 * err.mean() / radius < 2.0 and ang.mean() < 1.0
    print(f"结论             : {'PASS（注册全、误差达标）' if ok else 'CHECK（见上列指标）'}")


if __name__ == "__main__":
    main()
