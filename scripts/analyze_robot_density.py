#!/usr/bin/env python3
"""统计并可视化“机器人区域”的点云密度。

约定：相机绕机器人环绕拍摄 → 相机中心构成一个圆环，环心即机器人所在位置，
      环半径 R 与机器人尺度成正比（本场景 R≈2.0 m，机器人高≈1.3 m）。
      因此用「距环心 < 0.40·R」的点集近似机器人本体点云。

用法：
    python analyze_robot_density.py --model <模型目录> [--dense-ply <稠密点云>] [--out <前缀>]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pycolmap


def read_ply_xyz(path: Path, max_points: int = 4_000_000) -> np.ndarray:
    """读取 binary_little_endian PLY 顶点坐标（按 header 属性表解析）。"""
    with open(path, "rb") as f:
        header = b""
        while not header.endswith(b"end_header\n"):
            header += f.readline()
        lines = header.decode("ascii", "ignore").splitlines()
        count, props, in_vertex = 0, [], False
        types = {"float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
                 "uchar": "u1", "uint8": "u1", "int": "<i4", "uint": "<u4"}
        for line in lines:
            p = line.split()
            if len(p) >= 3 and p[0] == "element":
                in_vertex = p[1] == "vertex"
                if in_vertex:
                    count = int(p[2])
            elif in_vertex and len(p) >= 3 and p[0] == "property":
                props.append((p[2], types.get(p[1], "<f4")))
        arr = np.frombuffer(f.read(np.dtype(props).itemsize * count), dtype=np.dtype(props), count=count)
    xyz = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float64)
    if len(xyz) > max_points:
        xyz = xyz[np.random.default_rng(0).choice(len(xyz), max_points, replace=False)]
    return xyz


def robot_mask(xyz: np.ndarray, center: np.ndarray, radius: float, ratio: float = 0.40) -> np.ndarray:
    return np.linalg.norm(xyz - center, axis=1) < ratio * radius


def fit_plane_ransac(pts: np.ndarray, thresh: float, iters: int = 600) -> tuple[np.ndarray, float]:
    """RANSAC 拟合平面，返回 (单位法向量, 偏移)；平面上点满足 n·x + d ≈ 0。"""
    rng = np.random.default_rng(0)
    best_n, best_d, best_cnt = np.array([0, 0, 1.0]), 0.0, -1
    for _ in range(iters):
        idx = rng.choice(len(pts), 3, replace=False)
        p0, p1, p2 = pts[idx]
        n = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n = n / norm
        d = -float(n @ p0)
        cnt = int((np.abs(pts @ n + d) < thresh).sum())
        if cnt > best_cnt:
            best_n, best_d, best_cnt = n, d, cnt
    return best_n, best_d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--dense-ply", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None, help="散点图输出前缀（不含扩展名）")
    ap.add_argument("--ratio", type=float, default=0.40, help="机器人半径 = ratio × 相机环半径")
    ap.add_argument("--cam-radius-true", type=float, default=None,
                    help="渲染时相机到机器人中心的真实距离（如 2.0 / 3.0）。"
                         "给出后按真实尺度折算机器人半径，可跨数据集比较")
    ap.add_argument("--robot-radius-true", type=float, default=0.75,
                    help="机器人本体半径（真实尺度，米）；G1 站姿身高约 1.3 m")
    ap.add_argument("--exclude-floor", action="store_true",
                    help="额外剔除地面点（机器人邻域内做 RANSAC 平面拟合，只统计平面上方点）")
    args = ap.parse_args()

    rec = pycolmap.Reconstruction(str(args.model))
    centers = np.array([im.projection_center() for im in rec.images.values() if im.has_pose])
    center = centers.mean(0)
    radius = float(np.linalg.norm(centers - center, axis=1).mean())
    if args.cam_radius_true:
        # 重建尺度 = 模型环半径 / 真机位半径；据此把“真实 0.75 m 机器人半径”换算到模型坐标
        r_robot = args.robot_radius_true * radius / args.cam_radius_true
        ratio_used = r_robot / radius
    else:
        ratio_used = args.ratio

    sparse = np.array([p.xyz for p in rec.points3D.values()], dtype=float)
    m_sparse = robot_mask(sparse, center, radius, ratio_used)
    print(f"[robot] 模型：{args.model}")
    print(f"[robot] 相机环：半径 R={radius:.3f}，环心={np.round(center, 3).tolist()}")
    print(f"[robot] 机器人判定半径：{ratio_used * radius:.3f}（模型单位）= "
          f"{args.robot_radius_true if args.cam_radius_true else ratio_used * radius:.2f} m（真实）")
    print(f"[robot] 稀疏点：{len(sparse):>7d} 总数 | 机器人区域 {int(m_sparse.sum()):>6d} "
          f"({100 * m_sparse.mean():.1f}%)")

    dense_pts = None
    dense_body = None      # 剔除地面后的机器人本体点（用于制图）
    if args.dense_ply and Path(args.dense_ply).exists():
        dense_pts = read_ply_xyz(Path(args.dense_ply))
        # 稠密点云与稀疏模型同坐标系（同一 model 融合而来）
        m_dense = robot_mask(dense_pts, center, radius, ratio_used)
        print(f"[robot] 稠密点：{len(dense_pts):>7d} 总数 | 机器人区域 {int(m_dense.sum()):>6d} "
              f"({100 * m_dense.mean():.1f}%)")
        if args.exclude_floor:
            # 机器人邻域内 60% 左右是地面；用 RANSAC 找主平面，只保留平面上方 5 cm 以上的点
            near = dense_pts[m_dense]
            scale = radius / args.cam_radius_true       # 模型单位 / 米
            n, d = fit_plane_ransac(near, thresh=0.02 * scale)
            signed = near @ n + d                        # 有符号距离；用环心所在侧确定正方向
            if signed.mean() < 0:
                n, d, signed = -n, -d, -signed
            keep = signed > 0.05 * scale                 # 地面上方 5 cm
            print(f"[robot] 剔除地面后：{int(keep.sum()):>6d} 点"
                  f"（占该邻域 {100 * keep.mean():.1f}%；地面平面法向 {np.round(n, 3).tolist()}）")
            dense_body = near[keep]

    if args.out:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from colmap_pipeline import setup_cjk_font

        setup_cjk_font()
        sel = np.linalg.norm(sparse - center, axis=1) < ratio_used * radius
        fig = plt.figure(figsize=(16, 7.5))
        titles = ["机器人特写（3D）", "机器人特写（俯视 X–Z）"]
        data = [(sparse[sel], "3d"), (sparse[sel], "2d")]
        if dense_pts is not None:
            sel_d = np.linalg.norm(dense_pts - center, axis=1) < ratio_used * radius
            pts_plot = dense_body if dense_body is not None else dense_pts[sel_d]
            label = "机器人本体（已剔地面）" if dense_body is not None else "机器人邻域"
            data = [(pts_plot, "3d"), (pts_plot, "2d")]
            titles = [f"机器人特写 · 稠密点云（3D）", f"机器人特写 · 稠密点云（俯视 X–Z）"]
        for i, ((pts, kind), title) in enumerate(zip(data, titles), 1):
            ax = fig.add_subplot(1, 2, i, projection="3d" if kind == "3d" else None)
            if kind == "3d":
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=0.8, c="tab:blue", alpha=0.7, linewidths=0)
                ax.view_init(elev=12, azim=-90)
                ax.set_zlabel("Z [a.u.]")
            else:
                ax.scatter(pts[:, 0], pts[:, 2], s=0.8, c="tab:blue", alpha=0.7, linewidths=0)
                ax.set_aspect("equal", adjustable="box")
                ax.set_ylabel("Z [a.u.]")
                ax.grid(alpha=0.3)
            ax.set_xlabel("X [a.u.]")
            ax.set_title(f"{title} —— {len(pts)} 点", fontsize=12)
        fig.suptitle(f"机器人区域点云（< {ratio_used:.2f}·R，R={radius:.2f}）"
                     f"{' · ' + label if dense_pts is not None else ''}", fontsize=13)
        fig.tight_layout()
        pre = Path(args.out)
        fig.savefig(pre.with_suffix(".png"), dpi=135)
        print(f"[robot] 散点图 → {pre.with_suffix('.png')}")


if __name__ == "__main__":
    main()
