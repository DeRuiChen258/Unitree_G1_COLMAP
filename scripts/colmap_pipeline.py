#!/usr/bin/env python3
"""pycolmap 端到端流水线：视频/图像目录 → CUDA 特征 → 匹配 → 增量重建 → 3D 散点可视化。

用法示例：
    python colmap_pipeline.py --video ../test_video/g1_wave.mp4 --out ../py_out/g1_wave
    python colmap_pipeline.py --images ../movingcam_video/images --out ../py_out/movingcam

说明：
    * 特征提取/匹配默认走 CUDA（`pycolmap.Device.cuda`），失败时自动回退 CPU 并给出告警；
    * 单相机场景默认 CameraMode.SINGLE；
    * 可视化用 matplotlib 渲染 4 个视角的 3D 散点（点云按自身 RGB 着色 + 相机中心轨迹）。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pycolmap


class DeviceError(RuntimeError):
    """CUDA 路径不可用（用于触发 CPU 回退）；与『素材不可重建』区分开。"""


def setup_cjk_font() -> None:
    """让 matplotlib 正确渲染中文（系统装有 Noto CJK 时优先启用）。"""
    import matplotlib
    from matplotlib import font_manager

    wanted = ["Noto Sans CJK SC", "Noto Sans CJK HK", "Noto Sans CJK JP",
              "Source Han Sans SC", "WenQuanYi Zen Hei", "Droid Sans Fallback"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in wanted:
        if name in available:
            matplotlib.rcParams["font.family"] = [name, "DejaVu Sans"]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return


def read_ply_vertices(path: Path, max_points: int = 200000) -> np.ndarray | None:
    """读取 binary_little_endian PLY 的顶点坐标（按属性表解析，容忍法线/颜色等额外属性）。

    COLMAP 的 fused.ply 每顶点属性为 x,y,z,nx,ny,nz,red,green,blue —— 早先按“每顶点 3 个 float”
    硬解析会读到错位数据（坐标出现 1e38 级噪声），因此这里严格按照 header 构造结构化 dtype。
    """
    if path is None or not Path(path).exists():
        return None
    with open(path, "rb") as f:
        header = b""
        while not header.endswith(b"end_header\n"):
            line = f.readline()
            if not line:
                return None
            header += line
        lines = header.decode("ascii", "ignore").splitlines()
        if not any("binary_little_endian" in l for l in lines):
            print("[pipeline] 仅支持 binary_little_endian PLY，已跳过稠密点云")
            return None
        count, props = 0, []
        in_vertex = False
        type_map = {"float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
                    "uchar": "u1", "uint8": "u1", "int": "<i4", "uint": "<u4"}
        for line in lines:
            parts = line.split()
            if len(parts) >= 3 and parts[0] == "element":
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    count = int(parts[2])
            elif in_vertex and len(parts) >= 3 and parts[0] == "property":
                props.append((parts[2], type_map.get(parts[1], "<f4")))
        if count == 0 or not props:
            return None
        dtype = np.dtype(props)
        raw = f.read(dtype.itemsize * count)
        arr = np.frombuffer(raw, dtype=dtype, count=count)
    xyz = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float64)
    if len(xyz) > max_points:
        idx = np.random.default_rng(1).choice(len(xyz), max_points, replace=False)
        xyz = xyz[idx]
    return xyz


def extract_frames(video: Path, out_dir: Path, fps: float) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.png"):
        old.unlink()
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
         "-vf", f"fps={fps}", "-qscale:v", "2", str(out_dir / "frame_%04d.png")],
        check=True,
    )
    frames = sorted(out_dir.glob("*.png"))
    if not frames:
        raise SystemExit(f"[pipeline] 未能从 {video} 抽帧")
    return frames


def run_sfm(image_dir: Path, work: Path, device: str, matcher: str,
            max_image_size: int, max_features: int,
            peak_threshold: float = 0.0067,
            edge_threshold: float = 10.0) -> tuple[pycolmap.Reconstruction, dict]:
    work.mkdir(parents=True, exist_ok=True)
    db = work / "database.db"
    if db.exists():
        db.unlink()

    dev = pycolmap.Device.cuda if device == "cuda" else pycolmap.Device.cpu
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "SIMPLE_RADIAL"
    ex = pycolmap.FeatureExtractionOptions()
    ex.use_gpu = device == "cuda"
    ex.gpu_index = "0"  # pycolmap 4.x 中 gpu_index 是字符串（可为设备表达式）
    ex.max_image_size = max_image_size
    ex.sift.max_num_features = max_features
    ex.sift.peak_threshold = peak_threshold   # 调低 → 提取更多弱响应特征（提升点云密度）
    ex.sift.edge_threshold = edge_threshold   # 调高 → 保留更多边缘附近特征（机械结构更密）

    print(f"[pipeline] 特征提取：{len(list(image_dir.glob('*.png')))} 张，device={device}")
    try:
        pycolmap.extract_features(db, image_dir, camera_mode=pycolmap.CameraMode.SINGLE,
                                  reader_options=reader, extraction_options=ex, device=dev)

        mo = pycolmap.FeatureMatchingOptions()
        mo.use_gpu = device == "cuda"
        mo.gpu_index = "0"
        if matcher == "sequential":
            po = pycolmap.SequentialPairingOptions()
            po.overlap = 12
            po.loop_detection = True
            print("[pipeline] 匹配：sequential（带回环检测）")
            pycolmap.match_sequential(db, matching_options=mo, pairing_options=po, device=dev)
        else:
            print("[pipeline] 匹配：exhaustive")
            pycolmap.match_exhaustive(db, matching_options=mo, device=dev)
    except Exception as exc:  # 仅设备相关失败才回退 CPU
        raise DeviceError(f"{type(exc).__name__}: {exc}") from exc

    print("[pipeline] 增量式 SfM …")
    recs = pycolmap.incremental_mapping(db, image_dir, work / "sparse")
    if not recs:
        raise SystemExit(
            "[pipeline] SfM 未产出任何模型。常见原因：素材为零基线/纯旋转（固定机位视频）、"
            "或场景纹理缺失。可用 python/viz_keypoints.py 查看特征散点确认前端是否正常。"
        )
    best_id = max(recs, key=lambda k: recs[k].num_reg_images())
    rec = recs[best_id]
    model_dir = work / "model"
    model_dir.mkdir(exist_ok=True)
    rec.write(model_dir)
    rec.export_PLY(work / "sparse.ply")

    stats = {
        "num_images": len(list(image_dir.glob("*.png"))),
        "num_reg_images": rec.num_reg_images(),
        "num_points3D": rec.num_points3D(),
        "mean_reproj_error_px": float(rec.compute_mean_reprojection_error()),
        "mean_track_length": float(rec.compute_mean_track_length()),
        "num_models": len(recs),
        "best_model_id": int(best_id),
    }
    (work / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return rec, stats


def visualize(rec: pycolmap.Reconstruction, out_png: Path, title: str,
              max_points: int = 60000, dense_ply: Path | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    setup_cjk_font()
    pts = list(rec.points3D.values())
    xyz = np.array([p.xyz for p in pts], dtype=float)
    rgb = np.array([p.color for p in pts], dtype=float)
    rgb = np.clip(rgb / 255.0, 0.0, 1.0)
    # 稳健裁剪：稀疏重建中常有少量远处离群点，会把显示尺度拉坏（实测跨到 ±25）。
    # 按 0.5–99.5 分位裁剪，并把四张子图统一到同一坐标范围。
    lo_p = np.percentile(xyz, 0.5, axis=0)
    hi_p = np.percentile(xyz, 99.5, axis=0)
    inliers = np.all((xyz >= lo_p) & (xyz <= hi_p), axis=1)
    n_dropped = int((~inliers).sum())
    xyz, rgb = xyz[inliers], rgb[inliers]
    # 按文件名排序，轨迹连线才是相机真实的运动顺序（否则会画成乱麻）
    ordered = sorted([im for im in rec.images.values() if im.has_pose], key=lambda i: i.name)
    centers = np.array([im.projection_center() for im in ordered])
    if len(xyz) > max_points:
        idx = np.random.default_rng(0).choice(len(xyz), max_points, replace=False)
        xyz, rgb = xyz[idx], rgb[idx]
    # 显示范围 = 点云（稳健）× 相机中心 的并集：只用点云会把轨迹和外锥挤出画框
    lo = np.minimum(lo_p, centers.min(0))
    hi = np.maximum(hi_p, centers.max(0))
    pad = 0.04 * np.maximum(hi - lo, 1e-6)
    lo, hi = lo - pad, hi + pad

    dense = None
    try:  # 稠密点云：可选叠加显示，读取失败不影响主结果
        dense = read_ply_vertices(dense_ply)
    except Exception as exc:
        print(f"[pipeline] 稠密点云读取失败（忽略）：{exc}")

    span = hi - lo
    views = [("斜视 (azim=-60)", -60, 20), ("正视 (azim=0)", 0, 8),
             ("侧视 (azim=90)", 90, 8), ("俯视 (elev=80)", -90, 80)]
    frustum_depth = 0.12 * float(np.linalg.norm(hi - lo))

    def draw_frusta(ax) -> None:
        """绘制若干相机锥体（位姿来自 cam_from_world，按 COLMAP 相机坐标系 z 朝前）。"""
        if not ordered:
            return
        step = max(1, len(ordered) // 10)
        for im in ordered[::step]:
            m = im.cam_from_world().matrix()
            R_wc, t_wc = m[:3, :3], m[:3, 3]
            fx = float(im.camera.focal_length)
            half_w = 0.5 * im.camera.width / fx * frustum_depth
            half_h = 0.5 * im.camera.height / fx * frustum_depth
            corners_cam = np.array([[0, 0, 0],
                                    [-half_w, -half_h, frustum_depth], [half_w, -half_h, frustum_depth],
                                    [half_w, half_h, frustum_depth], [-half_w, half_h, frustum_depth]])
            world = (R_wc.T @ (corners_cam.T - t_wc[:, None])).T
            for a, b in [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]:
                ax.plot([world[a, 0], world[b, 0]], [world[a, 1], world[b, 1]],
                        [world[a, 2], world[b, 2]], color="darkorange", lw=0.7, alpha=0.85)

    fig = plt.figure(figsize=(17, 13))
    for i, (name, azim, elev) in enumerate(views, 1):
        ax = fig.add_subplot(2, 2, i, projection="3d")
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=rgb, s=1.2, marker=".", linewidths=0, alpha=0.9)
        if dense is not None:
            ax.scatter(dense[:, 0], dense[:, 1], dense[:, 2], c="tab:cyan", s=0.4, marker=".", alpha=0.25)
        draw_frusta(ax)
        ax.plot(centers[:, 0], centers[:, 1], centers[:, 2], "r.-", ms=6, lw=1.4,
                label="camera centers（按帧序）")
        ax.set_title(name, fontsize=11)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect(tuple(span))
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle(f"{title}  |  显示范围=0.5–99.5% 分位包围盒（裁剪离群点 {n_dropped}/{len(pts)}）", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"[pipeline] 可视化已保存：{out_png}")


def visualize_scatter(rec: pycolmap.Reconstruction, out_dir: Path, title: str,
                      max_points: int = 120000) -> None:
    """输出两张“纯散点图”：3D 立体散点 + 2D 俯视散点（含相机位姿）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    setup_cjk_font()

    xyz = np.array([p.xyz for p in rec.points3D.values()], dtype=float)
    rgb = np.clip(np.array([p.color for p in rec.points3D.values()], dtype=float) / 255.0, 0, 1)
    lo_p = np.percentile(xyz, 0.5, axis=0)
    hi_p = np.percentile(xyz, 99.5, axis=0)
    keep = np.all((xyz >= lo_p) & (xyz <= hi_p), axis=1)
    xyz, rgb = xyz[keep], rgb[keep]
    if len(xyz) > max_points:
        idx = np.random.default_rng(0).choice(len(xyz), max_points, replace=False)
        xyz, rgb = xyz[idx], rgb[idx]
    ordered = sorted([im for im in rec.images.values() if im.has_pose], key=lambda i: i.name)
    centers = np.array([im.projection_center() for im in ordered])
    lo = np.minimum(lo_p, centers.min(0))
    hi = np.maximum(hi_p, centers.max(0))
    pad = 0.04 * np.maximum(hi - lo, 1e-6)
    lo, hi = lo - pad, hi + pad

    # ① 3D 立体散点
    fig = plt.figure(figsize=(15, 11))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=rgb, s=2.0, marker=".", linewidths=0, alpha=0.95)
    ax.plot(centers[:, 0], centers[:, 1], centers[:, 2], "r.-", ms=7, lw=1.6, label="camera centers（按帧序）")
    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c="red", s=12, depthshade=False)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(tuple(hi - lo))
    ax.view_init(elev=22, azim=-58)
    ax.set_xlabel("X [a.u.]"); ax.set_ylabel("Y [a.u.]"); ax.set_zlabel("Z [a.u.]")
    ax.legend(loc="upper right")
    ax.set_title(f"{title}\n3D 散点：{len(xyz)} 个稀疏点（按 RGB 着色）+ 相机轨迹", fontsize=13)
    fig.tight_layout()
    out_3d = out_dir / "scatter_3d.png"
    fig.savefig(out_3d, dpi=140)
    plt.close(fig)

    # ② 2D 俯视散点（X–Z 平面）
    fig2, ax2 = plt.subplots(figsize=(13, 12))
    ax2.scatter(xyz[:, 0], xyz[:, 2], c=rgb, s=2.5, marker=".", linewidths=0, alpha=0.95)
    ax2.plot(centers[:, 0], centers[:, 2], "r.-", ms=8, lw=1.6, label="camera centers（按帧序）")
    ax2.set_aspect("equal", adjustable="box")
    ax2.set_xlabel("X [a.u.]"); ax2.set_ylabel("Z [a.u.]")
    ax2.grid(alpha=0.3)
    ax2.legend(loc="upper right")
    ax2.set_title(f"{title}\n俯视散点（X–Z 平面）：点云 + 环绕相机轨迹", fontsize=13)
    fig2.tight_layout()
    out_2d = out_dir / "scatter_top.png"
    fig2.savefig(out_2d, dpi=140)
    plt.close(fig2)
    print(f"[pipeline] 散点图已保存：{out_3d} / {out_2d}")


def main() -> None:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", type=Path, help="输入视频")
    src.add_argument("--images", type=Path, help="输入图像目录")
    src.add_argument("--model", type=Path, help="已保存的重建模型目录（跳过 SfM，只做可视化）")
    ap.add_argument("--out", type=Path, default=None, help="输出目录（--model 模式下默认取模型上级目录）")
    ap.add_argument("--fps", type=float, default=4.0, help="从视频抽帧的帧率")
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--matcher", choices=["exhaustive", "sequential"], default="exhaustive")
    ap.add_argument("--max-image-size", type=int, default=1600)
    ap.add_argument("--max-features", type=int, default=16384)
    ap.add_argument("--peak-threshold", type=float, default=0.0067, help="SIFT 峰值阈值（调低=特征更多）")
    ap.add_argument("--edge-threshold", type=float, default=10.0, help="SIFT 边缘阈值（调高=保留更多边缘特征）")
    ap.add_argument("--dense-ply", type=Path, default=None, help="可选：已有的稠密点云 PLY，叠加显示")
    ap.add_argument("--no-visualize", action="store_true")
    ap.add_argument("--scatter", action="store_true", help="额外输出两张纯散点图（3D + 俯视）")
    args = ap.parse_args()

    if args.model:
        rec = pycolmap.Reconstruction(str(args.model.resolve()))
        stats = {
            "num_images": len(rec.images),
            "num_reg_images": rec.num_reg_images(),
            "num_points3D": rec.num_points3D(),
            "mean_reproj_error_px": float(rec.compute_mean_reprojection_error()),
            "mean_track_length": float(rec.compute_mean_track_length()),
        }
        title = (f"pycolmap {pycolmap.__version__}（CUDA） |  注册图像 {stats['num_reg_images']}/{stats['num_images']}"
                 f"  |  稀疏点 {stats['num_points3D']}  |  平均重投影 {stats['mean_reproj_error_px']:.3f} px")
        if args.scatter:
            visualize_scatter(rec, args.model.resolve().parent, title)
        if not args.no_visualize:
            visualize(rec, args.model.resolve().parent / "visualization.png", title, dense_ply=args.dense_ply)
        return

    if args.out is None:
        if args.model is None:
            ap.error("--out 为必填（除非使用 --model）")
        args.out = args.model.resolve().parent
    work = args.out.resolve()
    if args.video:
        image_dir = work / "images"
        n = len(extract_frames(args.video, image_dir, args.fps))
        print(f"[pipeline] 抽帧 {n} 张 → {image_dir}")
    else:
        image_dir = args.images.resolve()

    try:
        rec, stats = run_sfm(image_dir, work, args.device, args.matcher,
                             args.max_image_size, args.max_features,
                             args.peak_threshold, args.edge_threshold)
    except DeviceError as exc:
        if args.device == "cuda":
            print(f"{exc}\n[pipeline] CUDA 路径失败，回退 CPU 重试", file=sys.stderr)
            rec, stats = run_sfm(image_dir, work, "cpu", args.matcher,
                                 args.max_image_size, args.max_features,
                                 args.peak_threshold, args.edge_threshold)
        else:
            raise

    print("[pipeline] 结果：")
    for k, v in stats.items():
        print(f"    {k:24s} = {v}")

    if not args.no_visualize:
        title = f"pycolmap {pycolmap.__version__}  |  注册图像 {stats['num_reg_images']}/{stats['num_images']}  |  " \
                f"稀疏点 {stats['num_points3D']}  |  平均重投影 {stats['mean_reproj_error_px']:.3f} px"
        visualize(rec, work / "visualization.png", title, dense_ply=args.dense_ply)


if __name__ == "__main__":
    main()
