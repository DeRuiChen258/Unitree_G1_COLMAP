#!/usr/bin/env python3
"""把稠密点云 PLY（如 fused.ply）转成 COLMAP 模型目录，方便在 COLMAP GUI 里直接查看。

背景：COLMAP 的 model_converter 只支持「导出」PLY，不支持把 PLY 读回模型；
      但 pycolmap 提供 Reconstruction.import_PLY()，可以完成 PLY → 模型目录的转换。

用法：
    # ① 仅点云（无相机位姿）
    python ply_to_colmap_model.py <fused.ply> <输出模型目录>

    # ② 与已有稀疏模型合并：保留相机位姿/图像，把点集换成稠密点云（推荐，GUI 里能同时看到相机与稠密点）
    python ply_to_colmap_model.py <fused.ply> <输出模型目录> \
        --base-model <稀疏模型目录> --database <database.db> --images <图像目录>
然后：
    bash $COLMAP_ROOT/open_in_colmap_gui.sh <输出模型目录>
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pycolmap


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ply", type=Path, help="输入稠密点云 PLY")
    ap.add_argument("out", type=Path, help="输出 COLMAP 模型目录")
    ap.add_argument("--base-model", type=Path, default=None,
                    help="基准稀疏模型目录（提供相机位姿与图像）；给出后仅替换点集为稠密点云")
    ap.add_argument("--database", type=Path, default=None, help="project.ini 里的 database_path")
    ap.add_argument("--images", type=Path, default=None, help="project.ini 里的 image_path")
    args = ap.parse_args()

    if not args.ply.exists():
        raise SystemExit(f"找不到 {args.ply}")
    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)

    if args.base_model:
        rec = pycolmap.Reconstruction(str(args.base_model))
        n_cam_before, n_img_before = len(rec.cameras), rec.num_reg_images()
    else:
        rec = pycolmap.Reconstruction()
        n_cam_before = n_img_before = 0
    rec.import_PLY(str(args.ply))
    rec.write(args.out)

    # 无图像模型：GUI 仍要求 database_path 选项存在（可以为空库），否则报
    # "the option 'database_path' is required but missing"。
    db = args.database if args.database else args.out / "empty.db"
    if args.database is None:
        Path(db).touch()
    img_path = args.images if args.images else args.out
    (args.out / "project.ini").write_text(
        "log_color=true\n"
        "log_level=0\n"
        "log_severity=0\n"
        "log_target=stderr_and_file\n"
        "log_path=\n"
        f"database_path={db}\n"
        f"image_path={img_path}\n",
        encoding="utf-8",
    )
    size_mb = sum(f.stat().st_size for f in args.out.iterdir()) / 1024 / 1024
    print(f"[ply2model] {args.ply.name} → {args.out}：{rec.num_points3D()} 点"
          f"（相机 {n_cam_before} / 图像 {n_img_before}，{size_mb:.1f} MB）")
    print(f"[ply2model] 在 GUI 中查看：bash open_in_colmap_gui.sh {args.out}")


if __name__ == "__main__":
    main()
