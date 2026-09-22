#!/usr/bin/env python3
"""把单帧（或固定机位视频帧）的 SIFT 特征"散点化"可视化。

用途：当素材是固定机位、无法做三维重建时，用 2D 散点展示特征提取结果——
     这是"视频 → 点"在二维上的真实成果，配合三维数据集才完整。

用法：
    python viz_keypoints.py --database ../py_out/g1_wave/database.db \
        --image ../py_out/g1_wave/images/frame_0001.png --out ../py_out/g1_wave/keypoints.png
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np


def load_keypoints(db: Path, image_name: str) -> tuple[np.ndarray, int]:
    con = sqlite3.connect(str(db))
    row = con.execute("SELECT image_id FROM images WHERE name = ?", (image_name,)).fetchone()
    if row is None:
        names = [r[0] for r in con.execute("SELECT name FROM images LIMIT 5")]
        raise SystemExit(f"[viz] 数据库中无图像 {image_name}；示例：{names}")
    blob, rows = con.execute("SELECT data, rows FROM keypoints WHERE image_id = ?", (row[0],)).fetchone()
    con.close()
    kp = np.frombuffer(blob, dtype="<f4").reshape(rows, -1)
    return kp, rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--database", type=Path, required=True)
    ap.add_argument("--image", type=Path, required=True, help="用于叠加的原始图像")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--title", default="")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    from colmap_pipeline import setup_cjk_font

    setup_cjk_font()
    kp, n = load_keypoints(args.database, args.image.name)
    img = np.asarray(Image.open(args.image).convert("RGB"))

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    axes[0].imshow(img)
    axes[0].set_title(f"原始帧 {args.image.name}  ({img.shape[1]}x{img.shape[0]})", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(img)
    sc = axes[1].scatter(kp[:, 0], kp[:, 1], c=kp[:, 2], s=3, cmap="turbo", linewidths=0)
    axes[1].set_title(f"SIFT 特征散点化：{n} 个关键点（颜色=尺度）", fontsize=11)
    axes[1].axis("off")
    fig.colorbar(sc, ax=axes[1], fraction=0.03, label="keypoint scale")
    fig.suptitle(args.title or f"固定机位素材的特征散点化（{n} keypoints）", fontsize=13)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"[viz] {n} 个关键点 → {args.out}")


if __name__ == "__main__":
    main()
