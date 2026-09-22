#!/usr/bin/env python3
"""生成非周期随机纹理（供 SfM 验证场景使用）。

为什么需要：
    MuJoCo menagerie 自带的 scene.xml 地面是**周期性棋盘格**，SIFT 在周期纹理上
    会出现整格错配，导致匹配虽多但几何退化解（实测重建相机轨迹非圆）。
    这里用固定随机种子生成多尺度噪声纹理，保证角点丰富且非重复。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def multi_scale_noise(seed: int, size: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size, 3), dtype=np.float64)
    total = 0.0
    for scale, weight in ((4, 0.25), (8, 0.2), (16, 0.2), (32, 0.2), (64, 0.35), (128, 0.3)):
        coarse = rng.random((scale, scale, 3))
        layer = np.array(
            Image.fromarray((coarse * 255).astype(np.uint8)).resize((size, size), Image.BICUBIC),
            dtype=np.float64,
        ) / 255.0
        img += weight * layer
        total += weight
    img /= total
    # 叠加高频斑点，增加可重复检测的角点
    speck = rng.random((size // 2, size // 2, 3))
    speck = np.array(Image.fromarray((speck * 255).astype(np.uint8)).resize((size, size), Image.NEAREST)) / 255.0
    img = 0.75 * img + 0.25 * speck
    return np.clip(img, 0.0, 1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--size", type=int, default=1024)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, seed in (("sfm_floor.png", 20260921), ("sfm_object.png", 987654321)):
        arr = (multi_scale_noise(seed, args.size) * 255).astype(np.uint8)
        Image.fromarray(arr).save(out / name)
        print(f"[texture] {out / name}")


if __name__ == "__main__":
    main()
