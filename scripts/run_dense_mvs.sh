#!/usr/bin/env bash
# ==============================================================================
# run_dense_mvs.sh —— 对已有稀疏模型做稠密重建（image_undistorter → PatchMatch → 融合）
#
# 用法：bash run_dense_mvs.sh <模型目录> <图像目录> <输出目录> [max_image_size]
# 说明：CUDA 路径（系统 /usr/local/cuda-13.2），几何一致性开启后按 geometric 融合。
# ==============================================================================
set -euo pipefail

COLMAP_BIN="${COLMAP_BIN:-${COLMAP_ROOT:-$HOME/colmap-deploy}/install/bin/colmap}"
MODEL_DIR="${1:?模型目录}"; IMAGES_DIR="${2:?图像目录}"; OUT_DIR="${3:?输出目录}"
MAX_SIZE="${4:-1000}"
LOGS="${OUT_DIR}/../logs"
mkdir -p "${OUT_DIR}" "${LOGS}"

log() { printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*" | tee -a "${LOGS}/dense_mvs.log"; }

log ">>> 1/3 image_undistorter（max_image_size=${MAX_SIZE}）"
"${COLMAP_BIN}" image_undistorter \
  --image_path "${IMAGES_DIR}" --input_path "${MODEL_DIR}" \
  --output_path "${OUT_DIR}" --output_type COLMAP \
  --max_image_size "${MAX_SIZE}" > "${LOGS}/dense_undistort.log" 2>&1

log ">>> 2/3 patch_match_stereo（CUDA，geom_consistency=true）"
"${COLMAP_BIN}" patch_match_stereo \
  --workspace_path "${OUT_DIR}" --workspace_format COLMAP \
  --PatchMatchStereo.geom_consistency true --PatchMatchStereo.gpu_index 0 \
  > "${LOGS}/dense_patch_match.log" 2>&1

log ">>> 3/3 stereo_fusion"
"${COLMAP_BIN}" stereo_fusion \
  --workspace_path "${OUT_DIR}" --workspace_format COLMAP \
  --input_type geometric --output_path "${OUT_DIR}/fused.ply" \
  > "${LOGS}/dense_fusion.log" 2>&1

log "完成：$(du -h "${OUT_DIR}/fused.ply" | cut -f1)  ${OUT_DIR}/fused.ply"
