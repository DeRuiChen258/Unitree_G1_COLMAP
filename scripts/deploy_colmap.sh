#!/usr/bin/env bash
# ==============================================================================
# deploy_colmap.sh —— COLMAP 4.2.0 本地源码部署（CUDA + GUI，可复现）
#
# 设计原则：
#   * 全部产物收敛到本目录：src/ (源码) + deps/ (依赖前缀) + build/ + install/ + logs/
#   * 不写系统路径、不用 sudo/apt、不改动既有 conda 环境 cuda_132
#   * 幂等：每一步先检查是否已完成，可安全重复执行（中断后从断点继续）
#
# 用法：bash deploy_colmap.sh [all|deps|configure|build|install|verify]
# ==============================================================================
set -euo pipefail

TARGET="${COLMAP_TARGET_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
DEPS="${TARGET}/deps"
SRC="${TARGET}/src/COLMAP"
BUILD="${TARGET}/build"
INSTALL="${TARGET}/install"
LOGS="${TARGET}/logs"
CUDA_ROOT="${CUDA_TOOLKIT_ROOT:-/usr/local/cuda-13.2}"
CONDA_BIN="${CONDA_BIN:-$(command -v conda)}"
COLMAP_TAG="${COLMAP_TAG:-4.2.0}"
JOBS="${JOBS:-16}"

STEP="${1:-all}"
mkdir -p "${TARGET}"/{src,build,install,data,logs} "${TARGET}/.agent"
say() { printf '\n\033[1m==> %s\033[0m\n' "$*" | tee -a "${LOGS}/deploy.log"; }

# ---- 依赖前缀（conda-forge，隔离于既有环境） ---------------------------------
step_deps() {
  if [ -x "${DEPS}/bin/cmake" ]; then
    say "依赖前缀已存在：${DEPS}（跳过创建）"
    return
  fi
  say "创建隔离依赖前缀 ${DEPS}（conda-forge）"
  "${CONDA_BIN}" create -y -p "${DEPS}" -c conda-forge --override-channels \
    "cmake>=3.28,<4" ninja eigen ceres-solver glog gflags metis suitesparse \
    openimageio sqlite libcurl openssl glew qt6-main cgal boost-cpp \
    2>&1 | tee "${LOGS}/conda_deps_install.log"
}

# ---- 源码 --------------------------------------------------------------------
step_src() {
  if [ -d "${SRC}/.git" ]; then
    say "源码已存在：${SRC} @ $(git -C "${SRC}" rev-parse --short HEAD)（跳过克隆）"
  else
    say "克隆 COLMAP ${COLMAP_TAG} 到 ${SRC}"
    # 本机对 GitHub 直连曾出现 TLS 中断，使用 HTTP/1.1 + 大 postBuffer 提升稳定性
    git -c http.version=HTTP/1.1 -c http.postBuffer=524288000 \
        -c http.lowSpeedLimit=1000 -c http.lowSpeedTime=60 \
        clone --depth 1 --branch "${COLMAP_TAG}" \
        https://github.com/colmap/colmap.git "${SRC}"
  fi
  # Eigen3 查找垫片：CGAL 会遮蔽 CMake 模块路径，导致 PoseLib 的 find_package(Eigen3) 失败
  say "安装 Eigen3 查找垫片到源码 cmake/ 目录（仅新增文件，不改上游文件）"
  cp "${TARGET}/cmake_helpers/FindEigen3.cmake" "${SRC}/cmake/FindEigen3.cmake"
}

# ---- 配置 --------------------------------------------------------------------
step_configure() {
  say "CMake 配置（CUDA + GUI + MVS + OpenGL + CGAL）"
  rm -f "${BUILD}/CMakeCache.txt"
  rm -rf "${BUILD}/CMakeFiles"
  "${DEPS}/bin/cmake" -S "${SRC}" -B "${BUILD}" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${INSTALL}" \
    -DCMAKE_PREFIX_PATH="${DEPS};${CUDA_ROOT}" \
    -DCMAKE_C_COMPILER=/usr/bin/gcc \
    -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
    -DCMAKE_CUDA_COMPILER="${CUDA_ROOT}/bin/nvcc" \
    -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++ \
    -DCMAKE_CUDA_ARCHITECTURES=120 \
    -DBOOST_ROOT="${DEPS}" -DBoost_NO_SYSTEM_PATHS=ON \
    -DTESTS_ENABLED=OFF \
    -DCUDA_ENABLED=ON -DGUI_ENABLED=ON -DOPENGL_ENABLED=ON -DMVS_ENABLED=ON \
    -DCGAL_ENABLED=ON -DONNX_ENABLED=OFF \
    -DCMAKE_INSTALL_RPATH="${DEPS}/lib;${CUDA_ROOT}/lib64" \
    -DCMAKE_BUILD_RPATH="${DEPS}/lib;${CUDA_ROOT}/lib64" \
    2>&1 | tee "${LOGS}/cmake_configure.log"
}

# ---- 编译 --------------------------------------------------------------------
step_build() {
  say "编译（-j ${JOBS}）"
  "${DEPS}/bin/cmake" --build "${BUILD}" -j "${JOBS}" 2>&1 | tee "${LOGS}/build.log"
}

# ---- 安装 --------------------------------------------------------------------
step_install() {
  say "安装到 ${INSTALL}"
  "${DEPS}/bin/cmake" --install "${BUILD}" 2>&1 | tee "${LOGS}/install.log"
}

# ---- 验证 --------------------------------------------------------------------
step_verify() {
  say "验证部署结果"
  bash "${TARGET}/verify_colmap.sh"
}

case "${STEP}" in
  deps)      step_deps ;;
  src)       step_src ;;
  configure) step_src; step_configure ;;
  build)     step_build ;;
  install)   step_install ;;
  verify)    step_verify ;;
  all)       step_deps; step_src; step_configure; step_build; step_install; step_verify ;;
  *) echo "用法：bash deploy_colmap.sh [all|deps|src|configure|build|install|verify]" >&2; exit 2 ;;
esac

say "完成：${STEP}"
