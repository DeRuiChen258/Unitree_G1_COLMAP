#!/usr/bin/env bash
# ==============================================================================
# build_pycolmap.sh —— 用「已本地安装的 COLMAP 4.2.0（CUDA）」构建 pycolmap wheel
#
# 关键点：
#   * python/CMakeLists.txt 只做 find_package(colmap REQUIRED) —— 复用 install/ 里的
#     静态库与依赖，**不重新编译整个 COLMAP**，因此只编译 pybind11 绑定层；
#   * CUDA 一律使用**系统 /usr/local/cuda-13.2**（不是 conda 里的 CUDA 包）；
#   * 通过 7897 代理拉取构建依赖（scikit-build-core / pybind11 / pybind11-stubgen）。
#
# 用法：bash build_pycolmap.sh [--install-into <conda-env-python>]
# ==============================================================================
set -euo pipefail

TARGET="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPS="${TARGET}/deps"
INSTALL="${TARGET}/install"
SRC="${TARGET}/src/COLMAP"
DIST="${TARGET}/dist"
CUDA_TOOLKIT_ROOT="${CUDA_TOOLKIT_ROOT:-/usr/local/cuda-13.2}"
LOGS="${TARGET}/logs"
PY_INSTALL="${PY_INSTALL:-${PY:-python3}}"
BUILD_PY="${BUILD_PY:-${TARGET}/build_venv/bin/python}"

# 代理（用户环境：verge-mihomo 7897）
export http_proxy="${http_proxy:-http://127.0.0.1:7897}"
export https_proxy="${https_proxy:-http://127.0.0.1:7897}"
export all_proxy="${all_proxy:-socks5://127.0.0.1:7897}"

mkdir -p "${DIST}" "${LOGS}"

echo "==> [1/3] 构建 wheel（CUDA=${CUDA_TOOLKIT_ROOT}，复用 ${INSTALL}）"
CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-20}" \
"${BUILD_PY}" -m pip wheel "${SRC}" --no-deps --no-build-isolation -w "${DIST}" \
  --config-settings=cmake.define.colmap_DIR="${INSTALL}/share/colmap" \
  --config-settings=cmake.define.CMAKE_PREFIX_PATH="${INSTALL};${DEPS};${CUDA_TOOLKIT_ROOT}" \
  --config-settings=cmake.define.pybind11_DIR="${TARGET}/build_venv/lib/python3.12/site-packages/pybind11/share/cmake/pybind11" \
  --config-settings=cmake.define.CUDA_TOOLKIT_ROOT_DIR="${CUDA_TOOLKIT_ROOT}" \
  --config-settings=cmake.define.CMAKE_CUDA_COMPILER="${CUDA_TOOLKIT_ROOT}/bin/nvcc" \
  --config-settings=cmake.define.CMAKE_CUDA_ARCHITECTURES=120 \
  --config-settings=cmake.define.CMAKE_INSTALL_RPATH="${DEPS}/lib;${CUDA_TOOLKIT_ROOT}/lib64" \
  --config-settings=cmake.define.CMAKE_INSTALL_RPATH_USE_LINK_PATH=ON \
  --config-settings=cmake.define.CMAKE_BUILD_WITH_INSTALL_RPATH=ON \
  2>&1 | tee "${LOGS}/pycolmap_build.log"

WHEEL=$(ls -t "${DIST}"/pycolmap-*.whl | head -1)
echo "==> wheel: ${WHEEL}"

echo "==> [2/3] 安装到 ${PY_INSTALL}（--no-deps：只新增 pycolmap，不动既有包）"
"${PY_INSTALL}" -m pip freeze > "${LOGS}/pip_freeze_before_pycolmap.txt"
"${PY_INSTALL}" -m pip install --no-deps --force-reinstall "${WHEEL}" \
  2>&1 | tee "${LOGS}/pycolmap_install.log"
"${PY_INSTALL}" -m pip freeze > "${LOGS}/pip_freeze_after_pycolmap.txt"
echo "--- 依赖差异（应只有 pycolmap）---"
diff "${LOGS}/pip_freeze_before_pycolmap.txt" "${LOGS}/pip_freeze_after_pycolmap.txt" || true

echo "==> [3/3] 导入与 CUDA 链接自检"
"${PY_INSTALL}" - <<'PY'
import pycolmap, subprocess, glob, os
print("pycolmap", pycolmap.__version__, "from", os.path.dirname(pycolmap.__file__))
so = glob.glob(os.path.join(os.path.dirname(pycolmap.__file__), "_core*.so"))
print("extension:", so)
if so:
    print(subprocess.run(["ldd", so[0]], capture_output=True, text=True).stdout.split("libcudart")[1][:80] if "libcudart" in subprocess.run(["ldd", so[0]], capture_output=True, text=True).stdout else "（未链接 libcudart，见下面完整 ldd）")
PY
