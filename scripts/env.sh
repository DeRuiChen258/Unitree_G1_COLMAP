#!/usr/bin/env bash
# ==============================================================================
# env.sh —— 本地 COLMAP（CUDA + C++）一键环境加载
#
# 用法：
#   source env.sh              # 激活 COLMAP 环境（叠加在当前 shell 之上）
#   source env.sh --quiet      # 静默模式
#
# 特性：
#   * 仅修改当前 shell 的环境变量，不改动 /etc、~/.bashrc、系统 pip / conda base；
#   * 重复 source 幂等（不会把同一路径反复前置）；
#   * 二进制自带 RPATH，即使不 source 也能直接运行；本脚本额外保证 GUI / Qt 插件 /
#     Python 绑定在同一次会话中都能找到依赖。
# ==============================================================================

_colmap_env_root="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

export COLMAP_ROOT="${_colmap_env_root}"
export COLMAP_INSTALL="${COLMAP_ROOT}/install"
export COLMAP_DEPS="${COLMAP_ROOT}/deps"
export COLMAP_SRC="${COLMAP_ROOT}/src/COLMAP"
export COLMAP_BUILD="${COLMAP_ROOT}/build"
export COLMAP_LOGS="${COLMAP_ROOT}/logs"

# CUDA 工具链（系统级只读引用，不复制、不修改）
export CUDA_TOOLKIT_ROOT="${CUDA_TOOLKIT_ROOT:-/usr/local/cuda-13.2}"
# 构建/运行目标 GPU 架构：RTX 5070 Laptop = Blackwell sm_120
export COLMAP_CUDA_ARCH="${COLMAP_CUDA_ARCH:-120}"

# ---- 去重前置函数 ------------------------------------------------------------
_colmap_prepend_path() {
  local __var="$1" __val="$2" __cur
  # 注意：用 ${var:-} 兜底，保证调用方开启 `set -u` 时也不会报 unbound variable
  eval "__cur=\"\${${__var}:-}\""
  case ":${__cur}:" in
    *":${__val}:"*) ;;
    *) if [ -n "${__cur}" ]; then
         eval "export ${__var}=\"${__val}:${__cur}\""
       else
         eval "export ${__var}=\"${__val}\""
       fi ;;
  esac
}

# ---- PATH / 动态库 / CMake / pkg-config --------------------------------------
_colmap_prepend_path PATH        "${COLMAP_INSTALL}/bin"
_colmap_prepend_path PATH        "${COLMAP_DEPS}/bin"          # cmake 3.31 / ninja（构建期用）
_colmap_prepend_path PATH        "${CUDA_TOOLKIT_ROOT}/bin"    # nvcc 13.2
_colmap_prepend_path LD_LIBRARY_PATH "${COLMAP_INSTALL}/lib"
_colmap_prepend_path LD_LIBRARY_PATH "${COLMAP_DEPS}/lib"
_colmap_prepend_path LD_LIBRARY_PATH "${CUDA_TOOLKIT_ROOT}/lib64"
_colmap_prepend_path CMAKE_PREFIX_PATH "${COLMAP_INSTALL}"
_colmap_prepend_path CMAKE_PREFIX_PATH "${COLMAP_DEPS}"
_colmap_prepend_path CMAKE_PREFIX_PATH "${CUDA_TOOLKIT_ROOT}"
_colmap_prepend_path PKG_CONFIG_PATH   "${COLMAP_DEPS}/lib/pkgconfig"

# Qt6 插件目录（GUI 依赖；conda-forge 的 Qt6 插件位于 lib/qt6/plugins，
# 若不显式设置，平台插件只在 Qt 自身前缀内解析，混用其它 Qt 时会找不到 xcb 插件）
if [ -d "${COLMAP_DEPS}/lib/qt6/plugins" ]; then
  export QT_PLUGIN_PATH="${COLMAP_DEPS}/lib/qt6/plugins${QT_PLUGIN_PATH:+:${QT_PLUGIN_PATH}}"
  export QT_QPA_PLATFORM_PLUGIN_PATH="${COLMAP_DEPS}/lib/qt6/plugins/platforms"
fi

# 让 CMake 在构建下游项目（例如把 COLMAP 当库链接）时能直接 find_package(colmap)
export colmap_DIR="${COLMAP_INSTALL}/share/colmap"

if [ "${1:-}" != "--quiet" ]; then
  echo "[colmap-env] COLMAP 本地环境已加载"
  echo "  colmap      : ${COLMAP_INSTALL}/bin/colmap  ($("${COLMAP_INSTALL}/bin/colmap" --help >/dev/null 2>&1 && echo runnable || echo '未就绪'))"
  echo "  deps 前缀   : ${COLMAP_DEPS}"
  echo "  CUDA        : ${CUDA_TOOLKIT_ROOT}"
fi

unset -f _colmap_prepend_path
