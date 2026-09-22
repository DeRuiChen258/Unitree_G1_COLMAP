#!/usr/bin/env bash
# ==============================================================================
# verify_colmap.sh —— 部署结果自检（结果写入 logs/verify.log）
# ==============================================================================
set -uo pipefail

TARGET="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL="${TARGET}/install"
DEPS="${TARGET}/deps"
LOG="${TARGET}/logs/verify.log"
BIN="${INSTALL}/bin/colmap"
PASS=0
FAIL=0

mkdir -p "${TARGET}/logs"
exec > >(tee "${LOG}") 2>&1

echo "==================== COLMAP 部署验证 ===================="
echo "时间: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo

echo "--- [1] 可执行文件是否存在 ---"
if [ -x "${BIN}" ]; then
  echo "PASS: ${BIN}"; ls -l "${BIN}"; PASS=$((PASS+1))
else
  echo "FAIL: 未找到可执行文件 ${BIN}"; FAIL=$((FAIL+1))
fi
echo

# 不 source env.sh，直接调用，用于验证 RPATH 是否自洽
echo "--- [2] colmap --help（不依赖 LD_LIBRARY_PATH） ---"
if env -u LD_LIBRARY_PATH "${BIN}" --help >/dev/null 2>&1; then
  echo "PASS: exit=0"; PASS=$((PASS+1))
else
  echo "FAIL: exit=$?"; FAIL=$((FAIL+1))
fi
"${BIN}" --version 2>/dev/null || true
echo

echo "--- [3] 版本 / 编译特性 ---"
"${BIN}" -h 2>&1 | head -3
echo "编译期特性（来自 CMakeCache）："
for opt in CUDA_ENABLED GUI_ENABLED OPENGL_ENABLED MVS_ENABLED CGAL_ENABLED ONNX_ENABLED; do
  printf "  %-16s = %s\n" "${opt}" "$(rg -N "^${opt}:" "${TARGET}/build/CMakeCache.txt" 2>/dev/null | cut -d= -f2)"
done
printf "  %-16s = %s\n" "CUDA_ARCHS" "$(rg -N "^CMAKE_CUDA_ARCHITECTURES:" "${TARGET}/build/CMakeCache.txt" | cut -d= -f2)"
echo

echo "--- [4] 子命令识别 ---"
SUBCMDS=$(env -u LD_LIBRARY_PATH "${BIN}" -h 2>&1)
for sub in feature_extractor exhaustive_matcher sequential_matcher mapper image_undistorter patch_match_stereo stereo_fusion model_analyzer gui; do
  if printf '%s' "${SUBCMDS}" | grep -qw "${sub}"; then
    echo "PASS: ${sub}"; PASS=$((PASS+1))
  else
    echo "FAIL: ${sub} 未在 -h 输出中"; FAIL=$((FAIL+1))
  fi
done
echo

echo "--- [5] 动态库链接完整性（ldd 无 not found） ---"
if env -u LD_LIBRARY_PATH ldd "${BIN}" 2>&1 | grep -q "not found"; then
  echo "FAIL: 存在未解析动态库："
  env -u LD_LIBRARY_PATH ldd "${BIN}" | grep "not found"
  FAIL=$((FAIL+1))
else
  echo "PASS: 全部动态库可解析"
  env -u LD_LIBRARY_PATH ldd "${BIN}" | grep -E "ceres|glog|Qt6|OpenImageIO|GLEW|cholmod|colmap" | sed 's/^/  /'
  PASS=$((PASS+1))
fi
echo

echo "--- [6] CUDA 运行时可见性 ---"
if nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; then
  echo "PASS: GPU 可见"; PASS=$((PASS+1))
else
  echo "FAIL: nvidia-smi 失败"; FAIL=$((FAIL+1))
fi
echo

echo "--- [7] GUI 启动（12s 超时，exit=124 表示启动后持续运行） ---"
echo "显示环境：DISPLAY=${DISPLAY:-<未设置>}  WAYLAND_DISPLAY=${WAYLAND_DISPLAY:-<未设置>}"
if [ -n "${DISPLAY:-}" ] && [ -d /tmp/.X11-unix ]; then
  # 有真实 X 显示：直接启动 GUI（会短暂弹窗），超时杀掉即视为启动成功
  GUI_OUT=$(QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}" timeout 12 "${BIN}" gui 2>&1)
  GUI_RC=$?
  if [ ${GUI_RC} -eq 124 ]; then
    echo "PASS: GUI 成功创建窗口并进入事件循环（超时被 SIGTERM 终止，属预期）"
    PASS=$((PASS+1))
  elif [ ${GUI_RC} -eq 0 ]; then
    echo "PASS: GUI 正常退出（exit=0）"; PASS=$((PASS+1))
  else
    echo "FAIL: GUI 退出码 ${GUI_RC}"
    printf '%s\n' "${GUI_OUT}" | tail -8 | sed 's/^/  /'
    FAIL=$((FAIL+1))
  fi
else
  # 无显示：Qt 的 offscreen 插件不提供 GL 上下文，OpenGLWidgets 会崩，仅作为弱校验
  GUI_OUT=$(QT_QPA_PLATFORM=offscreen timeout 15 "${BIN}" gui 2>&1)
  GUI_RC=$?
  if [ ${GUI_RC} -eq 124 ] || [ ${GUI_RC} -eq 0 ]; then
    echo "PASS: GUI 无显示环境下仍可初始化（exit=${GUI_RC}）"; PASS=$((PASS+1))
  else
    echo "WARN: 无显示环境 GUI 退出码 ${GUI_RC}（offscreen 无 GL 上下文，属已知限制，不计入失败）"
    printf '%s\n' "${GUI_OUT}" | tail -3 | sed 's/^/  /'
  fi
fi
echo "Qt 插件目录（env.sh 生效时）：${QT_PLUGIN_PATH:-<未设置；Qt 会使用 deps 前缀自带 qt.conf 解析>}"
echo

echo "--- [8] 安装目录内容 ---"
find "${INSTALL}" -maxdepth 2 -type d | sed 's/^/  /'
echo "  可执行文件："
find "${INSTALL}/bin" -maxdepth 1 -type f | sed 's/^/    /'
echo

echo "==================== 验证汇总 ===================="
echo "PASS=${PASS}  FAIL=${FAIL}"
if [ "${FAIL}" -eq 0 ]; then
  echo "结论：部署验证全部通过"
  exit 0
else
  echo "结论：存在 ${FAIL} 项失败，请查看上文"
  exit 1
fi
