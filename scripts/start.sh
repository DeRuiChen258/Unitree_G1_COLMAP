#!/usr/bin/env bash
# ==============================================================================
# start.sh —— 实验工程侧「一键启动」：加载本地 COLMAP（CUDA）并打开指定重建结果
#
# 位置：$WORKSPACE
# 部署：COLMAP 本体在 $COLMAP_ROOT（可用 COLMAP_ROOT 覆盖）
#
# 用法：
#   bash start.sh                 # 默认打开机器人加密结果（稠密 809,590 点）
#   bash start.sh dense           # 稠密点云模型（含 120 相机位姿）
#   bash start.sh sparse-robot    # 加密后的稀疏模型（108,261 点 / 120 视角）
#   bash start.sh sparse-video    # 视频重建稀疏模型（32,955 点 / 96 视角）
#   bash start.sh orbit           # 环绕数据集稀疏模型（16,075 点 / 36 视角）
#   bash start.sh ply             # 用系统里可用的方式打开稠密点云 PLY
#   bash start.sh raw             # 空 GUI（自行新建/导入项目）
#   bash start.sh cli             # 只加载环境并自检，不开窗口
#   bash start.sh --list          # 列出可打开的模型
#
# 实测的两个坑（脚本已处理并在 GUI 启动后给出指引）：
#   1) COLMAP 4.x 的 --project_path 必须是 **project.ini 文件路径**（给目录会 0 Points）；
#   2) 4.2 的启动参数 --import_path **不会**导入模型（多种模型实测均 0 Points），
#      需要在 GUI 里 File → Import model…（Ctrl+I）选一次模型目录。
# ==============================================================================
set -euo pipefail

TEST_ROOT="${WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
COLMAP_ROOT="${COLMAP_ROOT:-$HOME/colmap-deploy}"
COLMAP_BIN="${COLMAP_ROOT}/install/bin/colmap"

if [ ! -x "${COLMAP_BIN}" ]; then
  echo "[start] 找不到 COLMAP：${COLMAP_BIN}" >&2
  echo "        请先部署（bash ${COLMAP_ROOT}/deploy_colmap.sh）或用 COLMAP_ROOT=... 指定" >&2
  exit 1
fi

# ---- 模型注册表：key → 模型目录（目录内需有 cameras/images/points3D）----------
declare -A MODELS=(
  [dense]="${TEST_ROOT}/py_out/robot_close/dense_cli"
  [sparse-robot]="${TEST_ROOT}/py_out/robot_close/model"
  [sparse-video]="${TEST_ROOT}/py_out/movingcam/model"
  [orbit]="${TEST_ROOT}/orbit_multiview/sparse/0"
)
declare -A DESCS=(
  [dense]="机器人稠密点云：809,590 点（含 120 个相机位姿）"
  [sparse-robot]="机器人加密稀疏：108,261 点 / 120 视角 / 重投影 0.236 px"
  [sparse-video]="视频重建稀疏：32,955 点 / 96 视角 / 重投影 0.339 px"
  [orbit]="环绕数据集稀疏：16,075 点 / 36 视角 / 重投影 0.170 px"
)
DENSE_PLY="${TEST_ROOT}/py_out/robot_close/dense/fused.ply"

usage() { sed -n '5,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

case "${1:-}" in
  --list|-l)
    echo "可用模型："
    for k in dense sparse-robot sparse-video orbit; do
      printf "  %-13s %s\n            %s\n" "${k}" "${DESCS[$k]}" "${MODELS[$k]}"
    done
    printf "  %-13s %s\n" "ply" "稠密点云 PLY：${DENSE_PLY}"
    printf "  %-13s %s\n" "raw" "空 GUI，自行新建/导入"
    exit 0
    ;;
  -h|--help) usage; exit 0 ;;
esac

# ---- 环境 ----------------------------------------------------------------
# shellcheck disable=SC1091
source "${COLMAP_ROOT}/env.sh" --quiet
export DISPLAY="${DISPLAY:-:0}"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"

if [ "${1:-}" = "cli" ]; then
  echo "[start] 环境已加载"
  echo "  colmap : ${COLMAP_BIN}"
  "${COLMAP_BIN}" --version
  echo "  deps   : ${COLMAP_ROOT}/deps"
  echo "  工程根 : ${TEST_ROOT}"
  exit 0
fi

if [ "${1:-}" = "ply" ]; then
  echo "[start] 稠密点云：${DENSE_PLY}"
  for v in CloudCompare cloudcompare meshlab; do
    if command -v "$v" >/dev/null; then echo "[start] 用 $v 打开"; exec "$v" "${DENSE_PLY}"; fi
  done
  echo "[start] 未安装外部点云查看器，改用 COLMAP 导入："
  echo "        File → Import model from… → 选 ${DENSE_PLY}"
  set -- dense
fi

MODE="${1:-dense}"

# ---- 生成/校验 project.ini ------------------------------------------------
ensure_project_ini() {
  local dir="$1" ini="$1/project.ini" parent db img
  [ -f "${ini}" ] && return 0
  parent="$(dirname "${dir}")"
  db="${parent}/database.db"; img="${parent}/images"
  [ -d "${img}" ] || img="${parent}/../images"
  {
    echo "log_color=true"; echo "log_level=0"; echo "log_severity=0"; echo "log_target=stderr_and_file"
    echo "log_path="
    if [ -f "${db}" ]; then echo "database_path=${db}"; else echo "database_path=${dir}/empty.db"; : > "${dir}/empty.db"; fi
    if [ -d "${img}" ]; then echo "image_path=$(cd "${img}" && pwd)"; else echo "image_path=${dir}"; fi
    echo "[Mapper]"; echo "extract_colors=true"
  } > "${ini}"
  echo "[start] 已为 ${dir} 生成 project.ini"
}

cleanup_old() { pkill -x colmap 2>/dev/null || true; sleep 1; }

if [ "${MODE}" = "raw" ]; then
  cleanup_old
  echo "[start] 启动空白 COLMAP GUI"
  setsid nohup "${COLMAP_BIN}" gui > "${TEST_ROOT}/logs/gui_start.log" 2>&1 < /dev/null &
  sleep 6
  pgrep -x colmap >/dev/null && echo "[start] GUI 已启动" || { echo "[start] 启动失败，见 logs/gui_start.log" >&2; exit 1; }
  exit 0
fi

MODEL_DIR="${MODELS[${MODE}]:-}"
if [ -z "${MODEL_DIR}" ]; then echo "[start] 未知模式：${MODE}" >&2; usage >&2; exit 2; fi
if [ ! -f "${MODEL_DIR}/points3D.bin" ] && [ ! -f "${MODEL_DIR}/points3D.txt" ]; then
  echo "[start] 模型不存在或不完整：${MODEL_DIR}" >&2; exit 2
fi
mkdir -p "${TEST_ROOT}/logs"
ensure_project_ini "${MODEL_DIR}"

N_POINTS=$("${COLMAP_BIN}" model_analyzer --path "${MODEL_DIR}" 2>&1 | awk '/Points:/{print $NF}')
N_IMAGES=$("${COLMAP_BIN}" model_analyzer --path "${MODEL_DIR}" 2>&1 | awk '/Images:/{print $NF}')

# ---- 启动 GUI -------------------------------------------------------------
cleanup_old
echo "[start] 模型  : ${MODE} —— ${DESCS[${MODE}]:-}"
echo "[start] 目录  : ${MODEL_DIR}"
echo "[start] 规模  : ${N_IMAGES} 图像 / ${N_POINTS} 点"
setsid nohup "${COLMAP_BIN}" gui \
  --project_path "${MODEL_DIR}/project.ini" \
  --import_path "${MODEL_DIR}" \
  > "${MODEL_DIR}/gui.log" 2>&1 < /dev/null &
sleep 8
if ! pgrep -x colmap >/dev/null; then
  echo "[start] GUI 启动失败，见 ${MODEL_DIR}/gui.log" >&2; exit 1
fi

cat <<EOF

────────────────────────────────────────────────────────────────
[start] COLMAP GUI 已启动（若没看到窗口：Alt+Tab 或点任务栏 COLMAP）

  ⚠ 本版 COLMAP 4.2 的启动参数不会自动加载模型（会显示 0 Points），
    请在窗口里做一步载入：

        File → Import model…   （或按 Ctrl+I）
        粘贴下面这行路径后回车：

        ${MODEL_DIR}

  只看稠密点云（可选，更省事）：
        File → Import model from…  → 选
        ${DENSE_PLY}
────────────────────────────────────────────────────────────────
EOF
