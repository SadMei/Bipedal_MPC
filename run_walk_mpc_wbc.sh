#!/usr/bin/env bash

set -u
set -o pipefail

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

pause_before_exit() {
  echo
  read -r -p "按回车关闭终端..." _
}

run_inside_terminal() {
  cd "$REPO_ROOT" || return 1

  echo "[0/4] Repo: $REPO_ROOT"

  if [[ ! -f build/CMakeCache.txt ]]; then
    echo "[1/4] Configuring build directory..."
    if ! cmake -S . -B build; then
      echo "CMake configure failed."
      pause_before_exit
      return 1
    fi
  fi

  echo "[2/4] Building walk_mpc_wbc..."
  if ! cmake --build build -j4; then
    echo "Build failed."
    pause_before_exit
    return 1
  fi

  echo "[3/4] Preparing runtime config..."
  mkdir -p build
  ln -sf ../common/joint_ctrl_config.json build/joint_ctrl_config.json

  echo "[4/4] Running walk_mpc_wbc..."
  cd build || return 1
  ./walk_mpc_wbc
  local status=$?

  echo
  echo "walk_mpc_wbc exited with status $status"
  pause_before_exit
  return "$status"
}

open_new_terminal() {
  if [[ -n "${DISPLAY:-}" ]] && command -v gnome-terminal >/dev/null 2>&1; then
    exec gnome-terminal -- "$SCRIPT_PATH" --inside-terminal
  fi

  if [[ -n "${DISPLAY:-}" ]] && command -v x-terminal-emulator >/dev/null 2>&1; then
    exec x-terminal-emulator -e "$SCRIPT_PATH" --inside-terminal
  fi

  echo "No GUI terminal launcher found. Running in the current shell."
  run_inside_terminal
}

if [[ "${1:-}" == "--inside-terminal" ]]; then
  run_inside_terminal
else
  open_new_terminal
fi
