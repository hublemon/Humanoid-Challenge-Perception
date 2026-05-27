#!/usr/bin/env bash
set -euo pipefail

WS="${1:-/home/parkum/robotis_ros2_ws}"
USER_NAME="${SUDO_USER:-${USER}}"
GROUP_NAME="$(id -gn "${USER_NAME}")"

echo "Workspace: ${WS}"
echo "Owner    : ${USER_NAME}:${GROUP_NAME}"
echo

if [ ! -d "${WS}" ]; then
  echo "Workspace does not exist: ${WS}" >&2
  exit 1
fi

echo "[1/4] Changing ownership..."
sudo chown -R "${USER_NAME}:${GROUP_NAME}" "${WS}"

echo "[2/4] Making directories writable/searchable..."
find "${WS}" -xdev -type d -exec chmod u+rwx {} +

echo "[3/4] Making files readable/writable by owner..."
find "${WS}" -xdev -type f -exec chmod u+rw {} +

echo "[4/4] Restoring executable bits for scripts..."
find "${WS}" -xdev -type f \( -name "*.sh" -o -name "*.py" \) -exec chmod u+x {} +

echo
echo "Done."
echo "Note: read-only mounted directories such as .git/.codex/.agents may need the mount/IDE session fixed separately."
