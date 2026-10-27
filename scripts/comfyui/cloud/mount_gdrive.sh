#!/usr/bin/env bash
set -euo pipefail

MOUNT_DIR="${MOUNT_DIR:-$HOME/google_ai_cloud}"
REMOTE="${REMOTE:-gdrive:m2a_cloud}"
CACHE_DIR="${CACHE_DIR:-/run/media/hihry/OS/.cache/rclone}"
RCLONE_BIN="${RCLONE_BIN:-$HOME/.local/bin/rclone}"

case "${1:-mount}" in
  mount|start)
    mkdir -p "$MOUNT_DIR" "$CACHE_DIR"
    if mountpoint -q "$MOUNT_DIR"; then
      echo "[m2a-cloud] Already mounted at $MOUNT_DIR"
      exit 0
    fi
    if systemctl --user is-enabled rclone-m2a-cloud.service >/dev/null 2>&1; then
      systemctl --user start rclone-m2a-cloud.service
      echo "[m2a-cloud] Started systemd user service rclone-m2a-cloud.service -> $MOUNT_DIR"
    else
      "$RCLONE_BIN" mount "$REMOTE" "$MOUNT_DIR" \
        --daemon \
        --vfs-cache-mode writes \
        --cache-dir "$CACHE_DIR" \
        --vfs-cache-max-size 15G \
        --vfs-cache-max-age 1h \
        --dir-cache-time 72h \
        --poll-interval 15s \
        --buffer-size 32M \
        --vfs-read-chunk-size 64M \
        --vfs-read-chunk-size-limit 1G
      echo "[m2a-cloud] Mounted $REMOTE at $MOUNT_DIR"
    fi
    ;;
  unmount|stop)
    if systemctl --user is-active rclone-m2a-cloud.service >/dev/null 2>&1; then
      systemctl --user stop rclone-m2a-cloud.service
    elif mountpoint -q "$MOUNT_DIR"; then
      fusermount3 -uz "$MOUNT_DIR"
    fi
    echo "[m2a-cloud] Unmounted $MOUNT_DIR"
    ;;
  status)
    if mountpoint -q "$MOUNT_DIR"; then
      echo "[m2a-cloud] ACTIVE: $REMOTE is mounted at $MOUNT_DIR"
      df -h "$MOUNT_DIR"
    else
      echo "[m2a-cloud] INACTIVE: $MOUNT_DIR is not mounted"
    fi
    ;;
  *)
    echo "Usage: $0 {mount|unmount|status}" >&2
    exit 1
    ;;
esac
