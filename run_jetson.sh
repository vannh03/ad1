#!/bin/bash
CURRENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$CURRENT_DIR"

# Tự động kích hoạt môi trường ảo Python (.venv) nếu đã được tạo bởi setup_jetson.sh
if [ -f "$CURRENT_DIR/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$CURRENT_DIR/.venv/bin/activate"
fi

# Tự động nạp các biến cấu hình từ file .env nếu có
if [ -f "$CURRENT_DIR/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source <(grep -v '^#' "$CURRENT_DIR/.env" | sed 's/\r$//')
    set +a
fi

python3 run.py "$@"
