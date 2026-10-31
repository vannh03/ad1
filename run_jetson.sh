#!/bin/bash
CURRENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$CURRENT_DIR"

if [ -f "$CURRENT_DIR/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$CURRENT_DIR/.venv/bin/activate"
fi

python3 run.py "$@"
