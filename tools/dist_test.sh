#!/bin/bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 NUM_PROC MASTER_PORT -e CONFIG --gpu GPU_IDS [OPTIONS...]" >&2
    exit 2
fi

NUM_PROC=$1
MASTER_PORT=$2
shift 2
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"
exec torchrun --nproc_per_node="$NUM_PROC" --master_port="$MASTER_PORT" test.py "$@"
