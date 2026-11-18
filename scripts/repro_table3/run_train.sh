#!/usr/bin/env bash
set -Eeuo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
module=train
source scripts/repro_table3/run_common.sh train "$@"
