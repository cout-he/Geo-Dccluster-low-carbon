#!/usr/bin/env bash
set -Eeuo pipefail

stage=$1
shift
python_bin=${PYTHON:-python}
output_dir=outputs/table3_sac_repro
preflight_args=(--stage "$stage")
for arg in "$@"; do
    if [[ "$arg" == "--help" || "$arg" == "-h" ]]; then
        exec "$python_bin" -u -m "scripts.repro_table3.$module" "$@"
    fi
done

for ((i = 1; i <= $#; i++)); do
    arg=${!i}
    case "$arg" in
        --config-dir|--output-dir|--checkpoint|--resume|--smoke-steps)
            if (( i == $# )); then
                printf 'Missing value for %s\n' "$arg" >&2
                exit 2
            fi
            i=$((i + 1))
            value=${!i}
            preflight_args+=("$arg" "$value")
            if [[ "$arg" == "--output-dir" ]]; then
                output_dir=$value
            fi
            ;;
        --config-dir=*|--output-dir=*|--checkpoint=*|--resume=*|--smoke-steps=*)
            preflight_args+=("$arg")
            if [[ "$arg" == --output-dir=* ]]; then
                output_dir=${arg#*=}
            fi
            ;;
    esac
done

mkdir -p -- "$output_dir"
"$python_bin" -u -m scripts.repro_table3.preflight "${preflight_args[@]}" 2>&1 | tee -ia "$output_dir/run_$stage.log"
"$python_bin" -u -m "scripts.repro_table3.$module" "$@" 2>&1 | tee -ia "$output_dir/run_$stage.log"
