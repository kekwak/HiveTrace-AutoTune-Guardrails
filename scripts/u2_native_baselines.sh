#!/usr/bin/env bash
# Run on the RTX pod from a checkout of research/quantization.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-/workspace/u2-hf}
: "${U2_NATIVE_ROOT:=/workspace/u2-native}"
: "${U2_NATIVE_ENV:=$U2_NATIVE_ROOT/env}"
: "${U2_QUALITY_CORPUS:=$U2_NATIVE_ROOT/quality}"
PY="$U2_NATIVE_ENV/venv/bin/python"
DATA="$U2_NATIVE_ROOT/data"
CONFIGS="$U2_NATIVE_ROOT/configs"
RUNS="$U2_NATIVE_ROOT/runs"
BENCH=( "$PY" tools/benchmark.py )

prepare() {
    mkdir -p "$DATA" "$CONFIGS"
    if [[ ! -f "$U2_NATIVE_ENV/setup/status.json" ]]; then
        uv python install 3.13
        uv run --no-project --python 3.13 --with pip python tools/benchmark.py setup \
            --profile team-rtx --environment "$U2_NATIVE_ENV"
    fi
    "$PY" -c 'import json,sys; sys.exit(json.load(open(sys.argv[1]))["status"] != "passed")' \
        "$U2_NATIVE_ENV/setup/status.json"
    if [[ ! -f "$DATA/role-corpus/manifest.json" ]]; then
        "${BENCH[@]}" fetch-data --output "$DATA/role-corpus"
    fi
    "$PY" -m u2_quant.native_baselines reindex-spec --source "$DATA/role-corpus" \
        --out "$CONFIGS/reindex.json"
    if [[ ! -f "$DATA/speed/manifest.json" ]]; then
        "${BENCH[@]}" prepare-data --spec "$CONFIGS/reindex.json" --output "$DATA/speed"
    fi
    if [[ ! -f "$DATA/quality-gliner/manifest.json" ]]; then
        if [[ ! -f "$DATA/quality-gliner-src/import.json" ]]; then
            "$PY" -m u2_quant.gliner_corpus "$U2_QUALITY_CORPUS/corpus.jsonl" \
                "$DATA/quality-gliner-src"
        fi
        "${BENCH[@]}" prepare-data --spec "$DATA/quality-gliner-src/import.json" \
            --output "$DATA/quality-gliner"
    fi
    "$PY" -m u2_quant.native_baselines configs --environment "$U2_NATIVE_ENV" \
        --speed "$DATA/speed" --quality "$U2_QUALITY_CORPUS" \
        --gliner-quality "$DATA/quality-gliner" --out "$CONFIGS"
}

run_stage() {
    local model=$1 stage=$2 command=$3 config=$4
    local target="$RUNS/$model-$stage" attempt
    if [[ -f "$target.zip" ]] && \
        "$PY" -m u2_quant.native_baselines verify "$target.zip" "$stage" "$config"; then
        printf 'REUSE %s %s\n' "$model" "$stage"
        return
    fi
    if [[ -e "$target" || -e "$target.zip" ]]; then
        attempt=$(mktemp -d "$RUNS/$model-$stage-attempt.XXXXXX")
        [[ ! -e "$target" ]] || mv "$target" "$attempt/"
        [[ ! -e "$target.zip" ]] || mv "$target.zip" "$attempt/"
    fi
    printf 'START %s %s %s\n' "$model" "$stage" "$(date -u +%FT%TZ)"
    "${BENCH[@]}" "$command" --config "$config" --environment "$U2_NATIVE_ENV" --output "$target" &&
        "$PY" -m u2_quant.native_baselines verify "$target.zip" "$stage" "$config" || {
        printf 'FAILED %s %s\n' "$model" "$stage" >&2
        return 1
    }
    printf 'DONE %s %s %s\n' "$model" "$stage" "$(date -u +%FT%TZ)"
}

case "${1:-}" in
    prepare) prepare ;;
    smoke|run)
        mode=$1
        shift
        if [[ $# -eq 0 ]]; then set -- laya gliner koala llama halo shield hive; fi
        for model in "$@"; do
            case "$model" in laya|gliner|koala|llama|halo|shield|hive) ;;
                *) printf 'Unknown model: %s\n' "$model" >&2; exit 2 ;;
            esac
        done
        mkdir -p "$RUNS"
        failed=0
        for model in "$@"; do
            speed="$CONFIGS/$model-baseline.json"
            run_stage "$model" preflight preflight "$speed" || { failed=1; continue; }
            run_stage "$model" quality-smoke quality "$CONFIGS/$model-quality-smoke.json" ||
                { failed=1; continue; }
            if [[ "$mode" == run ]]; then
                run_stage "$model" baseline run "$speed" || failed=1
                run_stage "$model" quality quality "$CONFIGS/$model-quality.json" || failed=1
            fi
        done
        exit "$failed"
        ;;
    *) printf 'Usage: %s prepare | smoke [model...] | run [model...]\n' "$0" >&2; exit 2 ;;
esac
