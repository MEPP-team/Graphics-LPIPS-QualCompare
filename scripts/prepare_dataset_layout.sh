#!/usr/bin/env bash
# Expose the QualCompare rendered dataset in the Source/<N>VP + Distorted/<N>VP
# layout expected by Light_GraphicsLPIPS_csv.py, train.py and correlation_VP.py.
#
# The published rendered dataset extracts to a flat, per-dataset layout:
#     <DATASET_ROOT>/<NAME>_source/<REF_OBJECT>/{views,masks,patchs}
#     <DATASET_ROOT>/<NAME>_distorted/<DISTORTED_OBJECT>/{views,masks}
# but the tools expect:
#     <SRC_ROOT>/Source/<N>VP/...   and   <SRC_ROOT>/Distorted/<N>VP/...
#
# This script creates symlinks (no copy). Two layouts:
#   default   -> <OUT_ROOT>/<DB>/Source/<N>VP           (use: --src_root <OUT_ROOT>/<DB>)
#   --forbat  -> <OUT_ROOT>/<DB>/<RENDER>/<VIEW>/Source/<N>VP  (for the .bat presets;
#                then: export QUALCOMPARE_OUT_ROOT=<OUT_ROOT>)
#
# Usage:
#   scripts/prepare_dataset_layout.sh <DATASET_ROOT> [OUT_ROOT] [--forbat] [--remove]
set -euo pipefail

DATASET_ROOT="${1:?Usage: prepare_dataset_layout.sh <DATASET_ROOT> [OUT_ROOT] [--forbat] [--remove]}"
OUT_ROOT="$DATASET_ROOT/_run"
FORBAT=0
REMOVE=0
shift
while [ $# -gt 0 ]; do
  case "$1" in
    --forbat) FORBAT=1 ;;
    --remove) REMOVE=1 ;;
    *)        OUT_ROOT="$1" ;;
  esac
  shift
done

# DB:ARCHIVE_BASENAME:NUM_VIEWS  (matches dataset_info.json v1.1)
maps=(
  "TMQ:TMQ_Circle_0.3_8VP:8"
  "TSMD:TSMD_Circle_0.3_8VP:8"
  "SJTU-TMQA:SJTU-TMQA_Circle_0_8VP:8"
  "BASICS:BASICS_Circle_0.3_8VP_r003:8"
  "WPC:WPC_Circle_0.3_8VP_r001:8"
)

# DB -> "RENDER/VIEW[ RENDER/VIEW ...]" labels used by the paper_revalidation .bat presets.
bat_labels() {
  case "$1" in
    TMQ)        echo "New_Render/Y_fixed_0.3" ;;
    TSMD)       echo "New_Render/Y_fixed_0.3" ;;
    SJTU-TMQA)  echo "0_0_light/Y_fixed_0" ;;
    BASICS)     echo "SP_960x960/Y_fixed_0.3" ;;
    WPC)        echo "SP_960x960/Y_fixed_0.3" ;;
    *)          echo "" ;;
  esac
}

make_pair() { # base_dir vp src dis
  local base="$1" vp="$2" src="$3" dis="$4"
  if [ "$REMOVE" = "1" ]; then
    [ -L "$base/Source/$vp" ] && rm -f "$base/Source/$vp"
    [ -L "$base/Distorted/$vp" ] && rm -f "$base/Distorted/$vp"
    return
  fi
  mkdir -p "$base/Source" "$base/Distorted"
  ln -sfn "$src" "$base/Source/$vp"
  ln -sfn "$dis" "$base/Distorted/$vp"
}

for m in "${maps[@]}"; do
  IFS=: read -r db base views <<< "$m"
  src="$DATASET_ROOT/${base}_source"
  dis="$DATASET_ROOT/${base}_distorted"
  vp="${views}VP"

  if [ "$REMOVE" != "1" ] && { [ ! -d "$src" ] || [ ! -d "$dis" ]; }; then
    echo "WARN  $db: extracted archives not found (${base}_source / ${base}_distorted). Skipped." >&2
    continue
  fi

  if [ "$FORBAT" = "1" ]; then
    for rv in $(bat_labels "$db"); do
      base_dir="$OUT_ROOT/$db/$rv"
      make_pair "$base_dir" "$vp" "$src" "$dis"
      [ "$REMOVE" = "1" ] || echo "[ok] $db  $rv  --src_root \"$base_dir\""
    done
  else
    base_dir="$OUT_ROOT/$db"
    make_pair "$base_dir" "$vp" "$src" "$dis"
    [ "$REMOVE" = "1" ] || echo "[ok] $db  --src_root \"$base_dir\""
  fi
  [ "$REMOVE" = "1" ] && echo "[removed] $db"
done

if [ "$REMOVE" != "1" ]; then
  echo "Layout ready under: $OUT_ROOT"
  [ "$FORBAT" = "1" ] && echo "For the .bat presets: export QUALCOMPARE_OUT_ROOT=\"$OUT_ROOT\""
fi
