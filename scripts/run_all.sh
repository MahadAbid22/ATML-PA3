#!/usr/bin/env bash
# Run every experiment of the assignment. Two lanes run side by side:
#   GPU lane: all GPU work strictly one job at a time (so reported times / peak memory are not distorted)
#   CPU lane: CPU/network-only preparation (Task 2 data + image download, Task 3 offline analysis)
# Every stage writes a marker when it succeeds and is skipped on later invocations, and every training
# stage resumes from its last checkpoint, so after an interruption simply run this script again.
#   bash scripts/run_all.sh            (add OPTIONAL=0 to skip the optional 49-token Task 2 run)
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PA3_OUTPUTS="${PA3_OUTPUTS:-$PWD/outputs}"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_SYMLINKS_WARNING=1 PYTHONUNBUFFERED=1
OPTIONAL="${OPTIONAL:-1}"
M="$PA3_OUTPUTS/markers"
L="results/logs"
mkdir -p "$M" "$L"

stage() {  # stage <name> <command...>  (one automatic retry: training resumes from its checkpoint)
  local name=$1; shift
  [ -f "$M/$name.done" ] && return 0
  rm -f "$M/$name.failed"
  for attempt in 1 2; do
    echo "[$(date '+%F %T')] START $name (attempt $attempt)" | tee -a "$L/pipeline.log"
    if "$@" >> "$L/$name.log" 2>&1; then
      touch "$M/$name.done"
      echo "[$(date '+%F %T')] DONE  $name" | tee -a "$L/pipeline.log"
      return 0
    fi
    echo "[$(date '+%F %T')] FAIL  $name (see $L/$name.log)" | tee -a "$L/pipeline.log"
  done
  touch "$M/$name.failed"
  return 1
}

wait_for() {  # wait until another lane's stage has finished (or failed)
  while [ ! -f "$M/$1.done" ]; do
    [ -f "$M/$1.failed" ] && return 1
    sleep 30
  done
}

cpu_lane() {
  stage t2_annotations python -m task2_vlm.download --annotations \
    && stage t2_data python -m task2_vlm.data \
    && stage t2_images python -m task2_vlm.download --images --workers 16
  stage t3_data python -m task3_ttc.data \
    && stage t3_selection python -m task3_ttc.selection \
    && stage t3_signals python -m task3_ttc.signals \
    && stage t3_adaptive python -m task3_ttc.adaptive
  touch "$M/cpu_lane.finished"
}

gpu_lane() {
  python -c "from common.utils import environment_info, save_json, RESULTS; save_json(environment_info(), RESULTS / 'environment.json')"
  # ---------------- Task 1
  stage t1_data python -m task1_finetune.data || return 1
  stage t1_probes python -m task1_finetune.diagnostics --make-probes
  for m in full_ft l2sp lora_r4 lora_r16 oplora_r16; do
    stage "t1_train_$m" python -m task1_finetune.train --method "$m" || continue
    conds="$m"; [ "$m" = full_ft ] && conds="full_ft_mid full_ft"
    ok=1
    for c in $conds; do stage "t1_export_$c" python -m task1_finetune.export --condition "$c" || ok=0; done
    [ $ok = 1 ] && stage "t1_cleanup_$m" python -m task1_finetune.train --method "$m" --cleanup
  done
  for c in baseline full_ft_mid full_ft l2sp lora_r4 lora_r16 oplora_r16; do
    stage "t1_eval_$c" python -m task1_finetune.evaluate --condition "$c"
    stage "t1_qual_$c" python -m task1_finetune.qualitative --condition "$c"
  done
  stage t1_qual_collect python -m task1_finetune.qualitative --collect
  stage t1_diagnostics python -m task1_finetune.diagnostics --run
  stage t1_analyze python -m task1_finetune.analyze
  touch "$M/task1.finished"
  # ---------------- Task 2
  if wait_for t2_images; then
    stage t2_features python -m task2_vlm.features \
      && stage t2_train python -m task2_vlm.train \
      && stage t2_eval python -m task2_vlm.evaluate \
      && stage t2_analyze python -m task2_vlm.analyze
  fi
  touch "$M/task2.finished"
  # ---------------- Task 3 (live generation needs the split file from t3_data)
  wait_for t3_data && stage t3_live python -m task3_ttc.live
  while [ ! -f "$M/cpu_lane.finished" ]; do sleep 30; done
  stage t3_analyze python -m task3_ttc.analyze
  touch "$M/task3.finished"
  # ---------------- optional: 49 visual tokens (2x2-pooled), same recipe
  if [ "$OPTIONAL" = 1 ] && [ -f "$M/t2_features.done" ]; then
    stage t2_train_49 python -m task2_vlm.train --tokens 49 \
      && stage t2_eval_49 python -m task2_vlm.evaluate --tokens 49 \
      && stage t2_analyze_49 python -m task2_vlm.analyze --tokens 49
  fi
  touch "$M/all.finished"
}

echo "[$(date '+%F %T')] pipeline start on $(nvidia-smi --query-gpu=name --format=csv,noheader)" | tee -a "$L/pipeline.log"
cpu_lane &
gpu_lane
wait
echo "[$(date '+%F %T')] pipeline finished" | tee -a "$L/pipeline.log"
