#!/bin/bash
set -eu
while true; do
 state=$(systemctl --user show qda496-optimize-bench.service -p ActiveState --value 2>/dev/null || true)
 if [ "$state" != "active" ] && [ "$state" != "activating" ]; then break; fi
 sleep 2
done
export PYTHONPATH=/data/work/asr-pilot/VibeVoice
export HF_HOME=/data/work/asr-pilot/hf
export OMP_NUM_THREADS=2
for gpu in 0 1; do
 bucket=3600; tokens=65147
 if [ "$gpu" = 1 ]; then bucket=1800; tokens=36000; fi
 CUDA_VISIBLE_DEVICES=$gpu /data/work/asr-pilot/venv/bin/python -u /data/work/asr-pilot/benchmark.py --variant new --bucket "$bucket" --stress --stress-tokens "$tokens" > "/data/work/asr-optimization/stress-$gpu.log" 2>&1 &
 pids[$gpu]=$!
done
result=0
for pid in "${pids[@]}"; do wait "$pid" || result=1; done
exit "$result"
