#!/bin/bash
set -eu
export PYTHONPATH=/data/work/asr-pilot/VibeVoice
export HF_HOME=/data/work/asr-pilot/hf
export OMP_NUM_THREADS=2
while systemctl --user is-active --quiet qda496-training.service; do sleep 2; done
mkdir -p /data/work/asr-optimization
for bucket in 1800 3600; do
 CUDA_VISIBLE_DEVICES=0 /data/work/asr-pilot/venv/bin/python -u /data/work/asr-pilot/benchmark.py --variant old --bucket "$bucket" > "/data/work/asr-optimization/old-$bucket.log" 2>&1 &
 a=$!
 CUDA_VISIBLE_DEVICES=1 /data/work/asr-pilot/venv/bin/python -u /data/work/asr-pilot/benchmark.py --variant new --bucket "$bucket" > "/data/work/asr-optimization/new-$bucket.log" 2>&1 &
 b=$!
 result=0
 wait "$a" || result=1
 wait "$b" || result=1
 if [ "$result" != 0 ]; then exit 1; fi
done
