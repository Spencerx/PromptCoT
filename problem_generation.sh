#!/bin/bash

# Problem generation pipeline: run multiple inference seeds, then merge + deduplicate.
#
# Knobs:
# - BASE_SEED: first seed; run i uses seed = BASE_SEED + i.
# - N_RUNS: number of inference runs (and output shards).
#
# Notes:
# - In deduplicate_problems.py, `--indices` uses end-exclusive ranges (0-64 means 0..63).
# - `VLLM_WORKER_MULTIPROC_METHOD=spawn` is recommended for vLLM/torch multiprocessing.

BASE_SEED=8000
N_RUNS=64

for i in $(seq 0 $((N_RUNS - 1))); do
    SEED=$((BASE_SEED + i))
    echo "Running inference $i with seed $SEED..."
    # infer_split_merge.py:
    # - Input: JSONL with a `prompt` field per line (--data_path).
    # - Output: JSONL with an added `completion` field per line (--output_path).
    VLLM_WORKER_MULTIPROC_METHOD=spawn python infer_split_merge.py \
        --data_path "PromptCoT-2.0-Concepts/code.jsonl" \
        --output_path "output/promptcot_2_0_code_problems_${i}.jsonl" \
        --model_path "/path/to/the/problem_generation_model" \
        --n_gpus 8 \
        --n_splits 2 \
        --temperature 0.8 \
        --max_len 4096 \
        --expected_runs 1 \
        --seed $SEED
done

echo "Deduplicating problems..."
#
# deduplicate_problems.py:
# - Reads `completion` fields from the generated shards.
# - Extracts problem/rationale blocks and writes deduplicated `{"prompt": ...}` JSONL.
python deduplicate_problems.py \
    --pattern "output/promptcot_2_0_code_problems_{}.jsonl" \
    --indices "0-${N_RUNS}" \
    --output "output/promptcot_2_0_code_problems_deduplicated.jsonl" \
    --task-type "code"

echo "Done!"
