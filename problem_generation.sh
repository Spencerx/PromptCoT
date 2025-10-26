#!/bin/bash

BASE_SEED=8000
N_RUNS=64

for i in $(seq 0 $((N_RUNS - 1))); do
    SEED=$((BASE_SEED + i))
    echo "Running inference $i with seed $SEED..."
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
python deduplicate_problems.py \
    --pattern "output/promptcot_2_0_code_problems_{}.jsonl" \
    --indices "0-${N_RUNS}" \
    --output "output/promptcot_2_0_code_problems_deduplicated.jsonl" \
    --task-type "code"

echo "Done!"