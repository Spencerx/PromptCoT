import argparse
import json
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import math
from typing import List, Dict, Any
import time

from env_config import load_env, env_bool, env_float, env_int, env_str


"""
Run vLLM inference over a JSONL dataset.

Input JSONL: each line is a JSON object containing `prompt`.
Output JSONL: each line is the original object plus `completion`.
Uses a split/merge multiprocessing strategy; each worker binds to its assigned GPUs via CUDA_VISIBLE_DEVICES.
"""


def process_split_worker(split_id: int, data_split: List[Dict], args_dict: Dict, gpu_ids: List[int]) -> List[Dict]:
    """Worker function that sets GPU environment before importing CUDA libraries"""

    # Set CUDA_VISIBLE_DEVICES BEFORE importing torch/vllm
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))

    # Now import CUDA-dependent libraries
    import torch
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print(f"Split {split_id}: Processing {len(data_split)} items on GPUs {gpu_ids}")
    print(f"Split {split_id}: CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"Split {split_id}: Available CUDA devices: {torch.cuda.device_count()}")

    # Reconstruct args from dict
    args = argparse.Namespace(**args_dict)

    # Load tokenizer
    tokenizer_path = args.tokenizer_path if args.tokenizer_path else args.model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Use all visible GPUs for tensor parallelism
    tensor_parallel_size = len(gpu_ids)

    if args.factor > 1.0:
        hf_overrides = {
            "rope_scaling": {
                "rope_type": "yarn",
                "factor": args.factor,
                "original_max_position_embeddings": args.original_max_position_embeddings,
            },
            "max_model_len": int(args.original_max_position_embeddings * args.factor),
        }
    else:
        hf_overrides = {}

    # Load model
    if not args.use_mamba2:
        model = LLM(
            model=args.model_path,
            tokenizer=tokenizer_path,
            tokenizer_mode="slow",
            dtype=args.dtype,
            tensor_parallel_size=tensor_parallel_size,
            enforce_eager=True,
            disable_custom_all_reduce=True,
            gpu_memory_utilization=args.gpu_memory_utilization,
            trust_remote_code=False,
            hf_overrides=hf_overrides,
        )
    else:
        model = LLM(
            model=args.model_path,
            tokenizer=tokenizer_path,
            max_model_len=args.max_len + 4096,
            tokenizer_mode="slow",
            dtype=args.dtype,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_num_seqs=args.max_num_seqs,
            trust_remote_code=False,
            hf_overrides=hf_overrides,
        )

    # Setup sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_len,
        min_tokens=args.min_len,
        presence_penalty=args.presence_penalty,
        frequency_penalty=args.frequency_penalty,
        repetition_penalty=args.repetition_penalty,
        seed=None if args.seed == -1 else args.seed,
    )

    # Prepare prompts
    prompts = []
    for item in data_split:
        prompt = item["prompt"]
        if args.use_chat_template:
            messages = [{"role": "user", "content": prompt}]
            if args.reasoning_effort is not None:
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, reasoning_effort=args.reasoning_effort)
            else:
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)

    # Generate completions
    with torch.no_grad():
        completions = model.generate(prompts, sampling_params)
        completions = [completion.outputs[0].text for completion in completions]

    # Add completions to items
    results = []
    for item, completion in zip(data_split, completions):
        result_item = item.copy()
        result_item["completion"] = completion
        result_item["split_id"] = split_id
        results.append(result_item)

    print(f"Split {split_id}: Completed processing {len(results)} items")
    return results


def get_available_gpus():
    """Get list of available GPU IDs"""
    import torch
    return list(range(torch.cuda.device_count()))


def split_data(data: List[Dict], n_splits: int) -> List[List[Dict]]:
    """Split data into roughly equal chunks"""
    chunk_size = math.ceil(len(data) / n_splits)
    return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]


def distribute_gpus(total_gpus: int, n_splits: int) -> List[List[int]]:
    """Distribute GPUs among splits as evenly as possible"""
    if n_splits > total_gpus:
        gpu_assignments = []
        for i in range(n_splits):
            gpu_assignments.append([i % total_gpus])
        return gpu_assignments

    gpus_per_split = total_gpus // n_splits
    remaining_gpus = total_gpus % n_splits

    gpu_assignments = []
    gpu_idx = 0

    for split_idx in range(n_splits):
        split_gpus = list(range(gpu_idx, gpu_idx + gpus_per_split))
        gpu_idx += gpus_per_split

        if split_idx < remaining_gpus:
            split_gpus.append(gpu_idx)
            gpu_idx += 1

        gpu_assignments.append(split_gpus)

    return gpu_assignments


def merge_results(split_results: List[List[Dict]], output_path: str):
    """Merge results from all splits and write to output file"""
    print("Merging results from all splits...")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    all_results = []
    for split_result in split_results:
        all_results.extend(split_result)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in all_results:
            item_copy = item.copy()
            item_copy.pop("split_id", None)
            f.write(json.dumps(item_copy) + "\n")

    print(f"Merged {len(all_results)} results to {output_path}")


def main():
    # Import torch here in main to check available GPUs
    import torch
    from str2bool import str2bool

    load_env()

    # Namespaced env vars avoid collisions across scripts; unprefixed vars remain supported for compatibility.
    data_path_default = env_str("SPLIT_MERGE_DATA_PATH") or env_str("DATA_PATH")
    output_path_default = env_str("SPLIT_MERGE_OUTPUT_PATH") or env_str("OUTPUT_PATH")
    model_path_default = env_str("SPLIT_MERGE_MODEL_PATH") or env_str("MODEL_PATH")
    tokenizer_path_default = (
        env_str("SPLIT_MERGE_TOKENIZER_PATH")
        or env_str("TOKENIZER_PATH", default=None)
    )

    dtype_default = env_str("SPLIT_MERGE_DTYPE") or env_str("DTYPE", default="bfloat16")
    n_gpus_default = env_int("SPLIT_MERGE_N_GPUS", default=None)
    if n_gpus_default is None:
        n_gpus_default = env_int("N_GPUS", default=8)

    n_splits_default = env_int("SPLIT_MERGE_N_SPLITS", default=None)
    if n_splits_default is None:
        n_splits_default = env_int("N_SPLITS", default=None)
    if n_splits_default is None:
        n_splits_default = env_int("NUM_SPLITS", default=4)

    temperature_default = env_float("SPLIT_MERGE_TEMPERATURE", default=None)
    if temperature_default is None:
        temperature_default = env_float("TEMPERATURE", default=0.0)

    top_p_default = env_float("SPLIT_MERGE_TOP_P", default=None)
    if top_p_default is None:
        top_p_default = env_float("TOP_P", default=1.0)

    top_k_default = env_int("SPLIT_MERGE_TOP_K", default=None)
    if top_k_default is None:
        top_k_default = env_int("TOP_K", default=-1)

    presence_penalty_default = env_float("SPLIT_MERGE_PRESENCE_PENALTY", default=None)
    if presence_penalty_default is None:
        presence_penalty_default = env_float("PRESENCE_PENALTY", default=0.0)

    frequency_penalty_default = env_float("SPLIT_MERGE_FREQUENCY_PENALTY", default=None)
    if frequency_penalty_default is None:
        frequency_penalty_default = env_float("FREQUENCY_PENALTY", default=0.0)

    repetition_penalty_default = env_float("SPLIT_MERGE_REPETITION_PENALTY", default=None)
    if repetition_penalty_default is None:
        repetition_penalty_default = env_float("REPETITION_PENALTY", default=1.0)

    min_len_default = env_int("SPLIT_MERGE_MIN_LEN", default=None)
    if min_len_default is None:
        min_len_default = env_int("MIN_LEN", default=0)

    max_len_default = env_int("SPLIT_MERGE_MAX_LEN", default=None)
    if max_len_default is None:
        max_len_default = env_int("MAX_LEN", default=2048)

    use_chat_template_default = env_bool("SPLIT_MERGE_USE_CHAT_TEMPLATE", default=None)
    if use_chat_template_default is None:
        use_chat_template_default = env_bool("USE_CHAT_TEMPLATE", default=False)

    seed_default = env_int("SPLIT_MERGE_SEED", default=None)
    if seed_default is None:
        seed_default = env_int("SEED", default=-1)

    use_mamba2_default = env_bool("SPLIT_MERGE_USE_MAMBA2", default=None)
    if use_mamba2_default is None:
        use_mamba2_default = env_bool("USE_MAMBA2", default=False)

    gpu_memory_utilization_default = env_float("SPLIT_MERGE_GPU_MEMORY_UTILIZATION", default=None)
    if gpu_memory_utilization_default is None:
        gpu_memory_utilization_default = env_float("GPU_MEMORY_UTILIZATION", default=0.9)

    max_num_seqs_default = env_int("SPLIT_MERGE_MAX_NUM_SEQS", default=None)
    if max_num_seqs_default is None:
        max_num_seqs_default = env_int("MAX_NUM_SEQS", default=256)

    factor_default = env_float("SPLIT_MERGE_FACTOR", default=None)
    if factor_default is None:
        factor_default = env_float("FACTOR", default=1.0)

    original_max_pos_default = env_int("SPLIT_MERGE_ORIGINAL_MAX_POSITION_EMBEDDINGS", default=None)
    if original_max_pos_default is None:
        original_max_pos_default = env_int("ORIGINAL_MAX_POSITION_EMBEDDINGS", default=131072)

    expected_runs_default = env_int("SPLIT_MERGE_EXPECTED_RUNS", default=None)
    if expected_runs_default is None:
        expected_runs_default = env_int("EXPECTED_RUNS", default=1)

    reasoning_effort_default = env_str("SPLIT_MERGE_REASONING_EFFORT")
    if reasoning_effort_default is None:
        reasoning_effort_default = env_str("REASONING_EFFORT", default=None)

    debug_default = env_bool("SPLIT_MERGE_DEBUG", default=None)
    if debug_default is None:
        debug_default = env_bool("DEBUG", default=False)

    parser = argparse.ArgumentParser(description="Generate completions using split-and-merge with vLLM.")
    parser.add_argument(
        "--data_path",
        type=str,
        default=data_path_default,
        required=data_path_default is None,
        help="Path to the dataset file (JSONL with a `prompt` field).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=output_path_default,
        required=output_path_default is None,
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=model_path_default,
        required=model_path_default is None,
        help="Path to the pretrained model.",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=tokenizer_path_default,
        help="Tokenizer path (defaults to --model_path).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=dtype_default,
        help="Data type to use for the model.",
    )
    parser.add_argument(
        "--n_gpus",
        type=int,
        default=n_gpus_default,
        help="Total number of GPUs to use.",
    )
    parser.add_argument(
        "--n_splits",
        type=int,
        default=n_splits_default,
        help="Number of data splits/parallel processes.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=temperature_default,
        help="Sampling temperature for generation.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=top_p_default,
        help="Top-p sampling for generation.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=top_k_default,
        help="Top-k sampling for generation (-1 disables).",
    )
    parser.add_argument(
        "--presence_penalty",
        type=float,
        default=presence_penalty_default,
        help="Presence penalty for sampling.",
    )
    parser.add_argument(
        "--frequency_penalty",
        type=float,
        default=frequency_penalty_default,
        help="Frequency penalty for sampling.",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=repetition_penalty_default,
        help="Repetition penalty for sampling.",
    )
    parser.add_argument(
        "--min_len",
        type=int,
        default=min_len_default,
        help="Minimum number of tokens to generate.",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=max_len_default,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--use_chat_template",
        type=str2bool,
        default=use_chat_template_default,
        help="Whether to apply chat template to prompts.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=seed_default,
        help="Seed for sampling (-1 for non-deterministic). When --expected_runs > 1, seed must be -1 (non-deterministic).",
    )
    parser.add_argument(
        "--use_mamba2",
        type=str2bool,
        default=use_mamba2_default,
        help="Whether to use the Mamba2 variant code path.",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=gpu_memory_utilization_default,
        help="GPU memory utilization for vLLM.",
    )
    parser.add_argument(
        "--max_num_seqs",
        type=int,
        default=max_num_seqs_default,
        help="vLLM max_num_seqs (used when --use_mamba2 is true).",
    )

    # context extension
    parser.add_argument(
        "--factor",
        type=float,
        default=factor_default,
        help="RoPE scaling factor (YARN).",
    )
    parser.add_argument(
        "--original_max_position_embeddings",
        type=int,
        default=original_max_pos_default,
        help="Original max position embeddings (used when --factor > 1.0).",
    )

    parser.add_argument(
        "--expected_runs",
        type=int,
        default=expected_runs_default,
        help="Number of completions to generate per item (expanded by repetition).",
    )
    parser.add_argument(
        "--reasoning_effort",
        type=str,
        default=reasoning_effort_default,
        help="Optional arg passed to tokenizer chat template (requires --use_chat_template).",
    )
    parser.add_argument(
        "--debug",
        type=str2bool,
        default=debug_default,
        help="If true, only process a small subset for quick debugging.",
    )

    args = parser.parse_args()

    if "codeforces" in args.data_path:
        assert args.expected_runs == 8

    if args.expected_runs > 1:
        assert args.seed == -1

    # Validate GPU availability
    available_gpus = get_available_gpus()
    if args.n_gpus > len(available_gpus):
        print(f"Warning: Requested {args.n_gpus} GPUs but only {len(available_gpus)} available")
        args.n_gpus = len(available_gpus)

    print(f"Using {args.n_gpus} GPUs across {args.n_splits} splits")

    # Load data
    print("Loading data...")
    items = []
    with open(args.data_path, encoding="utf-8") as f:
        for line in f.readlines():
            for _ in range(args.expected_runs):
                item = json.loads(line)
                items.append(item)
            if args.debug and len(items) >= 100:
                break

    print(f"Loaded {len(items)} items")

    # Split data and distribute GPUs
    data_splits = split_data(items, args.n_splits)
    gpu_assignments = distribute_gpus(args.n_gpus, args.n_splits)

    print(f"Data splits: {[len(split) for split in data_splits]}")
    print(f"GPU assignments: {gpu_assignments}")

    # Convert args to dict for pickling
    args_dict = vars(args)

    # Process splits in parallel
    print("Starting parallel processing...")
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=args.n_splits) as executor:
        future_to_split = {}

        for split_id, (data_split, gpu_ids) in enumerate(zip(data_splits, gpu_assignments)):
            future = executor.submit(process_split_worker, split_id, data_split, args_dict, gpu_ids)
            future_to_split[future] = split_id

        split_results = [None] * args.n_splits
        for future in as_completed(future_to_split):
            split_id = future_to_split[future]
            try:
                result = future.result()
                split_results[split_id] = result
                print(f"Split {split_id} completed successfully")
            except Exception as exc:
                print(f"Split {split_id} generated an exception: {exc}")
                raise exc

    processing_time = time.time() - start_time
    print(f"All splits completed in {processing_time:.2f} seconds")

    # Merge results
    merge_results(split_results, args.output_path)

    total_time = time.time() - start_time
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Average time per item: {total_time / len(items):.3f} seconds")


if __name__ == "__main__":
    main()
