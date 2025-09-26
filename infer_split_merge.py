import argparse
import json
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import math
from typing import List, Dict, Any
import time



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

    parser = argparse.ArgumentParser(description="Evaluate large language models using split-and-merge.")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--n_gpus", type=int, default=8)
    parser.add_argument("--n_splits", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--presence_penalty", type=float, default=0.0)
    parser.add_argument("--frequency_penalty", type=float, default=0.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--min_len", type=int, default=0)
    parser.add_argument("--max_len", type=int, default=2048)
    parser.add_argument("--use_chat_template", type=str2bool, default=False)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--use_mamba2", type=str2bool, default=False)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_num_seqs", type=int, default=256)

    # context extension
    parser.add_argument("--factor", type=float, default=1.0)
    parser.add_argument("--original_max_position_embeddings", type=int, default=131072)

    parser.add_argument("--use_ring", type=str2bool, default=False)
    parser.add_argument("--expected_runs", type=int, default=1)
    parser.add_argument("--reasoning_effort", type=str, default=None)
    parser.add_argument("--debug", type=str2bool, default=False)

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
