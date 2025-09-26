import argparse
import json
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import math
from typing import List, Dict
import time


def get_after_think(text):
    parts = text.split("</think>", 1)
    if len(parts) > 1:
        return parts[1]
    else:
        return text


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
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # Use all visible GPUs for tensor parallelism
    tensor_parallel_size = len(gpu_ids)

    # Load model
    if not args.use_mamba2:
        model = LLM(
            model=args.model_path,
            tokenizer=args.model_path,
            tokenizer_mode="slow",
            dtype=args.dtype,
            tensor_parallel_size=tensor_parallel_size,
            seed=None if args.seed == -1 else args.seed + split_id,  # Different seed per split
            enforce_eager=True,
            disable_custom_all_reduce=True,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        model = LLM(
            model=args.model_path,
            tokenizer=args.model_path,
            max_model_len=args.max_len + 4096,
            tokenizer_mode="slow",
            dtype=args.dtype,
            tensor_parallel_size=tensor_parallel_size,
            seed=None if args.seed == -1 else args.seed + split_id,  # Different seed per split
        )

    # Setup sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_len,
    )

    # Filter items that need processing
    remaining_items = []
    remaining_indices = []

    for idx, item in enumerate(data_split):
        # Initialize fields if not present
        if "completions" not in item:
            item["completions"] = []

        # Check if we need more completions
        if not item.get("solved", False) and len(item["completions"]) < args.num_completions:
            remaining_items.append(item)
            remaining_indices.append(idx)

    print(f"Split {split_id}: {len(remaining_items)} items need completions out of {len(data_split)} total")

    if not remaining_items:
        print(f"Split {split_id}: No items to process")
        return data_split

    # Prepare prompts for items that need processing
    prompts = []
    for item in remaining_items:
        prompt = item["prompt"]
        if args.use_chat_template:
            messages = [{"role": "user", "content": prompt}]
            if args.reasoning_effort is not None:
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, reasoning_effort=args.reasoning_effort)
            else:
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            # prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)

    # Generate completions
    with torch.no_grad():
        completions = model.generate(prompts, sampling_params)
        completions = [completion.outputs[0].text for completion in completions]

    print(f"Split {split_id}: Generated {len(completions)} completions")

    # Update items with new completions
    for completion, item in zip(completions, remaining_items):
        item["completions"].append(completion)

    # Update the original data split with processed items
    updated_data_split = data_split.copy()
    for idx, item in zip(remaining_indices, remaining_items):
        updated_data_split[idx] = item

    # Add split_id for tracking
    for item in updated_data_split:
        item["split_id"] = split_id

    print(f"Split {split_id}: Completed processing {len(remaining_items)} items")
    return updated_data_split


def get_available_gpus():
    """Get list of available GPU IDs"""
    import torch
    return list(range(torch.cuda.device_count()))


def split_data(data: List[Dict], num_splits: int) -> List[List[Dict]]:
    """Split data into roughly equal chunks"""
    chunk_size = math.ceil(len(data) / num_splits)
    return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]


def distribute_gpus(total_gpus: int, num_splits: int) -> List[List[int]]:
    """Distribute GPUs among splits as evenly as possible"""
    if num_splits > total_gpus:
        gpu_assignments = []
        for i in range(num_splits):
            gpu_assignments.append([i % total_gpus])
        return gpu_assignments

    gpus_per_split = total_gpus // num_splits
    remaining_gpus = total_gpus % num_splits

    gpu_assignments = []
    gpu_idx = 0

    for split_idx in range(num_splits):
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

    # Flatten and maintain order
    all_results = []
    for split_result in split_results:
        all_results.extend(split_result)

    # Remove split_id tracking field
    for item in all_results:
        item.pop("split_id", None)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in all_results:
            f.write(json.dumps(item) + "\n")

    print(f"Merged {len(all_results)} results to {output_path}")


def main():
    # Import torch here in main to check available GPUs
    import torch
    from str2bool import str2bool

    parser = argparse.ArgumentParser(
        description="Generate completions for large language models using split-and-merge.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset file.")
    parser.add_argument("--output_path", type=str, required=True, help="Directory to store cached outputs.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pretrained model.")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Data type to use for the model.")
    parser.add_argument("--n_gpus", type=int, default=8, help="Total number of GPUs to use.")
    parser.add_argument("--num_splits", type=int, default=4, help="Number of data splits/parallel processes.")
    parser.add_argument("--num_completions", type=int, default=1, help="Number of completions to generate per item.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for generation.")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling for generation.")
    parser.add_argument("--max_len", type=int, default=2048, help="Maximum number of tokens to generate.")
    parser.add_argument("--use_chat_template", type=str2bool, default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_mamba2", type=str2bool, default=False)
    parser.add_argument("--trust_remote_code", type=str2bool, default=False)
    parser.add_argument("--reasoning_effort", type=str, default=None)
    parser.add_argument("--debug", type=str2bool, default=False)

    args = parser.parse_args()

    # Validate GPU availability
    available_gpus = get_available_gpus()
    if args.n_gpus > len(available_gpus):
        print(f"Warning: Requested {args.n_gpus} GPUs but only {len(available_gpus)} available")
        args.n_gpus = len(available_gpus)

    print(f"Using {args.n_gpus} GPUs across {args.num_splits} splits")

    # Load data
    print("Loading data...")
    items = []
    with open(args.data_path, encoding="utf-8") as f:
        for line in f.readlines():
            item = json.loads(line)
            items.append(item)
            if args.debug and len(items) >= 100:
                break

    print(f"Loaded {len(items)} items")

    # Count items that need processing
    items_needing_processing = 0
    for item in items:
        if "completions" not in item:
            item["completions"] = []

        if not item.get("solved", False) and len(item["completions"]) < args.num_completions:
            items_needing_processing += 1

    print(f"{items_needing_processing} items need completions out of {len(items)} total")

    if items_needing_processing == 0:
        print("No items need processing. Writing original data to output.")
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        with open(args.output_path, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item) + "\n")
        return

    # Split data and distribute GPUs
    data_splits = split_data(items, args.num_splits)
    gpu_assignments = distribute_gpus(args.n_gpus, args.num_splits)

    print(f"Data splits: {[len(split) for split in data_splits]}")
    print(f"GPU assignments: {gpu_assignments}")

    # Convert args to dict for pickling
    args_dict = vars(args)

    # Process splits in parallel
    print("Starting parallel processing...")
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=args.num_splits) as executor:
        future_to_split = {}

        for split_id, (data_split, gpu_ids) in enumerate(zip(data_splits, gpu_assignments)):
            future = executor.submit(process_split_worker, split_id, data_split, args_dict, gpu_ids)
            future_to_split[future] = split_id

        split_results = [None] * args.num_splits
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

    if items_needing_processing > 0:
        print(f"Average time per processed item: {total_time / items_needing_processing:.3f} seconds")


if __name__ == "__main__":
    main()