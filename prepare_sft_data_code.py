import json
import random
import argparse
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
from utils import fix_gptoss_completion, get_after_think
from transformers import AutoTokenizer


def qwq_prompt(problem):
    """Generate prompt template for QWQ model."""
    template = (
        f"<|im_start|>user\n{problem}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return template


def extract_code(text: str) -> str:
    outputlines = text.split("\n")
    indexlines = [i for i, line in enumerate(outputlines) if "```" in line]
    if len(indexlines) < 2:
        return ""
    return "\n".join(outputlines[indexlines[-2] + 1:indexlines[-1]])


def process_item_batch(items_batch, tokenizer_path, max_len, min_len):
    """Process a batch of items with tokenization."""
    # Initialize tokenizer in each process
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)

    batch_results = []
    batch_unique_prompts = set()

    for item in items_batch:
        try:
            prompt = item["prompt"]

            # Fix completion
            try:
                completion = fix_gptoss_completion(item["completions"][0])
            except:
                continue

            code = extract_code(get_after_think(completion))
            if code is None or code.strip() == "":
                continue

            # Format prompt and completion
            formatted_prompt = qwq_prompt(prompt).strip()
            formatted_completion = "\n" + completion

            # Tokenize and check length
            tok_len = len(tokenizer(formatted_prompt + formatted_completion)['input_ids']) + 20
            if tok_len >= max_len or tok_len < min_len:
                continue

            # Check for uniqueness within this batch
            if formatted_prompt not in batch_unique_prompts:
                batch_unique_prompts.add(formatted_prompt)
                batch_results.append({
                    "prompt": formatted_prompt,
                    "completion": formatted_completion,
                })
        except Exception as e:
            # Silently skip problematic items
            continue

    return batch_results, batch_unique_prompts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process data files for training')
    parser.add_argument('--data_path', type=str, required=True, help='Input data file path')
    parser.add_argument('--output_path', type=str, required=True, help='Output file path')
    parser.add_argument('--tokenizer_path', type=str, default="/personal/xueliang/hf_models/Qwen2.5-7B-Instruct", help='Tokenizer path')
    parser.add_argument('--min_len', type=int, default=0, help='Minimum token length')
    parser.add_argument('--max_len', type=int, default=16384, help='Maximum token length')

    args = parser.parse_args()

    # Load all items from single input file
    all_items = []
    try:
        with open(args.data_path, encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    all_items.append(item)
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        print(f"Error: File {args.data_path} not found")
        exit(1)

    print(f"Loaded {len(all_items)} items total")

    # Determine number of processes and batch size
    num_processes = min(cpu_count(), 32)  # Limit to 32 processes to avoid memory issues
    batch_size = max(len(all_items) // num_processes, 1)

    # Split items into batches
    item_batches = [
        all_items[i:i + batch_size]
        for i in range(0, len(all_items), batch_size)
    ]

    print(f"Processing with {num_processes} processes, {len(item_batches)} batches")

    # Create partial function with fixed parameters
    process_func = partial(
        process_item_batch,
        tokenizer_path=args.tokenizer_path,
        max_len=args.max_len,
        min_len=args.min_len
    )

    # Process batches in parallel
    results = []
    global_unique_prompts = set()

    with Pool(processes=num_processes) as pool:
        batch_results = list(tqdm(
            pool.imap(process_func, item_batches),
            total=len(item_batches),
            desc="Processing batches"
        ))

    # Combine results and ensure global uniqueness
    for batch_result, batch_unique_prompts in batch_results:
        for item in batch_result:
            if item["prompt"] not in global_unique_prompts:
                global_unique_prompts.add(item["prompt"])
                results.append(item)

    print(f"Final results: {len(results)} unique items")

    # Shuffle results
    random.shuffle(results)

    # Print first result as example
    if results:
        print("Example result:")
        print(results[0])

    # Write to output file
    with open(args.output_path, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item) + "\n")

    print(f"Results written to {args.output_path}")