import argparse
import json
import os
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict
import time
import numpy as np

import threading
from concurrent.futures import TimeoutError as FuturesTimeoutError


def extract_code(text: str) -> str:
    outputlines = text.split("\n")
    indexlines = [i for i, line in enumerate(outputlines) if "```" in line]
    if len(indexlines) < 2:
        return ""
    return "\n".join(outputlines[indexlines[-2] + 1:indexlines[-1]])


def get_after_think(text):
    parts = text.split("</think>", 1)
    if len(parts) > 1:
        return parts[1]
    else:
        return text


def check_correctness(tests: dict, generation: str, timeout: int = 30):
    """Check correctness of code generation with a global timeout.
    The global timeout is to catch some extreme/rare cases not handled by the timeouts
    inside `run_test`"""
    from livecodebench_v5_utils.compute_code_generation_metrics import _temp_run

    manager = mp.Manager()
    result = manager.list()
    metadata_list = manager.list()
    p = mp.Process(
        target=_temp_run,
        args=(tests, generation, False, result, metadata_list, timeout),
    )
    p.start()
    p.join(timeout=(timeout + 1) * len(json.loads(tests["input_output"])["inputs"]) + 5)
    if p.is_alive():
        p.kill()
    if not result:
        in_outs = json.loads(tests["input_output"])
        # consider that all tests failed
        result = [[-1 for i in range(len(in_outs["inputs"]))]]
        metadata_list = [{"error_code": -3}]

    res, metadata = result[0], metadata_list[0]
    fixed = []
    for e in res:
        if isinstance(e, np.ndarray):
            e = e.item(0)
        if isinstance(e, np.bool_):
            e = bool(e)
        if e != True and e != False:
            e = False
        fixed.append(e)
    res = fixed

    if not np.all(res):
        return dict(ispass=0, results=res, metadata=metadata)
    else:
        return dict(ispass=1, results=res, metadata=metadata)


def run_code_test_cases(completion: str, test_cases: List[Dict]) -> str:
    """Test code completion against test cases"""
    gen_code = extract_code(get_after_think(completion))

    test_results = []
    for test_json in test_cases:
        tests = {
            "input_output": json.dumps({
                "inputs": [test_json['input']],
                "outputs": [test_json['output']],
                "fn_name": None,
            })
        }
        res = check_correctness(tests=tests, generation=gen_code)

        if str(res["ispass"]) == "0":
            test_results.append("fail")
        else:
            test_results.append("pass")

    if all([res == "pass" for res in test_results]):
        return "pass"
    else:
        return "fail"


def run_math_test(completion: str, answer) -> str:
    """Test math completion against expected answer"""
    try:
        from eval.qwen_math import math_equal, extract_answer, strip_string
        from math_verify import parse, verify

        res = math_equal(
            extract_answer(completion),
            answer,
        )

        if res:
            return "pass"
        else:
            return "fail"
    except Exception as e:
        print(f"Math evaluation failed with exception: {e}")
        return "fail"


def test_completion_worker_code(completion: str, test_cases: List[Dict]) -> str:
    """Worker function for testing a single code completion"""
    return run_code_test_cases(completion, test_cases)


def test_completion_worker_math(completion: str, answer) -> str:
    """Worker function for testing a single math completion"""
    return run_math_test(completion, answer)


def test_item_completions(item: Dict, num_workers: int, eval_type: str) -> Dict:
    """Test all completions for a single item and mark if any pass"""
    completions = item.get("completions", [])
    if not completions:
        completion = item.get("completion", "")
        if not completion:
            raise ValueError("Each item should contain at least a 'completions' or a 'completion' field.")
        completions = [completion]

    if not completions:
        item["solved"] = False
        item["test_results"] = []
        return item

    if eval_type == "code":
        test_cases = item.get("test_cases", [])
        if not test_cases:
            item["solved"] = False
            item["test_results"] = []
            return item

        # Test all completions concurrently for code
        test_results = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_completion = {}

            for idx, completion in enumerate(completions):
                future = executor.submit(test_completion_worker_code, completion, test_cases)
                future_to_completion[future] = idx

            # Collect results in order
            completion_results = [None] * len(completions)
            for future in as_completed(future_to_completion):
                completion_idx = future_to_completion[future]
                try:
                    result = future.result()
                    completion_results[completion_idx] = result
                except Exception as exc:
                    print(f"Code completion {completion_idx} test failed with exception: {exc}")
                    completion_results[completion_idx] = "fail"

            test_results = completion_results

    elif eval_type == "math":
        from eval.qwen_math import math_equal, extract_answer, strip_string

        answer = item.get("answer", "")
        completions = item.get("completions", [])

        # Check if completions exist when no answer is provided
        if not answer and not completions:
            raise ValueError("No answer provided and no completions available for majority voting")

        if item["source"] == "PromptCoT-Math":
            assert answer is not None and answer.strip() != ""

        if not answer:
            extracted_answers = []
            for completion in completions:
                extracted_answer = extract_answer(get_after_think(completion))
                extracted_answers.append(extracted_answer)

            # Perform majority voting using math_equal for comparison
            answer_counts = {}
            for extracted_answer in extracted_answers:
                found_match = False
                for existing_answer in answer_counts:
                    if math_equal(extracted_answer, existing_answer):
                        answer_counts[existing_answer] += 1
                        found_match = True
                        break
                if not found_match:
                    answer_counts[extracted_answer] = 1

            # Get the answer with the highest count
            if answer_counts:
                answer = max(answer_counts.items(), key=lambda x: x[1])[0]
            else:
                item["solved"] = False
                item["test_results"] = []
                return item

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_completion = {}

            for idx, completion in enumerate(completions):
                future = executor.submit(test_completion_worker_math, completion, answer)
                future_to_completion[future] = idx

            # Collect results in order with timeout protection
            completion_results = [None] * len(completions)
            timeout_per_task = 30  # 30 seconds max per task

            for future in as_completed(future_to_completion, timeout=timeout_per_task * len(completions)):
                completion_idx = future_to_completion[future]
                try:
                    # Add timeout to individual future.result() calls
                    result = future.result(timeout=timeout_per_task)
                    completion_results[completion_idx] = result
                except TimeoutError:
                    print(f"Math completion {completion_idx} timed out after {timeout_per_task} seconds")
                    completion_results[completion_idx] = "timeout"
                    # Cancel the future to prevent resource leaks
                    future.cancel()
                except Exception as exc:
                    print(f"Math completion {completion_idx} test failed with exception: {exc}")
                    completion_results[completion_idx] = "fail"

            # Handle any remaining None results (in case some tasks never completed)
            for idx, result in enumerate(completion_results):
                if result is None:
                    print(f"Math completion {idx} never completed, marking as timeout")
                    completion_results[idx] = "timeout"

            test_results = completion_results

    else:
        raise ValueError(f"Unknown evaluation type: {eval_type}. Must be 'code' or 'math'")

    # Mark item as solved if any completion passes
    item["solved"] = any(result == "pass" for result in test_results)
    item["test_results"] = test_results

    return item


def main():
    from str2bool import str2bool

    parser = argparse.ArgumentParser(
        description="Run test cases on generated completions.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset file with completions.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to store results.")
    parser.add_argument("--eval_type", type=str, required=True, choices=["code", "math"], help="Type of evaluation: 'code' for code testing or 'math' for math equivalence")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of concurrent workers for test execution.")
    parser.add_argument("--max_items", type=int, default=None, help="Maximum number of items to process (for debugging).")
    parser.add_argument("--debug", type=str2bool, default=False)

    args = parser.parse_args()

    # Validate eval_type and check required dependencies
    if args.eval_type == "math":
        try:
            from eval.qwen_math import math_equal, extract_answer, strip_string
        except ImportError:
            print(
                "Error: Math evaluation dependencies not found. Please ensure eval.qwen_math is available.")
            return
    elif args.eval_type == "code":
        try:
            from livecodebench_v5_utils.compute_code_generation_metrics import _temp_run
        except ImportError:
            print("Error: Code evaluation dependencies not found. Please ensure livecodebench_v5_utils is available.")
            return

    # Load data
    print("Loading data...")
    items = []
    with open(args.data_path, encoding="utf-8") as f:
        for line in f.readlines():
            item = json.loads(line)
            if "solved" in item:
                item["solved"] = False
            items.append(item)
            if args.max_items and len(items) >= args.max_items:
                break

    print(f"Loaded {len(items)} items")

    # Count items that need testing
    items_needing_testing = 0
    items_already_tested = 0
    for item in items:
        if "solved" not in item or not item["solved"]:
            items_needing_testing += 1
        else:
            items_already_tested += 1

    print(f"{items_already_tested} items already tested")
    print(f"{items_needing_testing} items need testing out of {len(items)} total")

    if items_needing_testing == 0:
        print("No items need testing. Writing original data to output.")
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        with open(args.output_path, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item) + "\n")
        return

    print(f"Using {args.num_workers} concurrent workers for {args.eval_type} evaluation")

    # Process items
    print(f"Starting {args.eval_type} test execution...")
    start_time = time.time()

    processed_items = []
    solved_count = 0

    for idx, item in enumerate(items):
        if idx % 10 == 0:
            print(f"Processing item {idx + 1}/{len(items)}")

        # Test the item if needed
        if "solved" not in item or not item["solved"]:
            processed_item = test_item_completions(item, args.num_workers, args.eval_type)
        else:
            processed_item = item

        if processed_item.get("solved", False):
            solved_count += 1

        processed_items.append(processed_item)

    processing_time = time.time() - start_time
    print(f"Testing completed in {processing_time:.2f} seconds")

    # Write results
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output_path, "w", encoding="utf-8") as f:
        for item in processed_items:
            f.write(json.dumps(item) + "\n")

    print(f"Results written to {args.output_path}")
    print(f"Total solved items: {solved_count}/{len(items)} ({solved_count / len(items) * 100:.1f}%)")

    if items_needing_testing > 0:
        print(f"Average time per tested item: {processing_time / items_needing_testing:.3f} seconds")


if __name__ == "__main__":
    main()