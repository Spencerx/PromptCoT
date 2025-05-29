import argparse
import json
from tqdm import tqdm
import numpy as np
import re
from collections import defaultdict

from math_opensource_utils.math_equivalence import is_equiv_minerva as is_equiv
from math_opensource_utils.util import last_boxed_only_string, first_boxed_only_string, remove_boxed
from math_opensource_utils.qwen_math import math_equal, extract_answer, strip_string


def extract_parentheses(text):
    pattern = r'The correct answer is \((.*?)\)'
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    return ''


def is_valid(completion):
    if "<think>" not in completion or "</think>" not in completion:
        return False
    think = completion.split("<think>")[1].split("</think>")[0]
    solution = completion.split("</think>")[1]
    final_answer = remove_boxed(last_boxed_only_string(solution))
    if final_answer is None:
        return False
    if final_answer.strip() == "":
        return False
    return True


def calculate_stats(results):
    """Calculate mean and standard error from a list of binary outcomes"""
    if not results:
        return 0.0, 0.0

    mean = np.mean(results)
    std_err = np.std(results, ddof=1) / np.sqrt(len(results)) if len(results) > 1 else 0.0

    return mean, std_err


def main():
    parser = argparse.ArgumentParser(description="Evaluate large language models on critical datasets.")
    parser.add_argument("--input_path", type=str, required=True, help="Directory to store cached outputs.")
    parser.add_argument("--expected_runs", type=int, default=8)

    args = parser.parse_args()

    # First, organize completions by source and prompt to determine run_index
    # source -> prompt -> list of completions
    completions_by_prompt = defaultdict(lambda: defaultdict(list))

    # Read all items first to organize by prompt
    all_items = []
    with open(args.input_path, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            all_items.append(item)

            if "prompt" not in item:
                raise ValueError(f"Item is missing required 'prompt' field")

            # Store completion information by source and prompt
            source = item["source"]
            prompt = item["prompt"]
            completions_by_prompt[source][prompt].append(item)

    # Now, determine run_index based on order of completions for each prompt
    # source -> run_index -> prompt -> correct or not
    results = defaultdict(lambda: defaultdict(dict))

    # Keep track of prompts by source
    prompts_by_source = defaultdict(set)

    for source in completions_by_prompt:
        for prompt in tqdm(completions_by_prompt[source]):
            # Sort completions if there's any specific ordering required
            # (assuming they're already in the right order in the file)
            completions = completions_by_prompt[source][prompt]

            prompts_by_source[source].add(prompt)

            # Process each completion with its calculated run_index
            for run_idx, item in enumerate(completions):
                completion = item["completion"]
                reference_solution = item.get("reference_solution", item.get("solution"))

                valid = is_valid(completion)

                if source in ["aime2024", "aime2025", "math500"]:
                    correct = math_equal(
                        extract_answer(completion),
                        strip_string(reference_solution.split("####")[1].strip()),
                        timeout=False,
                    ) or is_equiv(
                        remove_boxed(last_boxed_only_string(completion)),
                        reference_solution.split("####")[-1].strip() if "####" in reference_solution else (
                            remove_boxed(last_boxed_only_string(reference_solution)),
                        )
                    )
                else:
                    raise NotImplementedError(f"Source '{source}' is not implemented")

                # Store the correctness result with calculated run_index
                results[source][run_idx][prompt] = int(correct)

    # Calculate and print statistics for each source
    print("\nRESULTS BY SOURCE:")
    print("-" * 80)
    print(f"{'Source':<15} {'Accuracy':<20} {'Num Prompts':<15} {'Runs':<10}")
    print("-" * 80)

    for source in sorted(results.keys()):
        # We expect 8 runs (0-7)
        expected_runs = args.expected_runs
        prompts = sorted(prompts_by_source[source])

        # Calculate accuracy for each run
        run_accuracies = []
        run_details = []

        for run_idx in range(expected_runs):
            if run_idx not in results[source]:
                print(f"Warning: Source '{source}' is missing run index {run_idx}")
                continue

            correct_count = 0
            total_count = 0

            for prompt in prompts:
                # Some prompts might be missing in some runs
                if prompt in results[source][run_idx]:
                    correct_count += results[source][run_idx][prompt]
                    total_count += 1

            # Calculate accuracy for this run
            if total_count > 0:
                run_accuracy = correct_count / total_count
                run_accuracies.append(run_accuracy)
                run_details.append(f"Run {run_idx}: {run_accuracy:.4f} ({correct_count}/{total_count})")

        # Calculate stats across all runs
        if run_accuracies:
            mean_accuracy = np.mean(run_accuracies)
            # std_error = np.std(run_accuracies, ddof=1) / np.sqrt(len(run_accuracies))  # Standard error of the mean
            std_dev = np.std(run_accuracies, ddof=1)

            # Format the result as mean ± standard error with percentage
            # accuracy_str = f"{mean_accuracy:.2%} ± {std_dev:.2%}"

            mean_pct = round(mean_accuracy * 100, 1)
            std_dev_pct = round(std_dev * 100, 1)

            accuracy_str = f"{mean_pct:.1f}% ± {std_dev_pct:.1f}%"

            print(f"{source:<15} {accuracy_str:<20} {len(prompts):<15} {len(run_accuracies)}")

            # Print individual run details
            for detail in run_details:
                print(f"  {detail}")

            print("-" * 80)


if __name__ == "__main__":
    main()
