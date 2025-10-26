import argparse
import json
import os
from pathlib import Path
from tqdm import tqdm
from utils import DuplicateChecker


def format_code_prompt(problem):
    """Format a problem with the code prompt template."""
    if "starter code" in problem:
        return f"""You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.\n\n{problem.lstrip()}"""
    else:
        return f"""You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.\n\n{problem.strip()}\n\nRead the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.\n```python\n# YOUR CODE HERE\n```\n\n"""


def format_math_prompt(problem):
    """Format a problem with the math prompt template."""
    if "Please reason step by step, and put your final answer within \\boxed{}." in problem:
        return problem
    else:
        return f"""{problem.strip()}\nPlease reason step by step, and put your final answer within \\boxed{{}}."""


def extract_problem_and_rationale(completion):
    """Extract problem and rationale from completion text."""
    rationale = None
    problem = None

    if "<!-- BEGIN RATIONALE -->" in completion and "<!-- END RATIONALE -->" in completion:
        rationale = completion.split("<!-- BEGIN RATIONALE -->")[1].split("<!-- END RATIONALE -->")[0].strip()

    if "<!-- BEGIN PROBLEM -->" in completion and "<!-- END PROBLEM -->" in completion:
        problem = completion.split("<!-- BEGIN PROBLEM -->")[1].split("<!-- END PROBLEM -->")[0].strip()

    return problem, rationale


def process_files(data_files, duplicate_checker, task_type="code"):
    """Process input files and return deduplicated problems."""
    results = []
    unique_problems = set()

    # Select the appropriate prompt formatter
    if task_type == "code":
        format_prompt = format_code_prompt
    elif task_type == "math":
        format_prompt = format_math_prompt
    else:
        raise ValueError(f"Unknown task_type: {task_type}. Must be 'code' or 'math'.")

    for file in tqdm(data_files, desc="Processing files"):
        if not os.path.exists(file):
            print(f"Warning: {file} not found.")
            continue

        with open(file, encoding="utf-8") as f:
            for line in tqdm(f.readlines(), desc=f"Reading {Path(file).name}", leave=False):
                item = json.loads(line)
                completion = item["completion"]

                problem, rationale = extract_problem_and_rationale(completion)

                if rationale is None or problem is None:
                    continue

                # Check for exact duplicates
                if problem not in unique_problems:
                    unique_problems.add(problem)

                    # Check for fuzzy duplicates
                    if duplicate_checker.add_problem(problem):
                        continue

                    # Add to results with prompt formatting
                    results.append({"prompt": format_prompt(problem)})

    return results


def write_results(results, output_file):
    """Write results to output file."""
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item) + "\n")

    print(f"Written {len(results)} items to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Deduplicate problems from JSONL files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process code problems
  python deduplicate_problems.py \\
    --pattern "output/problem_generation/code_concepts_trace{}.jsonl" \\
    --indices 0-64 \\
    --output "output/problem_generation/code_deduplicated.jsonl" \\
    --task-type code

  # Process math problems
  python deduplicate_problems.py \\
    --pattern "output/problem_generation/math_concepts_trace{}.jsonl" \\
    --indices 0-64 \\
    --output "output/problem_generation/math_deduplicated.jsonl" \\
    --task-type math

  # Process with custom duplicate threshold
  python deduplicate_problems.py \\
    --pattern "output/problem_generation/math_concepts_trace{}.jsonl" \\
    --indices 0-64 \\
    --exclude 0,2 \\
    --output "output/problem_generation/math_deduplicated.jsonl" \\
    --task-type math \\
    --threshold 0.3 \\
    --min-word-length 5
        """
    )

    parser.add_argument(
        "--pattern",
        type=str,
        required=True,
        help="File pattern with {} placeholder for index (e.g., 'data/trace{}.jsonl')"
    )

    parser.add_argument(
        "--indices",
        type=str,
        required=True,
        help="Range of indices to process (e.g., '0-60' or '0,2,4-10,15')"
    )

    parser.add_argument(
        "--exclude",
        type=str,
        default="",
        help="Comma-separated indices to exclude (e.g., '0,2,5')"
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output file path (e.g., 'output/deduplicated.jsonl')"
    )

    parser.add_argument(
        "--task-type",
        type=str,
        choices=["code", "math"],
        default="code",
        help="Type of task: 'code' or 'math' (default: code)"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.2,
        help="Duplicate detection threshold (default: 0.2)"
    )

    parser.add_argument(
        "--min-word-length",
        type=int,
        default=4,
        help="Minimum word length for duplicate detection (default: 4)"
    )

    args = parser.parse_args()

    # Parse indices
    indices = []
    for part in args.indices.split(','):
        if '-' in part:
            start, end = map(int, part.split('-'))
            indices.extend(range(start, end))
        else:
            indices.append(int(part))

    # Parse exclusions
    exclude = set()
    if args.exclude:
        exclude = set(map(int, args.exclude.split(',')))

    # Filter indices
    indices = [idx for idx in indices if idx not in exclude]

    # Generate file list
    data_files = [args.pattern.format(idx) for idx in indices]

    print(f"Processing {len(data_files)} files...")
    print(f"Task type: {args.task_type}")
    print(f"Duplicate threshold: {args.threshold}")
    print(f"Min word length: {args.min_word_length}")

    # Initialize duplicate checker
    duplicate_checker = DuplicateChecker(
        min_word_length=args.min_word_length,
        threshold=args.threshold
    )

    # Process files
    results = process_files(
        data_files,
        duplicate_checker,
        task_type=args.task_type
    )

    print(f"\nTotal unique problems after deduplication: {len(results)}")

    # Write results
    write_results(results, args.output)

    print("\nDone!")


if __name__ == "__main__":
    main()
