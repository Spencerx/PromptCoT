import argparse
import json
import re
import os
from tqdm import tqdm


def extract_json_from_output(output_text):
    """
    Extracts the JSON test case from the output text and validates that it
    contains exactly the required keys: 'input' and 'output'.

    Args:
        output_text (str): The complete output text containing reasoning and tests sections

    Returns:
        dict: The extracted JSON test case as a Python dictionary with only 'input' and 'output' keys,
              or None if extraction or validation fails
    """

    # Find the JSON content between ```json and ``` markers
    json_pattern = r'```json\s*(.*?)\s*```'
    match = re.search(json_pattern, output_text, re.DOTALL)

    if not match:
        return None

    json_str = match.group(1)
    try:
        # Parse the JSON string into a Python dictionary
        test_case = json.loads(json_str)

        # Validate that the test case has the required keys
        if not isinstance(test_case, dict):
            return None

        # Check for required keys
        if "input" not in test_case or "output" not in test_case:
            return None

        # Extract only the required keys
        valid_test_case = {
            "input": test_case["input"],
            "output": test_case["output"]
        }

        return valid_test_case

    except json.JSONDecodeError:
        return None


def generate_code_prompt(problem):
    """
    Generates a standardized prompt for code generation tasks.

    Args:
        problem (str): The problem specification

    Returns:
        str: The formatted prompt
    """
    problem = f"""You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.\n\n{problem.strip()}\n\nRead the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.\n```python\n# YOUR CODE HERE\n```\n\n"""
    return problem


def get_after_think(text):
    parts = text.split("</think>", 1)
    if len(parts) > 1:
        return parts[1]
    else:
        return text


def main():
    parser = argparse.ArgumentParser(description="Process test cases and generate prompts for code evaluation.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input test case file.")
    parser.add_argument("--output_file", type=str, required=True,  help="Path to the output file.")

    args = parser.parse_args()

    results = []

    with open(args.input_file, encoding="utf-8") as f:
        for line in tqdm(f.readlines()):
            item = json.loads(line)
            prompt = generate_code_prompt(item["problem"])
            test_cases = [extract_json_from_output(get_after_think(completion)) for completion in item["completions"]]
            test_cases = [test_json for test_json in test_cases if test_json is not None]

            if len(test_cases) == 0:
                print("skip")
                continue

            results.append({
                "prompt": prompt,
                "test_cases": test_cases,
            })

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    with open(args.output_file, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    main()
