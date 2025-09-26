import argparse
import json
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from str2bool import str2bool
import os

def test_case_generation_prompt(prompt_text):

    refined_prompt = (
        "As an expert in software testing and quality assurance, generate ONE carefully designed test case for the following programming problem:\n\n"
        f"=== PROBLEM DESCRIPTION ===\n{prompt_text}\n\n"

        "=== EVALUATION CRITERIA ===\n"
        "Your single test case should be selected based on one of these testing priorities:\n"
        "1. EDGE CASE - A boundary condition that tests limits of the solution\n"
        "2. COMPLEX SCENARIO - A challenging but valid input that tests multiple aspects\n"
        "3. ERROR-PRONE CONDITION - An input likely to reveal common implementation mistakes\n"
        "4. CRITICAL PATH - A test of the most important core functionality\n\n"

        "=== REQUIRED OUTPUT FORMAT ===\n"
        "Provide your response as a JSON object enclosed within code fence markers:\n\n"
        
        "```json\n"
        "{\"input\": \"<entire input as a single string>\", \"output\": \"<expected output as a single string>\"}\n"
        "```\n\n"
        
        "=== FORMAT REQUIREMENTS ===\n"
        "- The JSON must be enclosed with ```json at the beginning and ``` at the end\n"
        "- The JSON object must contain ONLY two fields: \"input\" and \"output\"\n"
        "- The input and output must be represented as STRING VALUES, not structured objects\n"
        "- For multi-line inputs, include newline characters (\\n) in the string\n"
        "- Example of correct format:\n"
        "  ```json\n"
        "  {\"input\": \"12\\n1 5 9 13\\n2 6 10 14\\n3 7 11 15\", \"output\": \"479001600\"}\n"
        "  ```\n"
        "- DO NOT use this format (incorrect):\n"
        "  ```json\n"
        "  {\"input\": {\"n\": 5, \"k\": 3, \"points\": [1, 2, 3, 4, 5]}, \"output\": 332748118}\n"
        "  ```\n\n"
        
        "=== IMPORTANT NOTES ===\n"
        "- Your response should consist ONLY of the code-fenced JSON object\n"
        "- Do not include any explanatory text before or after the JSON\n"
        "- Input/output must exactly match how data would be read/written in a standard input/output stream"
    )

    return refined_prompt

def main():
    parser = argparse.ArgumentParser(description="Evaluate large language models on critical datasets.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset file.")
    parser.add_argument("--output_path", type=str, required=True, help="Directory to store cached outputs.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pretrained model.")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to the pretrained model.")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Data type to use for the model (e.g., fp16, bf16, etc.).")
    parser.add_argument("--n_gpus", type=int, default=8, help="Number of GPUs to use for tensor parallelism.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for generation.")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling for generation.")
    parser.add_argument("--top_k", type=int, default=-1, help="Top-k sampling for generation.")  # todo: 40
    parser.add_argument("--max_len", type=int, default=2048, help="Maximum number of tokens to generate.")
    parser.add_argument("--use_chat_template", type=str2bool, default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable_thinking", type=str2bool, default=True)

    args = parser.parse_args()

    if args.tokenizer_path is None:
        args.tokenizer_path = args.model_path

    # Load the tokenizer for LLaMA or any model
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    model = LLM(
        model=args.model_path,
        tokenizer=args.tokenizer_path,
        tokenizer_mode="slow",
        dtype=args.dtype,
        tensor_parallel_size=args.n_gpus,
        enforce_eager=True,
        disable_custom_all_reduce=True,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_len,
        seed=args.seed,
    )

    prompts = []
    items = []
    with open(args.data_path, encoding="utf-8") as f:
        for line in f.readlines():
            item = json.loads(line)
            problem = item["problem"].strip()
            prefix = "You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests."
            suffix = "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.\n```python\n# YOUR CODE HERE\n```"
            if problem.startswith(prefix):
                problem = problem[len(prefix):]
            if problem.endswith(suffix):
                problem = problem[:-len(suffix)]
            problem = problem.strip()
            prompt = test_case_generation_prompt(problem)
            if args.use_chat_template:
                messages = [
                    {"role": "user", "content": prompt}
                ]
                if not args.enable_thinking:
                    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                else:
                    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(prompt)
            items.append(item)

    with torch.no_grad():
        completions = model.generate(prompts, sampling_params)
        completions = [completion.outputs[0].text for completion in completions]

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        for item, completion in zip(items, completions):
            if "completion" in item:
                old_completion = item.pop("completion")
                item["completions"] = [old_completion, completion]
            elif "completions" in item:
                item["completions"].append(completion)
            else:
                item["completions"] = [completion]
            f.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    main()
