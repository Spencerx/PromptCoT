import json
import re
from tqdm import tqdm
import random

import argparse
from utils import load_test_problems, fix_qwq_completion, get_after_think
from eval.qwen_math import extract_answer, strip_string


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--output_path", type=str, default="")

    args = parser.parse_args()

    # Chinese (Han) ranges + Compatibility + most extensions (BMP + SMP)
    _CHINESE_RE = re.compile(r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF\U00020000-\U0002EBEF]")
    # Korean (Hangul) ranges: Jamo, Compatibility Jamo, Extended-A/B, Syllables
    _KOREAN_RE = re.compile(r"[\u1100-\u11FF\u3130-\u318F\uA960-\uA97F\uAC00-\uD7AF\uD7B0-\uD7FF]")


    def contains_chinese_or_korean(seq) -> bool:
        """Return True if the given sequence (string or iterable of strings/chars)
        contains any Chinese (Han) or Korean (Hangul) characters."""
        if isinstance(seq, str):
            text = seq
        else:
            try:
                text = "".join(seq)
            except TypeError:
                text = str(seq)
        return bool(_CHINESE_RE.search(text) or _KOREAN_RE.search(text))


    def extract_code(text: str) -> str:
        outputlines = text.split("\n")
        indexlines = [i for i, line in enumerate(outputlines) if "```" in line]
        if len(indexlines) < 2:
            return ""
        return "\n".join(outputlines[indexlines[-2] + 1:indexlines[-1]])


    def is_valid(text, source="PromptCoT-Math"):
        pattern = r'^<think>(.*?)</think>(.+)$'
        match = re.match(pattern, text, re.DOTALL)

        if not match:
            return False

        # Extract the content inside and after the think tags
        inside_content = match.group(1)
        after_content = match.group(2)

        # Verify inside content is not empty
        if not inside_content.strip():
            return False

        # Verify neither part contains additional think tags
        if '<think>' in inside_content or '</think>' in inside_content:
            return False

        if '<think>' in after_content or '</think>' in after_content:
            return False

        text = text.split("</think>")[-1]
        if contains_chinese_or_korean(text):
            return False

        if source == "PromptCoT-Math" or source == "OpenSource-Math":
            prediction = extract_answer(text)
            prediction = strip_string(prediction)
            if prediction is None or not prediction:
                return False
        elif source == "PromptCoT-Code" or source == "OpenSource-Code":
            prediction = extract_code(text)
            if prediction is None or not prediction:
                return False
        else:
            raise ValueError

        return True


    def qwq_prompt(problem):
        template = (
            f"<|im_start|>user\n{problem}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        return template


    def extract_problem(problem):
        if problem.endswith("\nPlease reason step by step, and put your final answer within \\boxed{}."):
            problem = problem[:-len("\nPlease reason step by step, and put your final answer within \\boxed{}.")]
        return problem.strip()


    def no_solution(answer):
        if "none" in answer.lower():
            return True
        if "no" in answer.lower():
            return True
        return False


    def trivial_solution(answer):
        if answer in ["0", "1"]:
            return True
        return False


    test_problems = load_test_problems()

    results = []

    promptcot_count = 0
    opensource_count = 0
    with open(args.data_path, encoding="utf-8") as f:
        for line in tqdm(f.readlines()):
            item = json.loads(line)
            source = item["source"]
            prompt = item["prompt"]
            completions = item["completions"]
            test_results = item["test_results"]

            if source == "PromptCoT-Math":
                if extract_problem(item["prompt"]) in test_problems:
                    print("promptcot hit!")
                    continue

                if len(test_results) == 0:
                    print("No answers.")
                    continue

                if "the following conditions:\nPlease reason step by step" in prompt:
                    print("No solutions.")
                    continue

                if "the following properties:\nPlease reason step by step" in prompt:
                    print("No solutions.")
                    continue

                if "solution to\nPlease reason step by step" in prompt:
                    print("No solutions.")
                    continue

                if "the following properties:\nFind" in prompt:
                    print("No solutions.")
                    continue

                if "prove that" in prompt or "Prove that" in prompt:
                    print("No solutions.")
                    continue

                # answer = None
                # for completion, res in zip(completions, test_results):
                #     if res == "pass":
                #         answer = extract_answer(get_after_think(completion))
                #         break

                if item["answer"] is None or item["answer"].strip() == "":
                    continue

                if no_solution(item["answer"]):
                    print("No solutions.")
                    continue

                if trivial_solution(item["answer"]):
                    print("Trivial solutions.")
                    continue

                max_count = 10000  ### Note: if the answer is got through majority voting, max_count should be 2 or 4

            elif source == "OpenSource-Math":
                if extract_problem(item["prompt"]) in test_problems:
                    print("opensource hit!")
                    continue
                if len(test_results) == 0:
                    print("No answers.")
                    continue
                if item["answer"] is None or item["answer"].strip() == "":
                    continue
                max_count = 10000

            elif source == "PromptCoT-Code":
                if extract_problem(item["prompt"]) in test_problems:
                    print("promptcot hit!")
                    continue

                if len(test_results) == 0:
                    print("No answers.")
                    continue
                max_count = 10000

            elif source == "OpenSource-Code":
                if extract_problem(item["prompt"]) in test_problems:
                    print("opensource hit!")
                    continue

                if len(test_results) == 0:
                    print("No answers.")
                    continue
                max_count = 10000

            else:
                raise ValueError

            if sum([res == "pass" for res in test_results]) / len(test_results) > 1 / 2:
                continue

            if sum([res == "pass" for res in test_results]) == 0:
                continue

            chosen, rejected = set(), set()
            for completion, test_res in zip(completions, test_results):
                completion = fix_qwq_completion(completion)
                if test_res == "pass":
                    if is_valid(completion, source):
                        chosen.add(completion)
                else:
                    if is_valid(completion, source):
                        rejected.add(completion)
            chosen, rejected = list(chosen), list(rejected)

            # if len(chosen) == 1 and source == "PromptCoT-Math":
            #     print("No majority voted answers.")
            #     continue

            if len(chosen) == 0 or len(rejected) == 0:
                print("No valid completions.")
                continue

            n_pairs = min(max_count, min(len(chosen), len(rejected)))

            for i in range(n_pairs):
                if source == "PromptCoT-Math" or source == "PromptCoT-Code":
                    promptcot_count += 1
                else:
                    opensource_count += 1

                results.append({
                    "source": source,
                    "prompt": qwq_prompt(prompt).strip(),
                    "chosen": "\n" + chosen[i],
                    "rejected": "\n" + rejected[i]
                })
    print("PromptCoT2:", promptcot_count)
    print("OpenSource: ", opensource_count)
    random.shuffle(results)
    with open(args.output_path, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item) + "\n")
