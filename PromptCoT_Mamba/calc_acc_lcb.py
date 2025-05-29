import json
import argparse
from tqdm import tqdm
import os
import shutil
import copy

from livecodebench_v5 import compute_scores as compute_scores_livecodebench_v5


def get_after_think(text):
    parts = text.split("\n</think>\n\n", 1)
    if len(parts) > 1:
        return parts[1]
    else:
        return text


def main():
    parser = argparse.ArgumentParser(description="Evaluate model outputs")
    parser.add_argument("--input_path", type=str, required=True, help="Path to input jsonl file")
    parser.add_argument("--cache_path", type=str, required=True, help="Path to save cache results")
    # parser.add_argument("--task_name", type=str, required=True, help="Task should be in ['math_opensource/aime24', 'math_opensource/aime25' ,'livecodebench', 'ifeval']")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.cache_path), exist_ok=True)

    if os.path.exists(args.cache_path):
        if os.path.isdir(args.cache_path):
            shutil.rmtree(args.cache_path)  # Remove directory and all contents
        else:
            os.remove(args.cache_path)  # Remove file

    data = []
    with open(args.input_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            item = json.loads(line)
            if "completion" in item:
                completion = item.pop("completion")
                item["gen"] = [completion]
                data.append(item)
            elif "completions" in item:
                completions = item.pop("completions")
                for completion in completions:
                    item_copy = copy.deepcopy(item)
                    item_copy["gen"] = [completion]
                    data.append(item_copy)
            else:
                raise NotImplementedError

    for item in data:
        item["task"] = "livecodebench"
        temp = get_after_think(item['gen'][0])
        item['gen'][0] = temp
    acc = compute_scores_livecodebench_v5(data, args.cache_path)
    print(f"Input: {args.input_path}, Pass@1: {acc}")
    print("Evaluation complete!")


if __name__ == "__main__":
    main()