import ast
import json
import math
import os
import re
import warnings
from collections import defaultdict
from typing import List, Union

import numpy as np
from datasketch import MinHash, MinHashLSH
from tqdm import tqdm

import torch.nn as nn

from transformers import AutoTokenizer


def get_model_device(model):
    """
    Get model device handling both nn.DataParallel and regular model cases
    Args:
        model: torch model (can be wrapped in DataParallel or not)
    Returns:
        device: torch.device
    """
    if isinstance(model, nn.DataParallel):
        return next(model.module.parameters()).device
    else:
        return next(model.parameters()).device


def extract_code_snippet(text):
    # Extract the code between ```python and the closing ```
    match = re.search(r'```python(.*?)```', text, re.DOTALL)
    if match:
        return match.group(1).strip()  # Return the Python code, removing surrounding whitespace
    match = re.search(r'```(.*?)```', text, re.DOTALL)
    if match:
        return match.group(1).strip()  # Return the Python code, removing surrounding whitespace
    return None


def extract_list_from_code(code_str):
    # Parse the cleaned code string into a Python AST
    try:
        parsed_code = ast.parse(code_str)
        # Loop through all top-level statements
        for node in parsed_code.body:
            if isinstance(node, ast.Assign):  # Look for assignments
                for target in node.targets:
                    if isinstance(node.value, ast.List):  # Check if the assigned value is a List
                        # Extract the list from the assignment
                        return [ast.literal_eval(e) for e in node.value.elts]  # Convert list elements to their literal values
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.List):
                # Extract and return the elements from the list
                return [ast.literal_eval(e) for e in node.value.elts]
    except Exception as e:
        # print(f"Error parsing code: {e}")
        pass
    return None


def extract_parentheses(text):
    pattern = r'The correct answer is \((.*?)\)'
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    return ''


class DuplicateChecker:
    def __init__(self, min_word_length=4, threshold=0.8, num_perm=128, num_bands=50):
        self.min_word_length = min_word_length
        self.threshold = threshold
        self.num_perm = num_perm  # Number of permutations for MinHash

        # LSH index for fast similarity search
        self.lsh = MinHashLSH(
            threshold=threshold,
            num_perm=num_perm,
            params=(num_bands, num_perm // num_bands)
        )

        # Store MinHash signatures and original problems
        self.minhashes = {}  # problem_id -> MinHash
        self.problems = {}  # problem_id -> preprocessed words
        self.current_id = 0

        # Cache for frequent words (optional)
        self.common_words_cache = set()
        self.word_frequency = defaultdict(int)
        self.min_freq_threshold = 1000  # Adjust based on your needs

    def preprocess_text(self, text):
        """Normalize text and extract significant words."""
        # Convert to lowercase and clean text
        text = text.lower()
        text = re.sub(r'[\(\)\{\}\[\]\$\^\+\-\*\/\=]', ' ', text)

        # Split and filter words
        words = []
        for w in text.split():
            if (len(w) >= self.min_word_length and
                    not w.isdigit() and
                    not all(c in '0123456789.,' for c in w) and
                    w not in self.common_words_cache):
                words.append(w)
                # Update word frequency
                self.word_frequency[w] += 1
                # If word becomes too common, add to cache
                if self.word_frequency[w] > self.min_freq_threshold:
                    self.common_words_cache.add(w)

        return words

    def create_minhash(self, words):
        """Create MinHash signature for a set of words."""
        mh = MinHash(num_perm=self.num_perm)
        for word in words:
            mh.update(word.encode('utf-8'))
        return mh

    def find_duplicates(self, problem_id):
        """Find all similar problems for a given problem ID."""
        duplicates = self.lsh.query(self.minhashes[problem_id])
        duplicates.remove(problem_id)  # Remove self from results
        return duplicates

    def add_problem(self, problem_text):
        """Add a problem and check for duplicates."""
        # Preprocess text
        words = self.preprocess_text(problem_text)
        if not words:
            return False

        # Create MinHash signature
        mh = self.create_minhash(words)

        # Check for duplicates before adding
        try:
            # Get candidates from LSH
            candidates = self.lsh.query(mh)

            # If any candidates found, verify similarity
            if candidates:
                for candidate_id in candidates:
                    candidate_words = self.problems[candidate_id]
                    # Exact similarity check for candidates
                    intersection = len(set(words) & set(candidate_words))
                    union = len(set(words) | set(candidate_words))
                    if intersection / union > self.threshold:
                        return True
        except:
            pass  # No candidates found

        # If no duplicates found, add to indexes
        problem_id = str(self.current_id)
        self.minhashes[problem_id] = mh
        self.problems[problem_id] = words
        self.lsh.insert(problem_id, mh)
        self.current_id += 1

        return False

    def batch_add_problems(self, problem_texts):
        """Efficiently add multiple problems at once."""
        results = []
        for problem_text in problem_texts:
            is_duplicate = self.add_problem(problem_text)
            results.append(is_duplicate)
        return results

    def clear_cache(self):
        """Clear the common words cache and frequency counter."""
        self.common_words_cache.clear()
        self.word_frequency.clear()

    def get_statistics(self):
        """Get statistics about the index."""
        return {
            'total_problems': self.current_id,
            'common_words_cached': len(self.common_words_cache),
            'unique_words': len(self.word_frequency)
        }


def get_after_think(text):
    parts = text.split("</think>", 1)
    if len(parts) > 1:
        return parts[1]
    else:
        return text


def extract_answer(response):
    from math_verify import parse, verify
    parsed = parse(get_after_think(response))
    assert isinstance(parsed, list)
    parsed = [a for a in parsed if isinstance(a, str)]
    if not parsed:
        return None
    return parsed[0]


def fix_qwq_completion(completion):
    completion = completion.strip()
    if completion.startswith("<think>"):
        completion = completion[len("<think>"):].strip()
    completion = "<think>\n" + completion.strip()
    return completion


def fix_gptoss_completion(completion):
    sentence = completion.strip()
    if not sentence.lower().startswith("analysis"):
        raise ValueError("Sentence must start with 'analysis'")

    assistant_count = sentence.lower().count("assistantfinal")
    if assistant_count != 1:
        raise ValueError(f"Sentence must contain 'assistantfinal' exactly once, found {assistant_count}")

    analysis_end = len("analysis")
    processed = "<think>\n" + sentence[analysis_end:]
    processed = re.sub(r'assistantfinal', r'\n</think>\n\n', processed, flags=re.IGNORECASE)

    return processed


def is_bad_prompt(prompt):
    if "the following conditions:\nPlease reason step by step" in prompt:
        return True

    if "the following properties:\nPlease reason step by step" in prompt:
        return True

    if "solution to\nPlease reason step by step" in prompt:
        return True

    if "the following properties:\nFind" in prompt:
        return True

    if "prove that" in prompt or "Prove that" in prompt:
        return True

    return False


def no_solution(answer):
    if "none" in answer.lower():
        return True
    if "no" in answer.lower():
        return True
    if "inf" in answer.lower():
        return True
    return False



if __name__ == "__main__":
    pass
