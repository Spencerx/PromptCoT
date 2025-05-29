# **Scaling Reasoning without Attention** 

---

## 🚀 Overview

**PromptCoT-Mamba** establishes the first **attention-free foundation model** capable of surpassing strong Transformer baselines across a broad suite of competition-level math and code reasoning tasks. Built on the **Mamba-2** architecture and trained through a structured, two-stage curriculum using the [**PromptCoT**](http://arxiv.org/abs/2503.02324) pipeline, it delivers **high accuracy with constant-memory inference**, eliminating the need for KV caching.



## 📈 Key Results

### 🔹 General Performance

| Model                  | MATH-500 | AIME 24  | AIME 25  | OlympiadBench | HumanEval | HumanEval+ | Livecodebench |
| ---------------------- | -------- | -------- | -------- | ------------- | --------- | ---------- | ------------- |
| **PromptCoT-Mamba-7B** | 84.6     | **35.2 🔥** | **24.6 🔥** | 50.7          | 81.7      | 75.0       | **29.9🔥**      |
| Gemma3-27B             | **89.0** | 32.6     | 24.0     | **54.2**      | **86.0**  | **78.0**   | 26.9          |
| Gemma3-12B             | 83.8     | 22.9     | 19.2     | 49.9          | 81.1      | 73.2       | 22.2          |
| Sky-T1-7B              | 85.0     | 19.2     | 19.2     | 49.2          | 41.5      | 37.2       | 18.3          |
| S1.1-7B                | 82.0     | 19.2     | 17.5     | 43.1          | 64.0      | 56.7       | 13.3          |
| Bespoke-Stratos-7B     | 81.2     | 18.3     | 16.3     | 45.0          | 73.2      | 68.3       | 8.6           |
| Nemotron-H-8B          | 77.6     | --       | --       | --            | 79.3      | 74.4       | --            |
| M1-3B                  | 81.7     | 23.0     | 22.0     | 43.6          | --        | --         | --            |

> 🔍 **PromptCoT-Mamba-7B** consistently outperforms all 7B-scale Transformer and hybrid Mamba-Transformer baselines across all tasks.

---

### 🔹 Math Specialization vs. Generalist

| Model                       | MATH-500 | AIME 24  | AIME 25  | OlympiadBench | HumanEval | HumanEval+ | Livecodebench |
| --------------------------- | -------- | -------- | -------- | ------------- | --------- | ---------- | ------------- |
| **PromptCoT-Mamba-Math-7B** | **88.0 🔥** | **42.9 🔥** | **30.8 🔥** | **52.1 🔥**      | 71.3      | 66.5       | 20.3          |
| PromptCoT-Mamba-7B          | 84.6     | 35.2     | 24.6     | 50.7          | **81.7**  | **75.0**   | **29.9**      |

> 🎯 The math-specialized variant improves AIME 24 by **+7.7%** and AIME 25 by **+6.2%**, with a slight trade-off in code-related performance.

---

### ⚡ Inference Efficiency

Using `vLLM` under constrained memory, PromptCoT-Mamba-7B demonstrates substantial speedups over the S1.1-7B Transformer baseline:

* 💡 **3.66× faster** at long-sequence generation on **24GB GPU**
* 💡 **1.69× faster** under **72GB memory**

> ⚙️ Practical for cost-sensitive or long-context inference workloads at scale.


## 🧪 Quick Start

### Install Requirements

```bash
pip install -r requirements.txt
```

### Training with DeepSpeed

```bash
deepspeed --num_gpus=8 train.py \
    --adam_beta1=0.9 \
    --adam_beta2=0.95 \
    --bf16=True \
    --data_path=/path/to/sft_data \
    --ddp_find_unused_parameters=False \
    --deepspeed=configs/promptcot_mamba_7b_config.json \
    --fp16=False \
    --gradient_accumulation_steps=8 \
    --gradient_checkpointing=True \
    --learning_rate=5e-06 \
    --max_grad_norm=1.0 \
    --max_length=20480 \
    --model_name_or_path=/path/to/PromptCoT-Mamba-7B \
    --num_train_epochs=1 \
    --output_dir=/path/to/output_dir \
    --per_device_train_batch_size=1 \
    --save_steps=1000 \
    --save_strategy=steps \
    --save_total_limit=10 \
    --tokenizer_path=/path/to/PromptCoT-Mamba-7B \
    --warmup_steps=100 \
    --weight_decay=0.01
```

### AIME Test
```bash
python infer_longcot.py \
    --data_path data/aime_test.jsonl \
    --output_path data/aime_test_predictions.jsonl \
    --model_path /path/to/PromptCoT-Mamba-7B \
    --n_gpus 1 \
    --temperature 0.8 \
    --repetition_penalty 1.1 \
    --max_len 65536 \
    --n 16 \
    --max_retries 1 \
    --use_mamba2 True

python calc_acc_aime.py \
    --input_path data/aime_test_predictions.jsonl \
    --expected_runs 16
```

### LiveCodeBench Test

```bash
python infer_longcot.py \
    --data_path data/livecodebench_test.jsonl \
    --output_path data/livecodebench_test_predictions.jsonl \
    --model_path /path/to/PromptCoT-Mamba-7B \
    --n_gpus 1 \
    --temperature 0.8 \
    --repetition_penalty 1.1 \
    --max_len 65536 \
    --n 8 \
    --max_retries 1 \
    --use_mamba2 True

python calc_acc_lcb.py \
    --input_path data/livecodebench_test_predictions.jsonl \
    --cache_path cache/livecodebench_test_predictions.jsonl
```


## 📜 Citation

```bibtex
@article{zhao2025scaling,
  author    = {Xueliang Zhao and Wei Wu and Lingpeng Kong},
  title     = {Scaling Reasoning without Attention},
  journal   = {arXiv preprint arXiv:2505.22425},
  year      = {2025},
  url       = {https://arxiv.org/abs/2505.22425}
}
```
