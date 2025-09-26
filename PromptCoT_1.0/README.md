# **PromptCoT: Synthesizing Olympiad-Level Problems for Mathematical Reasoning in Large Language Models**

---

## **Highlights**
### **✨ The Missing Piece for Test-Time Scaling**  
A **lightweight yet powerful problem generation model** that enables the construction of **prompt sets at any scale with sufficient quality**—perfect for initializing your **post-training project**, whether it's **Supervised Fine-Tuning (SFT) or Reinforcement Learning (RL)**. Say goodbye to the limitations of open-source data!  

### **📖 A Fully Open Project**  
- **📂 Open-Source Problem Generation Model**  
  - **Model**: [Hugging Face](https://huggingface.co/xl-zhao/PromptCoT-Problem-Generation-Model) | [ModelScope](https://www.modelscope.cn/models/zhaoxlpku/PromptCoT-Problem-Generation-Model)  
  - **Training Data**: [Hugging Face](https://huggingface.co/datasets/xl-zhao/PromptCoT-Problem-Generation-Dataset) | [ModelScope](https://www.modelscope.cn/datasets/zhaoxlpku/PromptCoT-Problem-Generation-Dataset)  
- **🔹 Open-Source Distilled Models for Mathematical Reasoning**  
  - **PromptCoT-DS-1.5B** (**Distilled from DeepSeek-R1-Distill-Qwen-7B, 1.5B parameters**)  
    [Hugging Face](https://huggingface.co/xl-zhao/PromptCoT-DS-1.5B) | [ModelScope](https://www.modelscope.cn/models/zhaoxlpku/PromptCoT-DS-1.5B)  
  - **PromptCoT-DS-7B** (**Distilled from DeepSeek-R1-Distill-Qwen-7B, 7B parameters**)  
    [Hugging Face](https://huggingface.co/xl-zhao/PromptCoT-DS-7B) | [ModelScope](https://www.modelscope.cn/models/zhaoxlpku/PromptCoT-DS-7B)  
  - **[New] 🚀🚀🚀** **PromptCoT-QwQ-32B** (**Distilled from QwQ-32B, 32B parameters**)
    [Hugging Face](https://huggingface.co/xl-zhao/PromptCoT-QwQ-32B) | [ModelScope](https://www.modelscope.cn/models/zhaoxlpku/PromptCoT-QwQ-32B)  
  - **Training Data for Supervised Fine-Tuning (SFT) of PromptCoT-DS Series Models**  
    [Hugging Face](https://huggingface.co/datasets/xl-zhao/PromptCoT-DS-Dataset) | [ModelScope](https://www.modelscope.cn/datasets/zhaoxlpku/PromptCoT-DS-Dataset)  
  - **[New] 🚀🚀🚀** **Training Data for Supervised Fine-Tuning (SFT) of PromptCoT-QwQ-32B**  
    [Hugging Face](https://huggingface.co/datasets/xl-zhao/PromptCoT-QwQ-Dataset) | [ModelScope](https://www.modelscope.cn/datasets/zhaoxlpku/PromptCoT-QwQ-Dataset)  


### **🏆 Superior Performance**  
- **Consistent Improvements over Deepseek Counterparts**
  **PromptCoT-DS-7B** surpasses **DeepSeek-R1-Distill-Qwen-7B** across all major benchmarks, achieving consistent improvements in problem-solving accuracy. The results, averaged over **8 random seeds**, highlight the following gains:  
  - **+0.9%** absolute improvement on **MATH-500** (**93.7%** vs. **92.8%**)  
  - **+3.2%** absolute improvement on **AIME2024** (**58.7%** vs. **55.5%**)  
  - **+9.2%** absolute improvement on **AIME2025** (**49.2%** vs. **40.0%**)  

- **Competitive with 32B Models**  
  Despite having only 7B parameters, **PromptCoT-DS-7B** achieves results comparable to larger 32B models such as **S1-32B** and **LIMO-32B**.

**Performance Comparison of Different Models**

| **Model**                                    | **GSM8K**        | **MATH-500**        | **AIME2024**        | **AIME2025**        |
|----------------------------------------------|------------------|---------------------|---------------------|---------------------|
| **🔹 1.5B Models**                           |                  |                     |                     |                     |
| **DeepSeek-R1-Distill-Qwen-1.5B**            | -                | 83.9%               | 28.9%               | 28.1%               |
| **STILL-3-1.5B-preview**                     | -                | 85.5%               | 39.3%               | -                   |
| **DeepScaleR-1.5B-Preview**                  | -                | 🟢 **87.8%**         | 🟢 **43.1%**         | 🟢 **37.1%**         |
| **PromptCoT-DS-1.5B** (**ours**)             | 🟢 **87.6% ± 0.5%**       | **85.3% ± 1.1%**     | **41.2% ± 6.9%**     | **36.7% ± 6.2%**     |
| **🔹 7B Models**                             |                  |                     |                     |                     |
| **DeepSeek-R1-Distill-Qwen-7B**              | -                | 92.8%               | 55.5%               | 40.0%               |
| **Qwen2.5-7B-SimpleRL**                      | -                | 82.4%               | 26.7%               | -                   |
| **OpenThinker-7B**                           | -                | 89.6%               | 30.0%               | 33.3%               |
| **OpenR1-Qwen-7B**                           | -                | 90.6%               | 36.7%               | 40.0%               |
| **PromptCoT-DS-7B** (**ours**)               | 🔥 **92.8% ± 0.5%** | 🔥 **93.7% ± 0.7%**  | 🔥 **58.7% ± 3.1%**  | 🔥 **49.2% ± 7.9%**  |
| **🔹 32B Models**                            |                  |                     |                     |                     |
| **DeepSeek-R1-Distill-Qwen-32B**             | -                | 94.3%               | 72.6%               | -                   |
| **S1-32B**                                   | -                | 93.0%               | 56.7%               | 26.6%               |
| **LIMO-32B**                                 | -                | 94.8%               | 57.1%               | 46.6%               |
| **QwQ-32B**                                  | -                | -                   | 82.1%               | 70.8%               |
| **PromptCoT-QwQ-32B** (**ours**)             | 🔥🔥 **96.4% ± 0.2%** | 🔥🔥 **96.7% ± 0.5%**   | 🔥🔥 **83.8% ± 2.8%** | 🔥🔥 **75.4% ± 4.7%** |

- **Challenging RL-Based Methods Without RL**  
  Despite relying purely on distillation, **PromptCoT-DS-1.5B** achieves competitive results against RL-based models like **STILL-3-1.5B-preview** and **DeepScaleR-1.5B-Preview**, highlighting the strength of our problem generation pipeline.  

### **⚡ Efficiency Without Compromise**  
Compared to **DeepScaleR-1.5B-Preview**, **PromptCoT-DS-1.5B** achieves **40+% AIME scores** while using **over 15× fewer A100 GPU hours** (240 A100 hours vs. 3,800 A100 hours). This makes **PromptCoT-DS-1.5B** a highly efficient and cost-effective solution for mathematical reasoning.


---

## **Overview**  
Large language models (LLMs) have demonstrated remarkable advancements in mathematical reasoning. However, acquiring **challenging and high-quality Olympiad-level problems** at scale remains a significant challenge. Existing datasets often lack the necessary complexity to further enhance the capabilities of state-of-the-art models.

**PromptCoT** introduces a method to systematically generate high-quality Olympiad-level math problems by modeling the **rationale behind expert problem design**. This approach improves problem diversity and difficulty while ensuring **logically consistent problem construction**. 

📄 **Paper**: [🔗 PromptCoT: Synthesizing Olympiad-Level Problems for Mathematical Reasoning in Large Language Models](http://arxiv.org/abs/2503.02324).

### **Key Features**
- **Concept-Guided Problem Synthesis**: PromptCoT generates problems by systematically combining **mathematical concepts**, allowing for a **scalable** and **flexible** way to create a diverse range of challenging problems. 

- **Rationale-Driven Problem Formulation**: Instead of directly generating problems, PromptCoT first constructs an **intermediate reasoning process (rationale)**—a step-by-step thought process that mimics how expert problem designers craft questions. This rationale helps bridge the gap between abstract mathematical concepts and well-formed problems, ensuring logical consistency and problem difficulty.

- **Rejection Sampling for Quality Control**: Problems undergo an automated evaluation process where multiple reward models assess their quality. Only problems receiving the highest scores are retained, ensuring the final dataset consists of **challenging and high-quality** mathematical problems.

- **Scalability & Adaptability**: The method allows for **large-scale problem generation** across a wide range of mathematical domains. Additionally, the rationale-driven approach can be adapted to **other structured reasoning tasks** beyond mathematics.

---

## **Quick Start: Generating Olympiad-Level Problems**
Follow these steps to generate problems using **PromptCoT**.

### **1. Install Dependencies**
```bash
pip install sentence_transformers==3.2.1 scikit-learn==1.3.2 scipy==1.10.1 faiss-gpu==1.7.2 vllm==0.6.3 transformers==4.46.3 fire==0.7.0
pip install str2bool
```

### **2. Generating Problems**
#### **Step 1: Generate Concept Embeddings**
We first encode mathematical concepts into embeddings to enable efficient sampling:

```bash
python concept_encoding.py \
  --data_path data/mathematics_concepts.jsonl \
  --output_path data/embeddings.jsonl \
  --model_path /path/to/Llama-3.1-8B \
  --n_gpus 4
```

#### **Step 2: Sample Concept Combinations**
We then sample meaningful concept combinations for problem generation:

```bash
python concept_sampling.py \
  --data_path data/mathematics_concepts.jsonl \
  --output_path data/problem_generation_inputs.jsonl \
  --data_size 1000 \
  --embed_path data/embeddings.jsonl
```

#### **Step 3: Generate Math Problems**
Using the **pre-trained problem generation model** – available on [**Hugging Face**](https://huggingface.co/xl-zhao/PromptCoT-Problem-Generation-Model) | [**ModelScope**](https://www.modelscope.cn/models/zhaoxlpku/PromptCoT-Problem-Generation-Model) – we generate Olympiad-level math problems:  

```bash
python problem_generation.py \
  --data_path data/problem_generation_inputs.jsonl \
  --output_path data/problem_generation_outputs.jsonl \
  --model_path /path/to/problem_generation_model \
  --n_gpus 4 \
  --temperature 0.6 \
  --max_len 4096 \
  --seed 8000
```

#### **Step 4: Reward-Based Filtering**
To ensure high-quality problem selection, we compute **reward scores** using two evaluation models:

```bash
python rejection_sampling_reward.py \
  --data_path data/problem_generation_outputs.jsonl \
  --output_path data/problem_generation_outputs_reward0.jsonl \
  --model_path /path/to/Llama-3.1-70B-Instruct \
  --n_gpus 4 \
  --temperature 0.6 \
  --use_chat_template True \
  --seed 8000

python rejection_sampling_reward.py \
  --data_path data/problem_generation_outputs.jsonl \
  --output_path data/problem_generation_outputs_reward1.jsonl \
  --model_path /path/to/Qwen2.5-72B-Instruct \
  --n_gpus 4 \
  --temperature 0.6 \
  --use_chat_template True \
  --seed 8000
```

#### **Step 5: Select High-Quality Problems**
To ensure only the **highest-quality** problems are used for training, we apply a filtering process based on reward scores. Problems that receive **perfect ratings from multiple evaluators** are retained.

```bash
python problem_filtering.py \
  --template data/problem_generation_outputs_reward{}.jsonl \
  --output_path data/problem_generation_training.jsonl \
  --only_perfect True \
  --n_rewards 2
```

📌 **Our curated dataset of high-quality problems** (where each problem received **perfect ratings** across all evaluation criteria) is available here: [Hugging Face](https://huggingface.co/datasets/xl-zhao/PromptCoT-Problem-Generation-Dataset) | [ModelScope](https://www.modelscope.cn/datasets/zhaoxlpku/PromptCoT-Problem-Generation-Dataset)

---

## **Distillation**
After generating high-quality problems, we distill the knowledge into **smaller models** using **Deepseek-R1-Distill-Qwen-7B** as the teacher model. We train:
- **PromptCoT-DS-1.5B** (Student: Deepseek-R1-Distill-Qwen-1.5B)
- **PromptCoT-DS-7B** (Student: Deepseek-R1-Distill-Qwen-7B)


---

## **Reproducing Our Results**
To reproduce the results, follow these steps.

#### **Step 1: Install Dependencies**
```bash
conda create -n promptcot python=3.10.14
conda activate promptcot
pip install -r requirements.txt --ignore-installed --no-deps
```

#### **Step 2: Run Inference on Benchmark Datasets**
To run inference for the PromptCoT-DS series models, use the following command:

```bash
python infer_longcot.py \
  --data_path data/{dataset_name}.jsonl \
  --output_path data/{dataset_name}_predictions.jsonl \
  --model_path /path/to/{model_name} \
  --tokenizer_path /path/to/Deepseek-R1-Distill-Qwen-1.5B \
  --n_gpus 1 \
  --temperature 0.6 \
  --max_len 32768
  --n 8
```
where `{dataset_name}` can be:
- `gsm8k`
- `math500`
- `aime2024`
- `aime2025`

and `{model_name}` can be:
- `PromptCoT-DS-1.5B`
- `PromptCoT-DS-7B`

To run inference for PromptCoT-QwQ-32B, use the following command:

```bash
python infer_longcot.py \
  --data_path data/qwq/qwq_{dataset_name}_test.jsonl \
  --output_path data/qwq/qwq_{dataset_name}_predictions.jsonl \
  --model_path /path/to/PromptCoT-QwQ-32B \
  --tokenizer_path /path/to/QwQ-32B \
  --n_gpus 2 \
  --temperature 0.6 \
  --max_len 16384
  --n 8
```
where `{dataset_name}` can be:
- `gsm8k`
- `math500`
- `aime2024`
- `aime2025`

#### **Step 3: Compute Accuracy**
```bash
python calc_acc.py \
  --output_path data/{dataset_name}_predictions.jsonl
```

#### **[New]** **Step 4: Train with DeepSpeed**

You can reproduce the training process for the model using DeepSpeed with the following commands. Make sure to replace the paths with your own data and model paths.

- **For PromptCoT-DS-1.5B**:

```bash
deepspeed --num_gpus=8 train.py --bf16=True --data_path=/path/to/PromptCoT-DS-Dataset --ddp_find_unused_parameters=False --deepspeed=configs/promptcot_ds_1_5b_config.json --evaluation_strategy=no --fp16=False --gradient_accumulation_steps=8 --gradient_checkpointing=True --learning_rate=5e-06 --load_best_model_at_end=False --logging_steps=1 --model_max_length=16384 --model_name_or_path=/path/to/DeepSeek-R1-Distill-Qwen-1.5B --num_train_epochs=2 --output_dir=/path/to/PromptCoT-DS-1.5B --per_device_train_batch_size=1 --resume_from_checkpoint=False --save_steps=500 --save_strategy=steps --save_total_limit=6 --tokenizer_path=/path/to/DeepSeek-R1-Distill-Qwen-1.5B --warmup_steps=100 --weight_decay=0.01
```

- **For PromptCoT-DS-7B**:

```bash
deepspeed --num_gpus=8 train.py --bf16=True --data_path=/path/to/PromptCoT-DS-Dataset --ddp_find_unused_parameters=False --deepspeed=configs/promptcot_ds_7b_config.json --evaluation_strategy=no --fp16=False --gradient_accumulation_steps=8 --gradient_checkpointing=True --learning_rate=5e-06 --load_best_model_at_end=False --logging_steps=1 --model_max_length=16384 --model_name_or_path=/path/to/DeepSeek-R1-Distill-Qwen-7B --num_train_epochs=2 --output_dir=/path/to/PromptCoT-DS-7B --per_device_train_batch_size=1 --resume_from_checkpoint=False --save_steps=500 --save_strategy=steps --save_total_limit=6 --tokenizer_path=/path/to/DeepSeek-R1-Distill-Qwen-7B --warmup_steps=100 --weight_decay=0.01
```

- **For PromptCoT-QwQ-32B**:

```bash
deepspeed --num_gpus=8 train.py --bf16=True --data_path=/path/to/PromptCoT-QwQ-Dataset --ddp_find_unused_parameters=False --deepspeed=configs/promptcot_qwq_32b_config.json --evaluation_strategy=no --fp16=False --gradient_accumulation_steps=2 --gradient_checkpointing=True --learning_rate=2e-06 --load_best_model_at_end=False --logging_steps=1 --model_max_length=16384 --model_name_or_path=/path/to/QwQ-32B --num_train_epochs=2 --output_dir=/path/to/PromptCoT-QwQ-32B --per_device_train_batch_size=1 --resume_from_checkpoint=False --save_steps=500 --save_strategy=steps --save_total_limit=6 --tokenizer_path=/path/to/QwQ-32B --warmup_steps=100 --weight_decay=0.01
```


---

## **Citation**
If you find **PromptCoT** useful, please consider citing:

```
@article{zhao2025promptcot,
  author    = {Zhao, Xueliang and Wu, Wei and Guan, Jian and Kong, Lingpeng},
  title     = {PromptCoT: Synthesizing Olympiad-Level Problems for Mathematical Reasoning in Large Language Models},
  year      = {2025},
  journal   = {arXiv preprint arXiv:2503.02324},
  url       = {http://arxiv.org/abs/2503.02324}
}
```
