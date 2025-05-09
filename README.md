# COMS6998-gp-MoE
## Introduction
This is the groupwork of the course COMS6998 High-Performance Machine Learning in Columbia University, Spring 2025. The group Menbers are Tom, Layton and Andy.

This project explores the integration of Mixture-of-Experts (MoE) architectures into a TinyLlama-based LLM to improve training efficiency without sacrificing performance. MoE models activate only a subset of the network’s parameters per input, enabling high-capacity modeling with reduced compute overhead. By applying sparse expert routing, we aim to build a more scalable model that is friendly to resource-constrained environments.

Our goals are twofold:

Evaluate whether a 1B-parameter MoE-enhanced TinyLlama can match or outperform its dense version on NER tasks.

Investigate how optimization techniques like DeepSpeed ZeRO-3 and expert offloading can further reduce training costs and memory usage.

To ensure valid evaluation, we fine-tune on AskNews-NER-v0, a curated dataset not seen during TinyLlama’s pretraining. This ensures improvements come from actual learning rather than memorization.

We conduct head-to-head comparisons of dense and MoE variants under identical training setups, tracking both performance (e.g., F1-score) and efficiency metrics (e.g., GPU memory usage, convergence speed). Our findings aim to inform the design of smaller, deployable LLMs that remain performant even under tight resource constraints.

This repository includes all code, training scripts, and configuration files for reproducing our experiments.

## Code Repository:
Main components:
project_run.py is the main code used to train the models.  

req.txt and set_up.sh are used for set up the environment for training.  
ds_config.json is the configuration file for DeepSpeed, defining optimization strategies and hyperparameters for distributed training.  
train_all.sh and train_single+eval.sh are the scripts for running the training.
1.eval.py in the folder named "eval" is the code used for evaluation.
1.data_preprocessing.py is the file for data preprocessing.

On the Virtual Machine: We put these codes in /root/V429.

## Setup (Windows)
1. Install conda
2. Create conda env: `conda create --name <env> --file req.txt` or use set_up.sh by `./set_up.sh` (May need to use `chmod +x` first)
4. Download `https://huggingface.co/datasets/cognitivecomputations/dolphin/blob/main/flan1m-alpaca-uncensored-deduped.jsonl` and save to `./dataset`
5. Run `1.data_preprocessing.py`
6. Run one of the following commands.

## Commands
- Training all the models:
``
./train_all.sh
``

- Train a specific model and evaluate it:
``
./train_single+eval.sh
``

## Results & Evaluation

We trained and evaluated three variants of TinyLlama-1.1B on the **AskNews-NER-v0** dataset to compare dense and sparse (MoE) fine-tuning strategies under compute constraints. Each model was assessed on quality, training efficiency, and resource usage.

### Overall Performance Metrics

| Model                     | BLEU              | ROUGE-L           | METEOR            | Cosine Sim.      | NER Overlap       |
| ------------------------- | ----------------- | ----------------- | ----------------- | ---------------- | ----------------- |
| Dense (1 GPU)             | **0.256** (+848%) | **0.448** (+113%) | 0.382 (+101%)     | **0.556** (+39%) | **0.292** (+630%) |
| Dense (2 GPUs)            | 0.252 (+833%)     | 0.436 (+108%)     | **0.383** (+102%) | 0.551 (+37%)     | 0.291 (+628%)     |
| MoE (2 GPUs)              | 0.221 (+718%)     | 0.407 (+94%)      | 0.345 (+82%)      | 0.498 (+24%)     | 0.263 (+558%)     |
| TinyLlama-1.1B (original) | 0.027             | 0.210             | 0.190             | 0.401            | 0.040             |

> Source: `eval/summary_metrics.csv`

**Key Takeaways:**

* Dense models consistently outperform MoE across all metrics.
* All fine-tuned models substantially improve over the original TinyLlama base.
* MoE performs reasonably but lags behind in both quality and efficiency.

### Training Dynamics and Resource Usage (All 3 Models)

The plots below show **side-by-side comparisons of all three models** throughout training:

**Dense (1 GPU)**
**Dense (2 GPUs with ZeRO-2)**
**MoE (2 GPUs with top-1 routing)**

| Metric                  | Plot                                                    |
| ----------------------- | ------------------------------------------------------- |
| Training Loss Over Time | ![Training Loss](eval/training_loss_over_time.png)      |
| Step Time Over Time     | ![Step Time](eval/step_time_over_time.png)              |
| GPU Memory Reserved     | ![GPU Reserved](eval/gpu_memory_reserved_over_time.png) |
| GPU Memory Used         | ![GPU Used](eval/gpu_memory_used_over_time.png)         |
| Host RAM Usage          | ![RAM Usage](eval/ram_usage_over_time.png)              |

**Observations:**

* **Step Time:** Dense (1 GPU) is fastest (\~0.4s), while MoE is slowest (\~1.6s) due to routing overhead.
* **GPU Efficiency:** Dense (2 GPU) with ZeRO-2 uses the least GPU memory (\~10 GB) despite running across two devices.
* **MoE Overhead:** MoE uses more GPU memory (\~16.5 GB) and RAM (\~90 GB) due to expert routing and duplication.

### Profiler Insights

Using PyTorch Profiler + DeepSpeed tracing:

* **MoE vs. Dense:** MoE adds \~6.9s per profiling window from routing and tensor operations like `index_put`, `nonzero`, and extra `mm` ops.
* **Dense (2 GPUs):** ZeRO-2 improves memory use but introduces communication latency (\~8.5ms/step from `allreduce` and `record_param_comms`).

## WanDB Link
https://wandb.ai/6998gp_TLA/projects
