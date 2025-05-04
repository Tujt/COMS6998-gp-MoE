import os
import json
import torch
import shutil
import wandb
import time
import numpy as np
from transformers import default_data_collator
import logging
import argparse
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
import deepspeed
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaForCausalLM,
    TrainingArguments as HFTrainingArguments,
    HfArgumentParser,
)
from datasets import load_dataset, load_from_disk, concatenate_datasets
from modeling_file.modeling_llama_moe import LlamaMoEForCausalLM
from huggingface_hub import snapshot_download
from math import ceil

try:
    from safetensors import safe_open
except ImportError:
    safe_open = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

IGNORE_INDEX = -100

if torch.cuda.is_bf16_supported():
    logger.info('='*80)
    logger.info('Your GPU supports bfloat16, you can accelerate training with the argument --bf16')
    logger.info('='*80)

def _tokenize_fn(strings: Sequence[str], tokenizer) -> Dict:
    tokenized_list = [
        tokenizer(
            text,
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = [np.array(tokenized.input_ids) for tokenized in tokenized_list]
    input_ids_lens = [len(tokenized.input_ids) for tokenized in tokenized_list]
    return dict(
        input_ids=input_ids,
        input_ids_lens=input_ids_lens,
    )

def train_tokenize_function(examples, tokenizer):
    tokenized = tokenizer(
        examples['text'],
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_attention_mask=True,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

@dataclass
class ModelArguments:
    experiment_type: str = field(
        default="dense", metadata={"help": "模型类型：'dense' 或 'moe'"}
    )
    router_strategy: str = field(
        default="random", metadata={"help": "MoE 的 router 初始化策略：'random' 或 'mixtral'"}
    )
    model_name_or_path: str = field(
        default="/root/V429/hf_models/TinyLlama/TinyLlama-1.1B-Chat-v1.0", metadata={"help": "预训练模型路径"}
    )
    use_lora: bool = field(default=True, metadata={"help": "是否使用 LoRA 微调"})
    lora_trainable: str = field(default="q_proj,v_proj,k_proj,o_proj", metadata={"help": "LoRA 需要训练的模块"})
    lora_rank: int = field(default=8, metadata={"help": "LoRA 的 rank"})

@dataclass
class DataArguments:
    data_path: str = field(default="/root/V429/dataset/AskNews-NER-v0", metadata={"help": "训练数据路径"})
    eval_path: str = field(default="", metadata={"help": "评估数据路径，可选"})

@dataclass
class TrainingArguments(HFTrainingArguments):
    do_train: bool = field(default=True)
    do_eval: bool = field(default=False)
    model_max_length: int = field(default=1024, metadata={"help": "最大序列长度"})
    output_dir: str = field(default="/root/autodl-tmp/V429/output", metadata={"help": "Output directory for checkpoints"})
    remove_unused_columns: bool = field(default=False, metadata={"help": "保留数据集中未被模型 forward 使用的列"})
    deepspeed: str = field(default="/root/V429/ds_config.json", metadata={"help": "Path to DeepSpeed config file"})
    save_interval: int = field(default=500, metadata={"help": "Interval for saving checkpoints"})
    resume_from_checkpoint: bool = field(default=False, metadata={"help": "Resume from checkpoint"})
    per_device_train_batch_size: int = field(default=8, metadata={"help": "Batch size per GPU"})
    gradient_accumulation_steps: int = field(default=2, metadata={"help": "Gradient accumulation steps"})
    logging_steps: int = field(default=10, metadata={"help": "Steps between logging"})
    wandb_project: str = field(default="llama-training-20250429", metadata={"help": "WandB project name"})
    wandb_entity: str = field(default="6998gp_TLA", metadata={"help": "WandB entity name (team or organization)"})
    use_wandb: bool = field(default=True, metadata={"help": "Whether to use WandB for logging"})

def build_model(model_args: ModelArguments, training_args: TrainingArguments, checkpoint_dir: Optional[str] = None):
    if not os.path.isdir(model_args.model_name_or_path):
        logger.info(f"Downloading model from Hugging Face Hub: TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        model_args.model_name_or_path = snapshot_download(
            repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            cache_dir="/root/V429/hf_models"
        )
    if model_args.experiment_type == "dense":
        model = LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float32,
            trust_remote_code=True,
            use_cache=False,
        )
    else:
        moe_weights_dir = os.path.join("/root/V429/converted_moe", os.path.basename(model_args.model_name_or_path))
        if not os.path.exists(moe_weights_dir):
            logger.info("转换 dense 权重为 MoE 权重...")
            duplicate_mlp(
                ckpt_dir=model_args.model_name_or_path,
                moe_dir=moe_weights_dir,
                num_experts=2,
                num_experts_per_token=1,
                output_router_logits=True,
                router_aux_loss_coef=0.05,
            )
            if model_args.router_strategy == "mixtral":
                logger.info("执行 router 热启动：从 Mixtral 权重中加载 router 参数...")
                conver_router(
                    mixtral_model_path="/root/V429/hf_models/mixtral-instruct",
                    llama3_moe_router_warmboot=moe_weights_dir,
                )
        model = LlamaMoEForCausalLM.from_pretrained(
            moe_weights_dir,
            torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float32,
            trust_remote_code=True,
            use_cache=False,
        )
    if model_args.use_lora:
        try:
            from peft import get_peft_model, LoraConfig
        except ImportError:
            raise ImportError("请安装 peft 库以使用 LoRA 模块：pip install peft")
        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            target_modules=model_args.lora_trainable.split(","),
            inference_mode=False,
            r=model_args.lora_rank,
        )
        model = get_peft_model(model, peft_config)
    return model

def compute_metrics_(prediction):
    logits = prediction.predictions
    labels = prediction.label_ids
    pred_tokens = np.argmax(logits, axis=-1)
    accuracy = (pred_tokens == labels).mean()
    return {"accuracy": accuracy}

def train(model_args: ModelArguments, data_args: DataArguments, training_args: TrainingArguments):
    # Initialize WandB
    if training_args.use_wandb:
        wandb.init(
            project=training_args.wandb_project,
            entity=training_args.wandb_entity,
            config={
                "experiment_type": model_args.experiment_type,
                "router_strategy": model_args.router_strategy,
                "model_name_or_path": model_args.model_name_or_path,
                "use_lora": model_args.use_lora,
                "lora_trainable": model_args.lora_trainable,
                "lora_rank": model_args.lora_rank,
                "data_path": data_args.data_path,
                "model_max_length": training_args.model_max_length,
                "per_device_train_batch_size": training_args.per_device_train_batch_size,
                "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
                "logging_steps": training_args.logging_steps,
                "save_interval": training_args.save_interval,
                "bf16": training_args.bf16,
                "deepspeed": training_args.deepspeed
            }
        )
        wandb_start_time = time.time()

    logger.info(f"Using DeepSpeed config: {training_args.deepspeed}")
    if not os.path.exists(training_args.deepspeed):
        raise FileNotFoundError(f"DeepSpeed config file not found at {training_args.deepspeed}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    raw_train_dataset = load_from_disk(data_args.data_path)
    def concat_fields(example):
        return {
            "text": example["instruction"].strip() + " " + example["input"].strip() + " " + example["output"].strip()
        }

    # 数据集重复 1 次，总样本 1444
    raw_train_dataset = concatenate_datasets([raw_train_dataset, raw_train_dataset])
    logger.info(f"Dataset repeated 1 time, total samples: {len(raw_train_dataset)}")

    tokenized_dataset_path = os.path.join(data_args.data_path, "tokenized_repeated")
    if os.path.exists(tokenized_dataset_path):
        logger.info(f"Loading tokenized dataset from {tokenized_dataset_path}")
        train_dataset = load_from_disk(tokenized_dataset_path)
    else:
        logger.info("Tokenizing dataset and saving to disk...")
        train_dataset = raw_train_dataset.map(concat_fields, remove_columns=raw_train_dataset.column_names)
        train_dataset = train_dataset.map(
            lambda examples: train_tokenize_function(examples, tokenizer),
            batched=True,
            remove_columns=["text"],
            desc="Tokenizing dataset",
        )
        train_dataset.save_to_disk(tokenized_dataset_path)

    model = build_model(model_args, training_args, checkpoint_dir=None)

    # 修改：使用 requires_grad 筛选可训练参数
    model_parameters = [p for p in model.parameters() if p.requires_grad]
    model_engine, optimizer, train_dataloader, _ = deepspeed.initialize(
        args=training_args,
        model=model,
        model_parameters=model_parameters,
        training_data=train_dataset,
        collate_fn=default_data_collator,
        config=training_args.deepspeed
    )

    logger.info(f"DataLoader length: {len(train_dataloader)}")

    start_step = 0
    client_sd = {}
    if training_args.resume_from_checkpoint:
        logger.info(f"Resuming from checkpoint in {training_args.output_dir}")
        checkpoint_dir = os.path.join(training_args.output_dir, "checkpoint-500")
        if os.path.exists(checkpoint_dir):
            logger.info(f"Loading checkpoint from {checkpoint_dir}")
            _, client_sd = model_engine.load_checkpoint(training_args.output_dir, "checkpoint-500")
        else:
            logger.warning(f"No checkpoint found in {checkpoint_dir}, starting from scratch")
            training_args.resume_from_checkpoint = False

    if training_args.resume_from_checkpoint:
        start_step = client_sd.get('step', 0)
        for _ in range(start_step):
            next(train_dataloader, None)

    if training_args.do_train:
        logger.info("Starting training with DeepSpeed...")
        model_engine.train()
        num_epochs = 5  # 设置 5 个 epoch
        samples_per_epoch = len(train_dataset)  # 1444 样本
        steps_per_epoch = len(train_dataloader)  # 使用实际 DataLoader 长度
        total_steps = num_epochs * steps_per_epoch  # 5 × 361 ≈ 1805
        logger.info(f"Expected samples per epoch: {samples_per_epoch}, Steps per epoch: {steps_per_epoch}, Total steps: {total_steps}")

        global_step = start_step
        step_start_time = time.time()
        for epoch in range(num_epochs):
            logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")
            epoch_step = 0
            for batch in train_dataloader:
                batch = {k: v.to(model_engine.device) for k, v in batch.items()}
                outputs = model_engine(**batch)
                loss = outputs.loss
                model_engine.backward(loss)
                model_engine.step()

                global_step += 1
                epoch_step += 1

                if global_step % training_args.logging_steps == 0:
                    step_end_time = time.time()
                    step_time = (step_end_time - step_start_time) / training_args.logging_steps
                    grad_norm = sum(p.grad.norm().item() for p in model_parameters if p.grad is not None)
                    logger.info(f"Step: {global_step}, Epoch: {epoch + 1}, Epoch Step: {epoch_step}, Loss: {loss.item()}")
                    if training_args.use_wandb and model_engine.global_rank == 0:
                        wandb.log({
                            "step": global_step,
                            "epoch": epoch + 1,
                            "epoch_step": epoch_step,
                            "loss": loss.item(),
                            "learning_rate": optimizer.param_groups[0]['lr'],
                            "grad_norm": grad_norm,
                            "gpu_memory_allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                            "gpu_memory_reserved_mb": torch.cuda.memory_reserved() / 1024**2,
                            "epoch_progress": epoch_step / steps_per_epoch,
                            "step_progress": global_step / total_steps,
                            "training_time_sec": time.time() - wandb_start_time,
                            "step_time_sec": step_time
                        })
                    step_start_time = time.time()

                # 每 500 步输出专家频率（仅 MoE 模式）
                if model_args.experiment_type == "moe" and global_step % 500 == 0:
                    router_logits = outputs.router_logits if hasattr(outputs, "router_logits") else None
                    if router_logits is not None:
                        try:
                            if isinstance(router_logits, tuple):
                                router_logits = router_logits[0]  # 取第一个输出（如果为元组）
                            # 计算 top-k 专家索引（基于 num_experts_per_token）
                            _, selected_experts = torch.topk(router_logits, k=1, dim=-1)  # num_experts_per_token=1
                            expert_counts = torch.bincount(selected_experts.flatten(), minlength=2)  # num_experts=2
                            logger.info(f"Step: {global_step}, Expert selection counts: {expert_counts.tolist()}")
                            if training_args.use_wandb and model_engine.global_rank == 0:
                                wandb.log({"expert_selection_counts": expert_counts.tolist(), "step": global_step})
                        except Exception as e:
                            logger.warning(f"Failed to log expert selection counts: {str(e)}")

                if global_step % training_args.save_interval == 0 and global_step > start_step:
                    client_sd['step'] = global_step
                    ckpt_id = f"checkpoint-{global_step}"
                    model_engine.save_checkpoint(training_args.output_dir, ckpt_id)
                    tokenizer.save_pretrained(os.path.join(training_args.output_dir, ckpt_id))
                    client_sd_path = os.path.join(training_args.output_dir, ckpt_id, "client_sd.json")
                    with open(client_sd_path, 'w', encoding='utf8') as f:
                        json.dump(client_sd, f, indent=4)
                    logger.info(f"Checkpoint saved at {training_args.output_dir}/{ckpt_id}")
                    if training_args.use_wandb and model_engine.global_rank == 0:
                        wandb.log({"checkpoint": ckpt_id})

            logger.info(f"Epoch {epoch + 1} completed with {epoch_step} steps")

        # 保存最终模型
        client_sd['step'] = global_step
        ckpt_id = "final_model"
        model_engine.save_checkpoint(training_args.output_dir, ckpt_id)
        tokenizer.save_pretrained(os.path.join(training_args.output_dir, ckpt_id))
        client_sd_path = os.path.join(training_args.output_dir, ckpt_id, "client_sd.json")
        with open(client_sd_path, 'w', encoding='utf8') as f:
            json.dump(client_sd, f, indent=4)
        logger.info(f"Final model saved at {training_args.output_dir}/{ckpt_id}")
        if training_args.use_wandb and model_engine.global_rank == 0:
            wandb.log({"final_model": ckpt_id})

        # 训练完成后输出专家频率（仅 MoE 模式）
        if model_args.experiment_type == "moe":
            router_logits = outputs.router_logits if hasattr(outputs, "router_logits") else None
            if router_logits is not None:
                try:
                    if isinstance(router_logits, tuple):
                        router_logits = router_logits[0]
                    _, selected_experts = torch.topk(router_logits, k=1, dim=-1)
                    expert_counts = torch.bincount(selected_experts.flatten(), minlength=2)
                    logger.info(f"Final step: {global_step}, Expert selection counts: {expert_counts.tolist()}")
                    if training_args.use_wandb and model_engine.global_rank == 0:
                        wandb.log({"final_expert_selection_counts": expert_counts.tolist(), "step": global_step})
                except Exception as e:
                    logger.warning(f"Failed to log final expert selection counts: {str(e)}")

        if training_args.use_wandb:
            wandb.finish()

def duplicate_mlp(
        ckpt_dir: str,
        moe_dir: str,
        num_experts: int = 2,
        num_experts_per_token: int = 1,
        output_router_logits: bool = True,
        router_aux_loss_coef: float = 0.05,
):
    os.makedirs(moe_dir, exist_ok=True)
    for filename in tqdm(os.listdir(ckpt_dir), desc="Converting MLP to MoE experts"):
        filepath = os.path.join(ckpt_dir, filename)
        if filename in ["pytorch_model.bin.index.json", "model.safetensors.index.json"]:
            index_map = json.load(open(filepath, "r", encoding="utf8"))
            new_index_map = {
                "metadata": index_map["metadata"],
                "weight_map": {}
            }
            for k, v in index_map["weight_map"].items():
                if "safetensors" in filename:
                    v = "pytorch_" + v.replace("safetensors", "bin")
                if "mlp" in k:
                    for i in range(num_experts):
                        name = k.replace("mlp", f"block_sparse_moe.experts.{i}")
                        new_index_map["weight_map"][name] = v
                else:
                    new_index_map["weight_map"][k] = v
            new_index_path = os.path.join(moe_dir, "pytorch_model.bin.index.json")
            if not os.path.exists(new_index_path):
                json.dump(new_index_map, open(new_index_path, "w", encoding="utf8"), indent=4, ensure_ascii=False)
        elif (".bin" in filename) or (".safetensors" in filename):
            if ".bin" in filename:
                weights = torch.load(filepath, map_location="cpu")
            else:
                weights = safe_open_weight(ckpt_dir, filename)
            new_weights = {}
            for k, v in weights.items():
                if "mlp" in k:
                    for i in range(num_experts):
                        name = k.replace("mlp", f"block_sparse_moe.experts.{i}")
                        new_weights[name] = v
                else:
                    new_weights[k] = v
            if ".bin" in filename:
                new_path = os.path.join(moe_dir, filename)
            else:
                new_path = os.path.join(moe_dir, "pytorch_" + filename.replace("safetensors", "bin"))
            if not os.path.exists(new_path):
                torch.save(new_weights, new_path)
        elif filename == "config.json":
            config = json.load(open(filepath, "r", encoding="utf8"))
            config["num_local_experts"] = num_experts
            config["num_experts_per_tok"] = num_experts_per_token
            config["output_router_logits"] = output_router_logits
            config["router_aux_loss_coef"] = router_aux_loss_coef
            new_config_path = os.path.join(moe_dir, filename)
            if not os.path.exists(new_config_path):
                json.dump(config, open(new_config_path, "w", encoding="utf8"), indent=4, ensure_ascii=False)
        else:
            if os.path.isfile(filepath):
                shutil.copyfile(filepath, os.path.join(moe_dir, filename))

def conver_router(mixtral_model_path: str, llama3_moe_router_warmboot: str):
    mixtral_index_path = os.path.join(mixtral_model_path, "model.safetensors.index.json")
    moe_index_path = os.path.join(llama3_moe_router_warmboot, "pytorch_model.bin.index.json")
    mixtral_index = json.load(open(mixtral_index_path, "r", encoding="utf8"))
    moe_index = json.load(open(moe_index_path, "r", encoding="utf8"))
    for k, v in mixtral_index["weight_map"].items():
        if "gate" in k:
            layer_id = get_layer_id(k)
            v_replace = transfer_value(v, layer_id)
            moe_index["weight_map"][k] = v_replace
    json.dump(moe_index, open(moe_index_path, "w", encoding="utf8"), indent=4, ensure_ascii=False)
    weight_files = {
        "pytorch_model-00001-of-00004.bin": torch.load(os.path.join(llama3_moe_router_warmboot, "pytorch_model-00001-of-00004.bin"), map_location="cpu"),
        "pytorch_model-00002-of-00004.bin": torch.load(os.path.join(llama3_moe_router_warmboot, "pytorch_model-00002-of-00004.bin"), map_location="cpu"),
        "pytorch_model-00003-of-00004.bin": torch.load(os.path.join(llama3_moe_router_warmboot, "pytorch_model-00003-of-00004.bin"), map_location="cpu"),
    }
    for filename in os.listdir(mixtral_model_path):
        if (".bin" not in filename) and (".safetensors" not in filename):
            continue
        file_path = os.path.join(mixtral_model_path, filename)
        if ".bin" in filename:
            weights = torch.load(file_path, map_location="cpu")
        else:
            weights = safe_open_weight(mixtral_model_path, filename)
        for k, v in weights.items():
            if "gate" in k:
                layer_id = get_layer_id(k)
                if layer_id <= 8:
                    weight_files["pytorch_model-00001-of-00004.bin"][k] = v
                elif layer_id <= 20:
                    weight_files["pytorch_model-00002-of-00004.bin"][k] = v
                else:
                    weight_files["pytorch_model-00003-of-00004.bin"][k] = v
    for fname, w in weight_files.items():
        torch.save(w, os.path.join(llama3_moe_router_warmboot, fname))

def get_layer_id(key: str) -> int:
    try:
        return int(key.split(".")[2])
    except (IndexError, ValueError):
        return -1

def transfer_value(v: str, layer_id: int) -> str:
    if layer_id <= 8:
        return "pytorch_model-00001-of-00004.bin"
    elif layer_id <= 20:
        return "pytorch_model-00002-of-00004.bin"
    elif layer_id <= 31:
        return "pytorch_model-00003-of-00004.bin"
    else:
        return "pytorch_model-00004-of-00004.bin"

def safe_open_weight(model_path: str, filename: str) -> Dict:
    if safe_open is None:
        raise ImportError("请安装 safetensors 库：pip install safetensors")
    weights = {}
    file_path = os.path.join(model_path, filename)
    with safe_open(file_path, framework="pt") as f:
        for key in f.keys():
            weights[key] = f.get_tensor(key)
    return weights

def test_inference(model_args: ModelArguments, data_args: DataArguments, training_args: TrainingArguments):
    try:
        from transformers import AutoTokenizer
        model_class = LlamaMoEForCausalLM if model_args.experiment_type == "moe" else AutoModelForCausalLM
    except ImportError:
        print("请确保安装 transformers 库：pip install transformers")
        return

    model_ckpt = os.path.join(training_args.output_dir, "final_model")
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt, padding_side='left')
    model = model_class.from_pretrained(
        model_ckpt,
        torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float32,
        device_map="auto",
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("开始与模型对话（输入 'exit' 退出）：")
    history = []
    while True:
        user_input = input("您: ")
        if user_input.lower() == 'exit':
            break
        history.append({"role": "user", "content": user_input})
        prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                pad_token_id=tokenizer.eos_token_id,
                max_new_tokens=200,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                temperature=0.7
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"模型: {response}")
        history.append({"role": "assistant", "content": response})

    if training_args.use_wandb:
        wandb.init(project=training_args.wandb_project, entity=training_args.wandb_entity, name="test-run")
        wandb.log({"test_output": response})
        wandb.finish()

def main():
    parser = argparse.ArgumentParser(description="MoE vs Dense LLM Training Experiment")
    parser.add_argument("--run_mode", type=str, default="train", choices=["train", "test"],
                        help="运行模式：train 训练，test 推理测试")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    args, remaining_args = parser.parse_known_args()

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses(args=remaining_args)

    logger.info(f"Parsed TrainingArguments.deepspeed: {training_args.deepspeed}")

    if args.run_mode == "train":
        train(model_args, data_args, training_args)
    elif args.run_mode == "test":
        test_inference(model_args, data_args, training_args)
    else:
        print("无效的 run_mode 参数。")

if __name__ == "__main__":
    main()
