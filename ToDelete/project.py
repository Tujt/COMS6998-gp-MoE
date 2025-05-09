import os
import json
import torch
import shutil
from transformers import default_data_collator
import logging
import argparse
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence

from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments as HFTrainingArguments,
    HfArgumentParser, TrainerCallback, LlamaForCausalLM,
)
from datasets import load_dataset, load_from_disk

"""
from wandb_logger import init_wandb, log_metrics, save_checkpoint, finish_wandb at the top to import WandB utilities.
"""

from torch.profiler import profile, tensorboard_trace_handler
from modeling_file.modeling_llama_moe import LlamaMoEForCausalLM
from huggingface_hub import snapshot_download

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

# def build_instruction_prompt_llama3(examples, tokenizer):
#     PROMPT_FORMAT_SYSTEM = "<|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|>"
#     PROMPT_FORMAT_SINGLE = "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
#
#     sources = []
#     for instruction, user_input in zip(examples['instruction'], examples['input']):
#         system_msg = PROMPT_FORMAT_SYSTEM.format(instruction) if instruction.strip() else ""
#         user_msg = PROMPT_FORMAT_SINGLE.format(user_input)
#         sources.append(tokenizer.bos_token + system_msg + user_msg)
#     targets = [out + "<|eot_id|>" + tokenizer.eos_token for out in examples['output']]
#     data_dict = preprocess(sources, targets, tokenizer)
#     return data_dict
#
# def preprocess(sources: Sequence[str], targets: Sequence[str], tokenizer) -> Dict:
#     examples = [s + t for s, t in zip(sources, targets)]
#     examples_tokenized = _tokenize_fn(examples, tokenizer)
#     sources_tokenized = _tokenize_fn(sources, tokenizer)
#     input_ids = examples_tokenized["input_ids"]
#     labels = [np.copy(ids) for ids in input_ids]
#
#     for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
#         label[:source_len] = IGNORE_INDEX
#     return dict(input_ids=input_ids, labels=labels)
#
# @dataclass
# class DataCollatorForSupervisedDataset:
#     tokenizer: object
#
#     def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
#         input_ids = [torch.tensor(x) for x in [instance["input_ids"] for instance in instances]]
#         labels = [torch.tensor(x) for x in [instance["labels"] for instance in instances]]
#         input_ids = torch.nn.utils.rnn.pad_sequence(
#             input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
#         )
#         labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
#         return dict(
#             input_ids=input_ids,
#             labels=labels,
#             attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
#         )

def train_tokenize_function(examples, tokenizer):
    tokenized = tokenizer(
        examples['text'],
        padding=False,
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_attention_mask=True,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized
    # return build_instruction_prompt_llama3(examples, tokenizer)

@dataclass
class ModelArguments:
    experiment_type: str = field(
        default="dense", metadata={"help": "模型类型：'dense' 或 'moe'"}
    )
    router_strategy: str = field(
        default="random", metadata={"help": "MoE 的 router 初始化策略：'random' 或 'mixtral'"}
    )
    model_name_or_path: str = field(
        default="path/to/llama3-8b-instruct", metadata={"help": "预训练模型路径"}
    )
    use_lora: bool = field(default=False, metadata={"help": "是否使用 LoRA 微调"})
    lora_trainable: str = field(default="q_proj,v_proj,k_proj,o_proj", metadata={"help": "LoRA 需要训练的模块"})
    lora_rank: int = field(default=8, metadata={"help": "LoRA 的 rank"})

@dataclass
class DataArguments:
    data_path: str = field(default="data/train.json", metadata={"help": "训练数据路径"})
    eval_path: str = field(default="", metadata={"help": "评估数据路径，可选"})

@dataclass
class TrainingArguments(HFTrainingArguments):
    do_train: bool = field(default=True)
    do_eval: bool = field(default=False)
    model_max_length: int = field(default=1024, metadata={"help": "最大序列长度"})
    remove_unused_columns: bool = field(default=False, metadata={"help": "保留数据集中未被模型 forward 使用的列"})
    
    """
    wandb_project: str = field(default="llama-training", metadata={"help": "WandB project name"})  
    use_wandb: bool = field(default=False, metadata={"help": "Whether to use WandB for logging"})  
    """

def build_model(model_args: ModelArguments, training_args: TrainingArguments, checkpoint_dir: Optional[str] = None):
    if not os.path.isdir(model_args.model_name_or_path):
        logger.info(f"Downloading model from Hugging Face Hub: {model_args.model_name_or_path}")
        snapshot_download(repo_id=model_args.model_name_or_path,
                                                      cache_dir="./hf_models")
        model_args.model_name_or_path = model_args.model_name_or_path + "/hf_models"
    if model_args.experiment_type == "dense":
        model = LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            use_cache=False,
        )
    else:
        moe_weights_dir = os.path.join("converted_moe", os.path.basename(model_args.model_name_or_path))
        if not os.path.exists(moe_weights_dir):
            logger.info("转换 dense 权重为 MoE 权重...")
            duplicate_mlp(
                ckpt_dir=model_args.model_name_or_path,
                moe_dir=moe_weights_dir,
                num_experts=8,
                num_experts_per_token=2,
                output_router_logits=True,
                router_aux_loss_coef=0.02,
            )
            if model_args.router_strategy == "mixtral":
                logger.info("执行 router 热启动：从 Mixtral 权重中加载 router 参数...")
                conver_router(
                    mixtral_model_path="path/to/chinese-mixtral-instruct",
                    llama3_moe_router_warmboot=moe_weights_dir,
                )
        model = LlamaMoEForCausalLM.from_pretrained(
            moe_weights_dir,
            torch_dtype=torch.float32,
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

"""
class WandbCallback(TrainerCallback):  # Added for WandB
    def on_log(self, args, state, control, logs=None, **kwargs):
        if args.use_wandb and (state.is_local_process_zero or state.is_world_process_zero()):
            log_metrics(logs, step=state.global_step)

    def on_save(self, args, state, control, **kwargs):
        if args.use_wandb and (state.is_local_process_zero or state.is_world_process_zero()):
            ckpt_id = f"checkpoint-{state.global_step}"
            save_checkpoint(args.output_dir, ckpt_id)
"""

def train():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    """
    if training_args.use_wandb:
        init_wandb(training_args, model_args)
    """
    
    if not os.path.isdir(model_args.model_name_or_path):
        logger.info(f"Downloading model from Hugging Face Hub: {model_args.model_name_or_path}")
        model_args.model_name_or_path = snapshot_download(repo_id=model_args.model_name_or_path,
                                                          cache_dir="./hf_models")
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

    tokenized_dataset_path = os.path.join(data_args.data_path, "tokenized")
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


    # data_collator = default_data_collator(tokenizer)

    model = build_model(model_args, training_args, checkpoint_dir=None)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics_,
    )

    profiler = profile(
        activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=0, warmup=0, active=10, repeat=1),
        on_trace_ready=tensorboard_trace_handler(training_args.output_dir + "/profiler_logs"),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    )

    class ProfCallback(TrainerCallback):
        def __init__(self, prof):
            self.prof = prof

        def on_step_end(self, args, state, control, **kwargs):
            self.prof.step()

    with profiler:
        trainer.add_callback(ProfCallback(prof=profiler))
        
        """
        if training_args.use_wandb:  
            trainer.add_callback(WandbCallback())
        """
        
        logger.info("Starting training with Profiler...")
        if training_args.do_train:
            trainer.train()
            trainer.save_state()
            trainer.save_model(training_args.output_dir)
            tokenizer.save_pretrained(training_args.output_dir)

    if training_args.do_eval:
        metrics = trainer.evaluate()
        logger.info("Evaluation metrics: %s", metrics)
        
        """
        if training_args.use_wandb:  
            log_metrics(metrics)
        """
    """
    if training_args.use_wandb: 
        finish_wandb()
    """

def duplicate_mlp(
        ckpt_dir: str,
        moe_dir: str,
        num_experts: int = 8,
        num_experts_per_token: int = 2,
        output_router_logits: bool = True,
        router_aux_loss_coef: float = 0.02,
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

def test_inference():
    try:
        from modeling_file.llama3_moe.modeling_llama_moe import LlamaMoEForCausalLM
        from modeling_file.llama3_moe.tokenization_llama_fast import LlamaTokenizerFast
    except ImportError:
        print("请确保你已配置好自定义 LLaMA3-MoE 模型模块。")
        return

    model_ckpt = "path/to/your_saved_model"
    tokenizer = LlamaTokenizerFast.from_pretrained(model_ckpt, padding_side='left')
    model = LlamaMoEForCausalLM.from_pretrained(model_ckpt, device_map="auto", use_cache=False)

    text_list = ["Hello, what is your name?", "你好，你叫什么名字？", "今天天气怎么样？"]
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    inputs = tokenizer(text_list, return_tensors="pt", padding=True).to("cuda" if torch.cuda.is_available() else "cpu")
    output = model.generate(**inputs, pad_token_id=tokenizer.eos_token_id, max_new_tokens=100)
    print(tokenizer.batch_decode(output))


def main():
    parser = argparse.ArgumentParser(description="MoE vs Dense LLM Training Experiment")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"],
                        help="运行模式：train 训练，test 推理测试")
    args, remaining = parser.parse_known_args()

    """
    if args.local_rank != -1:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    """
    
    if args.mode == "train":
        train()
    elif args.mode == "test":
        test_inference()
    else:
        print("无效的 mode 参数。")

if __name__ == "__main__":
    main()
