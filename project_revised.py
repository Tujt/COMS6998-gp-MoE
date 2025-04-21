import os
import torch
import logging
import argparse
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence

import deepspeed
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    HfArgumentParser,
    TrainingArguments as HFTrainingArguments,
)
from datasets import load_from_disk
from huggingface_hub import snapshot_download

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
    input_ids = [tokenized.input_ids for tokenized in tokenized_list]
    input_ids_lens = [len(tokenized.input_ids) for tokenized in tokenized_list]
    return dict(
        input_ids=input_ids,
        input_ids_lens=input_ids_lens,
    )

def build_instruction_prompt_llama3(examples, tokenizer):
    PROMPT_FORMAT_SYSTEM = "<|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|>"
    PROMPT_FORMAT_SINGLE = "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

    sources = []
    for instruction, user_input in zip(examples['instruction'], examples['input']):
        system_msg = PROMPT_FORMAT_SYSTEM.format(instruction) if instruction.strip() else ""
        user_msg = PROMPT_FORMAT_SINGLE.format(user_input)
        sources.append(tokenizer.bos_token + system_msg + user_msg)
    targets = [out + "<|eot_id|>" + tokenizer.eos_token for out in examples['output']]
    data_dict = preprocess(sources, targets, tokenizer)
    return data_dict

def preprocess(sources: Sequence[str], targets: Sequence[str], tokenizer) -> Dict:
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized = _tokenize_fn(examples, tokenizer)
    sources_tokenized = _tokenize_fn(sources, tokenizer)
    input_ids = examples_tokenized["input_ids"]
    labels = [list(ids) for ids in input_ids]

    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = [IGNORE_INDEX] * source_len
    return dict(input_ids=input_ids, labels=labels)

@dataclass
class DataCollatorForSupervisedDataset:
    tokenizer: object

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [torch.tensor(x) for x in [instance["input_ids"] for instance in instances]]
        labels = [torch.tensor(x) for x in [instance["labels"] for instance in instances]]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def train_tokenize_function(examples, tokenizer):
    if 'instruction' in examples and 'input' in examples and 'output' in examples:
        return build_instruction_prompt_llama3(examples, tokenizer)
    else:
        tokenized = tokenizer(
            examples['text'],
            padding=False,
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_attention_mask=True,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="andrijdavid/Llama3-1B-Base", metadata={"help": "Path to pretrained model"}
    )
    use_lora: bool = field(default=False, metadata={"help": "Whether to use LoRA fine-tuning"})
    lora_trainable: str = field(default="q_proj,v_proj,k_proj,o_proj", metadata={"help": "LoRA trainable modules"})
    lora_rank: int = field(default=8, metadata={"help": "LoRA rank"})

@dataclass
class DataArguments:
    data_path: str = field(default="./dataset/AskNews-NER-v0", metadata={"help": "Path to training data"})
    eval_path: str = field(default="", metadata={"help": "Path to evaluation data, optional"})

@dataclass
class TrainingArguments(HFTrainingArguments):
    do_train: bool = field(default=True)
    do_eval: bool = field(default=False)
    model_max_length: int = field(default=1024, metadata={"help": "Maximum sequence length"})
    deepspeed: str = field(default="ds_config.json", metadata={"help": "Path to DeepSpeed config file"})
    save_interval: int = field(default=1000, metadata={"help": "Interval for saving checkpoints"})
    resume_from_checkpoint: bool = field(default=False, metadata={"help": "Resume from checkpoint"})
    per_device_train_batch_size: int = field(default=8, metadata={"help": "Batch size per GPU"})
    logging_steps: int = field(default=100, metadata={"help": "Steps between logging"})

def build_model(model_args: ModelArguments, training_args: TrainingArguments, checkpoint_dir: Optional[str] = None):
    if not os.path.isdir(model_args.model_name_or_path):
        logger.info(f"Downloading model from Hugging Face Hub: {model_args.model_name_or_path}")
        model_args.model_name_or_path = snapshot_download(
            repo_id=model_args.model_name_or_path,
            cache_dir="./hf_models"
        )
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        use_cache=False,
    )
    if model_args.use_lora:
        try:
            from peft import get_peft_model, LoraConfig
        except ImportError:
            raise ImportError("Please install peft library to use LoRA: pip install peft")
        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            target_modules=model_args.lora_trainable.split(","),
            inference_mode=False,
            r=model_args.lora_rank,
        )
        model = get_peft_model(model, peft_config)
    return model

def train():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Initialize distributed environment
    deepspeed.init_distributed()

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

    data_collator = DataCollatorForSupervisedDataset(tokenizer)
    model = build_model(model_args, training_args)

    # Initialize DeepSpeed
    model_parameters = [p for p in model.parameters() if p.requires_grad]
    model_engine, optimizer, train_dataloader, _ = deepspeed.initialize(
        args=training_args,
        model=model,
        model_parameters=model_parameters,
        training_data=train_dataset,
        collate_fn=data_collator,
    )

    # Load checkpoint if resuming
    start_step = 0
    client_sd = {}
    if training_args.resume_from_checkpoint:
        logger.info(f"Resuming from checkpoint in {training_args.output_dir}")
        _, client_sd = model_engine.load_checkpoint(training_args.output_dir, "latest")
        start_step = client_sd.get('step', 0)
        for _ in range(start_step):
            next(train_dataloader, None)

    # Training loop
    if training_args.do_train:
        logger.info("Starting training with DeepSpeed ZeRO-3...")
        model_engine.train()
        for step, batch in enumerate(train_dataloader, start=start_step):
            batch = {k: v.to(model_engine.device) for k, v in batch.items()}
            outputs = model_engine(**batch)
            loss = outputs.loss
            model_engine.backward(loss)
            model_engine.step()

            if step % training_args.logging_steps == 0 and model_engine.global_rank == 0:
                logger.info(f"Step: {step}, Loss: {loss.item()}")

            # Save checkpoint
            if step % training_args.save_interval == 0 and step > start_step and model_engine.global_rank == 0:
                client_sd['step'] = step
                ckpt_id = f"step_{step}"
                model_engine.save_checkpoint(training_args.output_dir, ckpt_id, client_sd=client_sd)
                tokenizer.save_pretrained(os.path.join(training_args.output_dir, ckpt_id))
                logger.info(f"Checkpoint saved at {training_args.output_dir}/{ckpt_id}")

def main():
    parser = argparse.ArgumentParser(description="Dense LLM Training with DeepSpeed ZeRO-3")
    parser.add_argument("--mode", type=str, default="train", choices=["train"], help="Run mode: train")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    args, _ = parser.parse_known_args()

    if args.local_rank != -1:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    if args.mode == "train":
        train()
    else:
        print("Invalid mode parameter.")

if __name__ == "__main__":
    main()
