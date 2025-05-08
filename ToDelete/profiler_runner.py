import os
import argparse
import torch
from torch.profiler import profile, tensorboard_trace_handler
from transformers import Trainer
import logging

# Import necessary functions and classes from project.py
from project import (
    build_model,
    train_tokenize_function,
    DataCollatorForSupervisedDataset,
    ModelArguments,
    DataArguments,
    TrainingArguments,
    HfArgumentParser,
    AutoTokenizer,
    load_from_disk,
    compute_metrics_,  # For evaluation metrics, if needed
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def run_profiler_training(model_path, data_path, output_dir):
    """Run training with PyTorch Profiler."""
    # Parse arguments using the same parser as in project.py
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Update paths based on command-line arguments
    model_args.model_name_or_path = model_path
    data_args.data_path = data_path
    training_args.output_dir = output_dir
    training_args.do_train = True
    training_args.do_eval = False  # Focus on training for profiling

    # Load tokenizer
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

    # Load and process dataset
    logger.info(f"Loading dataset from {data_args.data_path}")
    raw_train_dataset = load_from_disk(data_args.data_path)
    train_dataset = raw_train_dataset.map(
        lambda examples: train_tokenize_function(examples, tokenizer),
        batched=True,
        remove_columns=raw_train_dataset.column_names,
        desc="Tokenizing dataset",
    )

    # Data collator
    data_collator = DataCollatorForSupervisedDataset(tokenizer)

    # Build model
    logger.info(f"Loading model from {model_args.model_name_or_path}")
    model = build_model(model_args, training_args, checkpoint_dir=None)

    # Configure Profiler
    profiler = profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=tensorboard_trace_handler(os.path.join(training_args.output_dir, "profiler_logs")),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    )

    # Configure Trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics_,  # Include metrics function from project.py
    )

    # Start training with Profiler
    logger.info("Starting training with Profiler...")
    with profiler:
        trainer.train()
        profiler.step()  # Record the boundary of each training step

    # Save model and tokenizer
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    logger.info(f"Training completed, model and Profiler logs saved to {training_args.output_dir}")

def main():
    """Main function to parse arguments and run profiler training."""
    parser = argparse.ArgumentParser(description="Run model training with PyTorch Profiler")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pretrained model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the training dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for saving model and Profiler logs")
    args, remaining = parser.parse_known_args()

    # Set remaining arguments for HfArgumentParser
    import sys
    sys.argv = [sys.argv[0]] + remaining

    # Run training with Profiler
    run_profiler_training(args.model_path, args.data_path, args.output_dir)

if __name__ == "__main__":
    main()
