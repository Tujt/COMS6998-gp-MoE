# COMS6998-gp-MoE
This is the groupwork of the course COMS6998 High-Performance Machine Learning in Columbia University, Spring 2025. The group Menbers are Tom, Layton and Andy.

## Setup (Windows)
1. Install conda
2. Create conda env: `conda create --name <env> --file req.txt`(Simple Version: use set_up.sh. May need to use chmod +x)
4. Download `https://huggingface.co/datasets/cognitivecomputations/dolphin/blob/main/flan1m-alpaca-uncensored-deduped.jsonl` and save to `./dataset`
5. Run `1.data_preprocessing.py`
6. Run one of the following commands.

## Commands
- Run baseline (using `TinyLlama/TinyLlama-1.1B-Chat-v1.0`):
``
python project.py --experiment_type dense --model_name_or_path "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --data_path 
"dataset\flan1m_2.5percent" --output_dir "outputs\dense_baseline" --use_lora False 
--per_device_train_batch_size 1 --gradient_accumulation_steps 1 --learning_rate 2e-5 
--num_train_epochs 1 --logging_steps 10 --save_strategy epoch --bf16 False --fp16 True --do_train True --model_max_length 512 --gradient_checkpointing True
``

- Run MoE + Router Random (using `TinyLlama/TinyLlama-1.1B-Chat-v1.0`):
``
 python project.py --experiment_type moe --router_strategy random --model_name_or_path "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --data_path "dataset/flan1m_2.5percent" --output_dir "outputs/moe_router_random" --use_lora False --per_device_train_batch_size 1 --gradient_accumulation_steps 8 --learning_rate 2e-5 --num_train_epochs 1 --logging_steps 10 --save_strategy epoch --bf16 False --fp16 True --do_train True --model_max_length 512 --gradient_checkpointing True
``

- Run baseline, with wandb:
``
python <name_of_program> --experiment_type dense --model_name_or_path "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --data_path 
"dataset\flan1m_2.5percent" --output_dir "outputs\dense_baseline" --use_lora False 
--per_device_train_batch_size 1 --gradient_accumulation_steps 1 --learning_rate 2e-5 
--num_train_epochs 1 --logging_steps 10 --save_strategy epoch --bf16 False --fp16 True --do_train True --model_max_length 512 --gradient_checkpointing True
--wandb_project <name_of_project> --run_name <name_of_run> --wandb_entity <name of team>
``


Eval:
1. Raw model + AskNews-input
2. Baseline model (AskNews-input-output) + AskNews-input
3. MOE model (AskNews-input-output MOE)
[4]. LoRA model (Raw model + AskNews-input-output) + AskNews-input
[5]. Llama3.2 1b model + AskNews-input


1. (0) Fix project.py to train on TinyLlama 1b (Tom)
2. (0) Zero3 + 2 GPU configuration (test on andrijdavid/Llama3-1B-Base) (Layton + Jingtian)
2.2. (2) Double check WanDB Metrics
3. (1,2,2.2) Train on TinyLlama 1b + AskNews (Baseline)
4. (3) Eval Baseline model (AskNews-input-output) + AskNews-input
5. (3) Train on TinyLlama 1b + AskNews (MOE model)
6. (5) Eval MOE model (AskNews-input-output MOE)
[7]. (1) LoRA model (Raw model + AskNews-input-output) + AskNews-input
[8]. (0) Eval Llama3.2 1b model + AskNews-input (Tom)