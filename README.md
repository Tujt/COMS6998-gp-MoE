# COMS6998-gp-MoE
This is the groupwork of the course COMS6998 High-Performance Machine Learning in Columbia University, Spring 2025. The group Menbers are Tom, Layton and Andy.

## Setup (Windows)
1. Install conda
2. Create conda env: `conda create --name <env> --file req.txt`
3. Download `https://huggingface.co/datasets/cognitivecomputations/dolphin/blob/main/flan1m-alpaca-uncensored-deduped.jsonl` and save to `./dataset`
4. Run `1.data_preprocessing.py`
5. Run one of the following commands.

## Commands
- Run baseline (using `andrijdavid/Llama3-1B-Base`):
``
python project.py --experiment_type dense --model_name_or_path "andrijdavid/Llama3-1B-Base" --data_path 
"dataset\flan1m_2.5percent" --output_dir "outputs\dense_baseline" --use_lora False 
--per_device_train_batch_size 1 --gradient_accumulation_steps 1 --learning_rate 2e-5 
--num_train_epochs 1 --logging_steps 10 --save_strategy epoch --bf16 False --fp16 True --do_train True --model_max_length 512 --gradient_checkpointing True
``

- Run MoE + Router Random (using `andrijdavid/Llama3-1B-Base`):
``
 python project.py --experiment_type moe --router_strategy random --model_name_or_path "andrijdavid/Llama3-1B-Base" --data_path "dataset/flan1m_2.5percent" --output_dir "outputs/moe_router_random" --use_lora False --per_device_train_batch_size 1 --gradient_accumulation_steps 8 --learning_rate 2e-5 --num_train_epochs 1 --logging_steps 10 --save_strategy epoch --bf16 False --fp16 True --do_train True --model_max_length 512 --gradient_checkpointing True
``

- Run baseline, with wandb:
``
python <name_of_program> --experiment_type dense --model_name_or_path "andrijdavid/Llama3-1B-Base" --data_path 
"dataset\flan1m_2.5percent" --output_dir "outputs\dense_baseline" --use_lora False 
--per_device_train_batch_size 1 --gradient_accumulation_steps 1 --learning_rate 2e-5 
--num_train_epochs 1 --logging_steps 10 --save_strategy epoch --bf16 False --fp16 True --do_train True --model_max_length 512 --gradient_checkpointing True
--wandb_project <name_of_project> --run_name <name_of_run> --wandb_entity <name of team>
``
