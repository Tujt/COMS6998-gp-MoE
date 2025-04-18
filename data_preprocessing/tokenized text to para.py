import json
import os

def create_llm_finetune_dataset(input_json_path, output_jsonl_path, min_input_tokens=10, split_ratio=0.3):
    with open(input_json_path, 'r') as f:
        data = json.load(f)

    output_data = []

    for item in data["test"]:
        tokens = item.get("tokenized_text", [])
        if len(tokens) < min_input_tokens:
            continue  # Skip short articles

        split_index = max(min_input_tokens, int(len(tokens) * split_ratio))
        input_text = ' '.join(tokens[:split_index])
        output_text = ' '.join(tokens[split_index:])

        formatted_entry = {
            "instruction": "You are a news assistant. Given the first few sentences of a news article, your task is to generate the rest of the article in a coherent and informative way.",
            "input": input_text,
            "output": output_text
        }

        output_data.append(formatted_entry)

    with open(output_jsonl_path, 'w') as f:
        for entry in output_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return output_jsonl_path

input_json_path = "./5049-formatted-summaries_llama3-dataset_splits.json"
output_jsonl_path = "../dataset/AskNews-NER-v0.jsonl"

create_llm_finetune_dataset(input_json_path, output_jsonl_path)
