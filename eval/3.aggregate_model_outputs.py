import json
import os
from glob import glob
from tqdm import tqdm

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def aggregate_outputs(input_dir, output_file):
    all_files = glob(os.path.join(input_dir, '*.json'))

    model_outputs = {}
    for file_path in all_files:
        model_name = os.path.splitext(os.path.basename(file_path))[0]
        model_outputs[model_name] = load_json(file_path)

    model_names = list(model_outputs.keys())
    num_examples = len(next(iter(model_outputs.values())))

    aggregated = []

    for idx in tqdm(range(num_examples), desc="Aggregating outputs"):
        first_model = model_outputs[model_names[0]][idx]
        entry = {
            "input": first_model['input'],
            "reference_output": first_model['reference_output'],
            "models": []
        }
        for model_name in model_names:
            model_data = model_outputs[model_name][idx]
            entry['models'].append({
                "model_name": model_name,
                "generated_output": model_data['generated_output'],
                "metrics": model_data['metrics']
            })
        aggregated.append(entry)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(aggregated, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    input_directory = "./"
    output_path = "aggregated_outputs.json"
    aggregate_outputs(input_directory, output_path)
