import json
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import torch

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

input_jsonl_path = "../dataset/AskNews-NER-v0.jsonl"
output_jsonl_path = "llama3_1b_original_outputs.jsonl"

results = []

with open(input_jsonl_path, 'r') as f:
    for line in f:
        example = json.loads(line.strip())
        instruction = example["instruction"]
        user_input = example["input"]
        ground_truth = example["output"]

        prompt = f"{instruction}\nInput: {user_input}\nOutput:"

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        output_tokens = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            streamer=streamer
        )

        decoded = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

        generated = decoded.split("Output:")[-1].strip()

        print("=== Prompt ===")
        print(prompt)
        print("=== Generated ===")
        print(generated)
        print("="*50)

        results.append({
            "input": user_input,
            "generated_output": generated,
            "reference_output": ground_truth
        })

with open(output_jsonl_path, 'w') as f:
    for item in results:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"All outputs saved to: {output_jsonl_path}")
