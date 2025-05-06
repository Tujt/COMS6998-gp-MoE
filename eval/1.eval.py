import json
import os
import argparse
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from modeling_file.modeling_llama_moe import LlamaMoEForCausalLM

def load_model_and_tokenizer(model_dir):
    print(f"🔄 Loading model from: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 如果是MOE
    # model = LlamaMoEForCausalLM.from_pretrained(
    #     model_dir,
    #     torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    #     device_map="auto",
    #     trust_remote_code=True
    # )
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    return model, tokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--verbose", default=True, action="store_true", help="Print each prompt/output.")
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model_dir)

    with open(args.input_jsonl, 'r') as f:
        lines = [json.loads(line.strip()) for line in f if line.strip()]

    results = []
    batch_size = args.batch_size

    with torch.no_grad():
        for i in tqdm(range(0, len(lines), batch_size), desc="🚀 Generating responses"):
            batch = lines[i:i + batch_size]

            prompts = [f"{ex['instruction']}\nInput: {ex['input']}\nOutput:" for ex in batch]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)

            output_tokens = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )

            decoded_outputs = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)

            for ex, decoded in zip(batch, decoded_outputs):
                if "Output:" in decoded:
                    generated = decoded.split("Output:")[-1].strip()
                else:
                    generated = decoded.strip()
                if args.verbose:
                    print("=== Prompt ===")
                    print(f"{ex['instruction']}\nInput: {ex['input']}\nOutput:")
                    print("=== Generated ===")
                    print(generated)
                    print("=" * 50)

                results.append({
                    "input": ex["input"],
                    "generated_output": generated,
                    "reference_output": ex["output"]
                })

    os.makedirs(os.path.dirname(args.output_jsonl), exist_ok=True)
    with open(args.output_jsonl, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ All outputs saved to: {args.output_jsonl}")

if __name__ == "__main__":
    main()
