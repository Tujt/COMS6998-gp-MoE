# 手动下载下面的数据集
# https://huggingface.co/datasets/cognitivecomputations/dolphin/blob/main/flan1m-alpaca-uncensored-deduped.jsonl

def prepare_flan_subset(
        jsonl_path="./dataset/AskNews-NER-v0.jsonl",
        save_dir="./dataset",
        output_dir_name="AskNews-NER-v0"
):
    from datasets import load_dataset
    import os

    dataset = load_dataset("json", data_files=jsonl_path, split="train")
    print(f"原始数据量: {len(dataset)} 条")

    subset = dataset.train_test_split(test_size=0.01, seed=42)["train"]
    save_path = os.path.join(save_dir, output_dir_name)
    os.makedirs(save_path, exist_ok=True)

    print(f"保存 HuggingFace Dataset 到: {save_path}")
    subset.save_to_disk(save_path)

    print(f"样本数: {len(subset)}")


if __name__ == "__main__":
    prepare_flan_subset()
