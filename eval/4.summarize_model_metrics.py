import json
import pandas as pd
import matplotlib.pyplot as plt
import os
from collections import defaultdict
from tqdm import tqdm

def load_aggregated_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def compute_summary(aggregated_data):
    records = []

    for sample in tqdm(aggregated_data, desc="Processing samples"):
        for model_info in sample['models']:
            record = {
                'model_name': model_info['model_name'],
                'bleu': model_info['metrics']['bleu'],
                'rouge-1': model_info['metrics']['rouge-1'],
                'rouge-2': model_info['metrics']['rouge-2'],
                'rouge-l': model_info['metrics']['rouge-l'],
                'meteor': model_info['metrics']['meteor'],
                'cosine_similarity': model_info['metrics']['cosine_similarity'],
                'ner_overlap_ratio': model_info['metrics']['ner_overlap_ratio']
            }
            records.append(record)

    df = pd.DataFrame(records)
    return df

def plot_metric_means(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    metrics = ['bleu', 'rouge-1', 'rouge-2', 'rouge-l', 'meteor', 'cosine_similarity', 'ner_overlap_ratio']
    summary = df.groupby('model_name')[metrics].mean().reset_index()

    for metric in metrics:
        plt.figure()
        plt.bar(summary['model_name'], summary[metric])
        plt.title(f"Mean {metric.upper()} Score Across Models")
        plt.ylabel(metric.upper())
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{metric}_mean_plot.png"))
        plt.close()

    return summary

def save_summary(summary_df, output_path):
    summary_df.to_csv(output_path, index=False)

def plot_all_metrics_subplots(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    metrics = ['bleu', 'rouge-1', 'rouge-2', 'rouge-l', 'meteor', 'cosine_similarity', 'ner_overlap_ratio']
    summary = df.groupby('model_name')[metrics].mean().reset_index()

    num_metrics = len(metrics)
    cols = 3
    rows = (num_metrics + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]
        ax.bar(summary['model_name'], summary[metric])
        ax.set_title(f"{metric.upper()} Mean")
        ax.set_ylabel(metric.upper())
        ax.tick_params(axis='x', rotation=45)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "all_metrics_subplots.png"))
    plt.close()

if __name__ == "__main__":
    aggregated_path = "aggregated_outputs.json"
    output_summary_csv = "summary_metrics.csv"
    output_plot_dir = "plots"

    aggregated_data = load_aggregated_file(aggregated_path)
    df = compute_summary(aggregated_data)
    summary_df = plot_metric_means(df, output_plot_dir)
    plot_all_metrics_subplots(df, output_plot_dir)
    save_summary(summary_df, output_summary_csv)

    print("\nAll done! Summary CSV and plots saved.")