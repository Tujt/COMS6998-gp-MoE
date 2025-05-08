import os
import pandas as pd
import matplotlib.pyplot as plt

# === Load and Label Data ===
def load_labeled_history_summary(project_name, model_label, gpu_label, base_dir="./wandb"):
    history_dir = os.path.join(base_dir, project_name, "history")
    summary_dir = os.path.join(base_dir, project_name, "summary")

    history_dfs = []
    summary_dfs = []

    for f in os.listdir(history_dir):
        hist = pd.read_csv(os.path.join(history_dir, f))
        hist['run_file'] = f
        hist['model_label'] = model_label
        hist['gpu_label'] = gpu_label
        hist['config_label'] = f"{model_label}{gpu_label}"
        history_dfs.append(hist)

    for f in os.listdir(summary_dir):
        summ = pd.read_csv(os.path.join(summary_dir, f))
        summ['run_file'] = f
        summ['model_label'] = model_label
        summ['gpu_label'] = gpu_label
        summ['config_label'] = f"{model_label}{gpu_label}"
        summary_dfs.append(summ)

    history_df = pd.concat(history_dfs, ignore_index=True)
    summary_df = pd.concat(summary_dfs, ignore_index=True)

    return history_df, summary_df

# Define experiments
projects = [
    ("llama-training-20250504-ltx-exp-dense1gpu-batch-8", "Dense", "GPU1"),
    ("llama-training-20250504-ltx-exp-dense-batch-8", "Dense", "GPU2"),
    ("llama-training-20250504-ltx-exp-moe-batch-8", "MoE", "GPU2"),
]

# Load all data
all_history, all_summary = [], []
for project_name, model_label, gpu_label in projects:
    hist_df, summ_df = load_labeled_history_summary(project_name, model_label, gpu_label)
    all_history.append(hist_df)
    all_summary.append(summ_df)

history_df = pd.concat(all_history, ignore_index=True)
summary_df = pd.concat(all_summary, ignore_index=True)

# === Plotting Utility ===
config_colors = {
    "DenseGPU1": "blue",
    "DenseGPU2": "orange",
    "MoEGPU2": "green"
}
linestyles = ['-', '--', '-.', ':']

def plot_metric(metric_name, ylabel, title):
    plt.figure(figsize=(10, 6))
    legend_drawn = set()

    # Collect all unique epoch start steps
    epoch_transitions = history_df[['step', 'epoch']].dropna().drop_duplicates()
    transition_steps = epoch_transitions.groupby('epoch')['step'].min().sort_values().tolist()

    # Draw bold red vertical lines for epoch boundaries
    for s in transition_steps:
        plt.axvline(x=s, color='red', linestyle=':', linewidth=1.5)

    # Plot each config with color and line style
    for config_label, group in history_df.groupby('config_label'):
        color = config_colors.get(config_label, "black")
        for i, (run_file, df) in enumerate(group.groupby('run_file')):
            linestyle = linestyles[i % len(linestyles)]
            label = config_label if config_label not in legend_drawn else None
            plt.plot(df['step'], df[metric_name], color=color, linestyle=linestyle, label=label)
        legend_drawn.add(config_label)

    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# === Run Plots ===
plot_metric("loss", "Loss", "Training Loss Over Time")
plot_metric("step_time_sec", "Step Time (sec)", "Step Time Over Time")
plot_metric("grad_norm", "Gradient Norm", "Gradient Norm Over Time")
plot_metric("gpu_memory_allocated_mb", "Allocated GPU Memory (MB)", "GPU Memory Allocated Over Time")
plot_metric("gpu_memory_reserved_mb", "Reserved GPU Memory (MB)", "GPU Memory Reserved Over Time")
plot_metric("cpu_percent", "CPU Usage (%)", "CPU Utilization Over Time")
plot_metric("ram_used_mb", "RAM Used (MB)", "Host RAM Usage Over Time")
plot_metric("training_time_sec", "Training Time (sec)", "Cumulative Training Time")