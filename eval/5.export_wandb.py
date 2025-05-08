import wandb
import pandas as pd
import os
api = wandb.Api()

def export_wandb_runs_separately(entity, project_names, output_dir="./wandb_exports"):
    """
    Exports full metric history and summary+config separately for each run in each project.

    Args:
        entity (str): wandb entity (user/org).
        project_names (list of str): List of wandb project names.
        output_dir (str): Root directory to save CSVs.
    """
    api = wandb.Api()
    os.makedirs(output_dir, exist_ok=True)

    for project in project_names:
        runs = api.runs(f"{entity}/{project}")
        project_history_dir = os.path.join(output_dir, project, "history")
        project_summary_dir = os.path.join(output_dir, project, "summary")
        os.makedirs(project_history_dir, exist_ok=True)
        os.makedirs(project_summary_dir, exist_ok=True)

        for run in runs:
            try:
                run_id = run.id
                run_name = run.name.replace(" ", "_").replace("/", "_")
                filename = f"{run_id}_{run_name}.csv"

                # Save history
                history_df = run.history(samples=10000)
                history_path = os.path.join(project_history_dir, filename)
                history_df.to_csv(history_path, index=False)

                # Save summary + config
                summary = run.summary._json_dict
                config = {k: v for k, v in run.config.items() if not k.startswith('_')}
                meta_df = pd.DataFrame([{**config, **summary}])
                summary_path = os.path.join(project_summary_dir, filename)
                meta_df.to_csv(summary_path, index=False)

                print(f"Exported: {filename}")

            except Exception as e:
                print(f"Failed to export run {run.id}: {e}")

export_wandb_runs_separately(
    entity="6998gp_TLA",
    project_names=["llama-training-20250504-ltx-exp-dense1gpu-batch-8", "llama-training-20250504-ltx-exp-moe-batch-8", "llama-training-20250504-ltx-exp-dense-batch-8"],
    output_dir="./wandb"
)