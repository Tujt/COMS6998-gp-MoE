import wandb
import os

def init_wandb(training_args, model_args):
    """
    Initialize WandB, executed only on the main process.
    """
    if training_args.local_rank == -1 or training_args.local_rank == 0:
        wandb.init(project=training_args.wandb_project, config=vars(training_args))
        wandb.config.update(vars(model_args))

def log_metrics(metrics, step=None):
    """
    Log metrics to WandB.
    """
    if wandb.run is not None:
        wandb.log(metrics, step=step)

def save_checkpoint(output_dir, ckpt_id):
    """
    Save checkpoint and optionally upload to WandB.
    """
    if wandb.run is not None:
        wandb.save(os.path.join(output_dir, ckpt_id))

def finish_wandb():
    """
    Finish the WandB run.
    """
    if wandb.run is not None:
        wandb.finish()