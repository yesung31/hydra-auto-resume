import functools

from omegaconf import DictConfig, OmegaConf

from .cmd_line import bootstrap
from .resolver import resolve


def auto_resume(
    resume_arg_name="resume",
    checkpoint_dir="checkpoints",
    checkpoint_names=None,
    checkpoint_ext=".ckpt",
    wandb_artifact_type="model",
    wandb_artifact_alias="latest",
    wandb_ckpt_pattern="*.ckpt",
    wandb_ckpt_target_filename="wandb.ckpt",
    config_ckpt_path_key="ckpt_path",
    config_wandb_id_key="wandb_id",
):
    """
    Decorator for Hydra's main function to enable unified auto-resumption logic.

    This decorator handles three main resumption scenarios:
    1. WandB ID: Resumes a specific run by downloading config and weights from WandB.
    2. Checkpoint File: Starts a new run initialized with weights from a local file.
    3. Directory: Resumes a run in-place from an existing log directory.
    4. Slurm Preemption: Automatically detects and resumes from local checkpoints if a job restarts.

    Args:
        resume_arg_name (str): The name of the command-line argument to trigger resumption.
            Default is "resume" (usage: `python run.py resume=...`).
        checkpoint_dir (str): The name of the subdirectory within the log directory where checkpoints are stored.
            Default is "checkpoints".
        checkpoint_names (list[str] | None): A list of specific checkpoint filenames to prioritize for
            auto-recovery (e.g., Slurm preemption). Default is ["hpc_ckpt.ckpt", "last.ckpt"].
        checkpoint_ext (str): The file extension used to identify checkpoint files when a path is provided.
            Default is ".ckpt".
        wandb_artifact_type (str): The type of WandB artifact to search for when downloading a model.
            Default is "model".
        wandb_artifact_alias (str): The specific alias of the artifact version to download (e.g., "latest", "best").
            Default is "latest".
        wandb_ckpt_pattern (str): A glob pattern to identify the actual checkpoint file within the downloaded
            artifact directory. Default is "*.ckpt".
        wandb_ckpt_target_filename (str): The filename to rename the downloaded checkpoint to.
            Default is "wandb.ckpt".
        config_ckpt_path_key (str): The key in the Hydra configuration `cfg` where the resolved checkpoint
            path should be stored. Supports dot notation (e.g., "model.resume_path"). Default is "ckpt_path".
        config_wandb_id_key (str): The key in the Hydra configuration `cfg` where the resolved WandB run ID
            should be stored. Default is "wandb_id".

    Usage:
        @auto_resume(config_ckpt_path_key="model.weights", checkpoint_names=["last.pt"])
        @hydra.main(...)
        def main(cfg):
            ...
    """
    if checkpoint_names is None:
        checkpoint_names = ["hpc_ckpt.ckpt", "last.ckpt"]

    # 1. Bootstrapping (Runs immediately when module is imported/decorated)
    bootstrap(
        resume_arg_name=resume_arg_name,
        checkpoint_dir=checkpoint_dir,
        checkpoint_names=checkpoint_names,
        checkpoint_ext=checkpoint_ext,
        config_ckpt_path_key=config_ckpt_path_key,
        config_wandb_id_key=config_wandb_id_key,
    )

    def decorator(func):
        @functools.wraps(func)
        def wrapper(cfg: DictConfig, *args, **kwargs):
            # 2. Resolution (Runs inside the job)
            ckpt_path, wandb_id, saved_cfg = resolve(
                cfg,
                checkpoint_dir=checkpoint_dir,
                checkpoint_names=checkpoint_names,
                wandb_artifact_type=wandb_artifact_type,
                wandb_artifact_alias=wandb_artifact_alias,
                wandb_ckpt_pattern=wandb_ckpt_pattern,
                wandb_ckpt_target_filename=wandb_ckpt_target_filename,
                config_ckpt_path_key=config_ckpt_path_key,
                config_wandb_id_key=config_wandb_id_key,
            )

            # 3. Injection
            # If we have a saved config, we merge it IN FRONT of the current config
            # but AFTER the user's CLI overrides (which are already in cfg).
            # Hydra's merge logic: cfg.merge_with(other) means 'other' values take precedence.
            # Here, we want saved_cfg to take precedence over current defaults,
            # but user CLI overrides should still win.
            if saved_cfg:
                # We need to preserve the user's CLI overrides if they are present.
                # Actually, in most resume scenarios, we want the saved config to be the truth.
                OmegaConf.set_struct(cfg, False)
                # Merge saved_cfg into current cfg.
                # Note: saved_cfg is a full composed config.
                cfg.merge_with(saved_cfg)
                OmegaConf.set_struct(cfg, True)
                print("[HydraAutoResume] Merged saved configuration into current session.")

            updates = []
            if ckpt_path:
                updates.append(f"{config_ckpt_path_key}={ckpt_path}")
                print(f"[HydraAutoResume] Set cfg.{config_ckpt_path_key} = {ckpt_path}")

            if wandb_id:
                updates.append(f"{config_wandb_id_key}={wandb_id}")
                print(f"[HydraAutoResume] Set cfg.{config_wandb_id_key} = {wandb_id}")

            if updates:
                OmegaConf.set_struct(cfg, False)
                cfg.merge_with(OmegaConf.from_dotlist(updates))
                OmegaConf.set_struct(cfg, True)

            return func(cfg, *args, **kwargs)

        return wrapper

    return decorator
