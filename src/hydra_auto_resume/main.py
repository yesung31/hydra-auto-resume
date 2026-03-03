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
    no_log=False,
    use_saved_config=False,
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
        no_log (bool): If True, disables Hydra's log directory creation and in-place resumption.
            Useful for evaluation runs. Default is False.
        use_saved_config (bool): If True, loads the already-composed configuration from the
            resumed session's .hydra/config.yaml instead of re-composing it from the current project files
            and overrides. Default is False.

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
        no_log=no_log,
        use_saved_config=use_saved_config,
    )

    def decorator(func):
        @functools.wraps(func)
        def wrapper(cfg: DictConfig, *args, **kwargs):
            # 2. Resolution (Runs inside the job)
            ckpt_path, wandb_id, saved_cfg, saved_hydra_cfg = resolve(
                cfg,
                checkpoint_dir=checkpoint_dir,
                checkpoint_names=checkpoint_names,
                wandb_artifact_type=wandb_artifact_type,
                wandb_artifact_alias=wandb_artifact_alias,
                wandb_ckpt_pattern=wandb_ckpt_pattern,
                wandb_ckpt_target_filename=wandb_ckpt_target_filename,
                config_ckpt_path_key=config_ckpt_path_key,
                config_wandb_id_key=config_wandb_id_key,
                use_saved_config=use_saved_config,
            )

            # 3. Injection
            if saved_cfg:
                # We want saved_cfg to take precedence over current defaults,
                # BUT user CLI overrides should still win.
                
                # Let's try to get ONLY CLI overrides.
                try:
                    import hydra.core.hydra_config
                    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
                    overrides = [
                        str(o) for o in hydra_cfg.overrides.task 
                        if not (o.startswith(f"{resume_arg_name}=") or "hydra_auto_resume" in o)
                    ]
                except Exception:
                    overrides = []

                # Create a fresh config starting from the saved one
                new_cfg = OmegaConf.create(saved_cfg)
                OmegaConf.set_struct(new_cfg, False)
                
                # Now re-apply CLI overrides (they should win)
                if overrides:
                    new_cfg.merge_with(OmegaConf.from_dotlist(overrides))
                
                # Replace cfg with new combined state
                OmegaConf.set_struct(cfg, False)
                cfg.clear()
                cfg.merge_with(new_cfg)
                OmegaConf.set_struct(cfg, True)
                print(
                    f"[HydraAutoResume] Merged saved configuration (applied {len(overrides)} CLI overrides)."
                )

            # If we have saved hydra config, we should ideally restore parts of it
            # like choices, but HydraConfig is mostly read-only in many parts.
            # However, some parts can be updated.
            if saved_hydra_cfg:
                try:
                    from hydra.core.hydra_config import HydraConfig
                    current_hydra_cfg = HydraConfig.get()
                    OmegaConf.set_struct(current_hydra_cfg, False)
                    # We merge saved hydra choices into current one
                    if "runtime" in saved_hydra_cfg and "choices" in saved_hydra_cfg.runtime:
                        current_hydra_cfg.runtime.choices.merge_with(saved_hydra_cfg.runtime.choices)
                    OmegaConf.set_struct(current_hydra_cfg, True)
                    print("[HydraAutoResume] Restored Hydra choices from saved session.")
                except Exception as e:
                    print(f"[HydraAutoResume] Warning: Failed to restore Hydra choices: {e}")

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
