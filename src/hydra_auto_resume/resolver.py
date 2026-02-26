import os
from pathlib import Path

import hydra
from omegaconf import DictConfig

from .wandb_tools import download_ckpt, recover_id_from_dir


def resolve(
    cfg: DictConfig,
    checkpoint_dir="checkpoints",
    checkpoint_names=None,
    wandb_artifact_type="model",
    wandb_artifact_alias="latest",
    wandb_ckpt_pattern="*.ckpt",
    wandb_ckpt_target_filename="wandb.ckpt",
    config_ckpt_path_key="ckpt_path",
    config_wandb_id_key="wandb_id",
):
    """
    Resolves the checkpoint path and WandB ID for the current run.
    Prioritizes Slurm/Submitit auto-resume (local 'hpc_ckpt.ckpt' or 'last.ckpt').

    Returns:
        (ckpt_path: str | None, wandb_id: str | None, saved_cfg: DictConfig | None)
    """
    if checkpoint_names is None:
        checkpoint_names = ["hpc_ckpt.ckpt", "last.ckpt"]

    try:
        hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
        log_dir = Path(hydra_cfg.runtime.output_dir)
    except Exception:
        print("[HydraAutoResume] Warning: Could not retrieve Hydra output dir.")
        return None, None, None

    # --- Priority 1: Slurm/Submitit Auto-Resume or Manual AUTO Signal ---
    # We check if we are in a resumed job context.
    # The surest sign is if we are running in a directory that ALREADY has a checkpoint.
    # We allow this if:
    # 1. We are in a Slurm job (to avoid accidental resumes on local dev)
    # 2. We explicitly received 'AUTO' as ckpt_path (e.g. from multirun bootstrapper)

    is_slurm = "SLURM_JOB_ID" in os.environ
    is_manual_auto = cfg.get(config_ckpt_path_key) == "AUTO"

    candidates = checkpoint_names
    local_ckpt = None

    if is_slurm or is_manual_auto:
        for name in candidates:
            p = log_dir / checkpoint_dir / name
            if p.exists():
                local_ckpt = str(p)
                break

    # Verification: If we found a local checkpoint, we use it.
    if local_ckpt:
        print(
            f"[HydraAutoResume] Found existing checkpoint in output dir: {local_ckpt}"
        )
        if is_slurm:
            print("[HydraAutoResume] Assuming auto-resume from preemption/timeout.")

        # Try to recover WandB ID from this directory to continue the same run
        recovered_id = recover_id_from_dir(log_dir, checkpoint_dir_name=checkpoint_dir)

        # Try to recover original configuration
        saved_cfg_path = log_dir / ".hydra" / "config.yaml"
        saved_cfg = None
        if saved_cfg_path.exists():
            from omegaconf import OmegaConf

            try:
                saved_cfg = OmegaConf.load(saved_cfg_path)
                print(f"[HydraAutoResume] Loaded saved config from {saved_cfg_path}")
            except Exception as e:
                print(
                    f"[HydraAutoResume] Warning: Failed to load {saved_cfg_path}: {e}"
                )

        return local_ckpt, recovered_id, saved_cfg

    # --- Priority 2: Bootstrapped Intent (Manual Resume) ---
    # If no local crash file, we look at what was injected into the config
    # by our bootstrapper (or manually by the user).

    ckpt_path = cfg.get(config_ckpt_path_key)
    if ckpt_path == "AUTO":
        ckpt_path = None

    wandb_id = cfg.get(config_wandb_id_key)

    # Special Case: User provided WandB ID but NO checkpoint path.
    # This happens when we want to resume a specific run but start fresh
    # (or download weights).
    if wandb_id and not ckpt_path:
        print(
            f"[HydraAutoResume] WandB ID {wandb_id} provided. Downloading checkpoint..."
        )
        ckpt_dir = log_dir / checkpoint_dir
        downloaded = download_ckpt(
            wandb_id,
            ckpt_dir,
            artifact_type=wandb_artifact_type,
            alias=wandb_artifact_alias,
            ckpt_pattern=wandb_ckpt_pattern,
            target_filename=wandb_ckpt_target_filename,
        )
        if downloaded:
            # Rename to avoid confusion with locally generated files?
            # Or keep original name. Let's keep it simple.
            ckpt_path = downloaded
            print(f"[HydraAutoResume] Downloaded checkpoint to: {ckpt_path}")
        else:
            print(
                "[HydraAutoResume] Warning:"
                f"Could not download checkpoint for ID {wandb_id}"
            )

    return ckpt_path, wandb_id, None
