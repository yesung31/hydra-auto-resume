import os
from pathlib import Path

import hydra
from omegaconf import DictConfig

from .wandb_tools import download_ckpt, recover_id_from_dir


def resolve(cfg: DictConfig):
    """
    Resolves the checkpoint path and WandB ID for the current run.
    Prioritizes Slurm/Submitit auto-resume (local 'hpc_ckpt.ckpt' or 'last.ckpt').

    Returns:
        (ckpt_path: str | None, wandb_id: str | None)
    """
    try:
        hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
        log_dir = Path(hydra_cfg.runtime.output_dir)
    except Exception:
        print("[HydraAutoResume] Warning: Could not retrieve Hydra output dir.")
        return None, None

    # --- Priority 1: Slurm/Submitit Auto-Resume ---
    # We check if we are in a resumed job context.
    # The surest sign is if we are running in a directory that ALREADY has a checkpoint.
    # CRITICAL: We also ensure we are actually IN a Slurm job to avoid local accidents.

    is_slurm = "SLURM_JOB_ID" in os.environ
    candidates = ["hpc_ckpt.ckpt", "last.ckpt"]
    local_ckpt = None

    if is_slurm:
        for name in candidates:
            p = log_dir / "checkpoints" / name
            if p.exists():
                local_ckpt = str(p)
                break

    # Verification: If we found a local checkpoint AND we are in Slurm, we MUST use it.
    # This implies the job crashed/timed-out and is restarting in the same directory.
    if local_ckpt:
        print(
            f"[HydraAutoResume] Found existing checkpoint in output dir: {local_ckpt}"
        )
        print("[HydraAutoResume] Assuming auto-resume from preemption/timeout.")

        # Try to recover WandB ID from this directory to continue the same run
        recovered_id = recover_id_from_dir(log_dir)
        return local_ckpt, recovered_id

    # --- Priority 2: Bootstrapped Intent (Manual Resume) ---
    # If no local crash file, we look at what was injected into the config
    # by our bootstrapper (or manually by the user).

    ckpt_path = cfg.get("ckpt_path")
    wandb_id = cfg.get("wandb_id")

    # Special Case: User provided WandB ID but NO checkpoint path.
    # This happens when we want to resume a specific run but start fresh
    # (or download weights).
    if wandb_id and not ckpt_path:
        print(
            f"[HydraAutoResume] WandB ID {wandb_id} provided. Downloading checkpoint..."
        )
        ckpt_dir = log_dir / "checkpoints"
        downloaded = download_ckpt(wandb_id, ckpt_dir)
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

    return ckpt_path, wandb_id
