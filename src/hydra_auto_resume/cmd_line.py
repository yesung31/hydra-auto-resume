import shutil
import sys
from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf

from .wandb_tools import download_config


def normalize_key(arg):
    """Strips Hydra prefixes (+, ++, ~) and returns the base key."""
    if "=" not in arg:
        return arg
    key = arg.split("=", 1)[0]
    return key.lstrip("+").lstrip("++").lstrip("~")


def backup_hydra_configs(log_dir):
    """Backs up the .hydra directory to preserve old configurations."""
    hydra_dir = log_dir / ".hydra"
    if hydra_dir.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = log_dir / f".hydra.old_{timestamp}"
        print(f"[HydraAutoResume] Backing up existing Hydra configs to {backup_dir}")
        try:
            shutil.copytree(hydra_dir, backup_dir)
        except Exception as e:
            print(f"[HydraAutoResume] Warning: Failed to backup .hydra: {e}")


def bootstrap(
    resume_arg_name="resume",
    checkpoint_dir="checkpoints",
    checkpoint_names=None,
    checkpoint_ext=".ckpt",
    config_ckpt_path_key="ckpt_path",
    config_wandb_id_key="wandb_id",
):
    """
    Parses sys.argv for the resume argument and modifies it in-place to inject
    configuration based on the resume target.

    Injection Strategy:
    1.  Duplicate Avoidance: Uses a dictionary (injected_keys) and a set (user_keys)
        to ensure that no key is injected multiple times and that user-provided CLI
        arguments have priority.
    2.  Smart Overrides: When loading overrides from a previous run (multirun.yaml or
        hydra.yaml), it converts single '+' prefixes to '++'. This ensures that the
        injected arguments correctly override any defaults in the current configuration,
        preventing Hydra's "Multiple values" error.
    3.  Config Preservation: Before resuming a directory in-place, the existing .hydra
        directory is backed up to .hydra.old_TIMESTAMP to ensure no previous run
        configuration is lost.
    4.  Multirun Support: Automatically detects multirun directories and injects
        hydra.sweep.dir and the '-m' flag if necessary.
    5.  Argument Order: Injects new arguments after the script name but before the
        original user-provided CLI arguments.
    """
    if checkpoint_names is None:
        checkpoint_names = ["hpc_ckpt.ckpt", "last.ckpt"]

    # Simple manual parse to find resume=... without triggering Hydra
    resume_val = None
    for i, arg in enumerate(sys.argv):
        if arg.startswith(f"{resume_arg_name}="):
            resume_val = arg.split("=", 1)[1]
            break

    if not resume_val:
        return

    print(f"[HydraAutoResume] Detected resume request: {resume_val}")

    new_args = []
    # Track injected and user-provided keys to avoid duplicates
    injected_keys = {}
    user_keys = {normalize_key(arg) for arg in sys.argv if "=" in arg}

    def add_arg(arg, force_override=False):
        key = normalize_key(arg)
        if key == resume_arg_name:
            return
        if key not in user_keys and key not in injected_keys:
            # Resuming an override usually means we want to FORCE it (++)
            # to avoid "Multiple values" errors if the base config already defines it.
            # However, for config groups (which usually contain a '/'),
            # '++' is often not supported for adding/overriding them.
            if (
                force_override
                and "/" not in key
                and not any(arg.startswith(p) for p in ["+", "++", "~"])
            ):
                arg = "++" + arg
            new_args.append(arg)
            injected_keys[key] = arg

    # 1. Check if it's a WandB ID (Length 8 alphanumeric usually, not a path)
    if not Path(resume_val).exists() and len(resume_val) == 8:
        wandb_id = resume_val
        print(f"[HydraAutoResume] Resuming from WandB ID: {wandb_id}")

        # Download previous config args
        launch_args = download_config(wandb_id)
        for arg in launch_args:
            add_arg(arg, force_override=True)

        add_arg(f"++{config_wandb_id_key}={wandb_id}")

    # 2. Check if it's a Checkpoint File OR Log Directory
    else:
        path_val = Path(resume_val).resolve()

        if path_val.is_file() and str(path_val).endswith(checkpoint_ext):
            print(f"[HydraAutoResume] Resuming from Checkpoint File: {resume_val}")
            # Case A: Checkpoint File -> Fresh Run, Default Config, Load Weights
            add_arg(f"++{config_ckpt_path_key}={path_val}")

        elif path_val.is_dir():
            log_dir = path_val

            # Check for multirun
            if (log_dir / "multirun.yaml").exists():
                print(
                    f"[HydraAutoResume] Resuming from Multirun Directory: {resume_val}"
                )
                add_arg(f"hydra.sweep.dir={log_dir}")
                add_arg(f"++{config_ckpt_path_key}=AUTO")

                if "-m" not in sys.argv and "--multirun" not in sys.argv:
                    print(
                        "[HydraAutoResume] Resuming a multirun: "
                        "Automatically adding '-m' flag."
                    )
                    # We don't use add_arg here because we want to ensure it's first
                    if "-m" not in injected_keys:
                        new_args.insert(0, "-m")
                        injected_keys["-m"] = "-m"

                # Load overrides from multirun.yaml
                multirun_yaml = log_dir / "multirun.yaml"
                try:
                    resume_cfg = OmegaConf.load(multirun_yaml)
                    if (
                        "hydra" in resume_cfg
                        and "overrides" in resume_cfg.hydra
                        and "task" in resume_cfg.hydra.overrides
                    ):
                        overrides = resume_cfg.hydra.overrides.task
                        print(
                            "[HydraAutoResume] Loaded overrides from multirun.yaml: "
                            f"{overrides}"
                        )
                        for arg in overrides:
                            add_arg(arg, force_override=True)
                except Exception as e:
                    print(
                        "[HydraAutoResume] Warning: "
                        f"Failed to load overrides from {multirun_yaml}: {e}"
                    )
            else:
                print(f"[HydraAutoResume] Resuming from Directory: {resume_val}")
                # Case B: Directory -> In-Place Resume, Old Config, Append Logs

                # Backup configs before Hydra overwrites them
                backup_hydra_configs(log_dir)

                # 1. Force Hydra to run in the same directory
                add_arg(f"hydra.run.dir={log_dir}")

                # 2. Find checkpoint
                ckpt_path = None
                for name in checkpoint_names:
                    if (log_dir / checkpoint_dir / name).exists():
                        ckpt_path = str(log_dir / checkpoint_dir / name)
                        break
                if ckpt_path:
                    add_arg(f"++{config_ckpt_path_key}={ckpt_path}")

                # 3. Load config overrides from hydra.yaml
                hydra_yaml = log_dir / ".hydra" / "hydra.yaml"
                if hydra_yaml.exists():
                    try:
                        hydra_cfg = OmegaConf.load(hydra_yaml)
                        overrides = hydra_cfg.hydra.overrides.task
                        print(
                            f"[HydraAutoResume] Loaded from previous run: {overrides}"
                        )
                        for arg in overrides:
                            add_arg(arg, force_override=True)
                    except Exception as e:
                        print(
                            "[HydraAutoResume] Warning: "
                            f"Failed to load overrides from {hydra_yaml}: {e}"
                        )

    # Inject new args into sys.argv
    clean_argv = [arg for arg in sys.argv if not arg.startswith(f"{resume_arg_name}=")]
    if new_args:
        print(f"[HydraAutoResume] Injecting args: {new_args}")
        sys.argv = [clean_argv[0]] + new_args + clean_argv[1:]
    else:
        sys.argv = clean_argv
