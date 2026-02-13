import sys
from pathlib import Path

from omegaconf import OmegaConf

from .wandb_tools import download_config, recover_id_from_dir


def bootstrap(
    resume_arg_name="resume",
    checkpoint_dir="checkpoints",
    checkpoint_names=None,
    checkpoint_ext=".ckpt",
    config_ckpt_path_key="ckpt_path",
    config_wandb_id_key="wandb_id",
):
    """
    Parses sys.argv for the resume argument.
    Modifies sys.argv in-place to inject configuration based on the resume target.
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

    # Identify user keys to avoid overwriting user's current CLI args
    user_keys = {arg.split("=")[0] for arg in sys.argv if "=" in arg}

    # 1. Check if it's a WandB ID (Length 8 alphanumeric usually, not a path)
    if not Path(resume_val).exists() and len(resume_val) == 8:
        wandb_id = resume_val
        print(f"[HydraAutoResume] Resuming from WandB ID: {wandb_id}")

        # Download previous config args
        launch_args = download_config(wandb_id)

        for arg in launch_args:
            key = arg.split("=")[0]
            if (
                key not in user_keys
                and "resume" not in key
                and config_ckpt_path_key not in key
                and config_wandb_id_key not in key
            ):
                new_args.append(arg)

        new_args.append(f"++{config_wandb_id_key}={wandb_id}")

    # 2. Check if it's a Checkpoint File OR Log Directory
    else:
        path_val = Path(resume_val).resolve()

        if path_val.is_file() and str(path_val).endswith(checkpoint_ext):
            print(f"[HydraAutoResume] Resuming from Checkpoint File: {resume_val}")
            # Case A: Checkpoint File -> Fresh Run, Default Config, Load Weights
            new_args.append(f"++{config_ckpt_path_key}={path_val}")

        elif path_val.is_dir():
            log_dir = path_val

            # Check for multirun
            if (log_dir / "multirun.yaml").exists():
                print(
                    f"[HydraAutoResume] Resuming from Multirun Directory: {resume_val}"
                )
                new_args.append(f"hydra.sweep.dir={log_dir}")
                new_args.append(f"++{config_ckpt_path_key}=AUTO")
                if "-m" not in sys.argv and "--multirun" not in sys.argv:
                    print(
                        "[HydraAutoResume] Resuming a multirun: "
                        "Automatically adding '-m' flag."
                    )
                    new_args.insert(0, "-m")

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
                            "[HydraAutoResume] Loaded overrides from multirun.yaml:"
                            f"{overrides}"
                        )
                        for arg in overrides:
                            if "=" in arg:
                                key = arg.split("=")[0]
                                # Only add if user didn't explicitly set it in
                                # current command
                                if key not in user_keys and "resume" not in key:
                                    new_args.append(arg)
                except Exception as e:
                    print(
                        "[HydraAutoResume] Warning:"
                        f"Failed to load overrides from {multirun_yaml}: {e}"
                    )
            else:
                print(f"[HydraAutoResume] Resuming from Directory: {resume_val}")
                # Case B: Directory -> In-Place Resume, Old Config, Append Logs

                # 1. Force Hydra to run in the same directory
                new_args.append(f"hydra.run.dir={log_dir}")

                # 2. Find checkpoint
                ckpt_path = None
                for name in checkpoint_names:
                    if (log_dir / checkpoint_dir / name).exists():
                        ckpt_path = str(log_dir / checkpoint_dir / name)
                        break
                if ckpt_path:
                    new_args.append(f"++{config_ckpt_path_key}={ckpt_path}")

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
                            if "=" in arg:
                                key = arg.split("=")[0]
                                # Only add if user didn't explicitly set it in
                                # current command
                                if key not in user_keys and "resume" not in key:
                                    new_args.append(arg)
                    except Exception as e:
                        print(
                            "[HydraAutoResume] Warning:"
                            f"Failed to load overrides from {hydra_yaml}: {e}"
                        )

                # 4. Recover WandB ID to resume the same run
                wandb_id = recover_id_from_dir(
                    log_dir, checkpoint_dir_name=checkpoint_dir
                )
                if wandb_id:
                    new_args.append(f"++{config_wandb_id_key}={wandb_id}")

    # Inject new args into sys.argv
    if new_args:
        print(f"[HydraAutoResume] Injecting args: {new_args}")
        # Current sys.argv: [script, user_arg1, user_arg2, resume=...]
        # Desired: [script, old_arg1, ..., ++wandb_id=..., user_arg1, ..., resume=...]
        # Priority: new_args > resume_args > default_args

        # We need to remove the original 'resume=...' arg from sys.argv because
        # it might not be in the config structure.

        # 1. Filter out the resume argument from the ORIGINAL sys.argv
        clean_argv = [
            arg for arg in sys.argv if not arg.startswith(f"{resume_arg_name}=")
        ]

        # 2. Inject new args
        # logic: sys.argv = [script] + new_args + [clean_user_args...]
        sys.argv = [clean_argv[0]] + new_args + clean_argv[1:]
    else:
        # If no new args were generated (e.g. download failed), we should still remove
        # resume arg to avoid Hydra crash if it's not in config.
        sys.argv = [
            arg for arg in sys.argv if not arg.startswith(f"{resume_arg_name}=")
        ]
