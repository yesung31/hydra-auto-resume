import functools

from omegaconf import DictConfig, OmegaConf

from .cmd_line import bootstrap
from .resolver import resolve


def auto_resume(resume_arg_name="resume"):
    """
    Decorator for Hydra's main function to enable unified auto-resumption.

    Usage:
        @auto_resume(resume_arg_name="resume")
        @hydra.main(...)
        def main(cfg):
            ...

    Behavior:
    1. Pre-Hydra: Scans sys.argv for 'resume=...' and injects configs/checkpoints.
    2. Post-Hydra (Inside): Resolves checkpoint path (prioritizing Slurm auto-resume).
    3. Updates cfg.ckpt_path and cfg.wandb_id automatically.
    """

    # 1. Bootstrapping (Runs immediately when module is imported/decorated)
    # We call this here to ensure it runs before Hydra starts parsing args.
    # Note: This side-effect happens at import time if using the decorator.
    # However, Python decorators run at definition time.
    # To run strictly before main(), we can just run it here.
    bootstrap(resume_arg_name)

    def decorator(func):
        @functools.wraps(func)
        def wrapper(cfg: DictConfig, *args, **kwargs):
            # 2. Resolution (Runs inside the job)
            ckpt_path, wandb_id = resolve(cfg)

            # 3. Injection
            # We need to unlock the config to modify it
            OmegaConf.set_struct(cfg, False)

            if ckpt_path:
                cfg.ckpt_path = ckpt_path
                print(f"[HydraAutoResume] Set cfg.ckpt_path = {ckpt_path}")

            if wandb_id:
                cfg.wandb_id = wandb_id
                print(f"[HydraAutoResume] Set cfg.wandb_id = {wandb_id}")

            OmegaConf.set_struct(cfg, True)

            return func(cfg, *args, **kwargs)

        return wrapper

    return decorator
