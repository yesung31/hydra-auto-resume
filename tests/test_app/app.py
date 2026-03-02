import hydra
from omegaconf import DictConfig
from hydra_auto_resume import auto_resume
import os
from pathlib import Path

@hydra.main(config_path="configs", config_name="config", version_base="1.3")
@auto_resume(
    no_log=os.environ.get("TEST_NO_LOG") == "True",
    use_saved_config=os.environ.get("TEST_USE_SAVED") == "True"
)
def main(cfg: DictConfig):
    print(f"VAL_PARAM1: {cfg.param1}")
    print(f"VAL_PARAM2: {cfg.param2}")
    from hydra.core.hydra_config import HydraConfig
    print(f"VAL_OUTPUT_DIR: {HydraConfig.get().runtime.output_dir}")
    
    # Create a checkpoint if requested
    if os.environ.get("TEST_CREATE_CKPT") == "True":
        ckpt_dir = Path(HydraConfig.get().runtime.output_dir) / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)
        (ckpt_dir / "last.ckpt").write_text("dummy")

if __name__ == "__main__":
    main()
