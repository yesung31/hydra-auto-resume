# Hydra Auto Resume

A unified PyTorch Lightning + Hydra + WandB + Submitit plugin for seamless training resumption.

## Features

- **Slurm/Submitit Preemption Recovery**: Automatically detects if a job was preempted (or timed out) and resumes from the local `last.ckpt` or `hpc_ckpt.ckpt` in the *same* directory.
- **WandB ID Resumption**: `python run.py resume=WANDB_ID`
  - Downloads the original config arguments from WandB.
  - Downloads the model checkpoint.
  - Starts a **new** run with the old config + old weights.
- **Checkpoint File Resumption**: `python run.py resume=path/to/model.ckpt`
  - Starts a new run with the *current* config but initializes weights from the file.
- **Directory Resumption**: `python run.py resume=path/to/old/log_dir`
  - Finds the last checkpoint in that directory.
  - Recovers the WandB ID from that directory.
  - Starts a **new** run continuing that state.

## Installation

```bash
pip install -e /path/to/hydra-auto-resume
```

## Usage

Just add the decorator to your Hydra main function:

```python
import hydra
from hydra_auto_resume import auto_resume

@auto_resume(resume_arg_name="resume")
@hydra.main(config_path="configs", config_name="base", version_base="1.3")
def main(cfg):
    # cfg.ckpt_path and cfg.wandb_id are automatically populated!
    
    trainer.fit(model, dm, ckpt_path=cfg.ckpt_path)

if __name__ == "__main__":
    main()
```

## How it works

1.  **Bootstrapping (Pre-Hydra)**: The `@auto_resume` decorator scans `sys.argv` before Hydra parses them.
    *   If `resume=...` is found, it resolves the intent (download from WandB, find file, etc.).
    *   It modifies `sys.argv` to inject `++ckpt_path=...` and `++wandb_id=...` and potentially standard configuration overrides from the previous run.
2.  **Resolution (In-Job)**: Inside the executed job, it checks for **Slurm Auto-Resumption** first.
    *   If `hpc_ckpt.ckpt` exists in the *current* output directory, it ignores all other arguments and resumes from there. This is critical for cyclic Slurm jobs.
    *   Otherwise, it respects the `ckpt_path` injected by the bootstrapper.
