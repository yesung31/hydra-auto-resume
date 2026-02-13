import json
import tempfile
from pathlib import Path

import wandb


def _get_api():
    return wandb.Api()


def _get_project_name(project_name=None):
    return project_name or Path.cwd().name


def recover_id_from_dir(log_path, checkpoint_dir_name="checkpoints"):
    """
    Attempts to recover a WandB ID from a local log directory.
    Checks inside 'wandb/latest-run' or scans the 'wandb' directory.
    """
    path = Path(log_path)
    # Handle case where log_path is the 'checkpoints' dir or a file
    if path.name == checkpoint_dir_name:
        path = path.parent
    if path.is_file():
        path = path.parent

    w_dir = path / "wandb"
    if w_dir.exists():
        # Try latest-run symlink first
        latest = w_dir / "latest-run"
        run_dir = latest.resolve() if latest.exists() else None

        # Fallback to finding the most recent directory
        if not run_dir:
            runs = sorted(
                [x for x in w_dir.iterdir() if x.is_dir()],
                key=lambda x: x.stat().st_mtime,
            )
            run_dir = runs[-1] if runs else None

        if run_dir:
            # Format is usually run-YYYYMMDD_HHMMSS-ID
            return run_dir.name.split("-")[-1]
    return None


def download_ckpt(
    wandb_id,
    download_dir,
    project_name=None,
    artifact_type="model",
    alias="latest",
    ckpt_pattern="*.ckpt",
    target_filename="wandb.ckpt",
):
    """
    Downloads the model checkpoint from a specific WandB run.
    """
    project_name = _get_project_name(project_name)
    api = _get_api()
    try:
        run = api.run(f"{project_name}/{wandb_id}")
    except wandb.errors.CommError:
        print(f"Error: Could not access run {project_name}/{wandb_id}")
        return None

    artifacts = [a for a in run.logged_artifacts() if a.type == artifact_type]

    # Try to find alias (e.g., 'latest' or 'best')
    target = next((a for a in artifacts if alias in a.aliases), None)
    # Fallback to the last logged artifact
    if not target and artifacts:
        target = artifacts[-1]

    if target:
        download_dir = Path(download_dir)
        download_dir.mkdir(parents=True, exist_ok=True)
        target.download(root=str(download_dir))

        # Find the actual .ckpt file
        ckpts = list(download_dir.glob(ckpt_pattern))
        if ckpts:
            if target_filename:
                # Rename to target_filename
                ckpt_path = ckpts[0]
                new_path = download_dir / target_filename
                ckpt_path.rename(new_path)
                return str(new_path)
            else:
                return str(ckpts[0])

    return None


def download_config(wandb_id, project_name=None):
    """
    Downloads the config.yaml (or wandb-metadata.json args) from WandB.
    Returns a list of override arguments (e.g. ['model=resnet', 'lr=0.01']).
    """
    project_name = _get_project_name(project_name)
    api = _get_api()
    try:
        run = api.run(f"{project_name}/{wandb_id}")
    except Exception:
        return []

    launch_args = []

    # Try to get 'wandb-metadata.json'
    try:
        if "wandb-metadata.json" in [f.name for f in run.files()]:
            with tempfile.TemporaryDirectory() as tmpdir:
                run.file("wandb-metadata.json").download(root=tmpdir, replace=True)
                with open(Path(tmpdir) / "wandb-metadata.json") as f:
                    data = json.load(f)
                    launch_args = data.get("args", [])
    except Exception:
        pass

    return launch_args
