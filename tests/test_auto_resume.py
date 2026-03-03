import os
import shutil
import subprocess
from pathlib import Path
import pytest
import sys

APP_PATH = Path(__file__).parent / "test_app" / "app.py"

@pytest.fixture
def test_env(tmp_path):
    """Sets up a clean test environment with a dummy config and initial run."""
    # Copy test_app to tmp_path to run there
    app_dir = tmp_path / "test_app"
    shutil.copytree(Path(__file__).parent / "test_app", app_dir)
    
    # Create an initial run directory to resume from
    initial_log_dir = app_dir / "initial_run"
    initial_log_dir.mkdir()
    hydra_dir = initial_log_dir / ".hydra"
    hydra_dir.mkdir()
    (hydra_dir / "config.yaml").write_text("param1: saved\nparam2: saved")
    (hydra_dir / "hydra.yaml").write_text("hydra:\n  overrides:\n    task: []")
    
    ckpt_dir = initial_log_dir / "checkpoints"
    ckpt_dir.mkdir()
    (ckpt_dir / "last.ckpt").write_text("dummy")
    
    return {
        "app_dir": app_dir,
        "initial_log_dir": initial_log_dir,
        "ckpt_path": ckpt_dir / "last.ckpt"
    }

def run_app(app_dir, args, env_vars=None):
    env = os.environ.copy()
    if env_vars:
        env.update(env_vars)
    # Ensure hydra-auto-resume is in python path if not installed
    # But it should be installed in editable mode for these tests to work easily
    result = subprocess.run(
        [sys.executable, str(app_dir / "app.py")] + args,
        cwd=str(app_dir),
        capture_output=True,
        text=True,
        env=env
    )
    return result

def test_non_existing_target(test_env):
    """1) The not existing file or folder should raise error."""
    result = run_app(test_env["app_dir"], ["resume=non_existent"])
    assert result.returncode != 0
    assert "FileNotFoundError" in result.stderr or "FileNotFoundError" in result.stdout
    assert "Resume target not found: non_existent" in result.stderr or "Resume target not found: non_existent" in result.stdout

def test_no_log_true(test_env):
    """2) no_log=true should run in the original log directory if available."""
    # Run with no_log=True. We check that it runs in initial_log_dir.
    result = run_app(test_env["app_dir"], ["resume=initial_run"], env_vars={"TEST_NO_LOG": "True"})
    assert result.returncode == 0
    assert "VAL_OUTPUT_DIR:" in result.stdout
    assert "initial_run" in result.stdout
    # Check that no NEW .hydra was created in app_dir (since output_subdir=null)
    assert not (test_env["app_dir"] / ".hydra").exists()

def test_folder_resume_no_log_false(test_env):
    """2a) For the folder resume, it should resume to the same folder, keeping the older hydra config."""
    # We resume to initial_run. It should run IN initial_run.
    result = run_app(test_env["app_dir"], ["resume=initial_run"], env_vars={"TEST_NO_LOG": "False"})
    assert result.returncode == 0
    assert "initial_run" in result.stdout
    
    # Check that backup was created
    backups = list(test_env["initial_log_dir"].glob(".hydra.old_*"))
    assert len(backups) >= 1

def test_ckpt_resume_no_log_false(test_env):
    """2b) When it resumes from ckpt, it should create a new logging folder."""
    # Resuming from a file should NOT force hydra.run.dir, so Hydra creates a new one.
    result = run_app(test_env["app_dir"], [f"resume=initial_run/checkpoints/last.ckpt"], env_vars={"TEST_NO_LOG": "False"})
    assert result.returncode == 0
    # VAL_OUTPUT_DIR should NOT be initial_run
    assert "VAL_OUTPUT_DIR:" in result.stdout
    output_line = [l for l in result.stdout.split('\n') if "VAL_OUTPUT_DIR:" in l][0]
    assert "initial_run" not in output_line.split("VAL_OUTPUT_DIR:")[1]

def test_use_saved_config_true(test_env):
    """3) When use_saved_config=true, then it should load from .hydra/config.yaml."""
    # initial_run/.hydra/config.yaml has param1: saved
    result = run_app(test_env["app_dir"], ["resume=initial_run", "param2=overridden"], env_vars={"TEST_USE_SAVED": "True"})
    assert result.returncode == 0
    assert "VAL_PARAM1: saved" in result.stdout
    assert "VAL_PARAM2: overridden" in result.stdout

def test_use_saved_config_false(test_env):
    """3) When use_saved_config=false, it should start from the existing config files with overrides."""
    # It should load param1: default from root configs
    result = run_app(test_env["app_dir"], ["resume=initial_run", "param2=overridden"], env_vars={"TEST_USE_SAVED": "False"})
    assert result.returncode == 0
    assert "VAL_PARAM1: default" in result.stdout
    assert "VAL_PARAM2: overridden" in result.stdout

def test_restore_hydra_choices(test_env):
    """Verify that Hydra choices are restored from the saved session."""
    # initial_run/.hydra/hydra.yaml should have some choices
    # We add them to the fixture setup if needed, but the current fixture setup
    # only creates hydra.yaml with empty task overrides.
    hydra_yaml = test_env["initial_log_dir"] / ".hydra" / "hydra.yaml"
    hydra_yaml.write_text("hydra:\n  overrides:\n    task: []\n  runtime:\n    choices:\n      data: saved_data\n")
    
    result = run_app(test_env["app_dir"], ["resume=initial_run"], env_vars={"TEST_USE_SAVED": "True"})
    assert result.returncode == 0
    # The app doesn't print choices by default, but we added a print in our test app if we want.
    # Let's check the logs for "Restored Hydra choices"
    assert "Restored Hydra choices from saved session" in result.stdout

from unittest.mock import patch

def test_wandb_id_resume_creates_new_folder(test_env):
    """2c) When you resume from wandb_id, it should also create a new log folder."""
    # We mock download_config and download_ckpt to avoid network calls
    # We also mock WandBTool's download functions in the subprocess if we were running it
    # But it's easier to just check that hydra.run.dir is NOT in the injected args
    with patch("hydra_auto_resume.cmd_line.download_config", return_value=["param1=wandb"]):
        import sys
        original_argv = sys.argv[:]
        try:
            sys.argv = ["app.py", "resume=abcdefgh"]
            from hydra_auto_resume.cmd_line import bootstrap
            bootstrap()
            # Check injected args in sys.argv
            assert "++wandb_id=abcdefgh" in sys.argv
            assert "param1=wandb" in sys.argv or "++param1=wandb" in sys.argv
            # hydra.run.dir should NOT be here because it's not a directory resume
            assert not any("hydra.run.dir" in arg for arg in sys.argv)
        finally:
            sys.argv = original_argv
