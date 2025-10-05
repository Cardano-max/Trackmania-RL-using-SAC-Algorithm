import json
import os
import shutil
import sys
import time
from pathlib import Path


def ensure_config():
    data_dir = Path("/TmrlData")
    config_dir = data_dir / "config"
    config_path = config_dir / "config.json"
    template = Path("/app/tmrl_templates/config.trainer.json")
    config_dir.mkdir(parents=True, exist_ok=True)
    if not config_path.exists() and template.exists():
        shutil.copyfile(template, config_path)
    return config_path


def main():
    cfg = ensure_config()
    os.environ.setdefault("TMRL_DATA_PATH", "/TmrlData")
    
    # Wait for server to be ready
    print("Waiting for TMRL server to be ready...")
    time.sleep(10)
    
    try:
        from tmrl.networking import RolloutWorker
        from tmrl import get_environment
    except Exception as e:
        print(f"Failed to import tmrl modules: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Starting TMRL Rollout Worker with config: {cfg}")
    
    try:
        # Create environment for rollout
        env = get_environment()
        
        # Create rollout worker
        worker = RolloutWorker(
            env_cls=lambda: env,
            worker_id=0,
            max_samples_per_episode=1000,
            obs_preprocessor=None,
            device="cpu"
        )
        
        # Start collecting samples
        worker.run()
        
    except Exception as e:
        print(f"Error during rollout: {e}", file=sys.stderr)
        # Keep container running for debugging
        while True:
            time.sleep(60)


if __name__ == "__main__":
    main()