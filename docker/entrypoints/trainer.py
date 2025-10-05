import json
import os
import shutil
import sys
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
    try:
        from tmrl.training import trainer as tmrl_trainer
    except Exception as e:
        print(f"Failed to import tmrl trainer module: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Starting TMRL Trainer with config: {cfg}")
    tmrl_trainer.main()


if __name__ == "__main__":
    main()

