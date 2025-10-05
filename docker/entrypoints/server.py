import json
import os
import shutil
import sys
from pathlib import Path


def ensure_config():
    data_dir = Path("/TmrlData")
    config_dir = data_dir / "config"
    config_path = config_dir / "config.json"
    template = Path("/app/tmrl_templates/config.server.json")
    config_dir.mkdir(parents=True, exist_ok=True)
    if not config_path.exists() and template.exists():
        shutil.copyfile(template, config_path)
    return config_path


def main():
    cfg = ensure_config()
    # tmrl discovers TmrlData via env var or default path
    os.environ.setdefault("TMRL_DATA_PATH", "/TmrlData")
    try:
        # tmrl provides server runnable via tmrl.networking.start_server
        from tmrl.networking import server as tmrl_server
    except Exception as e:
        print(f"Failed to import tmrl server module: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Starting TMRL Server with config: {cfg}")
    # Fallback to default args; configuration is read from /TmrlData/config/config.json
    tmrl_server.main()


if __name__ == "__main__":
    main()

