#!/usr/bin/env python3
"""
Standalone trainer that doesn't require the full TMRL framework
Uses our custom SAC implementation with mock environment
"""

import os
import sys
import shutil
from pathlib import Path

def ensure_config():
    """Ensure configuration exists"""
    data_dir = Path("/TmrlData")
    config_dir = data_dir / "config"
    config_path = config_dir / "config.json"
    template = Path("/app/tmrl_templates/config.trainer.json")
    config_dir.mkdir(parents=True, exist_ok=True)
    if not config_path.exists() and template.exists():
        shutil.copyfile(template, config_path)
    return config_path

def main():
    """Main training function"""
    cfg = ensure_config()
    os.environ.setdefault("TMRL_DATA_PATH", "/TmrlData")
    
    print(f"Starting Standalone SAC Trainer with config: {cfg}")
    
    # Add scripts to path
    sys.path.append('/app/scripts')
    
    try:
        # Try to use TMRL first
        from tmrl.training import trainer as tmrl_trainer
        print("Using TMRL trainer...")
        tmrl_trainer.main()
    except Exception as e:
        print(f"TMRL trainer failed: {e}")
        print("Falling back to standalone SAC trainer...")
        
        try:
            from train_sac import main as sac_main
            sac_main()
        except Exception as e2:
            print(f"Standalone trainer also failed: {e2}")
            import traceback
            traceback.print_exc()
            
            # Keep container running for debugging
            print("Keeping container alive for debugging...")
            import time
            while True:
                time.sleep(60)

if __name__ == "__main__":
    main()