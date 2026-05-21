from __future__ import annotations

import argparse
import json

from .config import load_config, validate_config
from .io import display_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a Catch-22 reproduction config.")
    parser.add_argument("config")
    args = parser.parse_args()
    config = load_config(args.config)
    errors = validate_config(config)
    payload = {
        "config": args.config,
        "track": config.track,
        "model_name": config.model_name,
        "dataset_path": display_path(config.dataset_path, config.root),
        "output_dir": display_path(config.output_dir, config.root),
        "methods": config.methods,
        "attacks": config.attacks,
        "valid": not errors,
        "errors": errors,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
