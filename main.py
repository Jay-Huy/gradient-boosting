from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from src.orchestrator.runner import Runner
from src.utils.logging import get_logger


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return config or {}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Boosting project entrypoint")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to YAML config file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    runtime_log_file = config.get("runtime", {}).get("log_file")
    training_log_file = config.get("training", {}).get("log_file")
    logger = get_logger("main", runtime_log_file or training_log_file or "outputs/logs/main.log")
    logger.info("Loaded config:\n%s", yaml.safe_dump(config, sort_keys=False))

    runner = Runner(config)
    runner.run()


if __name__ == "__main__":
    main()
