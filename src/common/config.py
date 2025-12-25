"""Configuration management for loading and validating YAML configuration files."""

import os
import yaml
from pathlib import Path
from src.common.constants import DTYPE


class ConfigLoader:
    """Loads and manages configurations for the project.
    Args:
        config_path (Path | str): Path to the configuration file.
    """

    def __init__(self, config_path: Path | str):
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> dict:
        """Loads YAML configuration file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        try:
            with self.config_path.open("r", encoding="utf-8") as file:
                return yaml.safe_load(file) or {}
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML config: {e}")

    def validate_dtypes(self) -> None:
        """Validates and updates data types based on predefined constants"""
        for column, column_info in self.config.get("columns", {}).items():
            dtype = column_info.get("dtype")
            if dtype in DTYPE:
                column_info["dtype"] = DTYPE[dtype]
            else:
                raise ValueError(
                    f"Unknown dtype: {dtype}, column: {column}, available dtypes: {list(DTYPE.keys())}"
                )

    def get_config(self, segment: str) -> dict:
        """Returns the configuration dictionary"""
        if segment not in self.config:
            raise ValueError(
                f"Unknown config segment: {segment}, available segments: {list(self.config.keys())}"
            )
        return self.config[segment]


def get_config_path() -> Path:
    """Determines the configuration file path based on environment variables."""
    workdir = os.getenv("WORKDIR")
    return (
        Path(workdir, "config/default_config.yaml")
        if workdir
        else Path("config/default_config.yaml")
    )
