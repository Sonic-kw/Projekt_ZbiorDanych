import yaml
from pathlib import Path
from typing import Any, Dict

def load_config(config_path: str = "config/settings.yaml") -> Dict[str, Any]:
    """
    Loads configuration from a YAML file.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    
    with open(path, 'r') as file:
        return yaml.safe_load(file)
