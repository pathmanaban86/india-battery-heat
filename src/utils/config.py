"""Configuration loading. All parameters live in configs/*.yaml."""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load(name):
    """Load configs/<name>.yaml as a dictionary."""
    with open(ROOT / 'configs' / f'{name}.yaml') as f:
        return yaml.safe_load(f)


def path(rel):
    """Absolute path for a path given relative to the repository root."""
    return ROOT / rel
