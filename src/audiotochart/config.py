from __future__ import annotations

import json
from pathlib import Path

_CONFIG_DIR = Path.home() / ".config" / "audiotochart"
_CONFIG_PATH = _CONFIG_DIR / "config.json"

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_CONFIG: dict = {
    "backend": "model",
    "model_dir": str(_PACKAGE_ROOT / "models" / "finetuned"),
    "onset_decoder_dir": str(_PACKAGE_ROOT / "models" / "onset_decoder"),
    "device": "auto",
    "separate_drums": True,
    "quantize": "1/16",
    "tom_consistency": False,
    "charter": "AudioToChart",
    "output_dir": ".",
}


def load_config() -> dict:
    try:
        with open(_CONFIG_PATH) as f:
            data = json.load(f)
        merged = dict(DEFAULT_CONFIG)
        merged.update(data)
        return merged
    except (FileNotFoundError, json.JSONDecodeError):
        return dict(DEFAULT_CONFIG)


def save_config(cfg: dict) -> None:
    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(_CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)
    _CONFIG_PATH.chmod(0o600)


def config_exists() -> bool:
    return _CONFIG_PATH.is_file()
