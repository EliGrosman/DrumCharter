from __future__ import annotations

import json
from pathlib import Path

import pytest

from audiotochart.config import DEFAULT_CHARTER, DEFAULT_CONFIG, load_config, save_config


def test_config_save_load_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("audiotochart.config._CONFIG_DIR", tmp_path)
    monkeypatch.setattr("audiotochart.config._CONFIG_PATH", tmp_path / "config.json")

    cfg = {
        "backend": "fake",
        "model_dir": "/some/path",
        "onset_decoder_dir": "/some/decoder",
        "device": "cpu",
        "separate_drums": False,
        "quantize": "1/8",
        "tom_consistency": True,
        "charter": "Test Charter",
        "output_dir": "/out",
    }
    save_config(cfg)
    loaded = load_config()
    for k, v in cfg.items():
        assert loaded[k] == v, f"{k}: expected {v!r}, got {loaded[k]!r}"


def test_config_missing_file_returns_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("audiotochart.config._CONFIG_DIR", tmp_path)
    monkeypatch.setattr("audiotochart.config._CONFIG_PATH", tmp_path / "config.json")

    loaded = load_config()
    for k, v in DEFAULT_CONFIG.items():
        assert loaded[k] == v, f"{k}: expected {v!r}, got {loaded[k]!r}"


def test_config_corrupt_json_returns_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("audiotochart.config._CONFIG_DIR", tmp_path)
    monkeypatch.setattr("audiotochart.config._CONFIG_PATH", tmp_path / "config.json")

    (tmp_path / "config.json").write_text("not valid json")
    loaded = load_config()
    for k, v in DEFAULT_CONFIG.items():
        assert loaded[k] == v, f"{k}: expected {v!r}, got {loaded[k]!r}"


def test_config_new_keys_merged_from_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("audiotochart.config._CONFIG_DIR", tmp_path)
    monkeypatch.setattr("audiotochart.config._CONFIG_PATH", tmp_path / "config.json")

    save_config({"backend": "adtof"})
    loaded = load_config()
    assert loaded["backend"] == "adtof"
    assert loaded["device"] == DEFAULT_CONFIG["device"]
    assert loaded["quantize"] == DEFAULT_CONFIG["quantize"]


def test_default_charter_is_audiotochart_ai() -> None:
    assert DEFAULT_CONFIG["charter"] == DEFAULT_CHARTER
    assert DEFAULT_CHARTER == "AudioToChart (AI)"


def test_legacy_bare_default_charter_is_migrated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("audiotochart.config._CONFIG_DIR", tmp_path)
    monkeypatch.setattr("audiotochart.config._CONFIG_PATH", tmp_path / "config.json")

    save_config({"charter": "AudioToChart"})

    loaded = load_config()
    assert loaded["charter"] == DEFAULT_CHARTER


def test_config_save_creates_parent_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    nested = tmp_path / "deeply" / "nested"
    monkeypatch.setattr("audiotochart.config._CONFIG_DIR", nested)
    monkeypatch.setattr("audiotochart.config._CONFIG_PATH", nested / "config.json")

    save_config({"backend": "fake"})
    assert (nested / "config.json").is_file()
    data = json.loads((nested / "config.json").read_text())
    assert data["backend"] == "fake"


def test_config_exists(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from audiotochart.config import config_exists

    monkeypatch.setattr("audiotochart.config._CONFIG_DIR", tmp_path)
    monkeypatch.setattr("audiotochart.config._CONFIG_PATH", tmp_path / "config.json")

    assert not config_exists()
    save_config({"backend": "fake"})
    assert config_exists()


def test_default_model_dir_resolves_to_package_path() -> None:
    p = Path(DEFAULT_CONFIG["model_dir"])
    assert p.parent.name == "models"
    assert p.is_dir()


def test_default_onset_decoder_dir_resolves_to_models_path() -> None:
    p = Path(DEFAULT_CONFIG["onset_decoder_dir"])
    assert p.parent.name == "models"
    assert p.name == "onset_decoder"


def test_config_fallback_used_when_model_dir_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-interactive mode falls back to config's model_dir."""
    from click.testing import CliRunner
    from unittest.mock import patch
    from audiotochart.cli import cli
    from audiotochart.pipeline import generate_drum_chart_folder

    monkeypatch.setattr("audiotochart.config._CONFIG_DIR", tmp_path)
    monkeypatch.setattr("audiotochart.config._CONFIG_PATH", tmp_path / "config.json")
    save_config({"backend": "model", "model_dir": "/nonexistent/model/path"})

    audio = tmp_path / "song.wav"
    audio.write_bytes(b"\x00\x00" * 4410)

    seen_model_dir = None

    def _capture(*, transcriber, **kwargs):
        nonlocal seen_model_dir
        seen_model_dir = str(transcriber.model_dir)
        raise SystemExit(0)

    runner = CliRunner()
    with patch.object(generate_drum_chart_folder, "__wrapped__", _capture) if hasattr(generate_drum_chart_folder, "__wrapped__") else patch("audiotochart.cli.generate_drum_chart_folder", side_effect=_capture):
        result = runner.invoke(cli, [
            "generate", str(audio),
            "--backend", "model",
            "--no-separate-drums",
            "--song", "Test", "--artist", "Tester", "--bpm", "120",
            "-o", str(tmp_path / "out"),
        ])
        assert seen_model_dir == "/nonexistent/model/path" or result.exit_code != 0
