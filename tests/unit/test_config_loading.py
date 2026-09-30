"""Configuration composition: includes, precedence, failure modes."""

from __future__ import annotations

from pathlib import Path

import pytest

from xai_gov.io.yaml_loader import ConfigError, deep_merge, load_config, require


def write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_include_is_merged_and_local_keys_win(tmp_path: Path) -> None:
    write(tmp_path / "base.yaml", "sim:\n  periods: 30\n  echelons: 1\n")
    target = write(
        tmp_path / "exp.yaml",
        "_include_: base.yaml\nsim:\n  periods: 60\nname: exp\n",
    )
    config = load_config(target)
    assert config["sim"] == {"periods": 60, "echelons": 1}
    assert config["name"] == "exp"


def test_later_includes_win_over_earlier_ones(tmp_path: Path) -> None:
    write(tmp_path / "a.yaml", "k: 1\n")
    write(tmp_path / "b.yaml", "k: 2\n")
    target = write(tmp_path / "c.yaml", "_include_:\n  - a.yaml\n  - b.yaml\n")
    assert load_config(target)["k"] == 2


def test_missing_file_raises_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_config(tmp_path / "absent.yaml")


def test_empty_file_raises_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_config(write(tmp_path / "empty.yaml", ""))


def test_include_cycle_is_detected(tmp_path: Path) -> None:
    write(tmp_path / "x.yaml", "_include_: y.yaml\n")
    write(tmp_path / "y.yaml", "_include_: x.yaml\n")
    with pytest.raises(ConfigError):
        load_config(tmp_path / "x.yaml")


def test_require_names_the_full_missing_path() -> None:
    with pytest.raises(ConfigError, match=r"sim\.periods"):
        require({"sim": {}}, "sim", "periods")


def test_deep_merge_does_not_mutate_inputs() -> None:
    base = {"a": {"b": 1}}
    deep_merge(base, {"a": {"c": 2}})
    assert base == {"a": {"b": 1}}


def test_shipped_app_configs_load(project_root: Path) -> None:
    for name in ("paths.yaml", "logging.yaml", "runtime.yaml", "tracking.yaml"):
        assert load_config(project_root / "configs" / "app" / name)
