"""Config validation tests."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from local_model_router.config import ConfigError, load_config


def write(tmp_path: Path, content: dict | str) -> Path:
    p = tmp_path / "config.yaml"
    if isinstance(content, dict):
        p.write_text(yaml.safe_dump(content))
    else:
        p.write_text(content)
    return p


def minimal() -> dict:
    return {
        "backends": {"local": {"base_url": "http://localhost:8080"}},
        "tools": {
            "t": {
                "description": "t",
                "task_type": "x",
                "parameters": {"text": {"type": "string", "required": True}},
                "user_prompt": "{text}",
                "routing": {"backend": "local", "reason": "ok"},
            }
        },
    }


def test_load_minimal_valid(tmp_path: Path) -> None:
    p = write(tmp_path, minimal())
    cfg = load_config(p)
    assert "local" in cfg.backends
    assert "t" in cfg.tools
    assert cfg.tools["t"].routing.backend == "local"
    assert cfg.defaults.backend == "decline"  # default applied


def test_tool_routing_to_undefined_backend_rejected(tmp_path: Path) -> None:
    raw = minimal()
    raw["tools"]["t"]["routing"]["backend"] = "ghost"
    p = write(tmp_path, raw)
    with pytest.raises(ConfigError, match="ghost"):
        load_config(p)


def test_decline_is_valid_backend_value(tmp_path: Path) -> None:
    raw = minimal()
    raw["tools"]["t"]["routing"]["backend"] = "decline"
    p = write(tmp_path, raw)
    cfg = load_config(p)
    assert cfg.tools["t"].routing.backend == "decline"


def test_override_without_backend_or_decline_rejected(tmp_path: Path) -> None:
    raw = minimal()
    raw["overrides"] = [{"match": {"tool": "*"}, "reason": "broken"}]
    p = write(tmp_path, raw)
    with pytest.raises(ConfigError, match="decline"):
        load_config(p)


def test_override_with_decline_true_is_valid(tmp_path: Path) -> None:
    raw = minimal()
    raw["overrides"] = [
        {"match": {"tool": "*"}, "decline": True, "reason": "global cap"},
    ]
    p = write(tmp_path, raw)
    cfg = load_config(p)
    assert cfg.overrides[0].decline is True


def test_override_with_undefined_backend_rejected(tmp_path: Path) -> None:
    raw = minimal()
    raw["overrides"] = [
        {"match": {"tool": "*"}, "backend": "missing", "reason": "x"},
    ]
    p = write(tmp_path, raw)
    with pytest.raises(ConfigError, match="missing"):
        load_config(p)


def test_defaults_backend_must_exist(tmp_path: Path) -> None:
    raw = minimal()
    raw["defaults"] = {"backend": "ghost", "reason": "x"}
    p = write(tmp_path, raw)
    with pytest.raises(ConfigError, match="ghost"):
        load_config(p)


def test_unknown_fields_ignored_for_phase2_compatibility(tmp_path: Path) -> None:
    """Tool routing rules should accept future fields without erroring."""
    raw = minimal()
    raw["tools"]["t"]["routing"]["evaluate"] = {"json_must_parse": True}
    raw["tools"]["t"]["future_field"] = "something"
    p = write(tmp_path, raw)
    cfg = load_config(p)
    assert "t" in cfg.tools  # didn't blow up


def test_real_repo_config_is_valid() -> None:
    """The shipped config.yaml at the repo root must always validate."""
    repo_cfg = Path(__file__).resolve().parent.parent / "config.yaml"
    cfg = load_config(repo_cfg)
    assert "local" in cfg.backends
    assert {"summarize", "extract", "classify"} <= set(cfg.tools.keys())
