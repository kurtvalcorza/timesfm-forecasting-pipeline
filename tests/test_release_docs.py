"""Cross-document consistency: pins, lock, identity, status tokens, and the static validator itself."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _tool(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_static_release_validator_passes() -> None:
    failures = _tool("validate_release_assets").validate_all()
    assert failures == [], "\n".join(failures)


def test_pins_file_equals_pyproject_versions() -> None:
    pins = {}
    for line in (ROOT / "tutorials" / "requirements-colab.in").read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            name, version = line.split("==")
            pins[name.lower().replace("_", "-")] = version
    declared = dict(re.findall(r'"([A-Za-z0-9_.-]+)==([^"]+)"', (ROOT / "pyproject.toml").read_text()))
    declared = {k.lower().replace("_", "-"): v for k, v in declared.items()}
    assert pins == {k: declared[k] for k in pins}
    assert set(pins) == {"timesfm", "torch", "numpy", "pandas", "huggingface-hub", "safetensors", "matplotlib"}


def test_status_is_candidate_with_no_hosted_run() -> None:
    status = (ROOT / "STATUS.md").read_text(encoding="utf-8")
    assert "**Candidate" in status and "clean_runtime_evidence: false" in status
    rv = (ROOT / "docs" / "release-verification.md").read_text(encoding="utf-8")
    table = rv.split("## Recorded executions", 1)[1]
    rows = [line for line in table.splitlines() if re.match(r"\|\s*\d{4}-\d{2}-\d{2}\s*\|", line)]
    assert "| Date |" in table and "|---|---|---|---|---|---|\n|" in table  # rows follow the header directly
    assert all("**FAILED**" in row or "**PASSED" in row for row in rows), "every recorded run states its outcome"
    if any("**PASSED" in row for row in rows):
        assert "passed one hosted" in status.lower() and "pending" in status.lower()
    else:
        assert "no successful hosted run" in status.lower()


def test_readme_has_the_colab_link_and_names_the_boundary() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "https://colab.research.google.com/github/kurtvalcorza/timesfm-forecasting-pipeline/blob/main/tutorials/timesfm_forecasting_colab.ipynb" in readme
    assert "google/timesfm-3.0-pytorch" in readme
    assert "non-commercial" in readme and "non-production" in readme  # the 3.0 weights licence is stated
    assert "google/timesfm-2.5-200m-pytorch" in readme  # the Apache-2.0 fallback build is named


def test_model_card_states_metrics_and_quantile_semantics() -> None:
    card = (ROOT / "MODEL_CARD.md").read_text(encoding="utf-8")
    for required in ("**MASE**", "**sMAPE**", "**quantile loss**", "model quantiles", "median"):
        assert required in card, required
    assert 'model_card_spec: "1.2"' in card


def test_no_assistant_model_identifier_anywhere() -> None:
    pattern = re.compile(r"claude[-_ ]?(opus|sonnet|haiku|fable|\d)", re.I)
    for path in ROOT.rglob("*"):
        if path.is_file() and ".git" not in path.parts and path.suffix in {".md", ".py", ".toml", ".yml", ".json", ".txt", ".ipynb", ".in", ""}:
            assert not pattern.search(path.read_text(encoding="utf-8", errors="ignore")), path
