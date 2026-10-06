"""Notebook contract: generator parity, isolated runtime (no kernel install, no restart guard), guided
markers, no placeholders, no quality asserts, and an idempotent Section 1 executed against a stand-in
environment. None of this is clean-runtime execution evidence."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import re
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "timesfm_forecasting_colab.ipynb"
LOCK = ROOT / "tutorials" / "requirements-colab.lock.txt"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = _load("build_notebook")
TEMPLATE = _load("notebook_template").TEMPLATE


@pytest.fixture(scope="module")
def notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _source(cell: dict) -> str:
    src = cell["source"]
    return "".join(src) if isinstance(src, list) else src


def _code_cells(notebook: dict) -> list[dict]:
    return [c for c in notebook["cells"] if c["cell_type"] == "code"]


def _markdown(notebook: dict) -> str:
    return "\n".join(_source(c) for c in notebook["cells"] if c["cell_type"] == "markdown")


def kernel_cell(notebook: dict) -> dict:
    cells = [c for c in _code_cells(notebook) if "# dimer: kernel cell" in _source(c)]
    assert len(cells) == 1
    return cells[0]


# ---------------------------------------------------------------------------
# Parity (PAR1-PAR3)
# ---------------------------------------------------------------------------


def test_par3_generator_check_is_clean(notebook: dict) -> None:
    rendered = build.to_bytes(build.render(ROOT, TEMPLATE, build.recorded_revision(NOTEBOOK)))
    assert NOTEBOOK.read_bytes().replace(b"\r\n", b"\n") == rendered, "notebook is stale; run python tools/build_notebook.py"
    assert build.main(["--repo", str(ROOT), "--check"]) == 0


def test_par1_embedded_modules_equal_repository_modules(notebook: dict) -> None:
    tagged = [c for c in _code_cells(notebook) if c.get("metadata", {}).get("dimer", {}).get("embedded_module")]
    ctx = build.load_context(ROOT, TEMPLATE, build.recorded_revision(NOTEBOOK))
    assert [c["metadata"]["dimer"]["embedded_module"] for c in tagged] == ctx["module_rels"]
    assert len(tagged) == 7
    for cell, module in zip(tagged, ctx["modules"], strict=True):
        rel = f"{ctx['pkg_rel']}/{module}"
        assert cell["metadata"]["dimer"]["module_sha256"] == ctx["per_module_sha256"][rel]
        assert _source(cell).rstrip("\n") + "\n" == ctx["embedded"][module]
        assert cell["metadata"]["cellView"] == "form"  # GDL11: carried code collapsed


def test_par2_pins_manifest_and_lock_are_carried(notebook: dict) -> None:
    code = "\n".join(_source(c) for c in _code_cells(notebook))
    pins = re.search(r"^PINS = \[(.*?)^\]", code, re.M | re.S)
    assert pins and re.findall(r"'([^']+)'", pins.group(1)) == build._pins(ROOT, TEMPLATE)
    manifest = json.loads((ROOT / "weights" / TEMPLATE["weights_key"] / "dimer-base-manifest.json").read_text())
    inline = re.search(r"^MANIFEST = (\{.*?^\})$", code, re.M | re.S)
    assert inline and json.loads(inline.group(1)) == manifest
    lock_text = LOCK.read_text(encoding="utf-8")
    assert f"LOCK_SHA256 = '{hashlib.sha256(lock_text.encode()).hexdigest()}'" in code
    assert lock_text in code
    build.check_lock(build._pins(ROOT, TEMPLATE), lock_text)


# ---------------------------------------------------------------------------
# Isolated runtime: nothing installed into the kernel, no restart guard
# ---------------------------------------------------------------------------


def test_no_pip_install_into_the_kernel_and_no_restart_guard(notebook: dict) -> None:
    code = "\n".join(_source(c) for c in _code_cells(notebook))
    assert "subprocess.run([sys.executable, '-m', 'pip'" not in code
    assert "Restart the runtime" not in code
    assert "packages_distributions()" not in code
    assert "importlib.invalidate_caches" not in code
    for cell in _code_cells(notebook):
        for line in _source(cell).splitlines():
            assert not line.lstrip().startswith(("!", "%"))
    k = _source(kernel_cell(notebook))
    assert "'--require-hashes', '--only-binary', ':all:'" in k
    assert "MPLBACKEND='Agg'" in k and 'MPLBACKEND="Agg"' in k
    for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"):
        assert name in k
    assert "dimer_isolated_env_' + LOCK_SHA256[:12]" in k  # environment keyed on the lock digest
    assert "_isolated_environment_ready()" in k
    assert "worker_reused" in k
    assert notebook["metadata"]["dimer"]["environment"].startswith("isolated hash-locked uv environment")


def test_only_the_kernel_cell_runs_in_the_kernel(notebook: dict) -> None:
    cells = _code_cells(notebook)
    assert "# dimer: kernel cell" in _source(cells[0])
    assert all("# dimer: kernel cell" not in _source(c) for c in cells[1:])


# ---------------------------------------------------------------------------
# Guided layer, placeholders, quality asserts, form fields
# ---------------------------------------------------------------------------


GUIDED_MARKERS = (
    "## Who this notebook is for, and how to use it",
    "**How to use this notebook.**",
    "**Roadmap.**",
    "**Input → Model → Output.**",
    "## Glossary",
    "## Troubleshooting",
    "**Prediction.**",
    "**What to notice.**",
    "Check your reasoning",
    "Predict → Change one thing → Run → Observe → Explain",
    "**Conclude with evidence.**",
    "**Infrastructure.**",
    "**Profile:** `TASK-INFERENCE`",
    "**Mode:** `GUIDED`",
)


def test_guided_layer_markers_present(notebook: dict) -> None:
    md = _markdown(notebook)
    for marker in GUIDED_MARKERS:
        assert marker in md, marker
    assert md.count("**Prediction.**") >= 3 and md.count("**What to notice.**") >= 6
    assert md.count("<details><summary>Check your reasoning</summary>") >= 2
    assert notebook["metadata"]["dimer"]["notebook_mode"] == "GUIDED"
    assert notebook["metadata"]["dimer"]["notebook_profile"] == "TASK-INFERENCE"


def test_checkpoint_answers_invent_no_numbers(notebook: dict) -> None:
    md = _markdown(notebook)
    for block in re.findall(r"<details><summary>Check your reasoning</summary>(.*?)</details>", md, re.S):
        assert not re.search(r"\b\d+\.\d+\b", block.replace("0.8", "")), block  # no recorded metric values
    assert "No hosted run is recorded yet" in md


def test_infrastructure_cells_are_labelled_and_collapsed(notebook: dict) -> None:
    infra = [c for c in _code_cells(notebook) if _source(c).startswith("# @title Infrastructure")]
    assert len(infra) == 3  # install/route, record runtime, stage/verify/load
    for c in infra:
        assert c["metadata"].get("cellView") == "form"


def test_no_placeholders_or_assistant_identifiers(notebook: dict) -> None:
    text = NOTEBOOK.read_text(encoding="utf-8")
    assert not re.search(r"\b(TODO|TBD|FIXME)\b|# WRITE ME|Insert text here", text)
    assert not re.search(r"claude[-_ ]?(opus|sonnet|haiku|fable|\d)", text, re.I)


def test_no_quality_assert_only_contract_checks(notebook: dict) -> None:
    for cell in _code_cells(notebook):
        tree = ast.parse(_source(cell))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assert):
                text = ast.unparse(node.test)
                assert not re.search(r"(mase|smape|mae|loss|coverage)", text, re.I), text
    code = "\n".join(_source(c) for c in _code_cells(notebook))
    # verdicts are recorded
    assert "'verdict': 'recorded; no quality assertion'" in code
    assert "report['model_beats_on_mase']" in code
    # contract checks stay hard
    assert "raise RuntimeError(f'holdout leakage check failed" in code
    assert "check_reload_parity(forecast_frame, reloaded_forecast)" in code


def test_form_fields_do_not_shadow_carried_module_names(notebook: dict) -> None:
    """Every cell shares one namespace with the carried package modules, so a form field named like a module
    constant (for example VALUE_COLUMN) would silently override the package's behaviour."""
    carried = [c for c in _code_cells(notebook) if c.get("metadata", {}).get("dimer", {}).get("embedded_module")]
    module_names: set[str] = set()
    for c in carried:
        for node in ast.parse(_source(c)).body:
            if isinstance(node, ast.Assign):
                module_names |= {t.id for t in node.targets if isinstance(t, ast.Name)}
            elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                module_names.add(node.name)
    fields = re.findall(r"^([A-Z_]+) = .*  # @param", "\n".join(_source(c) for c in _code_cells(notebook)), re.M)
    assert fields
    assert not set(fields) & module_names, set(fields) & module_names
    # stage-cell assignments must not rebind a carried module's public names either
    stage_names: set[str] = set()
    for c in _code_cells(notebook):
        if c in carried or "# dimer: kernel cell" in _source(c):
            continue
        for node in ast.parse(_source(c)).body:
            if isinstance(node, ast.Assign):
                stage_names |= {t.id for t in node.targets if isinstance(t, ast.Name)}
    assert not stage_names & module_names, stage_names & module_names


def test_form_fields_default_to_the_sample_path_and_executor_can_set_them(notebook: dict) -> None:
    code = "\n".join(_source(c) for c in _code_cells(notebook))
    for field in ("USE_BYOD = False", "BYOD_PATH = ''", "HORIZON = 24", "CONTEXT_LENGTH = 312", "SEASON_LENGTH = 24", "ACTIVITY_CONTEXT_LENGTH = 48"):
        assert re.search(rf"^{re.escape(field)}  # @param", code, re.M), field
    # EXE2: the upload dialog is imported only on the BYOD branch without a path
    assert "from google.colab import files  # imported only on this branch" in code
    assert code.count("google.colab") == 1 + _source(kernel_cell(notebook)).count("google.colab")


# ---------------------------------------------------------------------------
# Section 1 executed against a stand-in isolated environment: idempotent and re-runnable
# ---------------------------------------------------------------------------


class _FakeShell:
    def __init__(self) -> None:
        self.input_transformers_cleanup: list = []


def _run_section_1(source: str, namespace: dict) -> dict:
    exec(compile(source, "<section-1>", "exec"), namespace, namespace)
    return namespace


def test_section_1_is_idempotent_against_a_standin_environment(notebook: dict, tmp_path: Path, monkeypatch, capsys) -> None:
    """Stand-in: a ready environment folder (this interpreter + the lock digest marker) and a fake IPython shell.
    Running the cell twice must reuse the environment, keep the live worker (same pid) and keep routing."""
    source = _source(kernel_cell(notebook))
    lock_sha = hashlib.sha256(LOCK.read_text(encoding="utf-8").encode()).hexdigest()
    env_dir = tmp_path / ("dimer_isolated_env_" + lock_sha[:12])
    (env_dir / "bin").mkdir(parents=True)
    os.symlink(sys.executable, env_dir / "bin" / "python")
    (env_dir / ".dimer-lock-sha256").write_text(lock_sha + "\n", encoding="utf-8")
    monkeypatch.setenv("DIMER_ISOLATED_ENV", str(env_dir))
    monkeypatch.delenv("DIMER_NOTEBOOK_CI_PREINSTALLED", raising=False)
    shell = _FakeShell()
    ipython = types.ModuleType("IPython")
    ipython.get_ipython = lambda: shell
    display_mod = types.ModuleType("IPython.display")
    shown: list = []
    display_mod.display = lambda data, raw=False: shown.append(data)
    ipython.display = display_mod
    monkeypatch.setitem(sys.modules, "IPython", ipython)
    monkeypatch.setitem(sys.modules, "IPython.display", display_mod)
    ns: dict = {"__name__": "__main__"}
    try:
        _run_section_1(source, ns)
        out1 = capsys.readouterr().out
        assert "'reused': True" in out1 and "'worker_reused': False" in out1
        runtime1 = ns["_DIMER_ISOLATED_RUNTIME"]
        assert runtime1.alive() and len(shell.input_transformers_cleanup) == 1
        # a routed cell runs in the worker and its output comes back
        runtime1.run("x = 21\nprint(x * 2)\n")
        assert "42" in capsys.readouterr().out
        _run_section_1(source, ns)
        out2 = capsys.readouterr().out
        assert "'reused': True" in out2 and "'worker_reused': True" in out2
        assert ns["_DIMER_ISOLATED_RUNTIME"] is runtime1 and ns["_DIMER_ISOLATED_RUNTIME"].proc.pid == runtime1.proc.pid
        assert len(shell.input_transformers_cleanup) == 1  # the transformer is replaced, not duplicated
        runtime1.run("print(x)\n")  # variables created before the re-run survive it
        assert "21" in capsys.readouterr().out
        # the router leaves kernel cells and empty cells alone and routes everything else
        route = shell.input_transformers_cleanup[0]
        assert route(["# dimer: kernel cell\n"]) == ["# dimer: kernel cell\n"]
        assert route(["\n"]) == ["\n"]
        assert route(["print(1)\n"]) == ["_DIMER_ISOLATED_RUNTIME.run('print(1)\\n')\n"]
        with pytest.raises(ns["IsolatedCellError"]):
            runtime1.run("raise ValueError('boom')\n")
    finally:
        runtime = ns.get("_DIMER_ISOLATED_RUNTIME")
        if runtime is not None and runtime.alive():
            runtime.close()


def test_section_1_preinstalled_mode_skips_everything(notebook: dict, monkeypatch, capsys) -> None:
    monkeypatch.setenv("DIMER_NOTEBOOK_CI_PREINSTALLED", "1")
    ns: dict = {"__name__": "__main__"}
    _run_section_1(_source(kernel_cell(notebook)), ns)
    out = capsys.readouterr().out
    assert "already installed" in out and "Routing disabled" in out
    assert "_DIMER_ISOLATED_RUNTIME" not in ns
