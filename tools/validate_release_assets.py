"""Static release-asset validation for the TimesFM zero-shot forecasting DIMER pipeline.

Checks the STANDALONE tutorial notebook (DIMER Notebook Specification 2.2 §4), the tutorial registry,
model card, README, STATUS.md, the weight manifest and the lock for source conformance and
cross-document identity consistency, and runs the generator parity check (PAR3).

This is source validation only. A PASS here is NOT clean-runtime execution evidence; the release gate
is defined in docs/release-verification.md.
"""
# ruff: noqa: E501  -- rule messages name the file and requirement in full; they are kept on one line
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "timesfm_forecasting"
REPO_NAME = "timesfm-forecasting-pipeline"
NOTEBOOK_NAME = "timesfm_forecasting_colab.ipynb"
EXPECTED_PROFILE = "TASK-INFERENCE"
EXPECTED_MODE = "GUIDED"
EXPECTED_MODEL_ID = "google/timesfm-3.0-pytorch"
# Pinned from the maintainer's hosted Colab run of 2026-10-06 (docs/execution-evidence/2026-10-06/
# timesfm_forecasting_colab_86f2213_run2_passed.ipynb): the commit the Hub served for ref main and the SHA-256
# computed in-run over the staged model.safetensors.
EXPECTED_MODEL_REVISION = "43046b85ec22d584a13f8098c2ed39c889e129c2"
EXPECTED_WEIGHTS_SHA256 = "a7592b0a8432baee54483254e5647856911ce69e09d09a9bb65904b2d98f17da"
MODEL_LOAD_EXPR = "load_pinned_model(weights_dir=WEIGHTS_DIR)"
NOTEBOOK_SPEC = "2.2"
GENERATOR_SPEC = "2.0"  # the vendored generator's metadata value (its NOTEBOOK_SPEC constant)
STATUS_TOKENS = ("Candidate", "Release-grade")
PLACEHOLDER = re.compile(r"\b(TODO|TBD|FIXME)\b|# WRITE ME|Insert text here|Tooltip:", re.I)
SHA40 = re.compile(r"\b[0-9a-f]{40}\b")
# 40-hex strings a document may legitimately contain: the commit of this repository the pinned sample URL names
# (also the TimesFM 2.5 Apache-2.0 build the README names as the fallback) and the pinned checkpoint commit.
KNOWN_SHAS = frozenset({"e47259ae75b93e7611c67e13a14d6607997c38f3", EXPECTED_MODEL_REVISION})
UNSUPPORTED_CLAIMS = re.compile(
    r"\b(production[- ]ready|battle[- ]tested|state[- ]of[- ]the[- ]art results (were|are) reproduced"
    r"|benchmark superiority (is|was) (shown|established)|is release-grade|now release-grade)\b",
    re.I,
)
BYOD_GATES = ("USE_BYOD",)
EXPECTED_OUTPUTS = (
    "outputs/timesfm_forecasting_input_manifest.json",
    "outputs/timesfm_forecasting_evaluation_report.json",
    "outputs/timesfm_forecasting_future_forecast.csv",
    "'outputs', 'timesfm_forecasting', forecast=forecast_frame",
)
CODE_MARKERS = (
    "USE_BYOD = False",
    "BYOD_PATH = ''",
    "from google.colab import files",
    "fetch_sample(SAMPLE_URL, SAMPLE_SHA256, SAMPLE_BYTES)",
    "frame = load_series_csv(data_source['path'], config)",
    "validation = validate_series(frame, config, source=data_source['path'])",
    "validate_series(probe, probe_config, source='probe-duplicate-timestamp.csv')",
    "split = chronological_holdout(validation.frame, config.horizon)",
    "leakage = split.leakage_check()",
    "naive_baseline(split.history, frequency, config.horizon)",
    "ses_baseline(split.history, frequency, config.horizon)",
    "seasonal_naive_baseline(split.history, frequency, config.horizon, config.season_length)",
    "result = forecast(history_validation, config, pipe)",
    "model_eval = evaluate_forecast(forecast_frame, split.truth, split.history, config)",
    "report = evaluation_report(model_eval, baseline_evals, horizon=config.horizon, sample_kind=sample_kind)",
    "ACTIVITY_CONTEXT_LENGTH = 48",
    "future_result = forecast(validation, config, pipe)",
    "write_result_bundle(",
    "reloaded_forecast, reloaded_result = reload_result_bundle('outputs', 'timesfm_forecasting')",
    "reload_parity = check_reload_parity(forecast_frame, reloaded_forecast)",
    "stage_missing_files(WEIGHTS_DIR, allow_download=True)",
    "snapshot = verify_snapshot(WEIGHTS_DIR)",
    f"pipe = {MODEL_LOAD_EXPR}",
)
MARKDOWN_MARKERS = (
    "**Profile:** `TASK-INFERENCE`",
    "**Mode:** `GUIDED`",
    "## Who this notebook is for, and how to use it",
    "**How to use this notebook.**",
    "**Roadmap.**",
    "**Input → Model → Output.**",
    "**Prediction.**",
    "**What to notice.**",
    "Check your reasoning",
    "Predict → Change one thing → Run → Observe → Explain",
    "## Troubleshooting",
    "## Glossary",
    "**Conclude with evidence.**",
    "**Infrastructure.**",
    "**The point forecast is the median, and the band is a model statement.**",
    "**Zero-shot does not mean untrained.**",
    "**Weights licence.**",
    "verdicts are recorded, not asserted",
    "not a benchmark",
    "**AI Assistance Disclosure:**",
    f"immutable revision `{EXPECTED_MODEL_REVISION}`",
    "(never `main`)",
)
# Direct library use that must stay inside the carried module cells (the notebook calls the package API).
FORBIDDEN_OUTSIDE_MODULE = (
    "from huggingface_hub import",
    "import huggingface_hub",
    "hf_hub_download(",
    "snapshot_download(",
    "import timesfm",
    "from timesfm",
    "TimesFM3Forecaster(",
    "TimesFM3Torch",
    "from_pretrained(",
    "from safetensors",
    "torch.load(",
    "pickle.load",
    "trust_remote_code",
    "extractall(",
    "git clone",
    "pip install",
)
REQUIRED_CARD_HEADINGS = [
    (4, "Description"),
    (4, "Intended Use and Limitations"),
    (6, "Primary Intended Uses"),
    (6, "Primary Intended Users"),
    (6, "Out-of-scope use cases"),
    (4, "Factors"),
    (6, "Groups"),
    (6, "Instrumentation"),
    (6, "Environment"),
    (4, "Metrics"),
    (6, "Performance Measures"),
    (6, "Decision thresholds"),
    (6, "Approaches to uncertainty and variability"),
    (4, "Ethical considerations and biases"),
    (6, "Data"),
    (6, "Human Life"),
    (6, "Mitigations"),
    (6, "Risks and harms"),
    (6, "Use cases"),
]

FAILURES: list[str] = []


def _check(condition: bool, message: str) -> None:
    if not condition:
        FAILURES.append(message)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_tool(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _package_identity() -> dict[str, str]:
    text = _read(ROOT / "src" / PACKAGE / "model.py")
    out = {}
    for name in ("MODEL_ID", "MODEL_REVISION", "MODEL_LICENSE", "MODEL_KEY"):
        m = re.search(rf'^{name} = "([^"]+)"$', text, re.M)
        _check(bool(m), f"src/{PACKAGE}/model.py: {name} not found as a top-level string constant")
        out[name] = m.group(1) if m else ""
    return out


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------


def validate_model_card() -> None:
    path = ROOT / "MODEL_CARD.md"
    text = _read(path)
    _check(text.startswith("---\n"), "MODEL_CARD.md: must open with a YAML front-matter block (G1)")
    front = text.split("---\n", 2)[1] if text.startswith("---\n") else ""
    for key in ("license:", "model_card_spec:", "pipeline_tag:", "base_model:", "date_published:"):
        _check(key in front, f"MODEL_CARD.md: front matter lacks {key} (G1)")
    _check('model_card_spec: "1.2"' in front, "MODEL_CARD.md: model_card_spec must be \"1.2\"")
    _check(f"base_model: {EXPECTED_MODEL_ID}" in front, "MODEL_CARD.md: base_model must name the pinned checkpoint")
    body = text[len(front) + 8 :] if front else text
    h1 = [line for line in body.splitlines() if line.startswith("# ")]
    _check(len(h1) == 1, f"MODEL_CARD.md: exactly one level-1 heading required, found {len(h1)} (G3)")
    _check(not PLACEHOLDER.search(text), "MODEL_CARD.md: placeholder/tooltip marker survives (G9/G11)")
    headings = [(len(m.group(1)), m.group(2).strip().strip("*").strip()) for m in re.finditer(r"^(#{4,6}) (.+)$", body, re.M)]
    wanted = [(lvl, name.lower()) for lvl, name in REQUIRED_CARD_HEADINGS]
    got = [(lvl, name.lower()) for lvl, name in headings if (lvl, name.lower()) in set(wanted)]
    _check(got == wanted, f"MODEL_CARD.md: required sections missing, misordered or at the wrong level (G4-G6): got {got}")
    for lvl, name in REQUIRED_CARD_HEADINGS:
        if lvl == 4 and name != "Description":
            continue  # containers carry no prose of their own (MODEL_CARD_SPEC §4); G13 binds content sections
        m = re.search(rf"^{'#' * lvl} \*?\*?{re.escape(name)}\*?\*?\s*$(.*?)(?=^#{{4,6}} |\Z)", body, re.M | re.S | re.I)
        if m:
            words = len(re.findall(r"\w+", m.group(1)))
            _check(words >= 40, f"MODEL_CARD.md: section {name!r} has {words} words (<40, G13)")
            _check(not re.fullmatch(r"\s*(N/A|None|Not applicable)\.?\s*", m.group(1), re.I), f"MODEL_CARD.md: section {name!r} answered with a bare N/A (G10)")
    _check(not UNSUPPORTED_CLAIMS.search(text), "MODEL_CARD.md: unsupported readiness/benchmark claim")
    _check("non-commercial" in text and "non-production" in text, "MODEL_CARD.md: must state the TimesFM 3.0 weights licence restriction (non-commercial, non-production) (LIC5)")
    _check("timesfm-non-commercial-license-v1.0" in text, "MODEL_CARD.md: must name the weights licence timesfm-non-commercial-license-v1.0")
    _check(EXPECTED_MODEL_REVISION in text, "MODEL_CARD.md: must name the pinned checkpoint commit")
    _check(EXPECTED_WEIGHTS_SHA256 in text, "MODEL_CARD.md: must record the pinned model.safetensors SHA-256")
    _check("to be confirmed on the first hosted run" not in text, "MODEL_CARD.md: stale pending-revision wording (the pin is recorded)")


def validate_identity_consistency() -> None:
    ident = _package_identity()
    _check(ident["MODEL_ID"] == EXPECTED_MODEL_ID, f"model.py MODEL_ID {ident['MODEL_ID']!r} != {EXPECTED_MODEL_ID!r}")
    _check(ident["MODEL_REVISION"] == EXPECTED_MODEL_REVISION, f"model.py MODEL_REVISION {ident['MODEL_REVISION']!r} != the commit observed on the 2026-10-06 hosted run")
    _check(bool(SHA40.fullmatch(ident["MODEL_REVISION"])), "model.py MODEL_REVISION must be an immutable 40-hex commit (MOD2)")
    manifest = json.loads(_read(ROOT / "weights" / ident["MODEL_KEY"] / "dimer-base-manifest.json"))
    _check(manifest["modelId"] == ident["MODEL_ID"] and manifest["revision"] == ident["MODEL_REVISION"], "manifest identity != module identity")
    pending = [f["path"] for f in manifest["files"] if not re.fullmatch(r"[0-9a-f]{64}", str(f.get("sha256")))]
    for f in manifest["files"]:
        if f["path"] in pending:
            _check(str(f.get("sha256", "")).startswith("pending:"), f"manifest: {f['path']} has neither a 64-hex digest nor the explicit 'pending:' sentinel")
    if not SHA40.fullmatch(ident["MODEL_REVISION"]):
        _check(
            "revisionStatus" in manifest and "to be confirmed on the first hosted run" in manifest["revisionStatus"],
            "manifest: a non-SHA revision must carry revisionStatus 'to be confirmed on the first hosted run'",
        )
        _check("STATUS.md" and "to be confirmed on the first hosted run" in _read(ROOT / "STATUS.md"), "STATUS.md must state the pending revision (MOD2 deviation)")
    else:
        _check(not pending, f"manifest: the revision is an immutable commit but {pending} carry no 64-hex digest")
    weights = next((f for f in manifest["files"] if f["path"] == "model.safetensors"), {})
    _check(weights.get("sha256") == EXPECTED_WEIGHTS_SHA256, "manifest: model.safetensors sha256 != the digest observed on the 2026-10-06 hosted run")
    _check("provenance" in manifest and "model.safetensors" in manifest["provenance"] and "revision" in manifest["provenance"], "manifest: must state where the revision and each digest came from (provenance block)")
    _check(manifest["totalBytes"] == sum(int(f["bytes"]) for f in manifest["files"]), "manifest totalBytes != sum of files")
    for doc in ("README.md", "MODEL_CARD.md", "STATUS.md", "tutorials/README.md"):
        text = _read(ROOT / doc)
        _check(EXPECTED_MODEL_ID in text, f"{doc}: must name the pinned model id {EXPECTED_MODEL_ID}")
        stray = {s for s in SHA40.findall(text)} - KNOWN_SHAS
        _check(not stray, f"{doc}: unexpected 40-hex revision(s) {sorted(stray)}; only the pinned checkpoint commit and the sample/fallback commit are known")
        if pending:
            _check("pending" in text.lower() or "to be confirmed" in text, f"{doc}: must mention that the weight digest is pending")


def validate_release_status() -> None:
    tokens = {}
    for doc in ("STATUS.md", "README.md", "tutorials/README.md"):
        text = _read(ROOT / doc)
        present = [t for t in STATUS_TOKENS if re.search(rf"\*\*{t}\b", text)]
        tokens[doc] = present
        _check(not UNSUPPORTED_CLAIMS.search(text), f"{doc}: unsupported readiness/benchmark claim")
        _check(not PLACEHOLDER.search(text), f"{doc}: placeholder marker survives")
    _check(all("Candidate" in v for v in tokens.values()), f"release status token must be **Candidate** in every document: {tokens}")
    _check(not any("Release-grade" in v for v in tokens.values()), f"no document may declare **Release-grade**: {tokens}")
    status = _read(ROOT / "STATUS.md")
    _check("clean_runtime_evidence: false" in status, "STATUS.md must state clean_runtime_evidence: false")
    rv = _read(ROOT / "docs" / "release-verification.md")
    _check("## Recorded executions" in rv, "docs/release-verification.md must carry the recorded-executions table")


def validate_pins_and_lock() -> None:
    build = _load_tool("build_notebook")
    template = _load_tool("notebook_template").TEMPLATE
    pins = build._pins(ROOT, template)
    lock = _read(ROOT / template["lock"])
    build.check_lock(pins, lock)
    pyproject = _read(ROOT / "pyproject.toml")
    declared = dict(re.findall(r'"([A-Za-z0-9_.-]+)==([^"]+)"', pyproject))
    for pin in pins:
        name, version = pin.split("==", 1)
        canon = name.replace("_", "-").lower()
        found = {k.replace("_", "-").lower(): v for k, v in declared.items()}.get(canon)
        _check(found == version, f"pins file pins {pin} but pyproject.toml declares {found}; keep them equal")
    _check("uv.lock" and (ROOT / "uv.lock").is_file(), "uv.lock missing (CI uses uv sync --locked)")


# ---------------------------------------------------------------------------
# Notebook
# ---------------------------------------------------------------------------


def _cell_source(cell: dict) -> str:
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else src


def _is_quality_assert(node: ast.Assert) -> bool:
    """An `assert` comparing a metric with a baseline or threshold (the pattern the fleet sweep removed)."""
    text = ast.unparse(node.test)
    return bool(re.search(r"(mase|smape|mae|rmse|loss|coverage|accuracy|f1)", text, re.I) and re.search(r"[<>]", text))


def validate_notebook() -> None:
    path = ROOT / "tutorials" / NOTEBOOK_NAME
    _check(path.is_file(), f"{path} missing")
    if not path.is_file():
        return
    nb = json.loads(_read(path))
    _check(nb.get("nbformat") == 4, "notebook: nbformat must be 4")
    meta = nb["metadata"].get("dimer", {})
    _check(meta.get("notebook_profile") == EXPECTED_PROFILE, f"notebook metadata profile {meta.get('notebook_profile')!r}")
    _check(meta.get("notebook_mode") == EXPECTED_MODE, f"notebook metadata mode {meta.get('notebook_mode')!r}")
    _check(meta.get("standalone") is True, "notebook metadata standalone != true")
    _check(meta.get("generated_from", {}).get("repository") == REPO_NAME, "notebook generated_from.repository mismatch")
    _check("isolated" in meta.get("environment", ""), "notebook metadata must declare the isolated environment")
    cells = nb["cells"]
    code_cells = [(i, c) for i, c in enumerate(cells) if c["cell_type"] == "code"]
    all_md = "\n".join(_cell_source(c) for c in cells if c["cell_type"] == "markdown")
    all_code = "\n".join(_cell_source(c) for _, c in code_cells)
    carried = [c for _, c in code_cells if c.get("metadata", {}).get("dimer", {}).get("embedded_module")]
    _check(len(carried) == 7, f"notebook: expected 7 carried module cells, found {len(carried)}")
    kernel_cells = [c for _, c in code_cells if "# dimer: kernel cell" in _cell_source(c)]
    _check(len(kernel_cells) == 1, "notebook: exactly one kernel cell (the isolated-environment install) expected")
    for i, c in code_cells:
        src = _cell_source(c)
        try:
            tree = ast.parse(src)
        except SyntaxError as exc:
            FAILURES.append(f"notebook cell {i}: does not compile: {exc}")
            continue
        _check(c.get("outputs") == [] and c.get("execution_count") is None, f"notebook cell {i}: persisted outputs (SRC4)")
        _check(not any(line.lstrip().startswith(("!", "%")) for line in src.splitlines()), f"notebook cell {i}: shell/magic line (SRC6)")
        _check(i > 0 and cells[i - 1]["cell_type"] == "markdown", f"notebook cell {i}: code cell without a preceding markdown cell")
        for node in ast.walk(tree):
            if isinstance(node, ast.Assert):
                _check(not _is_quality_assert(node), f"notebook cell {i}: quality assert `{ast.unparse(node.test)}` (record a verdict instead)")
        if c in carried or c in kernel_cells:
            continue
        for pattern in FORBIDDEN_OUTSIDE_MODULE:
            _check(pattern not in src, f"notebook cell {i}: {pattern!r} outside the carried module cells")
    _check("pip install" not in all_code.replace("'pip', 'install'", "").replace("uv pip install", "") or "'pip', 'install'" not in all_code, "notebook: pip install into the kernel")
    _check("subprocess.run([sys.executable, '-m', 'pip'" not in all_code, "notebook: pip install into the kernel (restart guard pattern)")
    _check("Restart the runtime" not in all_code, "notebook: restart guard survives")
    _check("MPLBACKEND" in all_code and "PYTHONSTARTUP" in all_code, "notebook: the isolated runtime must force MPLBACKEND=Agg and drop PYTHONSTARTUP")
    _check("--require-hashes" in all_code and "--only-binary" in all_code, "notebook: the lock must be installed with --require-hashes --only-binary :all:")
    lock_text = _read(ROOT / "tutorials" / "requirements-colab.lock.txt")
    _check(hashlib.sha256(lock_text.encode("utf-8")).hexdigest() in all_code, "notebook: carried lock digest != tutorials/requirements-colab.lock.txt")
    for gate in BYOD_GATES:
        _check(f"{gate} = False" in all_code, f"notebook: BYOD gate {gate} must default to False (RUN2)")
    for marker in CODE_MARKERS:
        _check(marker in all_code, f"notebook: code marker missing: {marker!r}")
    for marker in MARKDOWN_MARKERS:
        _check(marker in all_md, f"notebook: markdown marker missing: {marker!r}")
    for out in EXPECTED_OUTPUTS:
        _check(out in all_code, f"notebook: expected output path missing: {out!r}")
    _check(not PLACEHOLDER.search(all_md + all_code), "notebook: placeholder marker survives (SRC3)")
    _check("warnings.filterwarnings('ignore')" not in all_code and 'warnings.filterwarnings("ignore")' not in all_code, "notebook: global warning suppression (SRC11)")
    sample_url = _load_tool("notebook_template").SAMPLE_URL
    _check("github.com/kurtvalcorza" not in all_code.replace(sample_url, ""), "notebook: repository source fetched at runtime (ST3/ST4)")
    _check("non-commercial" in all_md and "non-production" in all_md, "notebook: must state the weights licence restriction before the download cell")
    # The generating repository commit (NOTEBOOK_SOURCE.repository_revision) is the only other 40-hex value allowed.
    stray = set(SHA40.findall(all_code + all_md)) - KNOWN_SHAS - {str(meta.get("generated_from", {}).get("revision"))}
    _check(not stray, f"notebook: unexpected 40-hex revision(s) {sorted(stray)}")
    claude = re.search(r"claude[-_ ]?(opus|sonnet|haiku|fable|\d)", all_code + all_md, re.I)
    _check(claude is None, "notebook: a model identifier of the authoring assistant must not appear")
    # Infrastructure cells are labelled and collapsed (GDL11).
    for _, c in code_cells:
        src = _cell_source(c)
        if src.startswith("# @title Infrastructure") or c in carried:
            _check(c.get("metadata", {}).get("cellView") == "form", "notebook: an infrastructure cell is not collapsed (cellView form)")
    # Parity (PAR3).
    build = _load_tool("build_notebook")
    template = _load_tool("notebook_template").TEMPLATE
    rendered = build.to_bytes(build.render(ROOT, template, build.recorded_revision(path)))
    _check(path.read_bytes().replace(b"\r\n", b"\n") == rendered, "notebook: stale; run python tools/build_notebook.py (PAR3)")
    registry = _read(ROOT / "tutorials" / "README.md")
    _check(NOTEBOOK_NAME in registry and f"`{EXPECTED_PROFILE}`" in registry and f"`{EXPECTED_MODE}`" in registry, "tutorials/README.md must list the notebook with its profile and mode")
    _check(not list((ROOT / "tutorials").glob("*.ipynb")) or [p.name for p in (ROOT / "tutorials").glob("*.ipynb")] == [NOTEBOOK_NAME], "tutorials/: exactly one notebook expected")


def validate_sample_data() -> None:
    sums = _read(ROOT / "examples" / "sample-data" / "SHA256SUMS").split()
    digest, name = sums[0], sums[1]
    actual = hashlib.sha256((ROOT / "examples" / "sample-data" / name).read_bytes()).hexdigest()
    _check(actual == digest, f"examples/sample-data/{name}: digest {actual} != SHA256SUMS {digest}")
    template = _load_tool("notebook_template")
    _check(digest == template.SAMPLE_SHA256, "notebook template SAMPLE_SHA256 != examples/sample-data/SHA256SUMS")
    _check((ROOT / "examples" / "sample-data" / name).stat().st_size == template.SAMPLE_BYTES, "notebook template SAMPLE_BYTES != committed sample size")


def validate_no_assistant_identifiers() -> None:
    pattern = re.compile(r"claude[-_ ]?(opus|sonnet|haiku|fable|\d)", re.I)
    for path in ROOT.rglob("*"):
        if path.is_file() and ".git" not in path.parts and path.suffix in {".md", ".py", ".toml", ".yml", ".yaml", ".json", ".txt", ".ipynb", ".in"}:
            _check(not pattern.search(_read(path)), f"{path.relative_to(ROOT)}: assistant model identifier present")


def validate_all() -> list[str]:
    FAILURES.clear()
    validate_model_card()
    validate_identity_consistency()
    validate_release_status()
    validate_pins_and_lock()
    validate_notebook()
    validate_sample_data()
    validate_no_assistant_identifiers()
    return list(FAILURES)


def main() -> int:
    failures = validate_all()
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        print(f"{len(failures)} failure(s)", file=sys.stderr)
        return 1
    print("OK: release assets pass static validation (NOT clean-runtime execution evidence; see docs/release-verification.md)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
