# Release verification

`tutorials/timesfm_forecasting_colab.ipynb` (`TASK-INFERENCE` / `GUIDED`) is a **release candidate** until
the exact notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, JSON
validation, code-cell compilation, the stand-in execution and `tools/validate_release_assets.py` are
necessary checks but are **not** runtime evidence under DIMER Notebook Specification 2.2 (REL8). This file
is the durable release-gate record.

## Automatic coverage (static and stand-in, every pull request)

CI (`.github/workflows/ci.yml`, job `lint-and-unit`) installs the locked environment **without** the model
stack (`uv sync --locked --extra dev`: numpy, pandas, pytest, ruff, nbformat, matplotlib) and runs:

- `ruff check src tests tools`;
- `pytest -m "not integration"`: unit tests for configuration, data validation (every rule, naming the file
  and the rule), baselines and metrics, the forecast frame contract with a stand-in model, the result bundle
  and reload parity, the manifest/staging/verification code with an injected downloader and a sparse
  placeholder weight file; notebook-contract tests (generator parity, no `pip install` into the kernel, no
  restart guard, the isolated runtime's `MPLBACKEND=Agg` and dropped `PYTHONPATH`/`PYTHONHOME`/
  `PYTHONSTARTUP`, guided markers, no placeholders, no quality assert, form fields, Section 1 executed twice
  against a stand-in environment and shown to reuse it and keep its worker); a BYOD matrix that executes the
  notebook's own Section 4 and 5 cell source over good and incompatible files; and a sequential execution of
  every notebook code cell against stub `torch`/`timesfm` modules that produces the full result bundle;
- `tools/validate_release_assets.py`: notebook structure, metadata declarations, carried modules, pins, lock
  digest, forbidden patterns, BYOD gate defaults, guided markers, model-card front matter and section order,
  identity consistency across documents, status tokens, sample digest, no invented revision;
- `tools/build_notebook.py --check` (PAR3);
- a dry-run `uv pip install --require-hashes --only-binary :all:` of `tutorials/requirements-colab.lock.txt`.

The `release-evidence-boundary` job prints that none of this is clean-runtime evidence. The `integration`
job (manual `workflow_dispatch` only) installs the `model` extra, runs the real-weight test and executes the
notebook cells with `DIMER_NOTEBOOK_CI_PREINSTALLED=1`; it is a pre-flight on the locked stack, not a
fresh-boundary run of the inline pins.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CPU or GPU runtime | The runtime the notebook is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle kernel | Kaggle CPU/GPU kernel, Linux x86_64 | Clean-room executor of the same class; the notebook is standalone, no checkout is needed |
| Repository CI integration job (`tools/run_notebook.py`) | GitHub-hosted Ubuntu runner, locked `uv` environment with the `model` extra, `DIMER_NOTEBOOK_CI_PREINSTALLED=1` | Pre-flight on the locked stack; not promotion evidence |
| Stand-in execution (`tests/test_notebook_standin_exec.py`) | CI runner, stub `torch`/`timesfm`, sparse placeholder weights | Proves the cells, contract and exports; **not** pretrained inference |

## Supported release verification procedure

Before changing the status from `Candidate` to `Release-grade`:

1. resolve the exact commit under review and confirm static CI is green;
2. open that exact notebook revision in a fresh runtime (Colab or Kaggle) with no repository checkout and a
   clean model cache;
3. choose **Run all** with every form field at its default (`USE_BYOD = False`, `BYOD_PATH = ''`,
   `HORIZON = 24`, `CONTEXT_LENGTH = 312`, `SEASON_LENGTH = 24`, `NAN_POLICY = 'refuse'`,
   `GAP_POLICY = 'refuse'`, `ACTIVITY_CONTEXT_LENGTH = 48`) and **in one pass** — a run that needs a manual
   restart is recorded as a failure;
4. verify Section 1 reports the isolated Python `3.12.12`, `locked_packages: 52`, and that the runtime record
   shows `NOTEBOOK_SOURCE.repository_revision` equal to `metadata.dimer.generated_from.revision`;
5. verify Section 3 prints the staged files, `digests_pending: ['model.safetensors']` (until pinned), the
   observed weight digest, and `resolved_revision` from `resolved-revision.json`; **copy both into the
   record below and pin them in `model.py` and the manifest in the follow-up commit**;
6. verify Sections 4–12 complete: the sample digest check, the input manifest and the named refusal probe,
   the leakage check, the baseline table, the forecast and plot, the evaluation report with its recorded
   verdict, the activity table, the future forecast marked `not-measurable`, and `reload_parity.ok: True`
   with the seven files listed;
7. run the BYOD branch once with a representative CSV (`USE_BYOD = True`, `BYOD_PATH` set) through export,
   and once with an incompatible CSV (for example a duplicated timestamp) and record the named refusal;
8. append the executed `.ipynb` under `docs/execution-evidence/<date>/` and fill the table below;
9. write the run as a verification record in `MODEL_CARD.md`.

## Recorded executions

| Date | Notebook revision (commit / blob) | Runtime | Procedure | Observed result | Caveats |
|---|---|---|---|---|---|

_No execution recorded yet. The table is filled only from hosted runs of the exact candidate revision._
