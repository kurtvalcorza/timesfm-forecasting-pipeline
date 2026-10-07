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
  and the rule), baselines and metrics, the forecast frame contract with a stand-in model (nine quantile
  slots, median at slot 4, no mean column), the result bundle and reload parity, the manifest/staging/
  verification code (sizes, the `config.json` digest, the pinned 3.0 architecture, the licence record) with
  an injected downloader and a sparse placeholder weight file; notebook-contract tests (generator parity, no `pip install` into the kernel, no
  restart guard, the isolated runtime's `MPLBACKEND=Agg` and dropped `PYTHONPATH`/`PYTHONHOME`/
  `PYTHONSTARTUP`, guided markers, no placeholders, no quality assert, form fields, Section 1 executed twice
  against a stand-in environment and shown to reuse it and keep its worker); a BYOD matrix that executes the
  notebook's own Section 4 and 5 cell source over good and incompatible files; and a sequential execution of
  every notebook code cell against stub `torch`/`timesfm3` modules that produces the full result bundle;
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
| Stand-in execution (`tests/test_notebook_standin_exec.py`) | CI runner, stub `torch`/`timesfm3`, sparse placeholder weights | Proves the cells, contract and exports; **not** pretrained inference |

## Supported release verification procedure

Before changing the status from `Candidate` to `Release-grade`:

1. resolve the exact commit under review and confirm static CI is green;
2. open that exact notebook revision in a fresh runtime (Colab or Kaggle) with no repository checkout and a
   clean model cache; the run downloads the TimesFM 3.0 weights under their non-commercial, non-production
   licence, which a verification run (testing and evaluation) is within;
3. choose **Run all** with every form field at its default (`USE_BYOD = False`, `BYOD_PATH = ''`,
   `HORIZON = 24`, `CONTEXT_LENGTH = 312`, `SEASON_LENGTH = 24`, `NAN_POLICY = 'refuse'`,
   `GAP_POLICY = 'refuse'`, `ACTIVITY_CONTEXT_LENGTH = 48`) and **in one pass** — a run that needs a manual
   restart is recorded as a failure;
4. verify Section 1 reports the isolated Python `3.12.12`, `locked_packages: 52`, and that the runtime record
   shows `NOTEBOOK_SOURCE.repository_revision` equal to `metadata.dimer.generated_from.revision`;
5. verify Section 3 prints the staged files with `verified_files: 2` and the pinned revision, and that
   Section 12 reports `revision` and `resolved_revision` both equal to `43046b85ec22…`,
   `weights_sha256` equal to `a7592b0a…` with `weights_sha256_verified_against_manifest: True`, and
   `license: timesfm-non-commercial-license-v1.0`; a different commit or digest is a failed run — record it
   and stop;
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
| 2026-10-06 | `c7bf4a6` / `45b62c0c` | Google Colab, CUDA (`cuda:0`), isolated env Python 3.12.12 built in 64 s (kernel 3.13.15) | maintainer Run all on a fresh runtime | **FAILED** at code cell 15 (Section 7, first forecast): `TypeError: TimesFM3Forecaster.predict_batch() got an unexpected keyword argument 'per_core_batch_size'`. Cells 1–14 completed: isolated env, snapshot staged (2 files, 1,322,900,097 bytes) and size-verified, model loaded on CUDA, sample fetched and digest-verified (SHA-256 `74163ee6…`), validation and split passed | Fixed in the next commit (the batch size moves to the constructor); evidence copy `docs/execution-evidence/2026-10-06/timesfm_forecasting_colab_c7bf4a6_run1_failed.ipynb`. Not promotion evidence |
| 2026-10-06 | `86f2213` / `c6a9263a` | Google Colab, CUDA (`cuda:0`), torch 2.14.0+cu130, timesfm 3.0.2, isolated env Python 3.12.12 built in 73 s (kernel 3.13.15) | maintainer Run all on a fresh runtime, defaults (sample path; BYOD off) | **PASSED (default path)**: 19/19 code cells, execution counts 1–19, no errors. Snapshot staged (2 files, 1,322,900,097 bytes), loaded on CUDA; Hub-resolved commit `43046b85ec22d584a13f8098c2ed39c889e129c2`; `model.safetensors` SHA-256 `a7592b0a8432baee54483254e5647856911ce69e09d09a9bb65904b2d98f17da` (observed; not yet pinned). Held-out 24 h × 3 series: TimesFM MASE 0.6943 / sMAPE 1.698 / MAE 0.4636 vs seasonal-naive 0.7417 / 1.837 / 0.4958, naive 3.2201, SES 3.2398; q0.1–q0.9 coverage 0.764 (nominal 0.8); quantile loss q0.1 0.1247, q0.5 0.2318, q0.9 0.1166. Context activity: 312 → 48 points raised MASE to 0.756 and band width 1.53 → 2.18. Bundle written (7 files); reload parity ok (72 rows, tol 1e-8) | Evidence copy `docs/execution-evidence/2026-10-06/timesfm_forecasting_colab_86f2213_run2_passed.ipynb`. Not assessed in this run: BYOD (REL12), an export-cell re-run, a repeated Run all in a warm runtime. Colab inserted `# @title` lines only (non-substantive). One seeded holdout; not a benchmark |

## Pin from the 2026-10-06 run

`MODEL_REVISION` (`src/timesfm_forecasting/model.py`) and the manifest's `revision` are pinned to
`43046b85ec22d584a13f8098c2ed39c889e129c2`, the commit the Hub served for ref `main` on the `86f2213` run above
(`resolved_revision` in its last cell), and the manifest's `model.safetensors` SHA-256 is pinned to
`a7592b0a8432baee54483254e5647856911ce69e09d09a9bb65904b2d98f17da`, the digest that run computed in-run over the
1,322,898,824-byte staged file (`weights_sha256`). Both values were read from
`docs/execution-evidence/2026-10-06/timesfm_forecasting_colab_86f2213_run2_passed.ipynb`. At `86f2213` the
module still named the ref `main` and the manifest held a pending sentinel, which is why that run reported
`weights_sha256_verified_against_manifest: False`; **the run therefore predates the pin**, and `verify_snapshot`
is now a hard size-and-digest check. The next hosted run on the pinned head is the one that re-verifies the
pinned digest and commit (step 5 above); until it is recorded, the pin rests on one observation.

Saved outputs were inspected; execution was not independently repeated. Source identity: the first run's cells equal the `c7bf4a6` notebook exactly; the second run's cells equal the `86f2213` notebook apart from `# @title` lines Colab inserted into collapsed cells (no `# @param` changes).
