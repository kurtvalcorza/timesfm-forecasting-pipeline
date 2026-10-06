"""Execute pure-Python notebook code cells sequentially in one shared namespace."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _source(cell: dict) -> str:
    source = cell.get("source", "")
    if isinstance(source, list):
        return "".join(source)
    if isinstance(source, str):
        return source
    raise TypeError(f"unsupported notebook cell source type: {type(source).__name__}")


def execute_notebook(path: Path, *, workdir: Path | None = None) -> None:
    resolved = path.resolve()
    notebook = json.loads(resolved.read_text(encoding="utf-8"))
    namespace = {
        "__name__": "__main__",
        "__file__": str(resolved),
        "NOTEBOOK_CI": True,
    }
    old_cwd = Path.cwd()
    try:
        start = workdir.resolve() if workdir is not None else resolved.parents[1]
        start.mkdir(parents=True, exist_ok=True)
        os.chdir(start)
        for index, cell in enumerate(notebook.get("cells", []), start=1):
            if cell.get("cell_type") != "code":
                continue
            source = _source(cell)
            if not source.strip():
                continue
            # Ordinary Python only. Shell escapes and notebook magics naturally
            # fail compilation instead of being skipped or matched inside strings.
            code = compile(source, f"{resolved}:cell-{index}", "exec")
            exec(code, namespace, namespace)
    finally:
        os.chdir(old_cwd)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebooks", nargs="+", type=Path)
    parser.add_argument(
        "--workdir",
        type=Path,
        default=None,
        help=(
            "Starting working directory. Use a clean directory to exercise "
            "clone/install bootstrap."
        ),
    )
    args = parser.parse_args()
    for notebook in args.notebooks:
        print(f"[notebook-exec] {notebook}")
        execute_notebook(notebook, workdir=args.workdir)


if __name__ == "__main__":
    main()
