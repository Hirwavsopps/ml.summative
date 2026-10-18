"""
Assemble the graded notebook from ``notebook/cells/*`` and execute it top to bottom.

Why a custom runner instead of ``jupyter nbconvert --execute``?  The sandbox this
project is built in denies the local sockets a Jupyter kernel needs.  Executing the
cells in a single shared namespace produces exactly the same semantics for a linear
notebook -- one process, one namespace, cells run in file order -- and lets us capture
stdout and every open matplotlib figure as genuine notebook ``outputs``.

Usage
-----
    python src/build_notebook.py
"""
from __future__ import annotations

import base64
import io
import json
import os
import sys
import time
import traceback
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CELL_DIR = PROJECT_ROOT / "notebook" / "cells"
OUT_PATH = PROJECT_ROOT / "notebook" / "diabetes_risk_screening.ipynb"
LOG_PATH = PROJECT_ROOT / "results" / "notebook_execution.log"

# Debug aid: NB_SMOKE_CELLS=8 stops after eight cells and writes a throwaway notebook,
# so a syntax error in an early cell does not cost a full 20-minute run.
SMOKE_CELLS = int(os.environ.get("NB_SMOKE_CELLS", "0"))
if SMOKE_CELLS:
    OUT_PATH = PROJECT_ROOT / "notebook" / "_smoke.ipynb"

FIGURE_DPI = 100  # inline image resolution; the on-disk copies keep the higher dpi


def load_cells() -> list[tuple[str, str]]:
    """Return [(cell_type, source), ...] sorted by the numeric filename prefix."""
    cells = []
    for path in sorted(CELL_DIR.iterdir()):
        if path.suffix not in (".md", ".py"):
            continue
        kind = "markdown" if path.suffix == ".md" else "code"
        cells.append((kind, path.read_text(encoding="utf-8").rstrip() + "\n"))
    if not cells:
        raise SystemExit(f"No cell files found in {CELL_DIR}")
    return cells


def as_lines(text: str) -> list[str]:
    """nbformat stores multi-line sources as a list of lines, each keeping its newline."""
    return text.splitlines(keepends=True)


def stream_output(text: str) -> dict:
    return {"output_type": "stream", "name": "stdout", "text": as_lines(text)}


def figure_output(fig) -> dict:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=FIGURE_DPI, bbox_inches="tight")
    buf.seek(0)
    return {
        "output_type": "display_data",
        "data": {"image/png": base64.b64encode(buf.read()).decode("ascii"),
                 "text/plain": ["<Figure>"]},
        "metadata": {},
    }


def main() -> int:
    cell_specs = load_cells()
    if SMOKE_CELLS:
        cell_specs = cell_specs[:SMOKE_CELLS]
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    namespace: dict = {"__name__": "__main__", "__builtins__": __builtins__}
    nb_cells: list[dict] = []
    exec_count = 0
    log = open(LOG_PATH, "w", encoding="utf-8")
    started = time.perf_counter()
    failures = 0

    os.chdir(PROJECT_ROOT)

    def emit(message: str) -> None:
        elapsed = time.perf_counter() - started
        line = f"[{elapsed:7.1f}s] {message}"
        print(line, file=sys.__stderr__, flush=True)
        log.write(line + "\n")
        log.flush()

    for index, (kind, source) in enumerate(cell_specs, start=1):
        label = f"cell {index:02d}/{len(cell_specs)} [{kind}] {source.strip().splitlines()[0][:70]}"

        if kind == "markdown":
            nb_cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": as_lines(source),
            })
            emit(f"{label} -- markdown")
            continue

        exec_count += 1
        emit(f"{label} -- running")
        outputs: list[dict] = []
        buffer = io.StringIO()
        status = "ok"
        try:
            with redirect_stdout(buffer):
                exec(compile(source, f"<cell {index}>", "exec"), namespace)
        except Exception:
            status = "error"
            failures += 1
            traceback.print_exc(file=sys.__stderr__)
            log.write(traceback.format_exc() + "\n")
        finally:
            captured = buffer.getvalue()

        if captured:
            outputs.append(stream_output(captured))
            log.write(captured)

        for fignum in plt.get_fignums():
            outputs.append(figure_output(plt.figure(fignum)))
        plt.close("all")

        if status == "error":
            last_line = traceback.format_exc().strip().splitlines()[-1]
            outputs.append({
                "output_type": "error",
                "ename": "ExecutionError",
                "evalue": last_line,
                "traceback": traceback.format_exc().splitlines(),
            })
            emit(f"{label} -- FAILED: {last_line}")

        nb_cells.append({
            "cell_type": "code",
            "execution_count": exec_count,
            "metadata": {},
            "outputs": outputs,
            "source": as_lines(source),
        })

    notebook = {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
            "title": "Non-invasive diabetes risk screening: classical ML vs deep learning",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    OUT_PATH.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
    elapsed = time.perf_counter() - started
    emit(f"wrote {OUT_PATH.name} ({OUT_PATH.stat().st_size/1e6:.2f} MB) in {elapsed/60:.1f} min; "
         f"{failures} failing cell(s)")
    log.close()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
