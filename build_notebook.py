from __future__ import annotations

import contextlib
import json
import traceback
from io import StringIO
from pathlib import Path


def execute_notebook(path: Path) -> None:
    nb = json.loads(path.read_text())
    env: dict[str, object] = {}
    execution_count = 1

    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue

        code = "".join(cell.get("source", []))
        cell["outputs"] = []
        cell["execution_count"] = execution_count
        execution_count += 1

        if not code.strip():
            continue

        stdout = StringIO()
        try:
            with contextlib.redirect_stdout(stdout):
                exec(compile(code, str(path), "exec"), env)
        except Exception as exc:  # noqa: BLE001
            text = stdout.getvalue()
            if text:
                cell["outputs"].append(
                    {
                        "output_type": "stream",
                        "name": "stdout",
                        "text": text,
                    }
                )
            cell["outputs"].append(
                {
                    "output_type": "error",
                    "ename": exc.__class__.__name__,
                    "evalue": str(exc),
                    "traceback": traceback.format_exception(exc.__class__, exc, exc.__traceback__),
                }
            )
        else:
            text = stdout.getvalue()
            if text:
                cell["outputs"].append(
                    {
                        "output_type": "stream",
                        "name": "stdout",
                        "text": text,
                    }
                )

    path.write_text(json.dumps(nb, indent=2, ensure_ascii=False))


def main() -> None:
    notebook_paths = [
        Path("prueba1/boletin1_python.ipynb"),
        Path("trabajo.ipynb"),
    ]
    for notebook_path in notebook_paths:
        if notebook_path.exists():
            execute_notebook(notebook_path)
            print(f"Notebook ejecutado y actualizado: {notebook_path}")


if __name__ == "__main__":
    main()
