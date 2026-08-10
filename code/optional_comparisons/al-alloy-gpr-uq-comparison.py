"""Stage the verified Comment 10 package and run the GPR-only comparison."""
from __future__ import annotations

import csv
import shutil
import time
import zipfile
from pathlib import Path

from gpr_uq_comparison_core import run_staged


def latest_valid_comment10(cwd):
    candidates = sorted(path for path in cwd.glob("reviewer2_comment10_uq_comparison_*") if path.is_dir())
    for path in reversed(candidates):
        validation = path / "validation_checks.csv"
        if validation.exists():
            with validation.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            if rows and all(row.get("Pass") == "True" for row in rows):
                continue
        checks = path / "comment10_validation_checks.csv"
        if checks.exists():
            with checks.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            if rows and all(row.get("Pass") == "True" for row in rows):
                return path
    raise FileNotFoundError("No validated Comment 10 package is available")


def main():
    cwd = Path.cwd()
    source = latest_valid_comment10(cwd)
    root = cwd / f"reviewer2_comment10_gpr_uq_{time.strftime('%Y%m%d_%H%M%S')}"
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    shutil.copytree(source, root / "preserved_comment10_package")
    (root / "scripts").mkdir()
    shutil.copy2(Path(__file__), root / "scripts" / Path(__file__).name)
    shutil.copy2(Path(__file__).with_name("gpr_uq_comparison_core.py"), root / "scripts" / "gpr_uq_comparison_core.py")
    checks, runtime, context = run_staged(root)
    archive = cwd / f"reviewer2_comment10_gpr_uq_{time.strftime('%Y%m%d_%H%M%S')}.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                handle.write(path, str(Path(root.name) / path.relative_to(root)))
    print(f"FINAL_RESULTS={root}")
    print(f"FINAL_ARCHIVE={archive}")
    print(f"RUNTIME_SECONDS={runtime:.2f}")
    print(checks.to_string(index=False))
    if not checks.Pass.all():
        raise RuntimeError("GPR UQ validation failed")


if __name__ == "__main__":
    main()
