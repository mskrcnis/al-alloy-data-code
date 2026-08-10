"""Stage the latest corrected package and run Reviewer 2 Comment 10 only."""
from __future__ import annotations

import shutil
import time
from pathlib import Path

from comment10_uq_comparison_core import run_staged


def main():
    cwd = Path.cwd()
    candidates = sorted(p for p in cwd.glob("reviewer_comments4_7_correction_*") if p.is_dir())
    if not candidates:
        raise FileNotFoundError("Latest corrected manuscript-analysis package is unavailable")
    source = candidates[-1]
    root = cwd / f"reviewer2_comment10_uq_comparison_{time.strftime('%Y%m%d_%H%M%S')}"
    if root.exists():
        raise FileExistsError(root)
    root.mkdir()
    shutil.copytree(source, root / "preserved_latest_corrected_package")
    (root / "scripts").mkdir()
    shutil.copy2(Path(__file__), root / "scripts" / Path(__file__).name)
    shutil.copy2(Path(__file__).with_name("comment10_uq_comparison_core.py"), root / "scripts" / "comment10_uq_comparison_core.py")
    checks, runtime = run_staged(root)
    import zipfile
    archive = cwd / f"reviewer2_comment10_uq_comparison_{time.strftime('%Y%m%d_%H%M%S')}.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(root.rglob("*")):
            if p.is_file(): zf.write(p, str(Path(root.name) / p.relative_to(root)))
    print(f"FINAL_RESULTS={root}")
    print(f"FINAL_ARCHIVE={archive}")
    print(f"RUNTIME_SECONDS={runtime:.2f}")
    print(checks.to_string(index=False))
    if not checks.Pass.all():
        raise RuntimeError("Comment 10 validation failed")


if __name__ == "__main__":
    main()
