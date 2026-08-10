"""Stage the latest verified package and run package-relative Comment 4--7 corrections."""
from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

from comment4_7_corrections_core import run_staged


def main():
    cwd = Path.cwd()
    candidates = sorted(p for p in cwd.glob("reviewer_comments4_7_results_*") if p.is_dir())
    if not candidates:
        raise FileNotFoundError("Latest Comments 4--7 output package is unavailable")
    source = candidates[-1]
    verified = cwd / "final_submission_results_k12_continuation_20260807_070705"
    proof = Path("/home/mskr/Downloads/ERX-121075_Proof_hi.pdf")
    if not verified.exists():
        raise FileNotFoundError(f"Verified Comments 1--3 package unavailable: {verified}")
    if not proof.exists():
        raise FileNotFoundError(f"Manuscript proof unavailable: {proof}")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    root = cwd / f"reviewer_comments4_7_correction_{stamp}"
    if root.exists():
        raise FileExistsError(root)
    root.mkdir()
    shutil.copytree(source, root / "preserved_previous_comments4_7")
    shutil.copytree(verified, root / "preserved_verified_comment1_3")
    (root / "preserved_sources").mkdir()
    shutil.copy2(proof, root / "preserved_sources" / proof.name)
    (root / "scripts").mkdir()
    shutil.copy2(Path(__file__), root / "scripts" / Path(__file__).name)
    shutil.copy2(Path(__file__).with_name("comment4_7_corrections_core.py"), root / "scripts" / "comment4_7_corrections_core.py")
    prior_corrections = sorted(p for p in cwd.glob("reviewer_comments4_7_correction_*") if p.is_dir() and (p / "comment5/comment5_corrected_outer_metrics.csv").exists())
    if prior_corrections:
        prior = prior_corrections[-1]
        shutil.copytree(prior / "comment4", root / "preserved_prior_correction/comment4")
        shutil.copytree(prior / "comment5", root / "preserved_prior_correction/comment5")
        (root / "preserved_prior_correction").joinpath("source.txt").write_text(str(prior), encoding="utf-8")
    validation, runtime = run_staged(root)
    import zipfile
    archive = cwd / f"al_alloy_reviewer_comments4_7_correction_{time.strftime('%Y%m%d_%H%M%S')}.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(root.rglob("*")):
            if p.is_file():
                zf.write(p, str(root.name / p.relative_to(root)))
    print(f"FINAL_RESULTS={root}")
    print(f"FINAL_ARCHIVE={archive}")
    print(f"RUNTIME_SECONDS={runtime:.2f}")
    print(validation.to_string(index=False))
    if not validation.Status.eq("PASS").all():
        raise RuntimeError("Correction validation failed")


if __name__ == "__main__":
    main()
