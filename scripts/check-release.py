"""Validate freshly compiled PDFs and prepare a traceable release bundle."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil

from pypdf import PdfReader


BLOCKING_LOG = re.compile(
    r"(?:LaTeX|Package [^\n]+) Warning: [^\n]*undefined"
    r"|There were undefined (?:references|citations)"
    r"|Rerun to get cross-references right"
    r"|Missing character:"
    r"|^! ",
    re.MULTILINE,
)
DOCUMENTS = (
    ("main", "model-study.pdf"),
    ("quick-guide", "model-study-quick.pdf"),
)


def validate_document(build_dir, stem):
    pdf = build_dir / f"{stem}.pdf"
    log = build_dir / f"{stem}.log"
    if not pdf.is_file() or not log.is_file():
        raise ValueError(f"Missing freshly built PDF or log: {stem}")
    if pdf.read_bytes()[:5] != b"%PDF-":
        raise ValueError(f"Invalid PDF header: {pdf}")
    log_text = log.read_text(encoding="utf-8", errors="replace")
    if "Output written on" not in log_text:
        raise ValueError(f"No successful TeX output recorded: {log}")
    match = BLOCKING_LOG.search(log_text)
    if match:
        line = log_text.count("\n", 0, match.start()) + 1
        raise ValueError(f"Blocking TeX diagnostic at {log}:{line}: {match[0]}")
    reader = PdfReader(pdf, strict=True)
    if reader.is_encrypted or len(reader.pages) == 0:
        raise ValueError(f"Encrypted or empty PDF: {pdf}")
    for page in reader.pages:
        content = page.get_contents()
        if content is not None:
            content.get_data()
    return pdf, len(reader.pages)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=Path("build/release"))
    parser.add_argument("--output-dir", type=Path, default=Path("dist/release"))
    args = parser.parse_args()
    # Validate every input before copying any release asset.
    checked = [(name, *validate_document(args.build_dir, stem)) for stem, name in DOCUMENTS]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "commit": os.environ.get("GITHUB_SHA", "local"),
        "run_id": os.environ.get("GITHUB_RUN_ID", "local"),
        "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT", "local"),
        "files": [],
    }
    checksums = []
    for name, source, pages in checked:
        target = args.output_dir / name
        shutil.copyfile(source, target)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        checksums.append(f"{digest}  {name}\n")
        manifest["files"].append({"name": name, "pages": pages, "sha256": digest, "bytes": target.stat().st_size})
        print(f"PASS: {name}, {pages} pages, SHA256 {digest}")
    (args.output_dir / "SHA256SUMS").write_text("".join(checksums), encoding="utf-8")
    (args.output_dir / "release-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
