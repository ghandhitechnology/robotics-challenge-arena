"""Package the scenes, sources, builders and evidence with portable paths."""

import hashlib
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
archive = OUT / "robotics_challenge_arena.zip"

deliverables = sorted(p for p in OUT.rglob("*") if p.is_file()
                      and p.suffix != ".zip" and p.name != "SHA256SUMS"
                      and not p.name.endswith(".blend1"))
checksums = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(OUT).as_posix()}"
             for p in deliverables]
(OUT / "SHA256SUMS").write_text("\n".join(checksums) + "\n")

files = [ROOT / name for name in ("README.md", "arena_spec.json", "requirements-qa.txt")]
for directory in ("scripts", "reference", "docs", "output"):
    files.extend(p for p in (ROOT / directory).rglob("*") if p.is_file()
                 and "__pycache__" not in p.parts and p.suffix not in {".zip", ".pyc"}
                 and not p.name.endswith(".blend1"))
with ZipFile(archive, "w", compression=ZIP_DEFLATED, compresslevel=6) as bundle:
    for path in sorted(files):
        bundle.write(path, "robotics_challenge_arena/" + path.relative_to(ROOT).as_posix())
with ZipFile(archive) as bundle:
    corrupt = bundle.testzip()
    if corrupt:
        raise RuntimeError(f"Archive corruption: {corrupt}")
print(f"Packaged {len(files)} files: {archive} ({archive.stat().st_size / 1024**2:.2f} MiB)")
