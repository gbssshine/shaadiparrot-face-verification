"""The Cloud Run image must hold every module the code imports and every data file it opens (plan §1 row 19).

A root module that is missing from the image takes every route down on import (porutham.py was the reminder),
so the COPY line is a glob over the root modules and data files, and this test keeps it honest.
"""
import ast
import fnmatch
import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = (ROOT / "Dockerfile").read_text(encoding="utf-8")
LINES = [l.strip() for l in DOCKERFILE.splitlines() if l.strip() and not l.strip().startswith("#")]
ROOT_MODULES = sorted(p for p in ROOT.glob("*.py"))
DATA_EXT = (".json", ".sqlite3", ".sqlite", ".db", ".csv", ".txt", ".yaml", ".yml", ".ttf", ".otf", ".png", ".jpg",
            ".jpeg", ".webp", ".svg", ".pdf")
IMPORT_RE = re.compile(r"^(?:import\s+([\w.]+(?:\s+as\s+\w+)?(?:\s*,\s*[\w.]+(?:\s+as\s+\w+)?)*)|from\s+([\w.]+)\s+import\b)",
                       re.M)


def copy_lines():
    """(sources, destination) of every COPY that takes files from the build context."""
    out = []
    for line in LINES:
        parts = line.split()
        if parts[0].upper() != "COPY" or any(p.startswith("--from") for p in parts[1:]):
            continue
        args = [p for p in parts[1:] if not p.startswith("--")]
        out.append((args[:-1], args[-1]))
    return out


def covered(path: str) -> bool:
    """Is a repo-relative path taken into the image by a COPY source? Docker globs never cross a '/', and a
    directory source ('assets/') takes everything under it."""
    segs = path.split("/")
    for sources, _ in copy_lines():
        for src in sources:
            if src.endswith("/"):
                if path.startswith(src):
                    return True
                continue
            pat = src.split("/")
            if len(pat) == len(segs) and all(fnmatch.fnmatchcase(s, p) for s, p in zip(segs, pat)):
                return True
    return False


def imported_names(source: str):
    names = set()
    for m in IMPORT_RE.finditer(source):
        if m.group(1):
            for part in m.group(1).split(","):
                names.add(part.strip().split()[0].split(".")[0])
        else:
            names.add(m.group(2).split(".")[0])
    return names


def data_files(source: str):
    """String literals that name a data file next to the code: 'places.sqlite3', 'fates_tests_catalog.json',
    'assets/biodata/...'. Prose, URLs and Firestore paths are skipped."""
    found = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            v = node.value.strip()
            if not v or any(ch.isspace() for ch in v) or "://" in v or "{" in v or v.startswith(("/", ".")):
                continue
            if v.startswith("assets/") or (v.lower().endswith(DATA_EXT) and re.fullmatch(r"[\w./-]+", v)):
                found.add(v)
    return found


def test_the_copy_lines_are_present():
    assert "COPY *.py *.json places.sqlite3 ./" in LINES
    assert "COPY assets/ ./assets/" in LINES
    for kept in ("FROM python:3.11-slim", "WORKDIR /app", "COPY requirements.txt .",
                 "RUN pip install --no-cache-dir -r requirements.txt",
                 "COPY tools/fetch_models.py tools/fetch_models.py", "RUN python tools/fetch_models.py /app/models",
                 'CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]'):
        assert kept in LINES, kept


def test_every_copy_source_exists():
    for sources, _ in copy_lines():
        for src in sources:
            if any(ch in src for ch in "*?["):
                assert list(ROOT.glob(src)), src            # a glob that matches nothing fails the build
            else:
                assert (ROOT / src).exists(), src
    assert (ROOT / "assets" / ".gitkeep").exists()


def test_every_import_resolves_and_is_in_the_image():
    assert ROOT_MODULES
    for path in ROOT_MODULES:
        for name in sorted(imported_names(path.read_text(encoding="utf-8"))):
            if (ROOT / f"{name}.py").exists():
                assert covered(f"{name}.py"), f"{path.name} imports {name}, which the Dockerfile does not copy"
            elif (ROOT / name / "__init__.py").exists():
                assert covered(f"{name}/__init__.py"), f"{path.name} imports the package {name}"
            else:
                spec = importlib.util.find_spec(name)
                assert spec is not None, f"{path.name} imports {name}, which is neither a root module nor installed"
    for path in ROOT_MODULES:
        assert covered(path.name), path.name


def test_main_and_astro_api_modules_are_copied():
    """The features-plan check: every root module main.py and astro_api.py import is in the image."""
    for entry in ("main.py", "astro_api.py"):
        names = imported_names((ROOT / entry).read_text(encoding="utf-8"))
        local = {n for n in names if (ROOT / f"{n}.py").exists()}
        assert local, entry
        assert all(covered(f"{n}.py") for n in local), (entry, local)
    assert "porutham" in imported_names((ROOT / "astro_api.py").read_text(encoding="utf-8"))


def test_every_data_file_the_code_opens_is_copied():
    found = {}
    for path in ROOT_MODULES:
        for f in data_files(path.read_text(encoding="utf-8")):
            found.setdefault(f, path.name)
    assert "places.sqlite3" in found and "fates_tests_catalog.json" in found
    for f, where in sorted(found.items()):
        assert covered(f), f"{where} opens {f}, which the Dockerfile does not copy"
    # The face models are not in the repo: the image downloads and checks them at build time.
    assert "RUN python tools/fetch_models.py /app/models" in LINES


def test_the_checks_catch_a_missing_file():
    assert covered("porutham.py") and covered("caste_list.json") and covered("assets/biodata/logo.png")
    assert not covered("tools/export_guna_table.py") and not covered("tests/test_astro.py")
    assert not covered("models/face.onnx") and not covered("places.db")
    assert imported_names("import a, b.c as d\nfrom e.f import g\n    import lazy\nfrom __future__ import x\n") == \
        {"a", "b", "e", "__future__"}
    assert data_files('P = "places.sqlite3"\nQ = "assets/biodata/x.png"\nR = "see docs/x.json for more"\n'
                      'U = "https://x.org/a.json"\n') == {"places.sqlite3", "assets/biodata/x.png"}
