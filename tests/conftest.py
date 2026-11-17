"""Make the variant modules importable without installing them as packages.

The shared core lives in ``src/stemenvelope`` (installed by ``pip install -e .``);
the three variants are scripts, not packages, so tests add them to ``sys.path``.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

for relative in ("src", "variants/s3/python", "variants/khadas", "variants/host"):
    candidate = str(ROOT / relative)
    if candidate not in sys.path:
        sys.path.insert(0, candidate)
