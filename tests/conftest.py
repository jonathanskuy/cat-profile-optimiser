"""Pytest configuration (puts src/ on the path so tests can
`import preprocess`, `import model`, etc., matching how the app and
notebooks import these modules)."""


import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))