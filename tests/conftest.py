import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REFACTOR_SRC = REPO_ROOT / "refactor" / "src"
if str(REFACTOR_SRC) not in sys.path:
    sys.path.insert(0, str(REFACTOR_SRC))
