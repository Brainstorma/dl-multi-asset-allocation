import sys
from pathlib import Path

# Make `src` importable when pytest is run from the repository root without
# an installed/editable package (belt and braces alongside pytest.ini).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
