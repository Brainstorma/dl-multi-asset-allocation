"""Static guardrails on the notebooks themselves.

The repo's protocol says every parameter lives in config/config.yaml; these
checks keep that true without needing to execute the (network-bound)
notebooks in CI.
"""

import json
from pathlib import Path

import nbformat
import pytest

NOTEBOOKS = sorted((Path(__file__).resolve().parents[1] / "notebooks").glob("*.ipynb"))


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_is_valid_ipynb(path):
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_no_hardcoded_relative_config_paths(path):
    nb = nbformat.read(path, as_version=4)
    code = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    assert '"../config/' not in code and "'../config/" not in code, (
        "use src.config.load_config() instead of relative paths"
    )
