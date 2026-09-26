"""Shared library for the MScFE 690 capstone pipeline.

Keeping the small amount of reusable logic (config loading, cleaning policy)
in a package instead of copy-pasted notebook cells means notebooks 01-06 and
the test suite exercise exactly the same code paths.
"""

from src.config import load_config, project_root

__all__ = ["load_config", "project_root"]
