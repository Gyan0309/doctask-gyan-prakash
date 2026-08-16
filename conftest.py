"""Make the repository root importable for tests.

Packages live flat at the root (`domain/`, `services/`, `models/`, …) and are imported
absolutely. Something has to put the root on `sys.path` for that to work, and which
"something" depends on how pytest is invoked:

    python -m pytest    CWD is prepended by the -m machinery, so it works by accident
    pytest              CWD is NOT added, and every import fails at collection

Local runs used the first form and CI used the second, so 180 tests passed here and 11
modules failed to import there. The discrepancy is invisible until it isn't — the local
command that verifies your work is not the command that runs it.

A root conftest.py is the conventional fix: pytest imports it before collection, which
is early enough to fix the path for every test module.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent)

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
