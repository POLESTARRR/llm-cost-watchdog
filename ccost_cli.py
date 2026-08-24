"""Console entry point for `ccost`.

A thin shim so `pip install -e .` yields a real `ccost` command rather than
`python scripts/ccost.py`. The implementation stays in scripts/ccost.py, which
remains runnable directly for anyone who has cloned the repo and does not want
to install anything.
"""

import pathlib
import sys

_SCRIPTS = pathlib.Path(__file__).resolve().parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from ccost import main  # noqa: E402,F401

__all__ = ["main"]

if __name__ == "__main__":
    sys.exit(main())
