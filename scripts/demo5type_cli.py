"""PyInstaller entry point for the Windows console executable."""

from __future__ import annotations

import sys
from pathlib import Path


# PyInstaller starts with this script's directory on sys.path. Add the checkout
# root so the sibling demo5type package is resolved both locally and in builds.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from demo5type.demo import main  # noqa: E402


if __name__ == "__main__":
    main()
