#!/usr/bin/env python3
"""Thin executable wrapper for ``python -m nflvalue.fantasy.cli``."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nflvalue.fantasy.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
