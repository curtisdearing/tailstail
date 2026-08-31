#!/usr/bin/env python3
"""CLI for the private raw-state boundary.

    python scripts/private_state.py restore --store .tailstail-state
    python scripts/private_state.py save    --store .tailstail-state

The copy itself, and every refusal in it, lives in
`nflvalue.fantasy.private_state`; this is only the entry point the workflow
calls. See that module for what may cross and what is refused outright.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nflvalue.fantasy.private_state import main

if __name__ == "__main__":
    raise SystemExit(main())
