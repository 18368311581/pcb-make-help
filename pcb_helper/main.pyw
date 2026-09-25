#! python3.12
"""Double-click entry point without a console window on Windows."""

import sys
from pathlib import Path

# Directly launching a file inside a package does not add its parent to sys.path.
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from pcb_helper.ui import main


if __name__ == '__main__':
    raise SystemExit(main())
