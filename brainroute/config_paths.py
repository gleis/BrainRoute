"""Filesystem locations.

State lives next to the code by default so a fresh clone works with no setup.
``BRAINROUTE_DATA_DIR`` moves it elsewhere - which is what you want when the
package is installed system-wide, when the checkout is read-only, or when state
belongs on a different volume from the code.
"""

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = Path(os.environ.get("BRAINROUTE_CONFIG_DIR") or ROOT / "config")
DATA_DIR = Path(os.environ.get("BRAINROUTE_DATA_DIR") or ROOT / "data")
