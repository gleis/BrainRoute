"""Filesystem locations.

State lives next to the code by default so a fresh clone works with no setup.
``MODELROUTE_DATA_DIR`` moves it elsewhere - which is what you want when the
package is installed system-wide, when the checkout is read-only, or when state
belongs on a different volume from the code.
"""

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHECKOUT_CONFIG = ROOT / "config"
DEFAULT_CONFIG_DIR = CHECKOUT_CONFIG if CHECKOUT_CONFIG.is_dir() else Path(__file__).resolve().parent / "defaults"
CONFIG_DIR = Path(os.environ.get("MODELROUTE_CONFIG_DIR") or DEFAULT_CONFIG_DIR)
DEFAULT_DATA_DIR = ROOT / "data" if CHECKOUT_CONFIG.is_dir() else Path.home() / ".local" / "share" / "modelroute"
DATA_DIR = Path(os.environ.get("MODELROUTE_DATA_DIR") or DEFAULT_DATA_DIR)
