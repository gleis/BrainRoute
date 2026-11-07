"""Test package bootstrap.

Runs before any test module imports ModelRoute, so state paths are redirected
to a temporary directory. Without this, the suite writes telemetry into the
developer's real ``data/modelroute.sqlite`` and test runs contaminate the
reliability signal that routing decisions depend on.
"""

import atexit
import os
import shutil
import tempfile


_TEST_DATA_DIR = tempfile.mkdtemp(prefix="modelroute-tests-")
os.environ["MODELROUTE_DATA_DIR"] = _TEST_DATA_DIR

atexit.register(shutil.rmtree, _TEST_DATA_DIR, True)
