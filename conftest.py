'''
pytest bootstrap.

config.py reads its config path from sys.argv[1], which under pytest is
whichever path/flag pytest was invoked with — so importing anything that pulls
in `config` fails during collection ("IsADirectoryError: tests/", or a JSON
decode error on a test file). Normalising argv before collection makes the
suite runnable with the usual `pytest tests/` invocation.

Tests that need a different config should construct their own Config rather
than relying on this default.
'''
import os
import sys

# must happen before any test module imports `config`
sys.argv = [sys.argv[0], os.environ.get("A0_TEST_CONFIG", "config/config.json")]

# never pop up a plot window during a test run
os.environ.setdefault("MPLBACKEND", "Agg")
