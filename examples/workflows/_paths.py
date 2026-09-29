"""Path helpers for examples that can run via pytest or ``verdi run``."""

import pathlib
import sys


def script_dir(file):
    """Return the directory of an example script.

    ``verdi run`` sets ``__file__`` to the basename, while the full script path
    is available as ``sys.argv[0]``.
    """
    path = pathlib.Path(file)
    if path.parent == pathlib.Path("."):
        argv_path = pathlib.Path(sys.argv[0])
        if argv_path.name == path.name:
            path = argv_path
    return path.resolve().parent
