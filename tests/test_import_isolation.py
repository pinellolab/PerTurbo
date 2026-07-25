"""The default package must remain independent of the legacy stack."""

from __future__ import annotations

import subprocess
import sys


def test_default_import_does_not_load_legacy_runtime() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import perturbo; "
            "assert all(name not in sys.modules for name in ('torch', 'pyro', 'scvi'))",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
