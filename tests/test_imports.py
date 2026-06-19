"""Architecture guard: importing the Runner must NOT pull the deprecated training
code (Trainer / Model / Validator). That is the coupling we control.

Runs in a FRESH subprocess so unrelated tests don't pollute sys.modules.

Note: we deliberately do NOT assert on `tqdm` -- it turns out `import torch`
pulls tqdm transitively in this environment, which is out of our control. What we
guarantee is that the Runner path doesn't import our deprecated modules (and thus
doesn't import `tqdm` *via* our validator).
"""

import os
import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("torch")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_runner_import_is_decoupled():
    code = textwrap.dedent(
        """
        import sys
        import src.core.runner  # noqa: F401
        forbidden = ("src.training.trainer", "src.training.model", "src.training.validator")
        leaked = [m for m in forbidden if m in sys.modules]
        print(";".join(leaked))
        """
    )
    res = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT,
                         capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    leaked = res.stdout.strip()
    assert leaked == "", f"Runner import pulled in: {leaked}"
