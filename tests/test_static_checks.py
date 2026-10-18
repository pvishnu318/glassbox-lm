"""Static analysis verification test for GlassBox-LM.

Runs pyflakes over src/, scripts/, and tests/ to catch undefined names,
unbound variables, and unused imports across the entire repository before execution.
"""

from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_pyflakes_static_analysis() -> None:
    """Run pyflakes across src, scripts, and tests, ensuring zero warnings or errors."""
    cmd = [
        sys.executable,
        "-m",
        "pyflakes",
        "src",
        "scripts",
        "tests",
    ]

    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    output = result.stdout.strip()
    errors = result.stderr.strip()

    assert result.returncode == 0, (
        f"pyflakes detected static analysis errors (exit code {result.returncode}):\n"
        f"STDOUT:\n{output}\n"
        f"STDERR:\n{errors}"
    )
    assert not output, f"pyflakes emitted warnings/errors:\n{output}"
