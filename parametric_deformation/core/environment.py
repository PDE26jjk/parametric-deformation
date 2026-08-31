"""Per-template Python environment management.

Modeled on the GNM Blender add-on managed environment: each template pack may
declare requirements.txt; the plugin creates an isolated venv and installs the
dependencies there instead of polluting the Blender/system Python.
"""

from __future__ import annotations

import os
import subprocess
import sys
import venv


def _env_root() -> str:
    base = os.environ.get("PD_TEMPLATE_ENVS")
    if base:
        return base
    return os.path.join(os.path.expanduser("~"), ".parametric_deformation", "envs")


def env_path_for(pack_id: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in pack_id)
    return os.path.join(_env_root(), safe)


def env_python(pack_id: str) -> str:
    if os.name == "nt":
        return os.path.join(env_path_for(pack_id), "Scripts", "python.exe")
    return os.path.join(env_path_for(pack_id), "bin", "python")


def ensure_template_env(pack_dir: str, pack_id: str, quiet: bool = False) -> str | None:
    """Create/refresh the pack venv if requirements.txt exists.

    Returns the env python path, or None when the pack has no requirements.
    """
    req = os.path.join(pack_dir, "requirements.txt")
    if not os.path.exists(req):
        return None
    target = env_path_for(pack_id)
    if not os.path.exists(env_python(pack_id)):
        os.makedirs(os.path.dirname(target), exist_ok=True)
        venv.EnvBuilder(with_pip=True).create(target)
    python = env_python(pack_id)
    proc = subprocess.run(
        [python, "-m", "pip", "install", "-q", "-r", req],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0 and not quiet:
        raise RuntimeError(f"pip install failed for {pack_id}: {proc.stderr[-2000:]}")
    return python
