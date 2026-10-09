"""Load a template driver module from a pack directory."""

from __future__ import annotations

import importlib.util
import os
import sys
import types

# Relative imports: an extension install imports this package as
# ``bl_ext.<repo>.parametric_deformation``, so the absolute package name is not
# available at runtime (docs/vroid.md 7.5).
from ..core.generic import GenericAdapter
from ..core.interfaces import TemplateAdapter


def load_driver(pack_dir: str) -> TemplateAdapter:
    """Import py/driver.py from the pack and instantiate its adapter class."""
    # Template drivers import ``parametric_deformation...``; make sure that name
    # resolves to this package even when it was installed as an extension.
    from .. import _publish_package_alias

    _publish_package_alias()
    py_dir = os.path.join(pack_dir, "py")
    driver_path = os.path.join(pack_dir, "py", "driver.py")
    pkg_name = "_pd_template"
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [py_dir]
    sys.modules[pkg_name] = pkg
    spec = importlib.util.spec_from_file_location(f"{pkg_name}.driver", driver_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load driver {driver_path}")
    module = importlib.util.module_from_spec(spec)
    module.__package__ = pkg_name
    spec.loader.exec_module(module)
    for value in vars(module).values():
        if (
            isinstance(value, type)
            and issubclass(value, TemplateAdapter)
            and value is not TemplateAdapter
            and value is not GenericAdapter
        ):
            return value()
    return GenericAdapter()
