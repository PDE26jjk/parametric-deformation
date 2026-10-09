"""Validate template packs from the command line (no Blender needed).

Usage:

    python -m parametric_deformation.tools.validate_pack <pack_dir> [<pack_dir> ...]

Exits non-zero when any pack reports a problem, so pack repositories can wire
it straight into CI.
"""

from __future__ import annotations

import sys

from ..core.config import validate_pack


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(__doc__.strip())
        return 2
    failed = 0
    for pack_dir in argv:
        issues = validate_pack(pack_dir)
        if issues:
            failed += 1
            print(f"INVALID {pack_dir}")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print(f"OK      {pack_dir}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
