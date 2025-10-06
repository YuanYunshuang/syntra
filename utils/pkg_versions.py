#!/usr/bin/env python3
"""
Print versions for a list of pip package names.

Usage:
  python pkg_versions.py requests numpy pandas
"""

from typing import Iterable, Dict, Optional

# Works on Python 3.8+; falls back to the backport if needed.
try:
    from importlib.metadata import version, PackageNotFoundError  # type: ignore
except ImportError:
    from importlib_metadata import version, PackageNotFoundError  # type: ignore


def versions_for(packages: Iterable[str]) -> Dict[str, Optional[str]]:
    """Return a mapping of package name -> version (or None if not installed)."""
    results: Dict[str, Optional[str]] = {}
    for raw in packages:
        name = raw.strip()
        if not name:
            continue
        try:
            results[name] = version(name)
        except PackageNotFoundError:
            results[name] = None
    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) <= 1:
        print("Provide package names as arguments, e.g.:")
        print("  python pkg_versions.py requests numpy pandas")
        sys.exit(1)

    pkgs = sys.argv[1:]
    for name, ver in versions_for(pkgs).items():
        print(f"{name}: {ver or 'NOT INSTALLED'}")
