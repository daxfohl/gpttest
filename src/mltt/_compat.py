"""Compatibility helpers for optionally using newer typing features.

This module makes it straightforward for the rest of the codebase to target
Python 3.14-only typing helpers while still running on older interpreters.
Currently we provide a thin shim for :data:`typing.TypeIs`, falling back to
``TypeGuard`` when the interpreter does not expose it yet.  Keeping the alias in
one place prevents scattering version checks throughout the type checking code.
"""

from __future__ import annotations

from typing import TypeGuard

try:  # pragma: no cover - exercised only on Python < 3.14
    from typing import TypeIs  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - exercised only on Python < 3.14
    # ``TypeIs`` is slated for Python 3.14 via PEP 742.  When unavailable we
    # treat it as ``TypeGuard`` so that type checkers still perform the intended
    # narrowing while the runtime simply sees a ``bool`` return type.
    from typing import TypeGuard as TypeIs  # type: ignore[assignment]

__all__ = ["TypeIs"]

