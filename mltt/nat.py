from __future__ import annotations

from .ast import App, Id, Lam, NatRec, NatType, Pi, Refl, Succ, Term, Var, Zero
from .eq import cong, sym, trans
from .eval import normalize
from .typing import type_check


add = Lam(
    NatType(),
    Lam(
        NatType(),
        NatRec(
            P=Lam(NatType()),
            z=Var(0),
            s=Lam(Lam(Succ(App(App(Var(3), Var(1)), Var(0))))),
            n=Var(1),
        ),
    ),
)
