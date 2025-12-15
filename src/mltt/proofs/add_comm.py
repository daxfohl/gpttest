"""Proofs around addition on natural numbers."""

from __future__ import annotations

from ..core.ast import Lam, Term, Var
from ..core.inductive_utils import nested_lam, apply_term
from ..inductive.eq import Refl, Id, ap, trans, sym
from ..inductive.nat import NatType, Succ, Zero, add_terms, add_n_0, NatRec


def add_zero_right() -> Term:
    """∀ n. add n 0 = n."""

    return add_n_0()


def add_zero_left() -> Term:
    """∀ n. add 0 n = n."""

    return Lam(
        NatType(),
        Refl(NatType(), add_terms(Zero(), Var(0))),
    )


def succ_add() -> Term:
    """∀ n m. add (Succ n) m = Succ (add n m)."""

    return nested_lam(
        NatType(),
        NatType(),
        body=Refl(NatType(), add_terms(Succ(Var(1)), Var(0))),
    )


def add_succ_right() -> Term:
    """∀ n m. add m (Succ n) = Succ (add m n)."""

    # P m = Id (add m (Succ n)) (Succ (add m n)), with ``n`` free.
    P = Lam(
        NatType(),  # m
        Id(
            NatType(),
            add_terms(Var(0), Succ(Var(1))),  # add m (Succ n)
            Succ(add_terms(Var(0), Var(1))),  # Succ (add m n)
        ),
    )

    # m = 0 ⇒ both sides reduce to Succ n
    base = Refl(NatType(), Succ(Var(1)))

    # In the step, the context after introducing k and ih is [ih, k, m, n].
    # ih : add (Succ k) (Succ n) = Succ (add (Succ k) n)
    # ap Succ ih witnesses the step for Succ k.
    step = nested_lam(
        NatType(),  # k
        apply_term(P, Var(0)),  # ih : add k (Succ n) = Succ (add k n)
        body=ap(
            f=Lam(NatType(), Succ(Var(0))),
            A=NatType(),
            B0=NatType(),
            x=add_terms(Var(1), Succ(Var(3))),
            y=Succ(add_terms(Var(1), Var(3))),
            p=Var(0),
        ),
    )

    return nested_lam(
        NatType(),  # n
        NatType(),  # m
        body=NatRec(
            P=P,
            base=base,
            step=step,
            n=Var(0),  # recurse on m
        ),
    )


def add_comm() -> Term:
    """Proof that addition is commutative."""

    # Q n = Id (add n m) (add m n) for fixed m
    Q = Lam(
        NatType(),
        Id(
            NatType(),
            add_terms(Var(0), Var(1)),  # add n m
            add_terms(Var(1), Var(0)),  # add m n
        ),
    )

    base = Lam(
        NatType(),  # m
        sym(
            NatType(),
            Var(0),
            add_terms(Var(0), Zero()),
            apply_term(add_zero_right(), Var(0)),  # add m 0 = m
        ),
    )

    step = nested_lam(
        NatType(),  # n
        apply_term(Q, Var(0)),  # ih : Id (add n m) (add m n)
        body=Lam(
            NatType(),  # m
            trans(
                NatType(),
                Succ(add_terms(Var(1), Var(0))),  # add (Succ n) m
                Succ(add_terms(Var(0), Var(1))),  # Succ (add n m)
                add_terms(Var(0), Succ(Var(1))),  # add m (Succ n)
                ap(
                    f=Lam(NatType(), Succ(Var(0))),
                    A=NatType(),
                    B0=NatType(),
                    x=add_terms(Var(1), Var(0)),
                    y=add_terms(Var(0), Var(1)),
                    p=apply_term(Var(1), Var(0)),  # ih m
                ),
                sym(
                    NatType(),
                    add_terms(Var(0), Succ(Var(1))),
                    Succ(add_terms(Var(0), Var(1))),
                    apply_term(add_succ_right(), Var(2), Var(0)),
                ),
            ),
        ),
    )

    return nested_lam(
        NatType(),  # n
        NatType(),  # m
        body=apply_term(
            NatRec(P=Q, base=base, step=step, n=Var(1)),
            Var(0),
        ),
    )


__all__ = [
    "add_zero_right",
    "add_zero_left",
    "succ_add",
    "add_succ_right",
    "add_comm",
]
