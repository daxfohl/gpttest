"""Proofs around addition on natural numbers."""

from __future__ import annotations

from ..core.ast import Lam, Term, Var, Pi
from ..core.debruijn import shift
from ..core.inductive_utils import nested_lam, apply_term
from ..inductive.eq import Refl, Id, ap, trans, sym
from ..inductive.nat import NatType, Succ, Zero, add, NatRec


def add_zero_right() -> Term:
    """∀ n. add n 0 = n."""

    return Lam(
        NatType(),  # n
        NatRec(
            # motive P(n) = Id Nat (add n 0) n
            P=Lam(
                NatType(),
                Id(NatType(), add(Var(0), Zero()), Var(0)),
            ),
            # base: add 0 0 ≡ 0  ⇒ refl
            base=Refl(ty=NatType(), t=Zero()),
            # step: ih : Id Nat (add k 0) k  ⇒ need Id Nat (add (Succ k) 0) (Succ k)
            # definitional eqn: add (Succ k) 0 ≡ Succ (add k 0)
            # so use ap Succ ih : Id Nat (Succ (add k 0)) (Succ k)
            step=nested_lam(
                NatType(),  # k
                Id(NatType(), add(Var(0), Zero()), Var(0)),  # ih
                body=ap(
                    f=Lam(NatType(), Succ(Var(0))),  # Succ as a function
                    A=NatType(),
                    B0=NatType(),
                    x=add(Var(1), Zero()),  # add k 0
                    y=Var(1),  # k
                    p=Var(0),  # ih
                ),
            ),
            n=Var(0),  # recurse on n
        ),
    )


def add_zero_left() -> Term:
    """∀ n. add 0 n = n."""

    return Lam(
        NatType(),
        Refl(NatType(), add(Zero(), Var(0))),
    )


def succ_add() -> Term:
    """∀ n m. add (Succ n) m = Succ (add n m)."""

    return nested_lam(
        NatType(),
        NatType(),
        body=Refl(NatType(), add(Succ(Var(1)), Var(0))),
    )


def add_succ_right() -> Term:
    """∀ n m. add m (Succ n) = Succ (add m n)."""

    # P m = Id (add m (Succ n)) (Succ (add m n)), with ``n`` free.
    P = Lam(
        NatType(),  # m
        Id(
            NatType(),
            add(Var(0), Succ(Var(1))),  # add m (Succ n)
            Succ(add(Var(0), Var(1))),  # Succ (add m n)
        ),
    )

    # m = 0 ⇒ both sides reduce to Succ n
    base = Refl(NatType(), Succ(Var(0)))

    # In the step, the context after introducing k and ih is [ih, k, n].
    # ih : add (Succ k) (Succ n) = Succ (add (Succ k) n)
    # ap Succ ih witnesses the step for Succ k.
    step = nested_lam(
        NatType(),  # k
        apply_term(shift(P, 1), Var(0)),  # ih : P k, with n still referring to n
        body=ap(
            f=Lam(NatType(), Succ(Var(0))),
            A=NatType(),
            B0=NatType(),
            x=add(Var(1), Succ(Var(2))),  # add k (Succ n)
            y=Succ(add(Var(1), Var(2))),  # Succ (add k n)
            p=Var(0),  # ih
        ),
    )

    return nested_lam(
        NatType(),  # n
        NatType(),  # m
        body=NatRec(
            P=shift(P, 1),  # <-- critical
            base=shift(base, 1),  # <-- critical
            step=shift(step, 1),  # <-- critical
            n=Var(0),  # recurse on m
        ),
    )


def add_comm() -> Term:
    """Proof that addition is commutative."""

    # Q n m = Id (add n m) (add m n)
    Q = Lam(
        NatType(),  # n
        Pi(
            NatType(),  # m
            Id(
                NatType(),
                add(Var(1), Var(0)),  # add n m
                add(Var(0), Var(1)),  # add m n
            ),
        ),
    )

    base = Lam(
        NatType(),  # m
        sym(
            NatType(),
            add(Var(0), Zero()),  # x = add m 0
            Var(0),  # y = m
            apply_term(add_zero_right(), Var(0)),  # p : add m 0 = m
        ),
    )

    step = nested_lam(
        NatType(),  # n
        apply_term(Q, Var(0)),  # ih
        body=Lam(
            NatType(),  # m
            trans(
                NatType(),
                add(Succ(Var(2)), Var(0)),  # add (Succ n) m
                Succ(add(Var(0), Var(2))),  # Succ (add m n)
                add(Var(0), Succ(Var(2))),  # add m (Succ n)
                trans(
                    NatType(),
                    add(Succ(Var(2)), Var(0)),  # add (Succ n) m
                    Succ(add(Var(2), Var(0))),  # Succ (add n m)
                    Succ(add(Var(0), Var(2))),  # Succ (add m n)
                    apply_term(succ_add(), Var(2), Var(0)),  # succ_add n m
                    ap(
                        f=Lam(NatType(), Succ(Var(0))),
                        A=NatType(),
                        B0=NatType(),
                        x=add(Var(2), Var(0)),  # add n m
                        y=add(Var(0), Var(2)),  # add m n
                        p=apply_term(Var(1), Var(0)),  # ih m
                    ),
                ),
                sym(
                    NatType(),
                    add(Var(0), Succ(Var(2))),  # x
                    Succ(add(Var(0), Var(2))),  # y
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
