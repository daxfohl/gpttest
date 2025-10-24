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


def make_add_right_zero() -> Term:
    P = Lam(Id(NatType(), App(App(add, Var(0)), Zero()), Var(0)))
    z = Refl(NatType(), Zero())
    s = Lam(
        Lam(
            cong(
                Lam(NatType(), Succ(Var(0))),
                NatType(),
                Lam(NatType(), NatType()),
                App(App(add, Var(1)), Zero()),
                Var(1),
                Var(0),
            )
        )
    )
    return Lam(NatType(), NatRec(P, z, s, Var(0)))


def make_add_succ_right() -> Term:
    m = Var(1)
    k = Var(0)
    P = Lam(
        Id(
            NatType(),
            App(App(add, Var(0)), Succ(Var(1))),
            Succ(App(App(add, Var(0)), Var(1))),
        )
    )
    z = Refl(NatType(), Succ(k))
    s = Lam(
        Lam(
            cong(
                Lam(NatType(), Succ(Var(0))),
                NatType(),
                Lam(NatType(), NatType()),
                App(App(add, Var(1)), Succ(Var(2))),
                Succ(App(App(add, Var(1)), Var(2))),
                Var(0),
            )
        )
    )
    return Lam(NatType(), Lam(NatType(), NatRec(P, z, s, m)))


def make_add_comm() -> Term:
    n = Var(1)
    m = Var(0)
    P = Lam(
        Lam(
            Id(
                NatType(),
                App(App(add, Var(1)), Var(0)),
                App(App(add, Var(0)), Var(1)),
            )
        )
    )
    z = Lam(
        NatType(),
        sym(
            NatType(),
            App(App(add, Var(0)), Zero()),
            Var(0),
            App(add_right_zero, Var(0)),
        ),
    )
    s = Lam(
        Lam(
            Lam(
                NatType(),
                trans(
                    NatType(),
                    Succ(App(App(add, Var(2)), Var(0))),
                    Succ(App(App(add, Var(0)), Var(2))),
                    App(App(add, Var(0)), Succ(Var(2))),
                    cong(
                        Lam(NatType(), Succ(Var(0))),
                        NatType(),
                        Lam(NatType(), NatType()),
                        App(App(add, Var(2)), Var(0)),
                        App(App(add, Var(0)), Var(2)),
                        App(Var(1), Var(0)),
                    ),
                    sym(
                        NatType(),
                        App(App(add, Var(0)), Succ(Var(2))),
                        Succ(App(App(add, Var(0)), Var(2))),
                        App(App(add_succ_right, Var(0)), Var(2)),
                    ),
                ),
            )
        )
    )
    return Lam(NatType(), Lam(NatType(), NatRec(P, z, s, n)))


def pretty_nat(n: Term) -> int:
    n = normalize(n)
    k = 0
    while True:
        match n:
            case Zero():
                return k
            case Succ(t):
                k += 1
                n = t
            case _:
                raise ValueError("Not a numeral")


add_right_zero = make_add_right_zero()
add_succ_right = make_add_succ_right()
add_comm = make_add_comm()


if __name__ == "__main__":
    zero = Zero()
    one = Succ(zero)
    two = Succ(one)
    three = Succ(two)

    add_2_1 = App(App(add, two), one)
    add_1_2 = App(App(add, one), two)
    print("normalize(2+1) =", pretty_nat(add_2_1))
    print("normalize(1+2) =", pretty_nat(add_1_2))

    lemma_rz_2 = App(add_right_zero, two)
    rz_ty_2 = Id(NatType(), App(App(add, two), Zero()), two)
    print("type_check(add_right_zero 2):", type_check(lemma_rz_2, rz_ty_2))

    lemma_asr_1_2 = App(App(add_succ_right, one), two)
    asr_ty_1_2 = Id(
        NatType(),
        App(App(add, one), Succ(two)),
        Succ(App(App(add, one), two)),
    )
    print("type_check(add_succ_right 1 2):", type_check(lemma_asr_1_2, asr_ty_1_2))

    theorem_ty = Pi(
        NatType(),
        Pi(
            NatType(),
            Id(
                NatType(),
                App(App(add, Var(1)), Var(0)),
                App(App(add, Var(0)), Var(1)),
            ),
        ),
    )
    print("type_check(add_comm):", type_check(add_comm, theorem_ty))

    proof_2_1 = App(App(add_comm, two), one)
    nf_proof_2_1 = normalize(proof_2_1)
    print("add_comm(2,1) normal form:", nf_proof_2_1)

    print("2+1 =", pretty_nat(App(App(add, two), one)))
    print("1+2 =", pretty_nat(App(App(add, one), two)))
