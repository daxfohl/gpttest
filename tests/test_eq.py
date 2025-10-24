import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mltt.ast import App, Id, IdElim, Lam, NatType, Refl, Succ, Var, Zero
from mltt.eq import cong, sym, trans


def test_cong_builds_identity_elimination_over_function_application():
    A = NatType()
    B = Lam(NatType(), Var(0))
    f = Lam(NatType(), Succ(Var(0)))
    x = Zero()
    y = Succ(Zero())
    p = Var(1)

    result = cong(f, A, B, x, y, p)

    P = Lam(
        A,
        Lam(
            Id(A, x, Var(1)),
            Id(App(B, Var(1)), App(f, x), App(f, Var(1))),
        ),
    )
    d = Refl(App(B, x), App(f, x))
    expected = IdElim(A, x, P, d, y, p)

    assert result == expected


def test_sym_builds_identity_elimination_with_swapped_arguments():
    A = NatType()
    x = Zero()
    y = Succ(Zero())
    p = Var(0)

    result = sym(A, x, y, p)

    P = Lam(
        A,
        Lam(
            Id(A, x, Var(1)),
            Id(A, Var(1), x),
        ),
    )
    d = Refl(A, x)
    expected = IdElim(A, x, P, d, y, p)

    assert result == expected


def test_trans_builds_identity_elimination_for_composition():
    A = NatType()
    x = Zero()
    y = Succ(Zero())
    z = Succ(Succ(Zero()))
    p = Var(0)
    q = Var(1)

    result = trans(A, x, y, z, p, q)

    Q = Lam(
        A,
        Lam(
            Id(A, y, Var(1)),
            Id(A, x, Var(1)),
        ),
    )
    expected = IdElim(A, y, Q, p, z, q)

    assert result == expected
