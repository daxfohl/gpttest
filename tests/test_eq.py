import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mltt.ast import App, Id, IdElim, Lam, NatType, Refl, Succ, Var, Zero
from mltt.eq import cong, sym, trans


def test_cong_builds_identity_elimination_over_function_application():
    domain = NatType()
    codomain_family = Lam(NatType(), Var(0))
    function = Lam(NatType(), Succ(Var(0)))
    left = Zero()
    right = Succ(Zero())
    witness = Var(1)

    result = cong(function, domain, codomain_family, left, right, witness)

    P = Lam(
        domain,
        Lam(
            Id(domain, left, Var(1)),
            Id(
                App(codomain_family, Var(1)),
                App(function, left),
                App(function, Var(1)),
            ),
        ),
    )
    d = Refl(App(codomain_family, left), App(function, left))
    expected = IdElim(domain, left, P, d, right, witness)

    assert result == expected


def test_sym_builds_identity_elimination_with_swapped_arguments():
    domain = NatType()
    left = Zero()
    right = Succ(Zero())
    witness = Var(0)

    result = sym(domain, left, right, witness)

    P = Lam(
        domain,
        Lam(
            Id(domain, left, Var(1)),
            Id(domain, Var(1), left),
        ),
    )
    d = Refl(domain, left)
    expected = IdElim(domain, left, P, d, right, witness)

    assert result == expected


def test_trans_builds_identity_elimination_for_composition():
    domain = NatType()
    left = Zero()
    middle = Succ(Zero())
    right = Succ(Succ(Zero()))
    first_witness = Var(0)
    second_witness = Var(1)

    result = trans(
        domain,
        left,
        middle,
        right,
        first_witness,
        second_witness,
    )

    Q = Lam(
        domain,
        Lam(
            Id(domain, middle, Var(1)),
            Id(domain, left, Var(1)),
        ),
    )
    expected = IdElim(domain, middle, Q, first_witness, right, second_witness)

    assert result == expected
