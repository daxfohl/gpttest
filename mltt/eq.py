from __future__ import annotations

from .ast import App, Id, IdElim, Lam, Refl, Term, Var


def cong(
    function: Term,
    domain: Term,
    codomain_family: Term,
    left: Term,
    right: Term,
    witness: Term,
) -> Term:
    """Lift an equality proof through function application.

    This matches the usual ``cong`` combinator from Martin-Löf Type Theory:

    ``cong : (f : (x : A) -> B x) -> Id_A x y -> Id_{B y} (f x) (f y)``

    The explicit ``domain`` and ``codomain_family`` parameters correspond to the
    type family annotations that are implicit in the traditional presentation.
    """

    predicate = Lam(
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
    refl_proof = Refl(App(codomain_family, left), App(function, left))
    return IdElim(domain, left, predicate, refl_proof, right, witness)


def sym(domain: Term, left: Term, right: Term, witness: Term) -> Term:
    """Symmetry of identity proofs in MLTT."""

    predicate = Lam(
        domain,
        Lam(
            Id(domain, left, Var(1)),
            Id(domain, Var(1), left),
        ),
    )
    refl_proof = Refl(domain, left)
    return IdElim(domain, left, predicate, refl_proof, right, witness)


def trans(
    domain: Term,
    left: Term,
    middle: Term,
    right: Term,
    first_witness: Term,
    second_witness: Term,
) -> Term:
    """Transitivity of identity proofs in MLTT."""

    predicate = Lam(
        domain,
        Lam(
            Id(domain, middle, Var(1)),
            Id(domain, left, Var(1)),
        ),
    )
    return IdElim(domain, middle, predicate, first_witness, right, second_witness)


__all__ = ["cong", "sym", "trans"]
