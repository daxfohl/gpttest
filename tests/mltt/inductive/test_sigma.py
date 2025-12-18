import mltt.inductive.sigma as sigma
from mltt.core.ast import App, Lam, Pi, Univ, Var
from mltt.core.debruijn import shift
from mltt.core.utli import nested_lam
from mltt.core.util import apply_term, nested_pi
from mltt.core.reduce.normalize import normalize
from mltt.core.typing import infer_type, type_check, type_equal
from mltt.inductive.fin import Fin, FZCtor
from mltt.inductive.nat import NatType, Succ, Zero, numeral


def test_infer_sigma_type_constructor() -> None:
    expected = nested_pi(Univ(0), Pi(Var(0), Univ(0)), return_ty=Univ(0))

    assert infer_type(sigma.Sigma) == expected


def test_pair_type_check() -> None:
    A = NatType()
    B = Lam(A, NatType())
    pair = sigma.Pair(A, B, Zero(), Zero())

    type_check(pair, sigma.SigmaType(A, B))


def test_sigmarec_returns_first_projection() -> None:
    A = NatType()
    B = Lam(A, NatType())

    pair = sigma.Pair(A, B, Succ(Zero()), Zero())

    P = Lam(sigma.SigmaType(A, B), A)
    pair_case = nested_lam(
        A,
        App(B, Var(0)),  # b : B a
        body=Var(1),  # return the first projection
    )

    fst = sigma.SigmaRec(P, pair_case, pair)

    assert normalize(fst) == Succ(Zero())

    type_check(fst, A)


def test_let_pair_iota_reduces() -> None:
    A = NatType()
    B = Lam(NatType(), NatType())  # constant family for simplicity
    a = numeral(3)
    b = numeral(5)
    p = sigma.Pair(A, B, a, b)

    # let (x,y)=p in x  ==>  a
    term = sigma.let_pair(A, B, C=NatType(), p=p, f=Var(1))  # Var(1)=a under [b,a]
    assert type_equal(normalize(term), normalize(a))

    # let (x,y)=p in y  ==>  b
    term2 = sigma.let_pair(A, B, C=NatType(), p=p, f=Var(0))  # Var(0)=b
    assert type_equal(normalize(term2), normalize(b))


def test_let_pair_term_typechecks() -> None:
    lp = sigma.let_pair_term()
    inferred = infer_type(lp)

    expected = nested_pi(
        Univ(0),  # A
        Pi(Var(0), Univ(0)),  # B
        Univ(0),  # C
        sigma.SigmaType(Var(2), Var(1)),  # Sigma A B
        Pi(Var(3), Pi(App(Var(3), Var(0)), Var(3))),  # Π a. Π b. C
        return_ty=Var(2),  # C
    )
    assert type_equal(inferred, expected)


def test_let_pair_term_iota_equation() -> None:
    A = NatType()
    B = Lam(NatType(), NatType())
    C = NatType()

    a = numeral(2)
    b = numeral(7)
    p = sigma.Pair(A, B, a, b)

    # f := λa:Nat. λb:Nat. a
    f = nested_lam(NatType(), NatType(), body=Var(1))

    term = apply_term(sigma.let_pair_term(), A, B, C, p, f)
    assert type_equal(normalize(term), normalize(a))


def test_let_pair_handles_nondependent_B() -> None:
    # B a := Fin (Succ a)  (or Vec Nat a if you have Vec)
    A = NatType()
    B = Lam(NatType(), App(Fin, Succ(Var(0))))  # B a = Fin (Succ a)

    a = numeral(3)
    b = apply_term(FZCtor, a)  # FZ : n -> Fin (Succ n), so b : Fin (Succ a)
    p = sigma.Pair(A, B, a, b)

    # body returns b, so C is B a = Fin (Succ a)
    C = NatType()
    term = sigma.let_pair(A, B, C=C, p=p, f=Zero())
    assert type_equal(infer_type(term), C)


def test_let_pair_handles_dependent_B() -> None:
    A = NatType()
    B = Lam(NatType(), App(Fin, Succ(Var(0))))

    a = numeral(3)
    b = apply_term(FZCtor, a)
    p = sigma.Pair(A, B, a, b)

    # C : Π a:A. Π b:B a. Type
    # C a b := B a
    C = Lam(
        A,
        Lam(
            App(shift(B, 1), Var(0)),  # b : B a
            App(shift(B, 1), Var(1)),  # result type = B a
        ),
    )

    term = sigma.let_pair_dep(A, B, C=C, p=p, f_body=Var(0))  # return b
    expected = App(B, a)  # B 3
    assert type_equal(infer_type(term), expected)


def test_let_pair_dep_term_typechecks() -> None:
    lp = sigma.let_pair_dep_term()
    infer_type(lp)


def test_fst_beta() -> None:
    A = NatType()
    B = Lam(NatType(), App(Fin, Succ(Var(0))))
    a = numeral(3)
    b = apply_term(FZCtor, a)
    p = sigma.Pair(A, B, a, b)

    assert type_equal(infer_type(sigma.fst(A, B, p)), A)
    assert type_equal(normalize(sigma.fst(A, B, p)), a)


def test_snd_beta() -> None:
    A = NatType()
    B = Lam(NatType(), App(Fin, Succ(Var(0))))
    a = numeral(3)
    b = apply_term(FZCtor, a)
    p = sigma.Pair(A, B, a, b)

    expected_ty = App(B, a)
    assert type_equal(infer_type(sigma.snd(A, B, p)), expected_ty)
    assert type_equal(normalize(sigma.snd(A, B, p)), b)
