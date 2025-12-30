import mltt.inductive.sigma as sigma
from mltt.inductive.fin import Fin, FZCtor
from mltt.inductive.nat import NatType, Succ, Zero, numeral
from mltt.kernel.ast import App, Lam, Pi, Univ, Var
from mltt.kernel.telescope import mk_app, mk_pis, mk_lams


def test_infer_sigma_type_constructor() -> None:
    expected = mk_pis(Univ(0), Pi(Var(0), Univ(0)), return_ty=Univ(0))

    assert sigma.Sigma.infer_type() == expected


def test_pair_type_check() -> None:
    A = NatType()
    B = Lam(A, NatType())
    pair = sigma.Pair(A, B, Zero(), Zero())

    pair.type_check(sigma.SigmaType(A, B))


def test_sigmarec_returns_first_projection() -> None:
    A = NatType()
    B = Lam(A, NatType())

    pair = sigma.Pair(A, B, Succ(Zero()), Zero())

    P = Lam(sigma.SigmaType(A, B), A)
    pair_case = mk_lams(
        A,
        App(B, Var(0)),  # b : B a
        body=Var(1),  # return the first projection
    )

    fst = sigma.SigmaElim(P, pair_case, pair)

    assert fst.normalize() == Succ(Zero())

    fst.type_check(A)


def test_let_pair_iota_reduces() -> None:
    A = NatType()
    B = Lam(NatType(), NatType())  # constant family for simplicity
    a = numeral(3)
    b = numeral(5)
    p = sigma.Pair(A, B, a, b)

    # let (x,y)=p in x  ==>  a
    term = sigma.let_pair(A, B, p=p, f=Var(1))  # Var(1)=a under [b,a]
    assert term.normalize().type_equal(a.normalize())

    # let (x,y)=p in y  ==>  b
    term2 = sigma.let_pair(A, B, p=p, f=Var(0))  # Var(0)=b
    assert term2.normalize().type_equal(b.normalize())


def test_let_pair_term_typechecks() -> None:
    lp = sigma.let_pair_term()
    inferred = lp.infer_type()

    expected = mk_pis(
        Univ(0),  # A
        Pi(Var(0), Univ(0)),  # B
        Univ(0),  # C
        sigma.SigmaType(Var(2), Var(1)),  # Sigma A B
        Pi(Var(3), Pi(App(Var(3), Var(0)), Var(3))),  # Π a. Π b. C
        return_ty=Var(2),  # C
    )
    assert inferred.type_equal(expected)


def test_let_pair_term_iota_equation() -> None:
    A = NatType()
    B = Lam(NatType(), NatType())
    C = NatType()

    a = numeral(2)
    b = numeral(7)
    p = sigma.Pair(A, B, a, b)

    # f := λa:Nat. λb:Nat. a
    f = mk_lams(NatType(), NatType(), body=Var(1))

    term = mk_app(sigma.let_pair_term(), A, B, C, p, f)
    assert term.normalize().type_equal(a.normalize())


def test_let_pair_handles_nondependent_B() -> None:
    # B a := Fin (Succ a)  (or Vec Nat a if you have Vec)
    A = NatType()
    B = Lam(NatType(), App(Fin, Succ(Var(0))))  # B a = Fin (Succ a)

    a = numeral(3)
    b = mk_app(FZCtor, a)  # FZ : n -> Fin (Succ n), so b : Fin (Succ a)
    p = sigma.Pair(A, B, a, b)

    # body returns b, so C is B a = Fin (Succ a)
    C = NatType()
    term = sigma.let_pair(A, B, p=p, f=Zero())
    assert term.infer_type().type_equal(C)


def test_let_pair_handles_dependent_B() -> None:
    A = NatType()
    B = Lam(NatType(), App(Fin, Succ(Var(0))))

    a = numeral(3)
    b = mk_app(FZCtor, a)
    p = sigma.Pair(A, B, a, b)

    term = sigma.let_pair_dep(A, B, p=p, f_body=Var(0))  # return b
    expected = App(B, a)  # B 3
    assert term.infer_type().type_equal(expected)


def test_let_pair_dep_term_typechecks() -> None:
    lp = sigma.let_pair_dep_term()
    lp.infer_type()


def test_fst_beta() -> None:
    A = NatType()
    B = Lam(NatType(), App(Fin, Succ(Var(0))))
    a = numeral(3)
    b = mk_app(FZCtor, a)
    p = sigma.Pair(A, B, a, b)

    assert sigma.fst(A, B, p).infer_type().type_equal(A)
    assert sigma.fst(A, B, p).normalize().type_equal(a)


def test_snd_beta() -> None:
    A = NatType()
    B = Lam(NatType(), App(Fin, Succ(Var(0))))
    a = numeral(3)
    b = mk_app(FZCtor, a)
    p = sigma.Pair(A, B, a, b)

    expected_ty = App(B, a)
    assert sigma.snd(A, B, p).infer_type().type_equal(expected_ty)
    assert sigma.snd(A, B, p).normalize().type_equal(b)
