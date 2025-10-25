from mltt.ast import App, Id, IdElim, Lam, NatRec, Pair, Pi, Sigma, Succ, TypeUniverse, Var, Zero
from mltt.debruijn import shift, subst


def test_shift_respects_cutoff():
    term = App(Var(1), Var(0))
    shifted = shift(term, by=2, cutoff=1)
    assert shifted == App(Var(3), Var(0))


def test_shift_through_lambda_increments_free_variable():
    term = Lam(TypeUniverse(), App(Var(1), Var(0)))
    shifted = shift(term, by=1, cutoff=0)
    assert shifted == Lam(TypeUniverse(), App(Var(2), Var(0)))


def test_subst_replaces_target_and_decrements_greater_indices():
    term = App(Var(1), Var(0))
    sub = Succ(Var(0))
    result = subst(term, sub)
    assert result == App(Var(0), Succ(Var(0)))


def test_subst_under_lambda_preserves_bound_variable():
    term = Lam(TypeUniverse(), App(Var(1), Var(0)))
    sub = Succ(Var(0))
    result = subst(term, sub)
    assert result == Lam(TypeUniverse(), App(Succ(Var(1)), Var(0)))


def test_shift_nested_binders():
    term = Lam(TypeUniverse(), Lam(TypeUniverse(), Var(2)))
    shifted = shift(term, by=1, cutoff=0)
    assert shifted == Lam(TypeUniverse(), Lam(TypeUniverse(), Var(3)))


def test_subst_nested_binder_chain():
    term = Lam(TypeUniverse(), Lam(TypeUniverse(), Var(2)))
    sub = Succ(Var(0))
    result = subst(term, sub)
    assert result == Lam(TypeUniverse(), Lam(TypeUniverse(), Succ(Var(2))))


def test_subst_pi_body():
    pi_term = Pi(TypeUniverse(), Pi(TypeUniverse(), Var(1)))
    sub = Succ(Var(0))
    result = subst(pi_term, sub)
    assert result == Pi(TypeUniverse(), Pi(TypeUniverse(), Var(1)))


def test_subst_sigma_pair():
    sigma_term = Sigma(TypeUniverse(), Var(0))
    pair = Pair(Succ(Var(1)), Var(0))
    result = subst(pair, sigma_term)
    assert result == Pair(Succ(Var(0)), Sigma(TypeUniverse(), Var(0)))


def test_subst_natrec_components():
    term = NatRec(Pi(TypeUniverse(), TypeUniverse()), Zero(), Lam(TypeUniverse(), Var(0)), Var(0))
    sub = Succ(Var(0))
    result = subst(term, sub)
    assert isinstance(result, NatRec)


def test_subst_identity_constructs():
    term = Id(TypeUniverse(), Var(0), Var(1))
    sub = Succ(Var(0))
    result = subst(term, sub)
    assert isinstance(result, Id)


def test_subst_idelim_components():
    term = IdElim(TypeUniverse(), Var(0), Var(1), Var(2), Var(3), Var(4))
    sub = Succ(Var(0))
    result = subst(term, sub)
    assert isinstance(result, IdElim)
