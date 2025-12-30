import pytest

import mltt.inductive.vec as vec
from mltt.inductive import list as lst
from mltt.inductive.fin import FinType, FZ, FS
from mltt.inductive.nat import NatType, Succ, Zero, add, numeral
from mltt.inductive.vec import VecType
from mltt.kernel.ast import Term, Univ, Var
from mltt.kernel.ind import Elim, Ctor, Ind
from mltt.kernel.telescope import mk_app, mk_pis, mk_lams, Telescope


def test_infer_vec_type() -> None:
    assert vec.Vec.infer_type() == mk_pis(Univ(0), NatType(), return_ty=Univ(0))


def test_infer_vec_type_at_level() -> None:
    assert vec.VecAt(1).infer_type() == mk_pis(Univ(1), NatType(), return_ty=Univ(1))


def test_nil_has_zero_length() -> None:
    elem_ty = NatType()
    nil = vec.Nil(elem_ty)

    nil.type_check(vec.VecType(elem_ty, Zero()))


def test_cons_increments_length() -> None:
    elem_ty = NatType()
    tail = vec.Nil(elem_ty)
    cons = vec.Cons(elem_ty, Zero(), Zero(), tail)

    cons.type_check(vec.VecType(elem_ty, Succ(Zero())))


def test_vec_rec_on_nil_reduces_to_zero() -> None:
    elem_ty = NatType()
    P = mk_lams(NatType(), vec.VecType(elem_ty, Var(0)), body=NatType())
    base = Zero()
    step = mk_lams(
        NatType(),
        elem_ty,
        vec.VecType(elem_ty, Var(1)),
        mk_app(P, Var(2), Var(0)),
        body=Var(0),
    )

    term = vec.VecElim(P, base, step, vec.Nil(elem_ty))
    assert term.whnf() == Zero()


@pytest.mark.parametrize("vec_len", range(4))
@pytest.mark.parametrize("b", range(4))
@pytest.mark.parametrize("v", range(4))
def test_vec_rec_preserves_length_index1(vec_len: int, b: int, v: int) -> None:
    elem_ty = NatType()
    P = mk_lams(NatType(), vec.VecType(elem_ty, Var(0)), body=NatType())

    base = numeral(b)
    step = mk_lams(
        NatType(),  # n : Nat
        elem_ty,  # x : A
        vec.VecType(elem_ty, Var(1)),  # xs : Vec A n (Var(1) = n)
        mk_app(P, Var(2), Var(0)),  # ih : P n xs
        body=add(Var(0), Var(2)),  # ih + x
    )

    xs: Term = vec.Nil(elem_ty)
    for i in range(vec_len):
        xs = vec.Cons(elem_ty, numeral(i), numeral(v), xs)
    rec = vec.VecElim(P, base, step, xs)
    normalized = rec.normalize()
    assert normalized == numeral(v * vec_len + b)

    rec.type_check(NatType())


def test_vec_rec_preserves_length_index() -> None:
    elem_ty = NatType()
    # Motive specialized to length 0 so it matches Nil's result index.
    P = mk_lams(
        NatType(),
        vec.VecType(elem_ty, Var(0)),
        body=NatType(),
    )

    base = Succ(Zero())  # P (Nil A) = Nat
    step = mk_lams(
        NatType(),
        elem_ty,
        vec.VecType(elem_ty, Var(1)),
        mk_app(P, Var(2), Var(0)),
        body=Var(2),  # ignore IH; return Nat
    )

    xs: Term = vec.Nil(elem_ty)
    # xs = vec.Cons(elem_ty, Zero(), Zero(), xs)  # say Vec A 1
    # xs = vec.Cons(elem_ty, Succ(Zero()), Succ(Zero()), xs)  # say Vec A 1

    rec = vec.VecElim(P, base, step, xs)
    normalized = rec.normalize()
    assert normalized == Succ(Zero())

    rec.type_check(NatType())


@pytest.mark.parametrize(
    "elem", (Zero(), Succ(Zero()), vec.Nil(NatType()), NatType(), Univ(0), Univ(55))
)
@pytest.mark.parametrize("n", range(5))
def test_infer_type(elem: Term, n: int) -> None:
    elem_ty = elem.infer_type()
    level = elem_ty.expect_universe()
    vector = vec.Nil(elem_ty, level=level)
    for i in range(n):
        vector = vec.Cons(elem_ty, numeral(i), elem, vector, level=level)
    t = vector.infer_type()
    assert t == VecType(elem_ty, numeral(n), level=level)


def test_ctor_type() -> None:
    t = vec.NilCtor.infer_type()
    # Pi x : Type. Vec x Zero
    assert t == mk_pis(Univ(0), return_ty=vec.VecType(Var(0), Zero()))
    t = vec.ConsCtor.infer_type()
    # Pi x : Type. Pi x1 : Nat. x -> Vec x x1 -> Vec x (Succ x1)
    assert t == mk_pis(
        Univ(0),
        NatType(),
        Var(1),
        vec.VecType(Var(2), Var(1)),
        return_ty=vec.VecType(Var(3), Succ(Var(2))),
    )


def test_scrut_type() -> None:
    scrut = vec.Nil(NatType())
    t = scrut.infer_type()
    assert t == mk_app(vec.Vec, NatType(), Zero())
    scrut = vec.Cons(NatType(), Zero(), Zero(), scrut)
    t = scrut.infer_type()
    assert t == mk_app(vec.Vec, NatType(), Succ(Zero()))


def _vec_len_recursor() -> Term:
    """Length via ``VecElim`` with IH bound to the recursive tail."""

    motive = mk_lams(
        NatType(),  # n
        vec.VecType(Var(3), Var(0)),  # xs : Vec A n; A is Var(3) in Γ,n
        body=NatType(),
    )
    step = mk_lams(
        NatType(),  # n : Nat
        Var(3),  # x : A (Var(3) = A in Γ,n,x)
        vec.VecType(Var(4), Var(1)),  # xs : Vec A n (Var(1) = n)
        mk_app(motive.shift(2), Var(2), Var(0)),  # ih : P n xs (shift motive under n,x)
        body=Succ(Var(0)),  # Succ ih
    )

    return mk_lams(
        Univ(0),  # A
        NatType(),  # n
        vec.VecType(Var(1), Var(0)),  # xs : Vec A n
        body=vec.VecElim(motive, Zero(), step, Var(0)),
    )


def test_vec_len_recursor_handles_field_indices() -> None:
    vec_len = _vec_len_recursor()
    expected_ty = mk_pis(
        Univ(0), NatType(), vec.VecType(Var(1), Var(0)), return_ty=NatType()
    )
    vec_len.type_check(expected_ty)

    elem_ty = NatType()
    scrut = vec.Cons(elem_ty, Zero(), Zero(), vec.Nil(elem_ty))
    reduced = mk_app(vec_len, elem_ty, Succ(Zero()), scrut).normalize()
    assert reduced == Succ(Zero())


def test_vec_len_recursor_shifts_open_param() -> None:
    vec_len = _vec_len_recursor()
    term = mk_lams(Univ(0), body=mk_app(vec_len, Var(0)))

    expected_ty = mk_pis(
        Univ(0), NatType(), vec.VecType(Var(1), Var(0)), return_ty=NatType()
    )
    term.type_check(expected_ty)


def test_vec_len_recursor_reduces_with_open_param() -> None:
    vec_len = _vec_len_recursor()
    term = mk_lams(
        Univ(0),  # A
        Var(0),  # a : A
        body=mk_app(
            vec_len,
            Var(1),
            Succ(Zero()),
            vec.Cons(Var(1), Zero(), Var(0), vec.Nil(Var(1))),
        ),
    )

    expected_ty = mk_pis(Univ(0), Var(0), return_ty=NatType())
    term.type_check(expected_ty)

    normalized = term.normalize()
    assert normalized == mk_lams(Univ(0), Var(0), body=Succ(Zero()))
    applied = mk_app(term, NatType(), Zero()).normalize()
    assert applied == Succ(Zero())


def test_recursive_detection_whnfs_field_head() -> None:
    lazy = Ind(name="Lazy", level=0)
    lazy_ctor = Ctor(
        name="Thunk",
        inductive=lazy,
        field_schemas=Telescope.of(mk_app(mk_lams(Univ(0), body=Var(0)), lazy)),
    )
    object.__setattr__(lazy, "constructors", (lazy_ctor,))

    motive = mk_lams(mk_app(lazy), body=Univ(1))
    branch = mk_lams(
        mk_app(mk_lams(Univ(0), body=Var(0)), lazy),
        Univ(1),  # ih : motive scrutinee
        body=Univ(0),
    )

    elim = Elim(inductive=lazy, motive=motive, cases=(branch,), scrutinee=Var(0))
    Lam_ty = mk_pis(mk_app(lazy), return_ty=Univ(1))

    term = mk_lams(mk_app(lazy), body=elim)
    term.type_check(Lam_ty)


def test_nat_and_list_elims_stay_sane() -> None:
    elem_ty = NatType()
    xs = lst.Cons(elem_ty, Zero(), lst.Nil(elem_ty))

    list_motive = mk_lams(lst.ListType(elem_ty), body=NatType())
    list_step = mk_lams(elem_ty, lst.ListType(elem_ty), NatType(), body=Succ(Var(0)))

    length = lst.ListElim(list_motive, Zero(), list_step, xs)
    assert length.normalize() == Succ(Zero())
    length.type_check(NatType())


def test_vec_to_fin() -> None:
    to_fin = vec.vec_to_fin_term()
    expected_ty = mk_pis(
        Univ(0), NatType(), vec.VecType(Var(1), Var(0)), return_ty=FinType(Succ(Var(1)))
    )
    to_fin.type_check(expected_ty)

    elem_ty = NatType()
    nil = vec.Nil(elem_ty)
    one = mk_app(to_fin, elem_ty, Zero(), nil).normalize()
    assert one == FZ(Zero())
    one.type_check(FinType(Succ(Zero())))

    xs = vec.Cons(elem_ty, Zero(), Zero(), nil)
    two = mk_app(to_fin, elem_ty, Succ(Zero()), xs).normalize()
    assert two == FS(Succ(Zero()), FZ(Zero()))
    two.type_check(FinType(Succ(Succ(Zero()))))
