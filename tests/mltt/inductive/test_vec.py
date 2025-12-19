import pytest

import mltt.inductive.vec as vec
from mltt.core.ast import Term, Univ, Var
from mltt.core.ind import Elim, Ctor, Ind
from mltt.core.util import apply_term, nested_pi, nested_lam
from mltt.inductive import list as lst
from mltt.inductive.fin import FinType, FZ, FS
from mltt.inductive.nat import NatType, Succ, Zero, add, numeral
from mltt.inductive.vec import VecType


def test_infer_vec_type() -> None:
    assert vec.Vec.infer_type() == nested_pi(Univ(0), NatType(), return_ty=Univ(0))


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
    P = nested_lam(NatType(), vec.VecType(elem_ty, Var(0)), body=NatType())
    base = Zero()
    step = nested_lam(
        NatType(),
        elem_ty,
        vec.VecType(elem_ty, Var(1)),
        apply_term(P, Var(2), Var(0)),
        body=Var(0),
    )

    term = vec.VecRec(P, base, step, vec.Nil(elem_ty))
    assert term.whnf() == Zero()


@pytest.mark.parametrize("vec_len", range(4))
@pytest.mark.parametrize("b", range(4))
@pytest.mark.parametrize("v", range(4))
def test_vec_rec_preserves_length_index1(vec_len: int, b: int, v: int) -> None:
    elem_ty = NatType()
    P = nested_lam(NatType(), vec.VecType(elem_ty, Var(0)), body=NatType())

    base = numeral(b)
    step = nested_lam(
        NatType(),  # n : Nat
        elem_ty,  # x : A
        vec.VecType(elem_ty, Var(1)),  # xs : Vec A n (Var(1) = n)
        apply_term(P, Var(2), Var(0)),  # ih : P n xs
        body=add(Var(0), Var(2)),  # ih + x
    )

    xs: Term = vec.Nil(elem_ty)
    for i in range(vec_len):
        xs = vec.Cons(elem_ty, numeral(i), numeral(v), xs)
    rec = vec.VecRec(P, base, step, xs)
    normalized = rec.normalize()
    assert normalized == numeral(v * vec_len + b)

    rec.type_check(NatType())


def test_vec_rec_preserves_length_index() -> None:
    elem_ty = NatType()
    # Motive specialized to length 0 so it matches Nil's result index.
    P = nested_lam(
        NatType(),
        vec.VecType(elem_ty, Var(0)),
        body=NatType(),
    )

    base = Succ(Zero())  # P (Nil A) = Nat
    step = nested_lam(
        NatType(),
        elem_ty,
        vec.VecType(elem_ty, Var(1)),
        apply_term(P, Var(2), Var(0)),
        body=Var(2),  # ignore IH; return Nat
    )

    xs: Term = vec.Nil(elem_ty)
    # xs = vec.Cons(elem_ty, Zero(), Zero(), xs)  # say Vec A 1
    # xs = vec.Cons(elem_ty, Succ(Zero()), Succ(Zero()), xs)  # say Vec A 1

    rec = vec.VecRec(P, base, step, xs)
    normalized = rec.normalize()
    assert normalized == Succ(Zero())

    rec.type_check(NatType())


@pytest.mark.parametrize(
    "elem", (Zero(), Succ(Zero()), vec.Nil(NatType()), NatType(), Univ(0), Univ(55))
)
@pytest.mark.parametrize("n", range(5))
def test_infer_type(elem: Term, n: int) -> None:
    elem_ty = elem.infer_type()
    vector = vec.Nil(elem_ty)
    for i in range(n):
        vector = vec.Cons(elem_ty, numeral(i), elem, vector)
    t = vector.infer_type()
    assert t == VecType(elem_ty, numeral(n))


def test_ctor_type() -> None:
    t = vec.NilCtor.infer_type()
    # Pi x : Type. Vec x Zero
    assert t == nested_pi(Univ(0), return_ty=vec.VecType(Var(0), Zero()))
    t = vec.ConsCtor.infer_type()
    # Pi x : Type. Pi x1 : Nat. x -> Vec x x1 -> Vec x (Succ x1)
    assert t == nested_pi(
        Univ(0),
        NatType(),
        Var(1),
        vec.VecType(Var(2), Var(1)),
        return_ty=vec.VecType(Var(3), Succ(Var(2))),
    )


def test_scrut_type() -> None:
    scrut = vec.Nil(NatType())
    t = scrut.infer_type()
    assert t == apply_term(vec.Vec, NatType(), Zero())
    scrut = vec.Cons(NatType(), Zero(), Zero(), scrut)
    t = scrut.infer_type()
    assert t == apply_term(vec.Vec, NatType(), Succ(Zero()))


def _vec_len_recursor() -> Term:
    """Length via ``VecRec`` with IH bound to the recursive tail."""

    motive = nested_lam(
        NatType(),  # n
        vec.VecType(Var(3), Var(0)),  # xs : Vec A n; A is Var(3) in Γ,n
        body=NatType(),
    )
    step = nested_lam(
        NatType(),  # n : Nat
        Var(3),  # x : A (Var(3) = A in Γ,n,x)
        vec.VecType(Var(4), Var(1)),  # xs : Vec A n (Var(1) = n)
        apply_term(
            motive.shift(2), Var(2), Var(0)
        ),  # ih : P n xs (shift motive under n,x)
        body=Succ(Var(0)),  # Succ ih
    )

    return nested_lam(
        Univ(0),  # A
        NatType(),  # n
        vec.VecType(Var(1), Var(0)),  # xs : Vec A n
        body=vec.VecRec(motive, Zero(), step, Var(0)),
    )


def test_vec_len_recursor_handles_field_indices() -> None:
    vec_len = _vec_len_recursor()
    expected_ty = nested_pi(
        Univ(0), NatType(), vec.VecType(Var(1), Var(0)), return_ty=NatType()
    )
    vec_len.type_check(expected_ty)

    elem_ty = NatType()
    scrut = vec.Cons(elem_ty, Zero(), Zero(), vec.Nil(elem_ty))
    reduced = apply_term(vec_len, elem_ty, Succ(Zero()), scrut).normalize()
    assert reduced == Succ(Zero())


def test_vec_len_recursor_shifts_open_param() -> None:
    vec_len = _vec_len_recursor()
    term = nested_lam(Univ(0), body=apply_term(vec_len, Var(0)))

    expected_ty = nested_pi(
        Univ(0), NatType(), vec.VecType(Var(1), Var(0)), return_ty=NatType()
    )
    term.type_check(expected_ty)


def test_vec_len_recursor_reduces_with_open_param() -> None:
    vec_len = _vec_len_recursor()
    term = nested_lam(
        Univ(0),  # A
        Var(0),  # a : A
        body=apply_term(
            vec_len,
            Var(1),
            Succ(Zero()),
            vec.Cons(Var(1), Zero(), Var(0), vec.Nil(Var(1))),
        ),
    )

    expected_ty = nested_pi(Univ(0), Var(0), return_ty=NatType())
    term.type_check(expected_ty)

    normalized = term.normalize()
    assert normalized == nested_lam(Univ(0), Var(0), body=Succ(Zero()))
    applied = apply_term(term, NatType(), Zero()).normalize()
    assert applied == Succ(Zero())


def test_recursive_detection_whnfs_field_head() -> None:
    lazy = Ind(name="Lazy", param_types=(), level=0)
    lazy_ctor = Ctor(
        name="Thunk",
        inductive=lazy,
        arg_types=(apply_term(nested_lam(Univ(0), body=Var(0)), lazy),),
    )
    object.__setattr__(lazy, "constructors", (lazy_ctor,))

    motive = nested_lam(apply_term(lazy), body=Univ(0))
    branch = nested_lam(
        apply_term(nested_lam(Univ(0), body=Var(0)), lazy),
        Univ(0),  # ih : motive scrutinee
        body=Univ(0),
    )

    elim = Elim(inductive=lazy, motive=motive, cases=(branch,), scrutinee=Var(0))
    Lam_ty = nested_pi(apply_term(lazy), return_ty=Univ(0))

    term = nested_lam(apply_term(lazy), body=elim)
    term.type_check(Lam_ty)


def test_nat_and_list_elims_stay_sane() -> None:
    elem_ty = NatType()
    xs = lst.Cons(elem_ty, Zero(), lst.Nil(elem_ty))

    list_motive = nested_lam(lst.ListType(elem_ty), body=NatType())
    list_step = nested_lam(elem_ty, lst.ListType(elem_ty), NatType(), body=Succ(Var(0)))

    length = lst.ListRec(list_motive, Zero(), list_step, xs)
    assert length.normalize() == Succ(Zero())
    length.type_check(NatType())


def test_vec_to_fin() -> None:
    to_fin = vec.vec_to_fin_term()
    expected_ty = nested_pi(
        Univ(0), NatType(), vec.VecType(Var(1), Var(0)), return_ty=FinType(Succ(Var(1)))
    )
    to_fin.type_check(expected_ty)

    elem_ty = NatType()
    nil = vec.Nil(elem_ty)
    one = apply_term(to_fin, elem_ty, Zero(), nil).normalize()
    assert one == FZ(Zero())
    one.type_check(FinType(Succ(Zero())))

    xs = vec.Cons(elem_ty, Zero(), Zero(), nil)
    two = apply_term(to_fin, elem_ty, Succ(Zero()), xs).normalize()
    assert two == FS(Succ(Zero()), FZ(Zero()))
    two.type_check(FinType(Succ(Succ(Zero()))))
