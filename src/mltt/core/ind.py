"""Shared helpers for inductive types."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import ClassVar, Self

from .ast import Term, Var, Pi, Univ, App, Reducer
from .debruijn import Ctx, mk_app, mk_pis, decompose_app, discharge_binders


def infer_ind_type(ctx: Ctx, ind: Ind) -> Term:
    """Compute the dependent function type of an inductive.

    The resulting Pi-tower has parameters outermost, then indices,
    finishing with the universe level.
    """

    for b in ind.all_binders:
        b.expect_universe(ctx)
        ctx = ctx.insert(b)
    return mk_pis(*ind.all_binders, return_ty=Univ(ind.level))


def infer_ctor_type(ctor: Ctor) -> Term:
    """Compute the dependent function type of a constructor.

    The resulting Pi-tower has parameters outermost, then constructor
    arguments, finishing with the inductive head applied to the result
    indices.
    """
    ind = ctor.inductive
    if len(ctor.result_indices) != len(ind.index_types):
        raise TypeError(
            "Constructor result indices must match inductive index arity:\n"
            f"  ctor = {ctor}\n"
            f"  expected arity = {len(ind.index_types)}\n"
            f"  found arity = {len(ctor.result_indices)}"
        )
    # Parameters bind outermost, then constructor arguments.
    #   [params][args] from outermost to innermost.
    offset = len(ctor.field_schemas)
    param_vars = [
        Var(i) for i in reversed(range(offset, offset + len(ind.param_types)))
    ]
    return mk_pis(
        *ind.param_types,
        *ctor.field_schemas,
        return_ty=mk_app(ctor.inductive, *param_vars, *ctor.result_indices),
    )


def infer_elim_type(elim: Elim, ctx: Ctx) -> Term:
    """Infer the type of an ``InductiveElim`` while checking its well-formedness."""
    # 1. Infer type of scrutinee and extract params/indices.

    scrut = elim.scrutinee
    ind = elim.inductive
    scrut_ty = scrut.infer_type(ctx).whnf()
    scrut_ty_head, scrut_ty_bindings = decompose_app(scrut_ty)
    if scrut_ty_head is not ind:
        raise TypeError(
            "Eliminator scrutinee not of the right inductive type:\n"
            f"  scrutinee = {scrut}\n"
            f"  expected head = {ind}\n"
            f"  found head = {scrut_ty_head}"
        )

    # 2.1 Partially apply motive to the actual indices
    motive = elim.motive
    p = len(ind.param_types)
    q = len(ind.index_types)
    params_actual = scrut_ty_bindings[:p]
    indices_actual = scrut_ty_bindings[p:]
    motive_applied = mk_app(motive, *indices_actual)

    # 2.2 Infer the type of this partially applied motive
    motive_applied_ty = motive_applied.infer_type(ctx).whnf()
    if not isinstance(motive_applied_ty, Pi):
        raise TypeError(
            "InductiveElim motive must take scrutinee after indices:\n"
            f"  motive = {motive}\n"
            f"  motive_applied = {motive_applied}\n"
            f"  motive_applied_ty = {motive_applied_ty}"
        )

    # 2.3 The scrutinee binder domain must match the scrutinee type
    scrut_dom = motive_applied_ty.arg_ty
    if not scrut_dom.type_equal(scrut_ty):
        raise TypeError(
            "InductiveElim motive scrutinee domain mismatch:\n"
            f"  expected scrut_ty = {scrut_ty}\n"
            f"  found scrut_dom = {scrut_dom}"
        )

    # 2.4 The motive codomain must be a universe
    body_ty = motive_applied_ty.return_ty.whnf()
    if not isinstance(body_ty, Univ):
        raise TypeError(
            "InductiveElim motive codomain must be a universe:\n"
            f"  motive_applied = {motive_applied}\n"
            f"  motive_applied_ty.return_ty = {motive_applied_ty.return_ty}\n"
            f"  normalized = {body_ty}"
        )

    # 3. For each constructor, compute the expected branch type and check
    for ctor, case in zip(ind.constructors, elim.cases, strict=True):
        # Context legend:
        #   Γ          = current context passed to this eliminator
        #   fields     = constructor arguments (outermost → innermost)
        #   ihs        = induction hypotheses for recursive fields
        #
        # Most ctor schemas are written in context (params)(fields). At this
        # point params are already fixed by the scrutinee in Γ, so we
        # instantiate only those param binders and keep fields as de Bruijn
        # variables.

        # 3.1 Instantiate ctor field types with the actual parameters.
        ctor_field_types = tuple(
            discharge_binders(schema, params_actual, depth_above=i).whnf()
            for i, schema in enumerate(ctor.field_schemas)
        )
        m = len(ctor_field_types)

        # 3.2 Work in context Γ,fields: shift Γ-level params and motive by m.
        params_in_fields_ctx = tuple(p.shift(m) for p in params_actual)
        motive_in_fields_ctx = motive.shift(m)

        # Build a scrutinee-shaped term: C params field_vars.
        # field_vars are Var(m-1) .. Var(0) in the Γ,fields context.
        field_vars = tuple(Var(j) for j in reversed(range(m)))
        scrut_like = mk_app(ctor, *params_in_fields_ctx, *field_vars)

        # 3.3 Instantiate ctor result indices under fields.
        result_indices = tuple(
            discharge_binders(schema, params_actual, depth_above=m + i)
            for i, schema in enumerate(ctor.result_indices)
        )

        # 3.4 Build IH types for recursive fields.
        # Each IH is in context Γ,fields,ihs_so_far, so shift by m + ri.
        r = len(ctor.rps)
        ih_types: list[Term] = []
        for ri, j in enumerate(ctor.rps):
            h, rec_field_args = decompose_app(ctor_field_types[j].shift(m - j))
            assert h is ind
            rec_params = rec_field_args[:p]
            rec_indices = rec_field_args[p : p + q]
            assert rec_params == params_in_fields_ctx
            ih_type = mk_app(motive_in_fields_ctx, *rec_indices, field_vars[j])
            ih_types.append(ih_type.shift(ri))

        # 3.5 Branch codomain in Γ,fields,IHs.
        codomain = mk_app(motive_in_fields_ctx, *result_indices, scrut_like).shift(r)

        # 3.6 Expected branch type: Π fields. Π ihs. codomain.
        branch_ty = mk_pis(*ctor_field_types, *ih_types, return_ty=codomain)
        case.type_check(branch_ty, ctx)

    u = motive_applied_ty.return_ty.expect_universe(ctx)  # cod should be Univ(u)

    # target type is P i⃗_actual scrut
    target_ty = App(motive_applied, scrut)

    # sanity check target_ty really is a type in Type u (or ≤ u with cumulativity)
    _ = target_ty.infer_type(ctx).expect_universe(ctx)

    if u < ind.level:
        raise TypeError(
            "Eliminator motive returns too small a universe:\n"
            f"  motive level = {u}\n"
            f"  inductive level = {ind.level}"
        )

    return target_ty


@dataclass(frozen=True, kw_only=True)
class Ind(Term):
    """A generalized inductive type with constructors."""

    name: str
    param_types: tuple[Term, ...] = field(repr=False, default=())
    index_types: tuple[Term, ...] = field(repr=False, default=())
    constructors: tuple["Ctor", ...] = field(repr=False, default=())
    level: int = 0
    is_terminal: ClassVar[bool] = True

    @cached_property
    def all_binders(self) -> tuple[Term, ...]:
        return self.param_types + self.index_types

    # Typing -------------------------------------------------------------------
    def _infer_type(self, ctx: Ctx) -> Term:
        return infer_ind_type(ctx, self)

    def _type_equal_with(self, other: Self, ctx: Ctx) -> bool:
        return other is self


@dataclass(frozen=True, kw_only=True)
class Ctor(Term):
    """A constructor for an inductive type."""

    name: str
    inductive: Ind = field(repr=False)
    field_schemas: tuple[Term, ...] = field(repr=False, default=())
    result_indices: tuple[Term, ...] = field(repr=False, default=())
    is_terminal: ClassVar[bool] = True

    @cached_property
    def index_in_inductive(self) -> int:
        for idx, ctor in enumerate(self.inductive.constructors):
            if ctor is self:
                return idx
        raise TypeError("Constructor does not belong to inductive type")

    @cached_property
    def rps(self) -> tuple[int, ...]:
        return tuple(
            j
            for j, s in enumerate(self.field_schemas)
            if decompose_app(s.whnf())[0] is self.inductive
        )

    def iota_reduce(
        self, cases: tuple[Term, ...], args: tuple[Term, ...], motive: Term
    ) -> Term:
        """Compute the iota-reduction of an eliminator on a fully-applied ctor."""

        ind = self.inductive
        p = len(ind.param_types)
        m = len(self.field_schemas)
        ctor_args = args[p : p + m]
        ihs = [Elim(ind, motive, cases, ctor_args[rp]) for rp in self.rps]
        case = cases[self.index_in_inductive]
        return mk_app(case, *ctor_args, *ihs)

    # Typing -------------------------------------------------------------------
    def _infer_type(self, ctx: Ctx) -> Term:
        return infer_ctor_type(self)

    def _type_equal_with(self, other: Self, ctx: Ctx) -> bool:
        return other is self


@dataclass(frozen=True)
class Elim(Term):
    """Elimination principle for an inductive type."""

    inductive: Ind
    motive: Term
    cases: tuple[Term, ...]
    scrutinee: Term

    # Reduction ----------------------------------------------------------------
    def whnf(self) -> Term:

        scrutinee_whnf = self.scrutinee.whnf()
        head, args = decompose_app(scrutinee_whnf)
        if isinstance(head, Ctor) and head.inductive is self.inductive:
            expected_args = len(self.inductive.param_types) + len(head.field_schemas)
            if len(args) != expected_args:
                raise ValueError()
            return head.iota_reduce(self.cases, args, self.motive).whnf()
        return Elim(self.inductive, self.motive, self.cases, scrutinee_whnf)

    # Typing -------------------------------------------------------------------
    def _infer_type(self, ctx: Ctx) -> Term:
        return infer_elim_type(self, ctx)

    def _type_equal_with(self, other: Self, ctx: Ctx) -> bool:
        if other.inductive is not self.inductive:
            return False
        if len(self.cases) != len(other.cases):
            return False
        return (
            self.motive.type_equal(other.motive, ctx)
            and all(
                case1.type_equal(case2, ctx)
                for case1, case2 in zip(self.cases, other.cases, strict=True)
            )
            and self.scrutinee.type_equal(other.scrutinee, ctx)
        )


__all__ = [
    "Ind",
    "Ctor",
    "Elim",
]
