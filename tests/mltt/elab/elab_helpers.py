from operator import methodcaller

from mltt.solver.solver import Solver
from mltt.elab.term import elab_infer
from mltt.elab.types import ElabEnv
from mltt.kernel.ast import Let, Term
from mltt.kernel.env import Env
from mltt.kernel.prelude import prelude_env
from mltt.surface.parse import parse_elab_term


def elab_ok(src: str) -> None:
    env = ElabEnv.from_env(prelude_env())
    solver = Solver()
    term = parse_elab_term(src)
    term_k, ty_k = elab_infer(term, env, solver)
    solver.solve(env.kenv)
    term_k = solver.zonk(term_k)
    ty_term = solver.zonk(ty_k.term)
    solver.ensure_solved()
    _ = (term_k, ty_term)


def elab_ok_in_env(src: str, env: Env) -> None:
    elab_env = ElabEnv.from_env(env)
    solver = Solver()
    term = parse_elab_term(src)
    term_k, ty_k = elab_infer(term, elab_env, solver)
    solver.solve(elab_env.kenv)
    term_k = solver.zonk(term_k)
    ty_term = solver.zonk(ty_k.term)
    solver.ensure_solved()
    _ = (term_k, ty_term)


def elab_with_state(src: str) -> Solver:
    env = ElabEnv.from_env(prelude_env())
    solver = Solver()
    term = parse_elab_term(src)
    elab_infer(term, env, solver)
    solver.solve(env.kenv)
    return solver


def elab_eval(src: str) -> Term:
    kenv = prelude_env()
    env = ElabEnv.from_env(kenv)
    solver = Solver()
    term = parse_elab_term(src)
    term_k, _ty_k = elab_infer(term, env, solver)
    solver.solve(env.kenv)
    term_k = solver.zonk(term_k)
    solver.ensure_solved()
    while isinstance(term_k, Let):
        kenv = kenv.push_let(term_k.arg_ty, term_k.value)
        term_k = term_k.body
    return _normalize_with_env(term_k, kenv)


def _normalize_with_env(term: Term, env: Env) -> Term:
    reduced = term._reduce_inside_step(methodcaller("whnf_step", env))
    if reduced != term:
        return _normalize_with_env(reduced, env)
    return reduced


def get_ctor(name: str) -> Term:
    env = prelude_env()
    decl = env.lookup_global(name)
    assert decl is not None
    assert decl.value is not None
    return decl.value
