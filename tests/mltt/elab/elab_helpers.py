from operator import methodcaller

from mltt.kernel.ast import Let, Term
from mltt.kernel.env import Env
from mltt.elab.elab_state import ElabState
from mltt.elab.etype import ElabEnv
from mltt.elab.sast import elab_infer
from mltt.surface.parse import parse_elab_term
from mltt.kernel.prelude import prelude_env


def elab_ok(src: str) -> None:
    env = ElabEnv.from_env(prelude_env())
    state = ElabState()
    term = parse_elab_term(src)
    term_k, ty_k = elab_infer(term, env, state)
    state.solve(env.kenv)
    term_k = state.zonk(term_k)
    ty_term = state.zonk(ty_k.term)
    state.ensure_solved()
    _ = (term_k, ty_term)


def elab_ok_in_env(src: str, env: Env) -> None:
    elab_env = ElabEnv.from_env(env)
    state = ElabState()
    term = parse_elab_term(src)
    term_k, ty_k = elab_infer(term, elab_env, state)
    state.solve(elab_env.kenv)
    term_k = state.zonk(term_k)
    ty_term = state.zonk(ty_k.term)
    state.ensure_solved()
    _ = (term_k, ty_term)


def elab_with_state(src: str) -> ElabState:
    env = ElabEnv.from_env(prelude_env())
    state = ElabState()
    term = parse_elab_term(src)
    elab_infer(term, env, state)
    state.solve(env.kenv)
    return state


def elab_eval(src: str) -> Term:
    kenv = prelude_env()
    env = ElabEnv.from_env(kenv)
    state = ElabState()
    term = parse_elab_term(src)
    term_k, _ty_k = elab_infer(term, env, state)
    state.solve(env.kenv)
    term_k = state.zonk(term_k)
    state.ensure_solved()
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
