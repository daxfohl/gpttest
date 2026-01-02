from types import MappingProxyType

from mltt.kernel.env import Env
from mltt.surface.etype import ElabEnv
from mltt.surface.elab_state import ElabState
from mltt.surface.parse import parse_term


def elab_ok_in_env(src: str, env: Env) -> None:
    elab_env = ElabEnv.from_env(env)
    state = ElabState()
    term = parse_term(src)
    term_k, ty_k = term.elab_infer(elab_env, state)
    state.solve(elab_env.kenv)
    term_k = state.zonk(term_k)
    ty_term = state.zonk(ty_k.term)
    state.ensure_solved()
    _ = (term_k, ty_term)


def test_surface_elim_add_comm() -> None:
    env = Env(globals=MappingProxyType({}))
    src = """
    inductive Nat: Type 0 := 
    | Zero
    | Succ(k: Nat);

    let succ(x: Nat): Nat := Nat.Succ(x);
    
    inductive Id{u}<A: Type(u)> (x: A): (y: A) -> Type(u) :=
    | Refl: Id(x, x);

    let sym{u}<A: Type(u), x: A, y: A>(p: Id(x, y)): Id(y, x) :=
      match p with
      | Refl => ctor Id.Refl;

    let trans{u}<A: Type(u), x: A, y: A, z: A>(p: Id(x, y), q: Id(y, z)): Id(x, z) :=
      match q with
      | Refl => p;

    let ap{u}<A: Type(u), B: Type(u)>(f: A -> B) <x: A, y: A>(p: Id(x, y)): Id(f(x), f(y)) :=
      match p with
      | Refl => ctor Id.Refl;
    
    let add(m: Nat, n: Nat): Nat := 
      match m with
      | Zero => n
      | Succ k => succ(add(k, n));

    let add_zero_right(n: Nat): Id(add(n, Nat.Zero), n) :=
      match n with
      | Zero => ctor Id.Refl
      | Succ k => ap(succ, add_zero_right(k));

    let succ_add(n: Nat, m: Nat): Id(add(succ(n), m), succ(add(n, m))) :=
      ctor Id.Refl;

    let add_succ_right(n: Nat, m: Nat): Id(add(m, succ(n)), succ(add(m, n))) :=
      match m with
      | Zero => ctor Id.Refl
      | Succ k => ap(succ, add_succ_right(n, k));

    let add_comm(n: Nat, m: Nat): Id(add(n, m), add(m, n)) :=
      match n with
      | Zero => sym(add_zero_right(m))
      | Succ k =>
        trans(
          trans(succ_add(k, m), ap(succ, add_comm(k, m))),
          sym(add_succ_right(k, m))
        );
    add_comm
    """
    elab_ok_in_env(src, env)


def test_surface_elim_as_return() -> None:
    env = Env(globals=MappingProxyType({}))
    src = """
    inductive Nat: Type 0 := 
    | Zero
    | Succ(k: Nat);
    let pred(n: Nat): Nat := 
      match n return Nat with
      | Zero => Nat.Zero
      | Succ k => k;
    pred
    """
    elab_ok_in_env(src, env)
