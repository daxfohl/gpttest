from elab_helpers import elab_ok


def test_explicit_id() -> None:
    src = """
    let id(A: Type 0, x: A): A := x;
    id
    """
    elab_ok(src)


def test_arrow_sugar() -> None:
    src = """
    let k(A: Type 0, B: Type 0, a: A, b: B): A := a;
    k
    """
    elab_ok(src)


def test_check_mode_lambda() -> None:
    src = """
    let id2(A: Type 0) := fun (x: A) => x;
    id2
    """
    elab_ok(src)


def test_lambda_type_params() -> None:
    src = """
    let id2<A>(impl x: A) := x;
    id2
    """
    elab_ok(src)


def test_typed_let() -> None:
    src = "let A := Type 0; A"
    elab_ok(src)
