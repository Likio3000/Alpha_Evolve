import numpy as np

from alpha_evolve.programs import AlphaProgram, Op, FINAL_PREDICTION_VECTOR_NAME


def build_simple_program():
    ops = [
        Op("twos", "add", ("const_1", "const_1")),
        Op("scaled", "vec_mul_scalar", ("opens_t", "twos")),
        Op(FINAL_PREDICTION_VECTOR_NAME, "vec_add_scalar", ("scaled", "const_neg_1")),
    ]
    return AlphaProgram(setup=[], predict_ops=ops, update_ops=[])


def test_eval_returns_correct_vector():
    """Evaluate a simple program to confirm vector math matches the hand calculation."""
    n_stocks = 5
    features = {
        "opens_t": np.arange(1, n_stocks + 1, dtype=float),
        "const_1": 1.0,
        "const_neg_1": -1.0,
    }
    state = {}
    prog = build_simple_program()

    result = prog.eval(features, state, n_stocks)

    expected = features["opens_t"] * 2 - 1
    assert isinstance(result, np.ndarray)
    assert result.shape == (n_stocks,)
    assert np.allclose(result, expected)


def test_scalar_vector_handling():
    """Ensure operations work with scalar/vector combinations."""
    n_stocks = 3
    features = {
        "opens_t": np.array([10.0, -2.0, 3.5], dtype=float),
        "const_1": 1.0,
        "const_neg_1": -1.0,
    }
    state = {}
    prog = build_simple_program()

    result = prog.eval(features, state, n_stocks)
    expected = features["opens_t"] * 2 - 1
    assert result.shape == (n_stocks,)
    assert np.allclose(result, expected)


def test_heaviside_op():
    """Check the Heaviside operator outputs 0/1 thresholds on vector inputs."""
    buf = {"s": np.array([-1.0, 0.5, 0.0])}
    Op("out", "heaviside", ("s",)).execute(buf, n_stocks=3)
    assert np.allclose(buf["out"], np.array([0.0, 1.0, 0.0]))


def test_relation_ops():
    """Validate relation rank/demean operate group-wise over repeated cohorts."""
    v = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
    groups = np.array([0, 0, 0, 1, 1, 1])
    buf = {"v": v, "g": groups}
    Op("rank_out", "relation_rank", ("v", "g")).execute(buf, n_stocks=6)
    Op("demean_out", "relation_demean", ("v", "g")).execute(buf, n_stocks=6)
    expected_rank = np.array([-1.0, 0.0, 1.0, -1.0, 0.0, 1.0])
    expected_demean = np.array([-1.0, 0.0, 1.0, -1.0, 0.0, 1.0])
    assert np.allclose(buf["rank_out"], expected_rank)
    assert np.allclose(buf["demean_out"], expected_demean)


def test_relation_ops_realistic_groups():
    """Ensure relation operators handle uneven group sizes with correct scaling."""
    v = np.array([10.0, 20.0, 30.0, 100.0, 80.0, 5.0, 15.0, 25.0])
    groups = np.array([1, 1, 1, 2, 2, 3, 3, 3])
    buf = {"v": v, "g": groups}
    Op("rank_out", "relation_rank", ("v", "g")).execute(buf, n_stocks=8)
    Op("demean_out", "relation_demean", ("v", "g")).execute(buf, n_stocks=8)
    expected_rank = np.array([-1.0, 0.0, 1.0, 1.0, -1.0, -1.0, 0.0, 1.0])
    expected_demean = np.array([-10.0, 0.0, 10.0, 10.0, -10.0, -10.0, 0.0, 10.0])
    assert np.allclose(buf["rank_out"], expected_rank)
    assert np.allclose(buf["demean_out"], expected_demean)


def test_norm_op():
    """Confirm the norm operator matches numpy's Frobenius norm for matrices."""
    m = np.array([[3.0, 4.0], [0.0, 0.0]])
    buf = {"m": m}
    Op("n", "norm", ("m",)).execute(buf, n_stocks=2)
    assert buf["n"] == np.linalg.norm(m)
