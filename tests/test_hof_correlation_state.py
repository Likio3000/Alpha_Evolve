from __future__ import annotations

import numpy as np

from alpha_evolve.evolution import hall_of_fame as hof


def test_import_export_correlation_state_roundtrip() -> None:
    snapshot = hof.export_correlation_state()
    try:
        state = {
            "corr_penalty_config": {"weight": 0.42, "cutoff": 0.11},
            "hof_state_version": 123,
            "corr_fingerprints": ["fp_a"],
            "rank_pred_matrix": [np.array([0.0, 1.0], dtype=float)],
            "raw_pred_matrix": [np.array([1.0, 2.0], dtype=float)],
        }
        hof.import_correlation_state(state)
        restored = hof.export_correlation_state()
        assert restored["corr_penalty_config"]["weight"] == 0.42
        assert restored["corr_penalty_config"]["cutoff"] == 0.11
        assert restored["hof_state_version"] == 123
        assert restored["corr_fingerprints"] == ["fp_a"]
        assert np.allclose(restored["rank_pred_matrix"][0], np.array([0.0, 1.0]))
        assert np.allclose(restored["raw_pred_matrix"][0], np.array([1.0, 2.0]))
    finally:
        hof.import_correlation_state(snapshot)


def test_export_correlation_state_can_omit_raw_vectors() -> None:
    snapshot = hof.export_correlation_state()
    try:
        state = {
            "corr_penalty_config": {"weight": 0.33, "cutoff": 0.12},
            "hof_state_version": 10,
            "corr_fingerprints": ["fp_rawless"],
            "rank_pred_matrix": [np.array([0.0, 1.0, -1.0], dtype=float)],
            "raw_pred_matrix": [np.array([1.0, 2.0, 3.0], dtype=float)],
        }
        hof.import_correlation_state(state)
        compact = hof.export_correlation_state(include_raw=False)
        assert "raw_pred_matrix" not in compact

        # Importing compact state should safely clear raw vectors.
        hof.import_correlation_state(compact)
        restored = hof.export_correlation_state()
        assert restored["corr_fingerprints"] == ["fp_rawless"]
        assert len(restored["rank_pred_matrix"]) == 1
        assert restored["raw_pred_matrix"] == []
    finally:
        hof.import_correlation_state(snapshot)
