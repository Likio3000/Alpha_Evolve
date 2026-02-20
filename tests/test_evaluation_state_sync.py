from __future__ import annotations

from alpha_evolve.evolution import evaluation as el


def test_config_version_increments_only_when_config_changes() -> None:
    snapshot = el.export_evaluation_state()
    try:
        kwargs = dict(
            parsimony_penalty=0.002,
            max_ops=32,
            xs_flatness_guard=0.05,
            temporal_flatness_guard=0.05,
            early_abort_bars=20,
            early_abort_xs=0.05,
            early_abort_t=0.05,
            flat_bar_threshold=0.25,
            scale_method="madz",
            sharpe_proxy_weight=0.1,
            ic_std_penalty_weight=0.05,
            turnover_penalty_weight=0.01,
            ic_tstat_weight=0.0,
            use_train_val_splits=True,
            train_points=100,
            val_points=50,
            split_weighting="equal",
            sector_neutralize=True,
            net_exposure_target=0.25,
            winsor_p=0.02,
        )
        el.configure_evaluation(**kwargs)
        v1 = int(el.export_evaluation_state()["config_version"])
        el.configure_evaluation(**kwargs)
        v2 = int(el.export_evaluation_state()["config_version"])
        assert v2 == v1

        kwargs["sharpe_proxy_weight"] = 0.2
        el.configure_evaluation(**kwargs)
        v3 = int(el.export_evaluation_state()["config_version"])
        assert v3 == v2 + 1
    finally:
        el.import_evaluation_state(snapshot)
