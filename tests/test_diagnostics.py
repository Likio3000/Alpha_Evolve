from alpha_evolve.utils import diagnostics as diag


def test_diagnostics_event_hub_records_and_enriches():
    """Ensure the diagnostics hub captures generation stats then enriches the latest entry."""
    diag.reset()
    diag.record_generation(generation=1, eval_stats={"a": 1}, eval_events=[{"e": 1}], best={"b": 2})
    diag.enrich_last(pop_quantiles={"median": 0.1}, ramp={"corr_w": 0.2}, gen_eval_seconds=1.5)
    all_diags = diag.get_all()
    assert len(all_diags) == 1
    entry = all_diags[0]
    assert entry["generation"] == 1
    assert entry["eval_stats"]["a"] == 1
    assert entry["events_sample"][0]["e"] == 1
    assert entry["best"]["b"] == 2
    assert entry["pop_quantiles"]["median"] == 0.1
    assert entry["ramp"]["corr_w"] == 0.2
    assert entry["gen_eval_seconds"] == 1.5
