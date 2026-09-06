from bf_tap.validation import evaluate_quality


POLICY = {
    "max_loss_relative_to_better_control": -1e-6,
    "per_target_max_absolute_wmape_regression_vs_B1": 0.01,
}


def metric(loss, iron, time):
    return {"loss": loss, "iron": {"wmape": iron}, "time": {"wmape": time}}


def test_quality_requires_better_combined_score_and_each_target_guardrail():
    values = {
        "catboost": metric(0.15, 0.15, 0.15),
        "B0": metric(0.18, 0.18, 0.18),
        "B1": metric(0.17, 0.17, 0.17),
    }
    assert evaluate_quality(values, POLICY)["pass"] is True
    values["catboost"] = metric(0.18, 0.16, 0.20)
    result = evaluate_quality(values, POLICY)
    assert result["combined_pass"] is False
    assert result["per_target_pass"]["time"] is False
