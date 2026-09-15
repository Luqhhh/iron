from bf_tap.optimization.v10_rebuild import (
    EXPECTED_RESULT_SHA256,
    FIT_MONTHS,
    QRF_PARAMETERS,
    recipe,
)


def test_v10_recipe_is_fixed_to_scored_target_composition():
    value = recipe()
    assert value["candidate"] == "V10_V6I_IRON_V8_TIME"
    assert value["stage"] == "test_a"
    assert value["fit_months"] == list(range(4, 12)) == list(FIT_MONTHS)
    assert value["recency"]["target"] == "tap_iron"
    assert value["qrf"]["statistic"] == "lower weighted median"
    assert value["qrf"]["parameters"] == QRF_PARAMETERS
    assert value["known_result_csv_sha256"] == EXPECTED_RESULT_SHA256
