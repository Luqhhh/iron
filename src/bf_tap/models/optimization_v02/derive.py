"""OOF selection and replay of the selected, fixed local candidates."""
from ...artifacts import stable_digest
from .m1_blend.model import blend_predictions, select_blend_weights
from .m4_ensemble.model import apply_ensemble, select_ensemble
from .oof import score_oof_predictions
from .selection import quality_evidence


def replay_selection(predictions, frozen):
    result = {
        "M1_BLEND": blend_predictions(predictions["CatBoost"], predictions["B1"], frozen["m1_weights"]),
        "M2_RECENCY": predictions[f"M2_H{frozen['m2_half_life_days']}"].copy(),
        "M3_RESIDUAL": predictions["M3_RESIDUAL"].copy(),
    }
    if frozen["m4"]["status"] == "ENABLED":
        result["M4_ENSEMBLE"] = apply_ensemble(result, frozen["m4"])
    return result


def freeze_selection(actual, predictions, half_lives, m1_config):
    selected_m1 = select_blend_weights(actual, predictions["CatBoost"], predictions["B1"],
                                       weights=m1_config["parameters"]["weights"],
                                       tie_tolerance=m1_config["parameters"]["tie_tolerance"])
    m2_scores = {h: score_oof_predictions(actual, predictions[f"M2_H{h}"])
                 for h in half_lives}
    half_life = min(half_lives, key=lambda h: (m2_scores[h]["pooled"]["loss"], -h))
    members = {"M1_BLEND": selected_m1["predictions"],
               "M2_RECENCY": predictions[f"M2_H{half_life}"],
               "M3_RESIDUAL": predictions["M3_RESIDUAL"]}
    m4 = select_ensemble(actual, members, predictions["CatBoost"], predictions["B1"])
    frozen = {
        "schema_version": 1,
        "selection_scope": "three_month_rolling_oof_only",
        "selection_bias_notice": "Weights and half-life are selected on these OOF rows; not independent test performance.",
        "confirmation_overlap_notice": "DEV_LONG/SHORT overlap development periods and are stability confirmation only.",
        "m1_weights": selected_m1["weights"], "m1_grid": selected_m1["grid"],
        "m2_half_life_days": half_life, "m2_grid": {str(h): score for h, score in m2_scores.items()}, "m4": m4,
    }
    members = replay_selection(predictions, frozen)
    frozen["quality"] = {
        name: quality_evidence(actual, pred, predictions["CatBoost"], predictions["B1"])
        for name, pred in members.items()}
    frozen["ranked_members"] = sorted(
        members, key=lambda name: (frozen["quality"][name]["metrics"]["pooled"]["loss"], name))
    frozen["selection_sha256"] = stable_digest(frozen)
    return frozen, members
