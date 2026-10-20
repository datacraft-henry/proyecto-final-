from src.feature_ablation import (
    ABLATION_EXTRAS,
    ablation_feature_lists,
)
from src.features import DEFAULT_ANCHOR
from src.training import selected_feature_lists


def test_reference_matches_current_selected_features():
    candidates = ablation_feature_lists(DEFAULT_ANCHOR)

    assert candidates["referencia"] == selected_feature_lists(
        DEFAULT_ANCHOR
    )


def test_reference_excludes_features_under_evaluation():
    numeric, _ = ablation_feature_lists(DEFAULT_ANCHOR)["referencia"]
    evaluated = set().union(*ABLATION_EXTRAS.values())

    assert evaluated.isdisjoint(numeric)


def test_each_candidate_adds_only_its_requested_feature():
    reference_numeric, reference_categorical = selected_feature_lists(
        DEFAULT_ANCHOR
    )
    candidates = ablation_feature_lists(DEFAULT_ANCHOR)

    for name, extras in ABLATION_EXTRAS.items():
        numeric, categorical = candidates[name]

        assert categorical == reference_categorical
        assert numeric == reference_numeric + list(extras)
