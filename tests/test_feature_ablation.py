from src.feature_ablation import (
    ABLATION_EXTRAS,
    ablation_feature_lists,
)
from src.features import DEFAULT_ANCHOR
from src.training import selected_feature_lists


def _reference_without_evaluated():
    numeric, categorical = selected_feature_lists(DEFAULT_ANCHOR)
    evaluated = set().union(*ABLATION_EXTRAS.values())
    return [f for f in numeric if f not in evaluated], categorical


def test_reference_is_selected_features_without_evaluated_ones():
    candidates = ablation_feature_lists(DEFAULT_ANCHOR)
    numeric, categorical = _reference_without_evaluated()

    assert candidates["referencia"] == (numeric, categorical)


def test_reference_excludes_features_under_evaluation():
    numeric, _ = ablation_feature_lists(DEFAULT_ANCHOR)["referencia"]
    evaluated = set().union(*ABLATION_EXTRAS.values())

    assert evaluated.isdisjoint(numeric)


def test_each_candidate_adds_only_its_requested_feature():
    reference_numeric, reference_categorical = _reference_without_evaluated()
    candidates = ablation_feature_lists(DEFAULT_ANCHOR)

    for name, extras in ABLATION_EXTRAS.items():
        numeric, categorical = candidates[name]

        assert categorical == reference_categorical
        assert numeric == reference_numeric + list(extras)
