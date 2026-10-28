import numpy as np
import pandas as pd
import pytest

import tests.test_baseline as synthetic
from src.feature_engineering import build_model_dataset
from src.features import PREDICTION_TIME_COL
from src.training import (
    MAIN_CANDIDATES,
    alert_threshold,
    compare_models,
    select_model,
    selected_feature_lists,
    temporal_cv_folds,
)
from src.validation import temporal_three_way_split

N_ORDERS = 1500


@pytest.fixture
def dataset(monkeypatch):
    monkeypatch.setattr(synthetic, "N_ORDERS", N_ORDERS)
    tables = synthetic._synthetic_tables(seed=1)
    rng = np.random.default_rng(1)
    zips = ["01000", "20000", "40000"]
    tables["customers"] = tables["customers"].assign(customer_zip_code_prefix=rng.choice(zips, N_ORDERS))
    tables["sellers"] = tables["sellers"].assign(seller_zip_code_prefix=["01000", "20000"])
    tables["products"] = tables["products"].assign(product_category_name="moveis")
    tables["geolocation"] = pd.DataFrame(
        {
            "geolocation_zip_code_prefix": zips,
            "geolocation_lat": [-23.5, -22.9, -12.9],
            "geolocation_lng": [-46.6, -43.2, -38.5],
        }
    )
    return build_model_dataset(tables)


def test_temporal_three_way_split_is_ordered_in_time(dataset):
    train, validation, test = temporal_three_way_split(dataset)
    assert len(train) + len(validation) + len(test) == len(dataset)
    assert train[PREDICTION_TIME_COL].max() < validation[PREDICTION_TIME_COL].min()
    assert validation[PREDICTION_TIME_COL].max() < test[PREDICTION_TIME_COL].min()


def test_cv_folds_train_only_on_the_past(dataset):
    folds = temporal_cv_folds(dataset, n_folds=4)
    assert len(folds) == 4
    for fit_part, eval_part in folds:
        assert fit_part[PREDICTION_TIME_COL].max() < eval_part[PREDICTION_TIME_COL].min()
    evaluated = pd.concat([eval_part for _, eval_part in folds])
    assert evaluated["order_id"].is_unique


def test_select_model_prefers_simpler_model_within_tolerance():
    results = {"logistica_seleccion": {"pr_auc": 0.205}, "arboles_seleccion": {"pr_auc": 0.212},
               "arboles_completa": {"pr_auc": 0.200}}
    assert select_model(results, tolerance=0.01) == "logistica_seleccion"
    assert select_model(results, tolerance=0.001) == "arboles_seleccion"


def test_alert_threshold_keeps_requested_volume():
    scores = np.linspace(0, 1, 1000)
    assert (scores >= alert_threshold(scores, volume=0.05)).mean() == pytest.approx(0.05, abs=0.002)


def test_selected_features_drop_month_and_add_distance():
    numeric, _ = selected_feature_lists()
    assert "purchase_month" not in numeric
    assert "max_distance_km" in numeric
    assert "promised_days" in numeric


def test_selected_features_add_only_upcoming_holidays_from_sprint2():
    numeric, _ = selected_feature_lists()
    assert "n_national_holidays_next_7d" in numeric
    assert "is_national_holiday" not in numeric
    assert not any(c.startswith("seller_recent_approved_orders") for c in numeric)


def test_compare_models_end_to_end(dataset):
    report, fitted = compare_models(dataset)
    assert report["selected_model"] in MAIN_CANDIDATES
    assert set(report["cross_validation"]) == set(fitted)
    for name, metrics in report["test"].items():
        for key in ["confusion_matrix", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
            assert key in metrics, (name, key)
    cohorts = report["cohorts"]
    assert cohorts["train"]["period"][1] <= cohorts["validation"]["period"][0]
    assert report["validation"][report["selected_model"]]["roc_auc"] > 0.6
