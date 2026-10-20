"""Mide el aporte individual de las variables del Sprint 2.

No cambia el modelo productivo ni SELECTED_EXTRA_FEATURES.
Uso:
    python -m src.feature_ablation
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data_loading import RAW_DIR, load_raw_tables
from src.feature_engineering import (
    build_model_dataset,
    load_geolocation,
    model_feature_lists,
)
from src.features import ANCHORS, DEFAULT_ANCHOR
from src.training import (
    cross_validate,
    make_logistic,
    selected_feature_lists,
    temporal_cv_folds,
)
from src.validation import temporal_three_way_split

ABLATION_EXTRAS: dict[str, tuple[str, ...]] = {
    "referencia": (),
    "carga_7d": ("seller_recent_approved_orders_7d",),
    "carga_14d": ("seller_recent_approved_orders_14d",),
    "carga_30d": ("seller_recent_approved_orders_30d",),
    "feriado_actual": ("is_national_holiday",),
    "feriados_proximos_7d": ("n_national_holidays_next_7d",),
}


def ablation_feature_lists(
    anchor: str = DEFAULT_ANCHOR,
) -> dict[str, tuple[list[str], list[str]]]:
    """Devuelve las columnas del modelo sin las variables evaluadas, mas una por vez.

    La referencia saca las variables del Sprint 2 aunque alguna ya este en el
    modelo productivo, para que la comparacion siga siendo la misma.
    """
    requested = set().union(*ABLATION_EXTRAS.values())
    model_numeric, categorical = selected_feature_lists(anchor)
    selected_numeric = [f for f in model_numeric if f not in requested]
    available_numeric, _ = model_feature_lists(anchor)

    missing = requested - set(available_numeric)
    if missing:
        raise ValueError(
            "Las variables de ablacion no estan disponibles: "
            f"{sorted(missing)}"
        )

    candidates = {}
    for name, extras in ABLATION_EXTRAS.items():
        numeric = selected_numeric + [
            feature
            for feature in extras
            if feature not in selected_numeric
        ]
        candidates[name] = (numeric, list(categorical))

    return candidates


def evaluate_feature_ablation(
    dataset: pd.DataFrame,
    anchor: str = DEFAULT_ANCHOR,
) -> pd.DataFrame:
    """Compara cada variable con la misma CV temporal del modelo principal."""
    train, validation, _ = temporal_three_way_split(dataset)
    train_validation = pd.concat(
        [train, validation],
        ignore_index=True,
    )
    folds = temporal_cv_folds(train_validation)

    rows = []
    for name, (numeric, categorical) in ablation_feature_lists(
        anchor
    ).items():
        columns = numeric + categorical
        metrics = cross_validate(
            make_logistic(numeric, categorical),
            columns,
            folds,
        )
        rows.append(
            {
                "candidate": name,
                "added_features": (
                    ", ".join(ABLATION_EXTRAS[name]) or "ninguna"
                ),
                **metrics,
            }
        )

    report = pd.DataFrame(rows)
    reference_pr_auc = float(
        report.loc[
            report["candidate"].eq("referencia"),
            "pr_auc",
        ].iloc[0]
    )
    report["delta_pr_auc"] = report["pr_auc"] - reference_pr_auc
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument(
        "--anchor",
        choices=list(ANCHORS),
        default=DEFAULT_ANCHOR,
    )
    args = parser.parse_args()

    tables = load_raw_tables(args.raw_dir)
    tables["geolocation"] = load_geolocation(args.raw_dir)
    dataset = build_model_dataset(tables, anchor=args.anchor)

    report = evaluate_feature_ablation(dataset, anchor=args.anchor)
    columns = [
        "candidate",
        "added_features",
        "pr_auc_folds",
        "pr_auc",
        "delta_pr_auc",
        "roc_auc",
    ]
    print(
        report[columns].to_string(
            index=False,
            float_format="{:.6f}".format,
        )
    )


if __name__ == "__main__":
    main()
