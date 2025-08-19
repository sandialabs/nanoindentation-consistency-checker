#  ___________________________________________________________________________
#  Copyright (c) 2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________


"""
Creates "too_deep_test_labels.csv"
"""

import pandas as pd

from src.utils.indent_manager import IndentManager
from src.utils.preprocess import ANOMALY_TYPES
from src.utils.setup_data import (
    get_best_threshold,
    get_loading,
    normalize,
    normalize_on_other_indent,
    pointwise_euclidean,
)


def create_too_deep_test_labels_csv(datapath, progress):
    im = IndentManager(datapath, ANOMALY_TYPES, progress=progress)

    loading_data = []

    task = progress.add_task(
        "[green]Processing samples...", total=len(im.all_sample_nums)
    )
    for i, sample_num in enumerate(im.all_sample_nums):
        avg_indent_loading = im.get_non_anomalous_average(sample_num, get_loading)

        for indent_num in range(im.sample_size):
            indent_data = im.get_indent_data(sample_num, indent_num, "too_deep", True)

            indent = im.indents[(sample_num, indent_num)][:, :2]
            loading_indent = get_loading(indent)

            # Get the distance between the loading portion and the average of its sample
            loading_distance = pointwise_euclidean(avg_indent_loading, loading_indent)
            indent_data["loading_distance"] = loading_distance

            # Normalize each indent on its sample's average
            # Get the distance between the loading portion and the average of its sample
            loading_indent = normalize_on_other_indent(
                loading_indent, avg_indent_loading
            )
            normalized_loading_distance = pointwise_euclidean(
                normalize(avg_indent_loading), loading_indent
            )
            indent_data["normalized_loading_distance"] = normalized_loading_distance

            loading_data.append(indent_data)
        progress.update(task, advance=1)

    loading_df = pd.DataFrame(loading_data)

    # Only work with non-anomalous / too deep indents to not skew towards any other anomalies
    excluded_mask = loading_df["too_deep"] | ~loading_df["any"]
    excluded_loading_df = loading_df.loc[excluded_mask]
    y_true = excluded_loading_df["too_deep"].astype(bool)

    # CONSTANT: 3 for beta to place more importance on recall. Higher recall causes more false
    # negatives for true labels.
    res = get_best_threshold(
        excluded_loading_df["normalized_loading_distance"].values,
        [
            (
                excluded_loading_df["normalized_loading_distance"].min(),
                excluded_loading_df["normalized_loading_distance"].max(),
            )
        ],
        excluded_loading_df["too_deep"].astype(bool),
        beta=3,
        direction="right",
    )
    norm_t = res.x[0]
    print(f"Normalized loading distance threshold based on f3 score: {norm_t}")

    # CONSTANT: 3 for beta to place more importance on recall. Higher recall causes more false
    # negatives for true labels.
    res = get_best_threshold(
        excluded_loading_df["loading_distance"].values,
        [
            (
                excluded_loading_df["loading_distance"].min(),
                excluded_loading_df["loading_distance"].max(),
            )
        ],
        excluded_loading_df["too_deep"].astype(bool),
        beta=3,
        direction="right",
    )
    t = res.x[0]
    print(f"Loading distance threshold based on f3 score: {t}")

    # Flag
    normalized_loading_predictions = (
        excluded_loading_df["normalized_loading_distance"] > norm_t
    )
    loading_predictions = excluded_loading_df["loading_distance"] > t

    normalized_loading_mask = (normalized_loading_predictions == True) & (
        y_true == False
    )
    print(
        f"{sum(normalized_loading_mask)} false negatives detected from normalized loading "
        + "distance"
    )
    loading_mask = (loading_predictions == True) & (y_true == False)
    print(f"{sum(loading_mask)} false negatives detected from loading distance")

    predictions = normalized_loading_mask | loading_mask
    print(f"{sum(predictions)} total visual false negatives detected")

    reverification_pairs = excluded_loading_df.loc[
        predictions, ["sample_num", "indent_num"]
    ]

    print(
        "Total indents (only including too deep and non-anomalous indents):",
        len(excluded_loading_df),
    )
    print("Total reverification indents:", len(reverification_pairs))
    print("-" * 7)
    print("Total previous anomalous:", sum(excluded_loading_df["too_deep"]))
    print("Total false anomalous:", 0)
    print("Total new anomalous:", sum(excluded_loading_df["too_deep"] | predictions))
    print("-" * 7)
    print("Total previous non anomalous:", sum(~excluded_loading_df["too_deep"]))
    print(
        "Total false non anomalous:",
        sum(predictions & ~excluded_loading_df["too_deep"]),
    )
    print(
        "Total new non anomalous:", sum(~predictions & ~excluded_loading_df["too_deep"])
    )

    im.export_results(reverification_pairs, "too_deep_test_labels.csv")
