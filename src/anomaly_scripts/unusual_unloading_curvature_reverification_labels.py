#  ___________________________________________________________________________
#  Copyright (c) 2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________


"""
Creates "unusual_unloading_curvature_test_labels.csv"
"""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from src.main.config import INDENT_END, UNLOADING_PORTION_START
from src.utils.indent_manager import IndentManager
from src.utils.preprocess import ANOMALY_TYPES
from src.utils.setup_data import (
    get_best_threshold,
    get_combined_average,
    pointwise_euclidean,
)


def create_unusual_unloading_curvature_test_labels_csv(datapath, progress):
    im = IndentManager(datapath, ANOMALY_TYPES, progress=progress)

    # Get features
    indent_range = range(UNLOADING_PORTION_START, INDENT_END)

    # Obtain all non anomalous unloading parts of indents and offset them to all start at 0
    all_non_anomalous_examples = im.all_samples_df.loc[
        (im.all_samples_df["fully_anomalous_sample"] == 0)
        & (im.all_samples_df["unusual_unloading_curvature"] == 0),
        ["sample_num", "indent_num"],
    ].values
    all_non_anomalous_indent_nums = all_non_anomalous_examples[:, 1]
    all_non_anomalous_indents = [
        im.indents[(sample_num, indent_num)][indent_range, :2]
        for sample_num, indent_num in all_non_anomalous_examples
    ]
    all_non_anomalous_indents = [
        indent - indent.min(axis=0) for indent in all_non_anomalous_indents
    ]
    all_non_anomalous_indents = np.array(all_non_anomalous_indents)

    # Obtain all anomalous unloading parts of indents and offset them to all start at 0
    all_anomalous_examples = im.all_samples_df.loc[
        (im.all_samples_df["fully_anomalous_sample"] == 0)
        & (im.all_samples_df["unusual_unloading_curvature"] != 0),
        ["sample_num", "indent_num"],
    ].values
    all_anomalous_indent_nums = all_anomalous_examples[:, 1]
    all_anomalous_indents = [
        im.indents[(sample_num, indent_num)][indent_range, :2]
        for sample_num, indent_num in all_anomalous_examples
    ]
    all_anomalous_indents = [
        indent - indent.min(axis=0) for indent in all_anomalous_indents
    ]
    all_anomalous_indents = np.array(all_anomalous_indents)

    # Obtain entire dataset average indents
    (
        all_non_anomalous_indent,
        all_anomalous_indent,
        all_normalized_non_anomalous,
        all_normalized_anomalous_indent,
    ) = get_combined_average(
        -1,
        all_non_anomalous_indents,
        all_anomalous_indents,
        all_non_anomalous_indent_nums,
        all_anomalous_indent_nums,
    )

    all_distance_data = []
    distance_data = []

    task = progress.add_task(
        "[green]Processing samples...", total=len(im.all_sample_nums)
    )
    for i, sample_num in enumerate(im.all_sample_nums):
        # Obtain non anomalous and anomalous indents within the sample
        (
            non_anomalous_indents,
            anomalous_indents,
            non_anomalous_indent_nums,
            anomalous_indent_nums,
        ) = im.get_indent_classifications(
            sample_num, "unusual_unloading_curvature", indent_range
        )
        for indent_num in range(im.sample_size):
            indent_data = im.get_indent_data(
                sample_num, indent_num, "unusual_unloading_curvature", False
            )

            indent, normalized_indent = im.get_offset_indent(
                sample_num, indent_num, indent_range
            )

            # Obtain per sample average indents
            (
                non_anomalous_indent,
                anomalous_indent,
                normalized_non_anomalous,
                normalized_anomalous_indent,
            ) = get_combined_average(
                indent_num,
                non_anomalous_indents,
                anomalous_indents,
                non_anomalous_indent_nums,
                anomalous_indent_nums,
            )

            # Per sample features
            indent_data["non_anomalous_distance"] = pointwise_euclidean(
                indent, non_anomalous_indent
            )
            indent_data["anomalous_distance"] = pointwise_euclidean(
                indent, anomalous_indent
            )
            indent_data["normalized_non_anomalous_distance"] = pointwise_euclidean(
                normalized_indent, normalized_non_anomalous
            )
            indent_data["normalized_anomalous_distance"] = pointwise_euclidean(
                normalized_indent, normalized_anomalous_indent
            )

            distance_data.append(indent_data.copy())

            # Entire dataset features
            indent_data["non_anomalous_distance"] = pointwise_euclidean(
                indent, all_non_anomalous_indent
            )
            indent_data["anomalous_distance"] = pointwise_euclidean(
                indent, all_anomalous_indent
            )
            indent_data["normalized_non_anomalous_distance"] = pointwise_euclidean(
                normalized_indent, all_normalized_non_anomalous
            )
            indent_data["normalized_anomalous_distance"] = pointwise_euclidean(
                normalized_indent, all_normalized_anomalous_indent
            )

            all_distance_data.append(indent_data.copy())

        progress.update(task, advance=1)

    all_distance_df = pd.DataFrame(all_distance_data)
    distance_df = pd.DataFrame(distance_data)

    all_distance_df["anomalous_to_non_anomalous_distance"] = (
        all_distance_df["normalized_anomalous_distance"]
        - all_distance_df["normalized_non_anomalous_distance"]
    )
    distance_df["anomalous_to_non_anomalous_distance"] = (
        distance_df["normalized_anomalous_distance"]
        - distance_df["normalized_non_anomalous_distance"]
    )

    all_distance_df["percent"] = all_distance_df["normalized_anomalous_distance"] / (
        all_distance_df["normalized_anomalous_distance"]
        + all_distance_df["normalized_non_anomalous_distance"]
    )
    distance_df["percent"] = distance_df["normalized_anomalous_distance"] / (
        distance_df["normalized_anomalous_distance"]
        + distance_df["normalized_non_anomalous_distance"]
    )

    # Flag
    # Get best threshold for separating based on entire dataset feature
    res = get_best_threshold(
        all_distance_df["percent"].values,
        [(0.0, 1.0)],
        all_distance_df["unusual_unloading_curvature"].astype(bool),
    )
    all_percent_threshold = res.x[0]

    # Get best threshold for separating based on per sample feature
    non_zero_distance_df = distance_df[
        distance_df["anomalous_to_non_anomalous_distance"] != 0
    ]
    res = get_best_threshold(
        non_zero_distance_df["percent"].values,
        [(0.0, 1.0)],
        non_zero_distance_df["unusual_unloading_curvature"].astype(bool),
    )
    percent_threshold = res.x[0]

    print(f"Percent threshold: {percent_threshold}")
    print(f"All percent threshold: {all_percent_threshold}")

    # Use per sample feature for indents with nonequal distance to sample averages
    zero_mask = distance_df["anomalous_to_non_anomalous_distance"] == 0
    predictions = distance_df["percent"] < percent_threshold
    # Use entire dataset feature for indents with equal distance to sample averages
    predictions[zero_mask] = (
        all_distance_df.loc[zero_mask, "percent"] < all_percent_threshold
    ).tolist()

    n = len(distance_df)
    false_positives = ~predictions & distance_df["unusual_unloading_curvature"]
    false_negatives = predictions & ~distance_df["unusual_unloading_curvature"]

    print(f"Total indents: {n}")
    print(f"Labeled as Anomalous but above threshold: {sum(false_positives)}")
    print(f"Labeled as Non-Anomalous but below threshold: {sum(false_negatives)}")

    y_true = distance_df["unusual_unloading_curvature"].astype(bool)
    predictions = predictions.astype(bool)
    print(f"f1 score: {f1_score(y_true, predictions)}")
    print(f"recall score: {recall_score(y_true, predictions)}")
    print(f"precision score: {precision_score(y_true, predictions)}")

    reverification_pairs = distance_df.loc[
        predictions != y_true, ["sample_num", "indent_num"]
    ]

    print("Total indents:", len(distance_df))
    print("Total reverification indents:", len(reverification_pairs))
    print("-" * 7)
    print("Total previous anomalous:", sum(distance_df["unusual_unloading_curvature"]))
    print("Total false anomalous:", sum(false_positives))
    print(
        "Total new anomalous:",
        sum(
            (distance_df["unusual_unloading_curvature"] | false_negatives)
            & ~false_positives
        ),
    )
    print("-" * 7)
    print(
        "Total previous non anomalous:",
        sum(~distance_df["unusual_unloading_curvature"]),
    )
    print("Total false non anomalous:", sum(false_negatives))
    print(
        "Total new non anomalous:",
        sum(
            (~distance_df["unusual_unloading_curvature"] | false_positives)
            & ~false_negatives
        ),
    )

    im.export_results(
        reverification_pairs, "unusual_unloading_curvature_test_labels.csv"
    )
