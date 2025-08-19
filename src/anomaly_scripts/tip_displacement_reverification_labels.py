#  ___________________________________________________________________________
#  Copyright (c) 2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________


"""
Creates "tip_displacement_test_labels.csv"
"""

import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr
from sklearn.metrics import f1_score, precision_score, recall_score

from src.main.config import LOADING_PORTION_END, UNLOADING_PORTION_START
from src.utils.indent_manager import IndentManager
from src.utils.preprocess import ANOMALY_TYPES
from src.utils.setup_data import get_best_threshold, normalize_0_max


def create_tip_displacement_test_labels_csv(datapath, progress):
    im = IndentManager(datapath, ANOMALY_TYPES, progress=progress)

    indent_range = range(LOADING_PORTION_END, UNLOADING_PORTION_START)

    offset_data = []

    task = progress.add_task(
        "[green]Processing samples...", total=len(im.all_sample_nums)
    )
    for i, sample_num in enumerate(im.all_sample_nums):
        for indent_num in range(im.sample_size):
            indent_data = im.get_indent_data(
                sample_num, indent_num, "tip_displacement_decreases_at_holding", False
            )

            # Get time and depth of holding period
            indent = im.indents[(sample_num, indent_num)][np.ix_(indent_range, [2, 0])]
            normalized_indent = normalize_0_max(indent)

            # Get mean last depth of the normalized holding period
            for j in range(1, LOADING_PORTION_END // 2 + 1):
                indent_data[f"normalized_offset_{j}"] = normalized_indent[-j:, 1].mean()

            offset_data.append(indent_data)

        progress.update(task, advance=1)

    offset_df = pd.DataFrame(offset_data)

    corrs = []
    y = offset_df["tip_displacement_decreases_at_holding"].astype(bool)
    for j in range(1, LOADING_PORTION_END // 2 + 1):
        x = offset_df[f"normalized_offset_{j}"].values
        # Normalized offset refers to percentage of how far the last depth point is in holding
        # is compared to the start and max depth in holding. If way less than 0, it can skew
        # the results.
        x[x < 0] = 0

        corr, _ = pointbiserialr(x, y)

        corrs.append(abs(corr))

    corrs = np.array(corrs)
    best_mean_j = corrs.argmax() + 1

    print(f"Selected: normalized_offset_{best_mean_j}")

    # Flag
    # Get best threshold for separating entire dataset
    res = get_best_threshold(
        offset_df[f"normalized_offset_{best_mean_j}"].values,
        [(0.0, 1.0)],
        offset_df["tip_displacement_decreases_at_holding"].astype(bool),
    )
    main_threshold = res.x[0]

    y_true = offset_df["tip_displacement_decreases_at_holding"].astype(bool)
    predictions = offset_df[f"normalized_offset_{best_mean_j}"] < main_threshold
    print(f"Main threshold at {round(main_threshold, 2)} metrics:")
    print(f"f1 score: {f1_score(y_true, predictions)}")
    print(f"recall score: {recall_score(y_true, predictions)}")
    print(f"precision score: {precision_score(y_true, predictions)}")

    # Split dataset based on threshold
    half1 = offset_df[offset_df[f"normalized_offset_{best_mean_j}"] < main_threshold]
    half2 = offset_df[offset_df[f"normalized_offset_{best_mean_j}"] >= main_threshold]

    # Get best threshold for left side of dataset
    res = get_best_threshold(
        half1[f"normalized_offset_{best_mean_j}"].values,
        [(0.0, main_threshold)],
        half1["tip_displacement_decreases_at_holding"].astype(bool),
    )
    half1_threshold = res.x[0]

    # Get best threshold for right side of dataset
    res = get_best_threshold(
        half2[f"normalized_offset_{best_mean_j}"].values,
        [(main_threshold, 1.0)],
        half2["tip_displacement_decreases_at_holding"].astype(bool),
    )
    half2_threshold = res.x[0]

    # Print metrics
    print("Uncertain range:")
    print(half1_threshold, half2_threshold)

    uncertain_mask = (
        half1_threshold < offset_df[f"normalized_offset_{best_mean_j}"]
    ) & (offset_df[f"normalized_offset_{best_mean_j}"] < half2_threshold)

    y_true = offset_df.loc[
        ~uncertain_mask, "tip_displacement_decreases_at_holding"
    ].astype(bool)
    predictions = (
        offset_df.loc[~uncertain_mask, f"normalized_offset_{best_mean_j}"]
        < main_threshold
    )
    print("Main threshold metrics without uncertain range")
    print(f"f1 score: {f1_score(y_true, predictions)}")
    print(f"recall score: {recall_score(y_true, predictions)}")
    print(f"precision score: {precision_score(y_true, predictions)}")

    uncertain_non_anomalous = offset_df.loc[
        uncertain_mask & (offset_df["tip_displacement_decreases_at_holding"] == False)
    ]
    uncertain_anomalous = offset_df.loc[
        uncertain_mask & (offset_df["tip_displacement_decreases_at_holding"] == True)
    ]
    print("Total percent uncertain:")
    print(sum(uncertain_mask) / len(offset_df))
    print("Uncertain non-anomalous percentage:")
    print(
        len(uncertain_non_anomalous)
        / sum(offset_df["tip_displacement_decreases_at_holding"] == False)
    )
    print("Uncertain anomalous percentage:")
    print(
        len(uncertain_anomalous)
        / sum(offset_df["tip_displacement_decreases_at_holding"] == True)
    )

    # Get false positives (initial label was anomalous, but evidence suggests it is non-anomalous)
    false_positives = (
        offset_df[f"normalized_offset_{best_mean_j}"] > half1_threshold
    ) & (offset_df["tip_displacement_decreases_at_holding"] == True)

    # Get false negatives (initial label was non-anomalous, but evidence suggests it is anomalous)
    false_negatives = (
        offset_df[f"normalized_offset_{best_mean_j}"] <= half2_threshold
    ) & (offset_df["tip_displacement_decreases_at_holding"] == False)

    confident_false_positives = (
        offset_df[f"normalized_offset_{best_mean_j}"] > half2_threshold
    ) & (offset_df["tip_displacement_decreases_at_holding"] == True)

    confident_false_negatives = (
        offset_df[f"normalized_offset_{best_mean_j}"] <= half1_threshold
    ) & (offset_df["tip_displacement_decreases_at_holding"] == False)

    reverification_pairs = offset_df.loc[
        false_positives | false_negatives, ["sample_num", "indent_num"]
    ]

    print("Total indents:", len(offset_df))
    print("Total reverification indents:", len(reverification_pairs))
    print("-" * 7)
    print(
        "Total previous anomalous:",
        sum(offset_df["tip_displacement_decreases_at_holding"]),
    )
    print("Total confident false anomalous:", sum(confident_false_positives))
    print(
        "Total confident new anomalous:",
        sum(
            (
                offset_df["tip_displacement_decreases_at_holding"]
                | confident_false_negatives
            )
            & ~confident_false_positives
        ),
    )
    print("-" * 7)
    print(
        "Total previous non anomalous:",
        sum(~offset_df["tip_displacement_decreases_at_holding"]),
    )
    print("Total confident false non anomalous:", sum(confident_false_negatives))
    print(
        "Total confident new non anomalous:",
        sum(
            (
                ~offset_df["tip_displacement_decreases_at_holding"]
                | confident_false_positives
            )
            & ~confident_false_negatives
        ),
    )
    print("-" * 7)
    print("Total uncertain:", sum(uncertain_mask))
    print("Total anomalous in uncertain region:", len(uncertain_anomalous))
    print("Total non anomalous in uncertain region:", len(uncertain_non_anomalous))

    im.export_results(reverification_pairs, "tip_displacement_test_labels.csv")
