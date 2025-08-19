#  ___________________________________________________________________________
#  Copyright (c) 2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________


"""
Creates "unusual_loading_curvature_test_labels.csv"
"""

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier

from src.utils.indent_manager import IndentManager
from src.utils.preprocess import ANOMALY_TYPES
from src.utils.setup_data import (
    compute_triangle_area,
    get_best_threshold,
    get_loading,
    max_pointwise_euclidean,
    normalize,
    pointwise_euclidean,
)


def create_unusual_loading_curvature_test_labels_csv(datapath, progress):
    im = IndentManager(datapath, ANOMALY_TYPES, progress=progress)

    loading_data = []

    task = progress.add_task(
        "[green]Processing samples...", total=len(im.all_sample_nums)
    )
    for i, sample_num in enumerate(im.all_sample_nums):
        avg_indent_loading = im.get_non_anomalous_average(sample_num, get_loading)
        avg_norm_indent_loading = im.get_non_anomalous_average(
            sample_num, lambda x: normalize(get_loading(x))
        )

        for indent_num in range(im.sample_size):
            indent_data = im.get_indent_data(
                sample_num, indent_num, "unusual_loading_curvature", True
            )

            indent = im.indents[(sample_num, indent_num)][:, :2]
            loading_indent = get_loading(indent)

            loading_distance = pointwise_euclidean(avg_indent_loading, loading_indent)
            indent_data["loading_distance"] = loading_distance
            max_loading_distance = max_pointwise_euclidean(
                avg_indent_loading, loading_indent
            )
            indent_data["max_loading_distance"] = max_loading_distance

            # Normalize each indent on its sample's average
            loading_indent = normalize(loading_indent)
            normalized_loading_distance = pointwise_euclidean(
                avg_norm_indent_loading, loading_indent
            )
            indent_data["normalized_loading_distance"] = normalized_loading_distance
            max_normalized_loading_distance = max_pointwise_euclidean(
                avg_norm_indent_loading, loading_indent
            )
            indent_data["max_normalized_loading_distance"] = (
                max_normalized_loading_distance
            )

            curvatures = compute_triangle_area(loading_indent[:, :2])
            indent_data["max_curvature"] = curvatures.max()
            indent_data["index_of_max_curvature"] = curvatures.argmax()
            indent_data["depth_at_max_curvature"] = loading_indent[
                curvatures.argmax(), 0
            ]
            indent_data["load_at_max_curvature"] = loading_indent[
                curvatures.argmax(), 1
            ]

            normalized_curvatures = compute_triangle_area(
                normalize(loading_indent[:, :2])
            )
            indent_data["normalized_max_curvature"] = normalized_curvatures.max()
            indent_data["normalized_index_of_max_curvature"] = (
                normalized_curvatures.argmax()
            )

            loading_data.append(indent_data)

        progress.update(task, advance=1)

    loading_df = pd.DataFrame(loading_data)

    # Flag
    excluded_mask = loading_df["unusual_loading_curvature"] | ~loading_df["any"]
    excluded_loading_df = loading_df.loc[excluded_mask]

    all_features = [
        "loading_distance",
        "max_loading_distance",
        "normalized_loading_distance",
        "max_normalized_loading_distance",
        "max_curvature",
        "index_of_max_curvature",
        "depth_at_max_curvature",
        "load_at_max_curvature",
        "normalized_max_curvature",
        "normalized_index_of_max_curvature",
    ]

    X = excluded_loading_df.loc[:, all_features]
    y_true = excluded_loading_df.loc[:, "unusual_loading_curvature"].astype(bool)

    print("Precision decision tree")
    # CONSTANT: 4 Max leaf nodes to not encourage overfitting.
    clf = DecisionTreeClassifier(max_leaf_nodes=4, random_state=5)
    clf.fit(X, y_true)
    predictions = clf.predict(X)
    print(f"f1 score: {f1_score(y_true, predictions)}")
    print(f"recall score: {recall_score(y_true, predictions)}")
    print(f"precision score: {precision_score(y_true, predictions)}")
    false_negatives_mask = (predictions == True) & (y_true == False)
    print(f"False negatives: {sum(false_negatives_mask)}")
    print("-" * 7)

    print("Recall decision tree")
    # CONSTANT: 5 Max leaf nodes to not encourage overfitting. One extra max leaf node because
    # precision needed to be increased.
    # There are very few unusual loading curvature anomalies in our dataset, so setting class weight
    # to balanced will increase recall.
    clf = DecisionTreeClassifier(
        max_leaf_nodes=5, random_state=5, class_weight="balanced"
    )
    clf.fit(X, y_true)
    predictions = clf.predict(X)
    print(f"f1 score: {f1_score(y_true, predictions)}")
    print(f"recall score: {recall_score(y_true, predictions)}")
    print(f"precision score: {precision_score(y_true, predictions)}")
    false_positives_mask = (predictions == False) & (y_true == True)
    print(f"False positives: {sum(false_positives_mask)}")
    print("-" * 7)

    # CONSTANT: 3 for beta to place more importance on recall. Higher recall causes less false
    # positives for true labels.
    res = get_best_threshold(
        excluded_loading_df["max_curvature"].values,
        [
            (
                excluded_loading_df["max_curvature"].min(),
                excluded_loading_df["max_curvature"].max(),
            )
        ],
        y_true,
        beta=3,
        direction="right",
    )
    fp_t = res.x[0]
    print(f"Max curvature threshold based on f3 score: {fp_t}")

    # False positives
    predictions = excluded_loading_df["max_curvature"] > fp_t
    visual_false_positives_mask = (predictions == False) & (y_true == True)
    print(f"Threshold false positives: {sum(visual_false_positives_mask)}")
    print("-" * 7)

    # CONSTANT: 3 for number of standard deviations. Focused on precision to decrease number of
    # false negatives for true labels.
    # Based on median instead of mean because outliers heavily skew mean.
    norm_max_curve_t = (
        excluded_loading_df["normalized_max_curvature"].median()
        + 3 * excluded_loading_df["normalized_max_curvature"].std()
    )
    print(
        f"Normalized max curvature threshold based on 3 standard deviations away from median: {norm_max_curve_t}"
    )

    # CONSTANT: 1/2 for beta to place more importance on precision. Higher precision causes less
    # false negatives for true labels.
    res = get_best_threshold(
        excluded_loading_df["normalized_loading_distance"].values,
        [
            (
                excluded_loading_df["normalized_loading_distance"].min(),
                excluded_loading_df["normalized_loading_distance"].max(),
            )
        ],
        y_true,
        beta=(1 / 2),
        direction="right",
    )
    fn_t = res.x[0]
    print(f"Normalized loading distance threshold based on f(1/2) score: {fn_t}")

    # False negatives
    predictions = (
        excluded_loading_df["normalized_max_curvature"] > norm_max_curve_t
    ) | (excluded_loading_df["normalized_loading_distance"] > fn_t)
    visual_false_negatives_mask = (predictions == True) & (y_true == False)
    print(f"Threshold false negatives: {sum(visual_false_negatives_mask)}")
    print("-" * 7)

    false_positives_mask |= visual_false_positives_mask
    print(f"Total false positives: {sum(false_positives_mask)}")
    false_negatives_mask |= visual_false_negatives_mask
    print(f"Total false negatives: {sum(false_negatives_mask)}")

    reverification_pairs = excluded_loading_df.loc[
        false_positives_mask | false_negatives_mask, ["sample_num", "indent_num"]
    ]

    print(
        "Total indents (only including unusual loading curvature and non-anomalous indents):",
        len(excluded_loading_df),
    )
    print("Total reverification indents:", len(reverification_pairs))
    print("-" * 7)
    print(
        "Total previous anomalous:",
        sum(excluded_loading_df["unusual_loading_curvature"]),
    )
    print("Total false anomalous:", sum(false_positives_mask))
    print(
        "Total new anomalous:",
        sum(
            (excluded_loading_df["unusual_loading_curvature"] | false_negatives_mask)
            & ~false_positives_mask
        ),
    )
    print("-" * 7)
    print(
        "Total previous non anomalous:",
        sum(~excluded_loading_df["unusual_loading_curvature"]),
    )
    print("Total false non anomalous:", sum(false_negatives_mask))
    print(
        "Total new non anomalous:",
        sum(
            (~excluded_loading_df["unusual_loading_curvature"] | false_positives_mask)
            & ~false_negatives_mask
        ),
    )

    im.export_results(reverification_pairs, "unusual_loading_curvature_test_labels.csv")
