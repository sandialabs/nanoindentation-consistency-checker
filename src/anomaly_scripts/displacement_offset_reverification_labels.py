#  ___________________________________________________________________________
#  Copyright (c) 2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________


"""
Creates "displacement_offset_test_labels.csv"
"""

import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr
from sklearn import svm
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.utils import compute_class_weight

from src.main.config import LOADING_PORTION_END
from src.utils.indent_manager import IndentManager
from src.utils.preprocess import ANOMALY_TYPES
from src.utils.setup_data import normalize


def create_displacement_offset_test_labels_csv(datapath, progress):
    """Creates "displacement_offset_test_labels.csv"""

    indent_range = range(0, LOADING_PORTION_END)

    im = IndentManager(datapath, ANOMALY_TYPES, progress=progress)

    offset_data = []

    task = progress.add_task(
        "[green]Processing samples...", total=len(im.all_sample_nums)
    )
    for i, sample_num in enumerate(im.all_sample_nums):

        for indent_num in range(im.sample_size):
            indent_data = im.get_indent_data(
                sample_num, indent_num, "displacement_offset", False
            )

            indent = im.indents[(sample_num, indent_num)][indent_range]

            indent_data["start depth"] = indent[0, 0]
            for j in range(2, LOADING_PORTION_END // 2 + 1):
                indent_data[f"depth std {j}"] = indent[:j, 0].std()

            normalized_indent = normalize(indent)

            diff = normalized_indent[1:] - normalized_indent[:-1]
            diff_change = diff[1:] - diff[:-1]

            min_depth_diff = diff_change[:, 0].argmin()
            max_load_diff = diff_change[:, 1].argmax()
            # If this feature is 0, the max second derivative of both load and depth are at the same
            # point, meaning there is an elbow there
            indent_data["index difference"] = abs(max_load_diff - min_depth_diff)

            # Max "velocity"
            indent_data["max depth diff"] = diff[:, 0].max()

            # Max "change in direction"
            indent_data["max depth diff change"] = diff_change[:, 0].min()

            offset_data.append(indent_data)

        progress.update(task, advance=1)

    offset_df = pd.DataFrame(offset_data)

    corrs = []
    y = offset_df["displacement_offset"].astype(bool)
    for j in range(2, LOADING_PORTION_END // 2 + 1):
        x = offset_df[f"depth std {j}"].values

        corr, _ = pointbiserialr(x, y)

        corrs.append(corr)

    corrs = np.array(corrs)
    best_std_j = corrs.argmax() + 2

    # Flag
    features = [
        "max depth diff",
        f"depth std {best_std_j}",
        "max depth diff change",
        "start depth",
    ]
    print(f"Features: {features}")

    X = offset_df.loc[:, features]
    X = StandardScaler().fit_transform(X)
    y = offset_df.loc[:, "displacement_offset"].astype(bool)

    class_weights = compute_class_weight("balanced", classes=np.unique(y), y=y)

    # CONSTANT: (1 / 6) Ensures that too much weight isn't focused on anomaly.
    # Class weight for our dataset (~1.6% displacement offset anomalies) should be around 1:10
    class_weights_high_recall = {0: 1, 1: class_weights[1] / class_weights[0] * (1 / 6)}

    # For high recall
    clf = svm.SVC(class_weight=class_weights_high_recall)
    clf.fit(X, y)
    predictions = clf.predict(X)
    print("Recall model")
    print(f"f1 score: {f1_score(y, predictions)}")
    print(f"recall score: {recall_score(y, predictions)}")
    print(f"precision score: {precision_score(y, predictions)}")

    false_positives = (predictions == False) & (
        offset_df["displacement_offset"] == True
    )
    print(f"{sum(false_positives)} false positives detected")

    # For high precision
    # CONSTANT: (1 / 6) Ensures that too much weight isn't focused on anomaly.
    # Class weight for our dataset (~1.6% displacement offset anomalies) should be around 1:5
    class_weights_high_precision = {
        0: 1,
        1: class_weights[1] / class_weights[0] * (1 / 12),
    }
    clf = svm.SVC(class_weight=class_weights_high_precision)
    clf.fit(X, y)
    predictions = clf.predict(X)
    print("-" * 7)
    print("Precision model")
    print(f"f1 score: {f1_score(y, predictions)}")
    print(f"recall score: {recall_score(y, predictions)}")
    print(f"precision score: {precision_score(y, predictions)}")

    false_negatives = (predictions == True) & (
        offset_df["displacement_offset"] == False
    )
    print(f"{sum(false_negatives)} false negatives detected")

    # False negatives based on visually inspecting data
    # CONSTANT: -10 Any negative number should work. Dependent how sensitive you want detection.
    visual_false_negatives = (
        ~offset_df["displacement_offset"]
        & (offset_df["start depth"] < -10)
        & (offset_df["index difference"] == 0)
    )
    false_negatives = false_negatives | visual_false_negatives
    print(f"{sum(visual_false_negatives)} visual false negatives detected")
    print(f"{sum(false_negatives)} total false negatives detected")

    reverification_pairs = offset_df.loc[
        false_positives | false_negatives, ["sample_num", "indent_num"]
    ]

    print("Total indents:", len(offset_df))
    print("Total reverification indents:", len(reverification_pairs))
    print("-" * 7)
    print("Total previous anomalous:", sum(offset_df["displacement_offset"]))
    print("Total false anomalous:", sum(false_positives))
    print(
        "Total new anomalous:",
        sum((offset_df["displacement_offset"] | false_negatives) & ~false_positives),
    )
    print("-" * 7)
    print("Total previous non anomalous:", sum(~offset_df["displacement_offset"]))
    print("Total false non anomalous:", sum(false_negatives))
    print(
        "Total new non anomalous:",
        sum((~offset_df["displacement_offset"] | false_positives) & ~false_negatives),
    )

    im.export_results(reverification_pairs, "displacement_offset_test_labels.csv")
