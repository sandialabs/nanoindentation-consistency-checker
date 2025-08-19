#  ___________________________________________________________________________
#  Copyright (c) 2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________


"""
Creates "force_offset_test_labels.csv"
"""

import os

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler as SS
from sklearn.svm import SVC

from src.main.config import LOADING_PORTION_END
from src.utils.indent_manager import IndentManager
from src.utils.preprocess import ANOMALY_TYPES


def create_force_offset_test_labels_csv(datapath, progress):
    im = IndentManager(datapath, ANOMALY_TYPES, progress=progress)
    full_data = np.load(os.path.join(datapath, "full_data.npy"))

    good_sample_ind = list(im.good_samples_df.index)

    # Get the mean and standard deviation for the first sets of depth and load displacement
    # for the samples that aren't "fully anomalous"
    zero_offset_numbers_data = {
        "sample_num": im.good_samples_df["sample_num"],
        "indent_num": im.good_samples_df["indent_num"],
    }
    init_values = np.arange(1, LOADING_PORTION_END // 2 + 1, dtype=int)
    task = progress.add_task("[green]Processing samples...", total=len(init_values))
    for init in init_values:
        meanvals = np.mean(full_data[good_sample_ind, :init], axis=1)
        stdvals = np.std(full_data[good_sample_ind, :init], axis=1)
        zero_offset_numbers_data[f"mean_{init}_depth_nm"] = meanvals[:, 0]
        zero_offset_numbers_data[f"std_{init}_depth_nm"] = stdvals[:, 0]
        zero_offset_numbers_data[f"mean_{init}_load_micro_N"] = meanvals[:, 1]
        zero_offset_numbers_data[f"std_{init}_load_micro_N"] = stdvals[:, 1]
        progress.update(task, advance=1)

    zero_offset_numbers = pd.DataFrame(zero_offset_numbers_data)

    # Standard scale before SVM
    scaler = SS()
    scaled_vals = scaler.fit_transform(zero_offset_numbers.iloc[:, 2:])
    zero_offset_numbers.iloc[:, 2:] = scaled_vals

    # Obtain the true force_offset labels,
    # assuming here that verify zero offset type refers to a force offset
    true_labels = 1 * np.array(
        (
            im.good_samples_df["verify_zero_offset_type"]
            + im.good_samples_df["force_offset"]
        )
        > 0
    )

    # Set to the value that maximizes the average f-score and roc-auc, X can of course be manipulated to
    # any combination of standard deviations or average values.
    load_std_fscores = []
    load_std_auc = []
    task = progress.add_task("[green]Fitting model...", total=len(init_values))
    for init in np.arange(1, LOADING_PORTION_END // 2 + 1, dtype=int):
        # Get the std vales
        X = np.array(zero_offset_numbers[f"std_{init}_load_micro_N"]).reshape(-1, 1)

        model = SVC(kernel="linear")
        model.fit(X, true_labels)
        pred_labels = model.predict(X)

        load_std_fscores.append(f1_score(true_labels, pred_labels))
        load_std_auc.append(roc_auc_score(true_labels, pred_labels))
        progress.update(task, advance=1)

    load_std_fscores = np.array(load_std_fscores)
    load_std_auc = np.array(load_std_auc)

    avg_fscores_auc = (load_std_fscores + load_std_auc) / 2

    BEST_INIT = avg_fscores_auc.argmax() + 1
    print(f"Selected: std_{BEST_INIT}_load_micro_N")

    X = np.array(zero_offset_numbers[f"std_{BEST_INIT}_load_micro_N"]).reshape(-1, 1)

    # Train the SVM
    model = SVC(kernel="linear")
    model.fit(X, true_labels)
    pred_labels = model.predict(X)

    # Obtain the dataframe of points that need to be reverified for their force offset label
    reverification_pairs = im.good_samples_df.loc[
        true_labels != pred_labels, ["sample_num", "indent_num"]
    ]

    print("Total indents:", len(true_labels))
    print("Total reverification indents:", len(reverification_pairs))
    print("-" * 7)
    print("Total previous anomalous:", sum(true_labels))
    print("Total false anomalous:", sum((true_labels == 1) & (pred_labels == 0)))
    print("Total new anomalous:", sum(pred_labels))
    print("-" * 7)
    print("Total previous non anomalous:", sum((true_labels == 0)))
    print("Total false non anomalous:", sum((true_labels == 0) & (pred_labels == 1)))
    print("Total new non anomalous:", sum((pred_labels == 0)))

    im.export_results(reverification_pairs, "force_offset_test_labels.csv")
