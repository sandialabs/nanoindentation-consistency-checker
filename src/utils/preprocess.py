#  ___________________________________________________________________________
#  Copyright (c) 2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________


"""
File for creating indent access
"""

import numpy as np
import pandas as pd

ANOMALY_TYPES = [
    "verify_zero_offset_type",
    "force_offset",
    "displacement_offset",
    "unusual_unloading_curvature",
    "unusual_loading_curvature",
    "tip_displacement_decreases_at_holding",
    "too_deep",
]


class IndentDict(dict):
    """Dictionary where key is (sample_num, indent_num) and return value is determined by function"""

    def __getitem__(self, key):
        item = dict.__getitem__(self, key)
        return item(*key)


def _create_loaded_indents_functions(all_samples_df, full_data, anomaly_types):
    def create_indent(sample_num, indent_num):
        index = np.where(
            (all_samples_df["sample_num"] == sample_num)
            & (all_samples_df["indent_num"] == indent_num)
        )[0][0]
        return full_data[index]

    def create_indent_info(sample_num, indent_num):
        indent_info = (
            all_samples_df.loc[
                (all_samples_df["sample_num"] == sample_num)
                & (all_samples_df["indent_num"] == indent_num),
                anomaly_types,
            ].astype(int)
            != 0
        )

        return indent_info.iloc[0]

    return create_indent, create_indent_info


def create_loaded_indents(
    all_samples_df: pd.DataFrame, full_data: np.ndarray, anomaly_types: list[str]
):
    """
    Create the indents and indent_infos Indent dictionaries

    Args:
        all_samples_df (pd.DataFrame) : DataFrame with all samples. Includes sample_num, indent_num,
            and anomalies
        full_data ([int, int, int]) : List with shape (total_indents, indent_points, dimensions)
        anomaly_types : All anomaly types

    Returns:
        IndentDict ([int, int], [int, int]) : indents - Access with (sample_num, indent_num). All
            the points in an indent.
        IndentDict : [int, int], pd.DataFrame
        indent_infos - Access with (sample_num, indent_num). Values are the anomalies of an indent.
    """
    if not isinstance(all_samples_df, pd.DataFrame):
        raise TypeError("all_samples_df must be pd.DataFrame")
    if not isinstance(full_data, np.ndarray):
        raise TypeError("full_data must be np.ndarray")
    if not isinstance(anomaly_types, list):
        raise TypeError("anomaly_types must be list")
    if any(not isinstance(x, str) for x in anomaly_types):
        raise ValueError("anomaly_types must be list of str")

    sample_sizes = all_samples_df["sample_num"].value_counts().unique()
    if len(sample_sizes) > 1:
        raise ValueError(
            f"All samples should have same amount of indents. Detected {len(sample_sizes)} different sizes."
        )
    all_sample_nums = all_samples_df["sample_num"].unique()

    if not set(anomaly_types).issubset(all_samples_df.columns):
        raise ValueError("anomaly_types should be a subset of all_samples_df.columns")

    if len(full_data) != len(all_samples_df):
        raise ValueError(
            "Full data should have same amount of total indents as all_samples_df"
        )

    indents = IndentDict()
    indent_infos = IndentDict()
    create_indent, create_indent_info = _create_loaded_indents_functions(
        all_samples_df, full_data, anomaly_types
    )
    for sample_num in all_sample_nums:
        for indent_num in all_samples_df.loc[
            all_samples_df["sample_num"] == sample_num, "indent_num"
        ]:
            indents[(sample_num, indent_num)] = create_indent
            indent_infos[(sample_num, indent_num)] = create_indent_info
    return indents, indent_infos
