#  ___________________________________________________________________________
#  Copyright (c) 2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________


"""
Moving common code from different setup files
into this file for centralization
"""

import os
import warnings

import numpy as np
from sklearn.metrics import fbeta_score
from skopt import gp_minimize

from src.main.config import LOADING_PORTION_END


def verify_indent(indent: np.ndarray):
    """
    Verifies that indent is correct type and shape

    Args:
        indent : numpy array of size (points x features)
    """
    if not isinstance(indent, np.ndarray):
        raise TypeError("indent must be np.array")
    if indent.ndim != 2:
        raise ValueError("indent must be 2 dimensions (points x features)")


def normalize(indent: np.ndarray):
    """
    Min-max normalization

    Args:
        indent : numpy array of size (points x features)

    Returns:
        Min-max normalized indent on each feature
    """
    verify_indent(indent)

    return np.nan_to_num(
        (indent - indent.min(axis=0)) / (indent.max(axis=0) - indent.min(axis=0)), nan=0
    )


def normalize_0_max(indent: np.ndarray):
    """
    Min-max normalization on indent with minimum replaced with first point of indent
    Indents start at (0, 0) and their maximum is (1, 1). Minimum can be lower than 0.

    Args:
        indent : numpy array of size (points x features)

    Returns:
        Min-max normalized indent on first point
    """
    verify_indent(indent)

    return np.nan_to_num(
        (indent - indent[0, :]) / (indent.max(axis=0) - indent[0, :]), nan=0
    )


def normalize_on_other_indent(indent: np.ndarray, reference_indent: np.ndarray):
    """
    Min-max normalization on indent based on a reference indent

    Args:
        indent : numpy array of size (points x features)
        reference_indent : numpy array of size (points x features)

    Returns:
        Min-max normalized indent on referenced indent
    """
    verify_indent(indent)
    verify_indent(reference_indent)

    return np.nan_to_num(
        (indent - reference_indent.min(axis=0))
        / (reference_indent.max(axis=0) - reference_indent.min(axis=0)),
        nan=0,
    )


def get_loading(indent):
    """Returns loading portion of indent"""
    verify_indent(indent)

    return indent[:LOADING_PORTION_END]


def pointwise_euclidean(indent_a, indent_b):
    """Calculates the sum of pointwise euclidean distance between two arrays"""
    verify_indent(indent_a)
    verify_indent(indent_b)

    return np.sum(np.linalg.norm(indent_a[:, :2] - indent_b[:, :2], axis=1))


def max_pointwise_euclidean(indent_a, indent_b):
    """Calculates the maximum pointwise euclidean distance between two arrays"""
    verify_indent(indent_a)
    verify_indent(indent_b)

    return max(np.linalg.norm(indent_a[:, :2] - indent_b[:, :2], axis=1))


def compute_triangle_area(points):
    """
    Returns array of curvatures at every point from 1 to n-1
    """
    verify_indent(points)

    curvatures = []
    for i in range(1, len(points) - 1):
        p = np.array(points[[0, i, -1]])

        area = (1 / 2) * abs(
            (p[0][0] - p[2][0]) * (p[1][1] - p[0][1])
            - (p[0][0] - p[1][0]) * (p[2][1] - p[0][1])
        )

        curvatures.append(area)

    return np.array(curvatures)


def get_combined_average(
    indent_num: int,
    non_anomalous_indents: np.ndarray,
    anomalous_indents: np.ndarray,
    non_anomalous_indent_nums: np.ndarray,
    anomalous_indent_nums: np.ndarray,
):
    """
    Return averaged non anomalous and anomalous indents within a sample without including
    indent_num.
    Also return the average normalzied version of these indents.

    Args:
        indent_num : Indent number
        the rest of the args : from
            get_indent_classiciations(sample_num, anomaly_type, indent_range):

    Returns:
        non_anomalous_indent, anomalous_indent, normalized_non_anomalous_indent, \
        normalized_anomalous_indent
            If there's no anomalous indents, then the anomalous indents will equal normal indents
    """
    if not isinstance(indent_num, (int, np.integer)):
        raise TypeError("indent_num must be int")
    if not isinstance(non_anomalous_indents, np.ndarray):
        raise TypeError("non_anomalous_indents must be np.ndarray")
    if not isinstance(anomalous_indents, np.ndarray):
        raise TypeError("anomalous_indents must be np.ndarray")
    if not isinstance(non_anomalous_indent_nums, np.ndarray):
        raise TypeError("non_anomalous_indent_nums must be np.ndarray")
    if not isinstance(anomalous_indent_nums, np.ndarray):
        raise TypeError("anomalous_indent_nums must be np.ndarray")

    if len(non_anomalous_indents) != len(non_anomalous_indent_nums):
        raise ValueError(
            "Size of non_anomalous_indents and size of non_anomalous_indent_nums should match"
        )
    if len(anomalous_indents) != len(anomalous_indent_nums):
        raise ValueError(
            "Size of anomalous_indents and size of anomalous_indent_nums should match"
        )

    index = np.where(non_anomalous_indent_nums == indent_num)[0]
    if len(index) != 0:
        index = index[0]
        non_anomalous_indents = np.delete(non_anomalous_indents, index, axis=0)
    else:
        index = np.where(anomalous_indent_nums == indent_num)[0]
        if len(index) != 0:
            index = index[0]
            anomalous_indents = np.delete(anomalous_indents, index, axis=0)

    if len(non_anomalous_indents) == 0 and len(anomalous_indents) == 0:
        raise ValueError("No indents remaining. Unable to return average of nothing")

    if len(non_anomalous_indents) != 0:
        non_anomalous_indent = non_anomalous_indents.mean(axis=0)
        normalized_non_anomalous_indent = np.array(
            [normalize(indent) for indent in non_anomalous_indents]
        ).mean(axis=0)
    if len(anomalous_indents) != 0:
        anomalous_indent = anomalous_indents.mean(axis=0)
        normalized_anomalous_indent = np.array(
            [normalize(indent) for indent in anomalous_indents]
        ).mean(axis=0)

    if len(non_anomalous_indents) == 0:
        non_anomalous_indent = anomalous_indent
        normalized_non_anomalous_indent = normalized_anomalous_indent
    if len(anomalous_indents) == 0:
        anomalous_indent = non_anomalous_indent
        normalized_anomalous_indent = normalized_non_anomalous_indent

    return (
        non_anomalous_indent,
        anomalous_indent,
        normalized_non_anomalous_indent,
        normalized_anomalous_indent,
    )


def get_best_threshold(
    array: np.ndarray,
    dims: list[tuple],
    y_true: list[bool],
    n_calls=75,
    beta=1,
    direction="left",
):
    """
    Returns threshold that maximizes f1 score over an array

    Args:
        array : Array of numbers
        dims : List of search space dimensions [(lower limit, upper limit)]
        y_true : boolean array
        n_calls : number of calls to adjust threshold

    Returns:
        OptimizeResult
    """
    if not isinstance(array, np.ndarray):
        raise TypeError("array must be np.ndarray")
    if not isinstance(dims, list):
        raise TypeError("dims must be list")
    if not isinstance(n_calls, (int, np.integer)):
        raise TypeError("n_calls must be int")

    if not isinstance(dims[0], tuple) or len(dims) > 1 or len(dims[0]) != 2:
        raise ValueError("dims must be list of format [(lower limit, upper limit)]")

    try:
        if len(y_true) != len(array):
            raise ValueError("y_true must be same size as array")
        if any(not isinstance(x, bool) for x in y_true):
            raise ValueError("y_true must be list of boolean")
    except TypeError:
        raise TypeError("y_true must be list")

    if direction == "left":

        def get_f1_score(t):
            t = t[0]
            y_pred = array <= t
            return -fbeta_score(y_true, y_pred, beta=beta)

    else:

        def get_f1_score(t):
            t = t[0]
            y_pred = array >= t
            return -fbeta_score(y_true, y_pred, beta=beta)

    with warnings.catch_warnings(record=True):
        res = gp_minimize(get_f1_score, dims, n_calls=n_calls, random_state=29)
    return res


def verify_anomaly_type(anomaly_type: str, include_all=False):
    anomaly_types = [
        "displacement_offset",
        "force_offset",
        "tip_displacement",
        "too_deep",
        "unusual_loading_curvature",
        "unusual_unloading_curvature",
    ]
    if include_all:
        anomaly_types.insert(0, "all")

    if anomaly_type not in anomaly_types:
        raise ValueError(
            "anomaly_type must be one of the following:\n{}".format(
                "\n".join(anomaly_types)
            )
        )


def verify_datapath(datapath: str):
    """
    Verifies that datapath contains required files and raises exceptions if not

    Args:
        datapath : path to data that contains "all_samples_labelled.csv" and "full_data.npy" (or
        "Nanoindent_data/" to generate "full_data.npy")
    """
    if not os.path.isdir(datapath):
        raise NotADirectoryError(f"{datapath} is not a directory")

    if not os.path.isfile(os.path.join(datapath, "all_samples_labelled.csv")):
        raise FileNotFoundError(
            f"{os.path.join(datapath, 'all_samples_labelled.csv')} is not found"
        )

    if not os.path.isfile(
        os.path.join(datapath, "full_data.npy")
    ) and not os.path.isdir(os.path.join(datapath, "Nanoindent_data")):
        raise FileNotFoundError(
            f"{os.path.join(datapath, 'full_data.npy')} or {os.path.join(datapath, 'Nanoindent_data')} is required"
        )


class CustomFileError(FileNotFoundError):
    def __init__(self, err: FileNotFoundError, filename: str):
        abspath = os.path.abspath(filename)
        message = str(err)
        message += "\n"
        message += f"The file, {abspath}, can not be saved likely because the length of the path ({len(abspath)}) exceeds the max path length."
        message += "\n"
        message += "On Windows, the max path length is 260."
        super().__init__(message)
