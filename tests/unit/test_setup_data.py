#  ___________________________________________________________________________
#  Copyright (c) 2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________


"""
Unit tests for utils/setup_data.py
"""

import unittest

import numpy as np
import pytest

import src.utils.setup_data as setup_data


class TestVerifyIndent(unittest.TestCase):

    def test_wrong_argument_type(self):
        with pytest.raises(TypeError):
            setup_data.verify_indent(0)

    def test_correct_usage(self):
        two_dim_array = np.ones((10, 5))
        setup_data.verify_indent(two_dim_array)

    def test_too_few_dimensions(self):
        one_dim_array = np.ones(20)
        with pytest.raises(
            ValueError, match=r"indent must be 2 dimensions \(points x features\)"
        ):
            setup_data.verify_indent(one_dim_array)

    def test_too_many_dimensions(self):
        three_dim_array = np.ones((20, 5, 7))
        with pytest.raises(
            ValueError, match=r"indent must be 2 dimensions \(points x features\)"
        ):
            setup_data.verify_indent(three_dim_array)


class TestGetBestThreshold(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.array = np.array([0, 0.5, 3])
        cls.dims = [(0.0, 1.0)]
        cls.y_true = [True, False, False]
        cls.n_calls = 10

    def test_wrong_argument_type(self):
        with pytest.raises(TypeError, match="array must be np.ndarray"):
            setup_data.get_best_threshold(0, self.dims, self.y_true, self.n_calls)
        with pytest.raises(TypeError, match="array must be np.ndarray"):
            setup_data.get_best_threshold([2, 3], self.dims, self.y_true, self.n_calls)
        with pytest.raises(TypeError, match="dims must be list"):
            setup_data.get_best_threshold(self.array, 0, self.y_true, self.n_calls)
        with pytest.raises(TypeError, match="y_true must be list"):
            setup_data.get_best_threshold(self.array, self.dims, 0, self.n_calls)
        with pytest.raises(TypeError, match="n_calls must be int"):
            setup_data.get_best_threshold(self.array, self.dims, self.y_true, "test")

    def test_correct_usage(self):
        setup_data.get_best_threshold(self.array, self.dims, self.y_true, self.n_calls)

    def test_incorrect_dim_format(self):
        with pytest.raises(
            ValueError,
            match=r"dims must be list of format \[\(lower limit, upper limit\)\]",
        ):
            setup_data.get_best_threshold(self.array, [1], self.y_true, self.n_calls)
        with pytest.raises(
            ValueError,
            match=r"dims must be list of format \[\(lower limit, upper limit\)\]",
        ):
            setup_data.get_best_threshold(
                self.array, [(1, 3, 5)], self.y_true, self.n_calls
            )

    def test_wrong_y_true_array_value_type(self):
        with pytest.raises(ValueError, match="y_true must be list of boolean"):
            setup_data.get_best_threshold(
                self.array, self.dims, [3, 4, 5], self.n_calls
            )

    def test_wrong_y_true_array_length(self):
        with pytest.raises(ValueError, match="y_true must be same size as array"):
            setup_data.get_best_threshold(
                self.array, self.dims, [True, True, False, False], self.n_calls
            )


class TestGetCombinedAverage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.indent_num = 1
        cls.non_anomalous_indents = np.array([np.ones((10, 5))] * 10)
        cls.anomalous_indents = np.array([np.ones((10, 5))] * 2)
        cls.non_anomalous_indent_nums = np.array([1, 2, 3, 4, 5, 6, 8, 9, 10, 11])
        cls.anomalous_indent_nums = np.array([7, 12])

    def test_wrong_argument_type(self):
        with pytest.raises(TypeError):
            setup_data.get_combined_average(
                "test",
                self.non_anomalous_indents,
                self.anomalous_indents,
                self.non_anomalous_indent_nums,
                self.anomalous_indent_nums,
            )
        with pytest.raises(TypeError):
            setup_data.get_combined_average(
                self.indent_num,
                "test",
                self.anomalous_indents,
                self.non_anomalous_indent_nums,
                self.anomalous_indent_nums,
            )
        with pytest.raises(TypeError):
            setup_data.get_combined_average(
                self.indent_num,
                self.non_anomalous_indents,
                "test",
                self.non_anomalous_indent_nums,
                self.anomalous_indent_nums,
            )
        with pytest.raises(TypeError):
            setup_data.get_combined_average(
                self.indent_num,
                self.non_anomalous_indents,
                self.anomalous_indents,
                "test",
                self.anomalous_indent_nums,
            )
        with pytest.raises(TypeError):
            setup_data.get_combined_average(
                self.indent_num,
                self.non_anomalous_indents,
                self.anomalous_indents,
                self.non_anomalous_indent_nums,
                "test",
            )

    def test_correct_usage(self):
        setup_data.get_combined_average(
            self.indent_num,
            self.non_anomalous_indents,
            self.anomalous_indents,
            self.non_anomalous_indent_nums,
            self.anomalous_indent_nums,
        )

    def test_correct_usage_non_existent_indent_num(self):
        setup_data.get_combined_average(
            -1,
            self.non_anomalous_indents,
            self.anomalous_indents,
            self.non_anomalous_indent_nums,
            self.anomalous_indent_nums,
        )

    def test_correct_usage_single_indent(self):
        setup_data.get_combined_average(
            1,
            np.array([np.ones((10, 5))]),
            np.array([np.ones((10, 5))] * 3),
            np.array([1]),
            np.array([2, 3, 4]),
        )
        setup_data.get_combined_average(
            1,
            np.array([np.ones((10, 5))] * 3),
            np.array([np.ones((10, 5))]),
            np.array([2, 3, 4]),
            np.array([1]),
        )

    def test_non_matching_non_anomalous_arrays(self):
        with pytest.raises(
            ValueError,
            match="Size of non_anomalous_indents and size of non_anomalous_indent_nums should match",
        ):
            setup_data.get_combined_average(
                self.indent_num,
                self.non_anomalous_indents,
                self.anomalous_indents,
                np.array([1, 2, 3]),
                self.anomalous_indent_nums,
            )

    def test_non_matching_anomalous_arrays(self):
        with pytest.raises(
            ValueError,
            match="Size of anomalous_indents and size of anomalous_indent_nums should match",
        ):
            setup_data.get_combined_average(
                self.indent_num,
                self.non_anomalous_indents,
                self.anomalous_indents,
                self.non_anomalous_indent_nums,
                np.array([1, 2, 3]),
            )

    def test_empty_arrays(self):
        with pytest.raises(
            ValueError,
            match="No indents remaining. Unable to return average of nothing",
        ):
            setup_data.get_combined_average(
                self.indent_num, np.array([]), np.array([]), np.array([]), np.array([])
            )

    def test_single_indent_deleted(self):
        with pytest.raises(
            ValueError,
            match="No indents remaining. Unable to return average of nothing",
        ):
            setup_data.get_combined_average(
                self.indent_num,
                np.array([np.ones((10, 5))]),
                np.array([]),
                np.array([1]),
                np.array([]),
            )
        with pytest.raises(
            ValueError,
            match="No indents remaining. Unable to return average of nothing",
        ):
            setup_data.get_combined_average(
                self.indent_num,
                np.array([]),
                np.array([np.ones((10, 5))]),
                np.array([]),
                np.array([1]),
            )


class TestVerifyAnomalyType(unittest.TestCase):

    def test_wrong_anomaly_type(self):
        with pytest.raises(ValueError):
            setup_data.verify_anomaly_type("BOGUS_ANOMALY")

    def test_correct_anomaly_type(self):
        anomaly_types = [
            "displacement_offset",
            "force_offset",
            "tip_displacement",
            "too_deep",
            "unusual_loading_curvature",
            "unusual_unloading_curvature",
        ]
        for anomaly in anomaly_types:
            setup_data.verify_anomaly_type(anomaly)
