#  ___________________________________________________________________________
#  Copyright (c) 2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________


"""
Integration tests for setup_data.py
"""

import itertools
import unittest

import numpy as np

import src.utils.setup_data as setup_data


class TestExample1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.zeros = np.zeros((400, 3))
        cls.ones = np.ones((400, 3))

        cls.linear_increase = np.zeros((400, 3))
        cls.linear_increase[:, 0] = range(1, 400 + 1)
        cls.linear_increase[:, 1] = range(1, 400 + 1)
        cls.linear_increase[:, 2] = range(1, 400 + 1)

        np.random.seed(257)
        cls.normal = np.random.randn(400, 3)

        cls.sin_increase = np.zeros((400, 3))
        x = np.linspace(-1, 10, 400)
        y = np.sin(x) + x
        time = np.linspace(0, 10, 400)
        cls.sin_increase[:, 0] = x
        cls.sin_increase[:, 1] = y
        cls.sin_increase[:, 2] = time

        cls.x_squared = np.zeros((400, 3))
        x = np.linspace(-10, 10, 400)
        y = x**2
        time = np.linspace(0, 10, 400)
        cls.x_squared[:, 0] = x
        cls.x_squared[:, 1] = y
        cls.x_squared[:, 2] = time

        cls.back_to_zero = np.zeros((400, 3))
        time = np.linspace(0, 400, 400)
        x = -1 / 200 * (time - 200) ** 2 + 200
        y = -1 / 100 * (time - 200) ** 2 + 400
        cls.back_to_zero[:, 0] = x
        cls.back_to_zero[:, 1] = y
        cls.back_to_zero[:, 2] = time

        cls.regular_indent = np.zeros((400, 3))
        x = np.zeros(400)
        x[:100] = np.linspace(-1, 80, 100)
        x[100:300] = np.linspace(80, 100, 200)
        x[300:] = np.linspace(100, 75, 100)
        y = np.zeros(400)
        y[:100] = x[:100] * 2
        y[100:300] = y[99] + np.sin(x[100:300])
        y[300:] = np.linspace(y[299], 0, 100)
        time = np.linspace(0, 5, 400)
        cls.regular_indent[:, 0] = x
        cls.regular_indent[:, 1] = y
        cls.regular_indent[:, 2] = time

        np.random.seed(408)
        cls.noisy_regular_indent = cls.regular_indent.copy()
        cls.noisy_regular_indent[:, :2] += np.random.randn(400, 2)

        cls.indents = np.array(
            [
                cls.zeros,
                cls.ones,
                cls.linear_increase,
                cls.normal,
                cls.sin_increase,
                cls.back_to_zero,
                cls.regular_indent,
                cls.noisy_regular_indent,
            ]
        )

    def test_normalize(self):
        for indent_num, indent in enumerate(self.indents):
            normalized_indent = setup_data.normalize(indent)

            assert (normalized_indent >= 0).all(), indent_num
            assert (normalized_indent <= 1).all(), indent_num

    def test_normalize_0_max(self):
        for indent_num, indent in enumerate(self.indents):
            normalized_indent = setup_data.normalize_0_max(indent)

            assert (normalized_indent[0, :2] == np.array([0, 0])).all(), indent_num
            assert (normalized_indent <= 1).all(), indent_num

    def test_normalize_on_other_indent(self):
        for indent_pair in itertools.permutations(self.indents, 2):
            indent, reference_indent = indent_pair

            normalized_indent = setup_data.normalize_on_other_indent(
                indent, reference_indent
            )

            upper = indent > reference_indent.max(axis=0)
            lower = indent < reference_indent.min(axis=0)
            middle = (reference_indent.min(axis=0) < indent) & (
                indent < reference_indent.max(axis=0)
            )

            assert (normalized_indent[upper] > 1).all()
            assert (normalized_indent[lower] < 0).all()
            assert (
                (0 < normalized_indent[middle]) & (normalized_indent[middle] < 1)
            ).all()

    def test_get_loading(self):
        for indent in self.indents:
            loading_indent = setup_data.get_loading(indent)
            assert loading_indent.shape == (setup_data.LOADING_PORTION_END, 3)

    def test_pointwise_euclidean(self):
        for indent_pair in itertools.permutations(self.indents, 2):
            indent1, indent2 = indent_pair

            pointwise_euclidean = 0
            for i in range(len(indent1)):
                point1 = indent1[i]
                point2 = indent2[i]
                pointwise_euclidean += (
                    (point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2
                ) ** 0.5

            np.testing.assert_almost_equal(
                setup_data.pointwise_euclidean(indent1, indent2), pointwise_euclidean
            )

    def test_max_pointwise_euclidean(self):
        for indent_pair in itertools.permutations(self.indents, 2):
            indent1, indent2 = indent_pair

            max_pointwise_euclidean = 0
            for i in range(len(indent1)):
                point1 = indent1[i]
                point2 = indent2[i]
                distance = (
                    (point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2
                ) ** 0.5
                if distance > max_pointwise_euclidean:
                    max_pointwise_euclidean = distance

            np.testing.assert_almost_equal(
                setup_data.max_pointwise_euclidean(indent1, indent2),
                max_pointwise_euclidean,
            )

    def test_compute_triangle_area_x_squared(self):
        curvatures = setup_data.compute_triangle_area(self.x_squared)
        for i in range(1, len(curvatures) // 2):
            assert curvatures[i - 1] < curvatures[i]
        for i in range(len(curvatures) // 2 + 1, len(curvatures)):
            assert curvatures[i - 1] > curvatures[i]

    def test_compute_triangle_area_flat_curves(self):
        for indent in [self.zeros, self.ones, self.linear_increase, self.back_to_zero]:
            curvatures = setup_data.compute_triangle_area(indent)
            assert (curvatures == np.zeros(len(curvatures))).all()

    def test_get_combined_average_remove_non_anomalous(self):
        non_anomalous_indents = np.array([self.zeros, self.ones, self.linear_increase])
        anomalous_indents = np.array([self.normal, self.noisy_regular_indent])
        non_anomalous_indent_nums = np.array([0, 1, 2])
        anomalous_indent_nums = np.array([3, 4])

        (
            non_anomalous_indent,
            anomalous_indent,
            normalized_non_anomalous_indent,
            normalized_anomalous_indent,
        ) = setup_data.get_combined_average(
            indent_num=1,
            non_anomalous_indents=non_anomalous_indents,
            anomalous_indents=anomalous_indents,
            non_anomalous_indent_nums=non_anomalous_indent_nums,
            anomalous_indent_nums=anomalous_indent_nums,
        )

        np.testing.assert_almost_equal(
            non_anomalous_indent, np.mean(non_anomalous_indents[[0, 2]], axis=0)
        )
        np.testing.assert_almost_equal(
            anomalous_indent, np.mean(anomalous_indents, axis=0)
        )
        np.testing.assert_almost_equal(
            normalized_non_anomalous_indent,
            np.mean(
                [
                    setup_data.normalize(indent)
                    for indent in non_anomalous_indents[[0, 2]]
                ],
                axis=0,
            ),
        )
        np.testing.assert_almost_equal(
            normalized_anomalous_indent,
            np.mean(
                [setup_data.normalize(indent) for indent in anomalous_indents], axis=0
            ),
        )

    def test_get_combined_average_remove_anomalous(self):
        non_anomalous_indents = np.array([self.zeros, self.ones, self.linear_increase])
        anomalous_indents = np.array([self.normal, self.noisy_regular_indent])
        non_anomalous_indent_nums = np.array([0, 1, 2])
        anomalous_indent_nums = np.array([3, 4])

        (
            non_anomalous_indent,
            anomalous_indent,
            normalized_non_anomalous_indent,
            normalized_anomalous_indent,
        ) = setup_data.get_combined_average(
            indent_num=4,
            non_anomalous_indents=non_anomalous_indents,
            anomalous_indents=anomalous_indents,
            non_anomalous_indent_nums=non_anomalous_indent_nums,
            anomalous_indent_nums=anomalous_indent_nums,
        )

        np.testing.assert_almost_equal(
            non_anomalous_indent, np.mean(non_anomalous_indents, axis=0)
        )
        np.testing.assert_almost_equal(
            anomalous_indent, np.mean(anomalous_indents[[0]], axis=0)
        )
        np.testing.assert_almost_equal(
            normalized_non_anomalous_indent,
            np.mean(
                [setup_data.normalize(indent) for indent in non_anomalous_indents],
                axis=0,
            ),
        )
        np.testing.assert_almost_equal(
            normalized_anomalous_indent,
            np.mean(
                [setup_data.normalize(indent) for indent in anomalous_indents[[0]]],
                axis=0,
            ),
        )

    def test_get_combined_average_remove_none(self):
        non_anomalous_indents = np.array([self.zeros, self.ones, self.linear_increase])
        anomalous_indents = np.array([self.normal, self.noisy_regular_indent])
        non_anomalous_indent_nums = np.array([0, 1, 2])
        anomalous_indent_nums = np.array([3, 4])

        (
            non_anomalous_indent,
            anomalous_indent,
            normalized_non_anomalous_indent,
            normalized_anomalous_indent,
        ) = setup_data.get_combined_average(
            indent_num=-1,
            non_anomalous_indents=non_anomalous_indents,
            anomalous_indents=anomalous_indents,
            non_anomalous_indent_nums=non_anomalous_indent_nums,
            anomalous_indent_nums=anomalous_indent_nums,
        )

        np.testing.assert_almost_equal(
            non_anomalous_indent, np.mean(non_anomalous_indents, axis=0)
        )
        np.testing.assert_almost_equal(
            anomalous_indent, np.mean(anomalous_indents, axis=0)
        )
        np.testing.assert_almost_equal(
            normalized_non_anomalous_indent,
            np.mean(
                [setup_data.normalize(indent) for indent in non_anomalous_indents],
                axis=0,
            ),
        )
        np.testing.assert_almost_equal(
            normalized_anomalous_indent,
            np.mean(
                [setup_data.normalize(indent) for indent in anomalous_indents], axis=0
            ),
        )

    def test_get_best_threshold_perfect(self):
        array = np.array([1.0, 2.3, 5.2, 9.2])
        dims = [(1.5, 10)]
        y_true = [True, True, False, False]
        n_calls = 10

        res = setup_data.get_best_threshold(array, dims, y_true, n_calls)

        assert 2.3 <= res.x[0] and res.x[0] <= 5.2
        assert res.fun == -1

    def test_get_best_threshold_mixed(self):
        array = np.array([1.0, 2.3, 5.2, 9.2])
        dims = [(1.5, 10)]
        y_true = [True, False, True, False]
        n_calls = 10

        res = setup_data.get_best_threshold(array, dims, y_true, n_calls)

        assert 5.2 <= res.x[0] and res.x[0] <= 9.2
        assert -res.fun == (2 * (2 / 2 * 2 / 3) / (2 / 2 + 2 / 3))
