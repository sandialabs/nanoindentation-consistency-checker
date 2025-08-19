#  ___________________________________________________________________________
#  Copyright (c) 2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________


"""
Integration tests for IndentManager.py
"""

import os
import tempfile
import unittest

import numpy as np
import pandas as pd

import src.utils.indent_manager as IndentManager


class TestExample1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_data_path = os.path.join("code", "tests", "integration", "test_data")
        # Check if the test data path exists; skip if not
        if not os.path.exists(cls.test_data_path):
            raise unittest.SkipTest(
                f"Test data path '{cls.test_data_path}' does not exist. Skipping tests."
            )
        anomaly_types = ["outlier", "sample_1_anomaly"]

        cls.im = IndentManager.IndentManager(cls.test_data_path, anomaly_types)

    @classmethod
    def tearDownClass(cls):
        full_data_path = os.path.join(cls.test_data_path, "full_data.npy")
        os.remove(full_data_path)

    def test_load_data(self):
        all_samples_df = pd.DataFrame(
            {
                "sample_num": [1, 1, 2, 2, 3, 3, 4, 4],
                "indent_num": [1, 2, 1, 2, 1, 2, 1, 2],
                "outlier": [False, False, False, True, True, True, False, False],
                "sample_1_anomaly": [
                    True,
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                ],
                "fully_anomalous_sample": [
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    True,
                    True,
                ],
            }
        )

        good_samples_df = all_samples_df[~all_samples_df["fully_anomalous_sample"]]

        assert (
            self.im.all_samples_df.isin(all_samples_df)[
                [
                    "sample_num",
                    "indent_num",
                    "outlier",
                    "sample_1_anomaly",
                    "fully_anomalous_sample",
                ]
            ]
            .all()
            .all()
        )
        assert (
            self.im.good_samples_df.isin(good_samples_df)[
                [
                    "sample_num",
                    "indent_num",
                    "outlier",
                    "sample_1_anomaly",
                    "fully_anomalous_sample",
                ]
            ]
            .all()
            .all()
        )

        assert self.im.all_sample_nums.shape == (3,)
        assert all(self.im.all_sample_nums == np.array([1, 2, 3]))
        assert self.im.sample_size == 2

        assert self.im.full_data.shape == (8, 3, 3)
        assert len(self.im.indents) == 8
        assert len(self.im.indent_infos) == 8

    def test_indent_reterival(self):
        for sample_num in self.im.all_sample_nums:
            for indent_num in [1, 2]:
                indent = self.im.indents[(sample_num, indent_num)]
                assert indent.shape == (3, 3)
                assert np.issubdtype(indent.dtype, float)

                indent_info = self.im.indent_infos[(sample_num, indent_num)]
                assert indent_info.shape == (2,)
                assert np.issubdtype(indent_info.dtype, bool)
                assert all(indent_info.index == ["outlier", "sample_1_anomaly"])

    def test_get_extra_sample_df(self):
        reverification_df = pd.DataFrame({"sample_num": [1, 2], "indent_num": [1, 1]})

        expected_pairs = [(1, 2), (2, 2), (3, 1), (3, 2), (4, 1), (4, 2)]
        for i in range(1, 6 + 1):
            extra_sample_df = self.im.get_extra_sample_df(reverification_df, i)
            extra_sample_df.set_index(["sample_num", "indent_num"], inplace=True)

            assert extra_sample_df.index.isin(expected_pairs).all()

    def test_get_non_anomalous_average(self):
        for sample_num in [1, 3]:
            assert np.isnan(self.im.get_non_anomalous_average(sample_num))

        non_anomalous_average = self.im.get_non_anomalous_average(2)
        assert (non_anomalous_average == self.im.indents[(2, 1)][:, :2]).all()

    def test_get_indent_classifications_all_anomalous(self):
        (
            non_anomalous_indents,
            anomalous_indents,
            non_anomalous_indent_nums,
            anomalous_indent_nums,
        ) = self.im.get_indent_classifications(1, "sample_1_anomaly", range(3))

        assert len(non_anomalous_indents) == len(non_anomalous_indent_nums)
        assert len(anomalous_indents) == len(anomalous_indent_nums)
        assert non_anomalous_indents.shape == (0,)
        assert anomalous_indents.shape == (2, 3, 2)

        assert (anomalous_indent_nums == np.array([1, 2])).all()
        assert (
            anomalous_indents
            == np.array(
                [
                    [[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]],
                    [[0.0, 0.0], [0.0, 1.0], [1.0, 2.0]],
                ]
            )
        ).all()

    def test_get_indent_classifications_all_non_anomalous(self):
        for sample_num in [2, 3, 4]:
            (
                non_anomalous_indents,
                anomalous_indents,
                non_anomalous_indent_nums,
                anomalous_indent_nums,
            ) = self.im.get_indent_classifications(
                sample_num, "sample_1_anomaly", range(2)
            )

            assert len(non_anomalous_indents) == len(non_anomalous_indent_nums)
            assert len(anomalous_indents) == len(anomalous_indent_nums)
            assert non_anomalous_indents.shape == (2, 2, 2)
            assert anomalous_indents.shape == (0,)

            assert (non_anomalous_indent_nums == np.array([1, 2])).all()
            assert (
                non_anomalous_indents
                == np.array(
                    [
                        self.im.get_offset_indent(sample_num, 1, range(2))[0],
                        self.im.get_offset_indent(sample_num, 2, range(2))[0],
                    ]
                )
            ).all()

    def test_get_indent_classifications_mix(self):
        outliers = [[], [2], [1, 2], []]
        non_outliers = [[1, 2], [1], [], [1, 2]]
        for sample_num in [1, 2, 3, 4]:
            (
                non_anomalous_indents,
                anomalous_indents,
                non_anomalous_indent_nums,
                anomalous_indent_nums,
            ) = self.im.get_indent_classifications(sample_num, "outlier", range(1, 3))

            assert len(non_anomalous_indents) == len(non_anomalous_indent_nums)
            assert len(anomalous_indents) == len(anomalous_indent_nums)
            assert len(non_anomalous_indents) == len(non_outliers[sample_num - 1])
            assert len(anomalous_indents) == len(outliers[sample_num - 1])

            assert (
                non_anomalous_indent_nums == np.array(non_outliers[sample_num - 1])
            ).all()
            assert (anomalous_indent_nums == np.array(outliers[sample_num - 1])).all()

            assert (
                non_anomalous_indents
                == np.array(
                    [
                        self.im.get_offset_indent(sample_num, indent_num, range(1, 3))[
                            0
                        ]
                        for indent_num in non_outliers[sample_num - 1]
                    ]
                )
            ).all()
            assert (
                anomalous_indents
                == np.array(
                    [
                        self.im.get_offset_indent(sample_num, indent_num, range(1, 3))[
                            0
                        ]
                        for indent_num in outliers[sample_num - 1]
                    ]
                )
            ).all()

    def test_get_offset_indent(self):
        for sample_num in self.im.all_sample_nums:
            for indent_num in [1, 2]:
                indent, normalized_indent = self.im.get_offset_indent(
                    sample_num, indent_num, range(1, 3)
                )

                assert (indent.min(axis=0) == np.array([0, 0])).all()

                assert (normalized_indent >= 0).all()
                assert (normalized_indent <= 1).all()

    def test_get_indent_data(self):
        anomalous_pairs = [(1, 1), (1, 2), (2, 2), (3, 1), (3, 2)]
        for sample_num in self.im.all_sample_nums:
            for indent_num in [1, 2]:
                indent_data = self.im.get_indent_data(
                    sample_num, indent_num, "sample_1_anomaly", True
                )
                assert indent_data["sample_num"] == sample_num
                assert indent_data["indent_num"] == indent_num
                if sample_num == 1:
                    assert indent_data["sample_1_anomaly"]
                else:
                    assert not indent_data["sample_1_anomaly"]

                if (sample_num, indent_num) in anomalous_pairs:
                    assert indent_data["any"]
                else:
                    assert not indent_data["any"]

    def test_export_results_no_extra(self):
        reverification_pairs = pd.DataFrame(
            {"sample_num": [1, 2, 3], "indent_num": [1, 1, 2]}
        )
        resultsdir = tempfile.mkdtemp()
        all_reverification_export_filename = "all_src.csv"

        self.im.export_results(
            reverification_pairs, all_reverification_export_filename, resultsdir, 0
        )
        all_reverification_df = pd.read_csv(
            os.path.join(resultsdir, all_reverification_export_filename)
        )

        assert (all_reverification_df["consistency_flag"] == 1).all()
        assert reverification_pairs.isin(all_reverification_df).all().all()
        assert len(all_reverification_df) == 3

    def test_export_results_with_extra(self):
        reverification_pairs = pd.DataFrame(
            {"sample_num": [1, 2, 3], "indent_num": [1, 1, 2]}
        )
        resultsdir = tempfile.mkdtemp()
        export_filename = "test.csv"

        self.im.export_results(reverification_pairs, export_filename, resultsdir, 2)
        reverification_df = pd.read_csv(os.path.join(resultsdir, export_filename))

        assert sum(reverification_df["consistency_flag"] == 1) == 3
        assert sum(reverification_df["consistency_flag"] == 0) == 2
        assert reverification_pairs.isin(reverification_df).all().all()
        assert len(reverification_df) == 5

    def test_export_results_with_no_directory_made(self):
        reverification_pairs = pd.DataFrame(
            {"sample_num": [1, 2, 3], "indent_num": [1, 1, 2]}
        )
        resultsdir = os.path.join(tempfile.mkdtemp(), "results")
        export_filename = "test.csv"

        self.im.export_results(reverification_pairs, export_filename, resultsdir, 2)
        reverification_df = pd.read_csv(os.path.join(resultsdir, export_filename))

        assert sum(reverification_df["consistency_flag"] == 1) == 3
        assert sum(reverification_df["consistency_flag"] == 0) == 2
        assert reverification_pairs.isin(reverification_df).all().all()
        assert len(reverification_df) == 5
