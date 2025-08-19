#  ___________________________________________________________________________
#  Copyright (c) 2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________


"""
Unit tests for IndentManager.py
"""

import os
import tempfile
import unittest
from shutil import rmtree

import numpy as np
import pandas as pd
import pytest

import src.utils.indent_manager as IndentManager
from src.utils.preprocess import IndentDict


class TestLoadData(unittest.TestCase):
    def test_no_path(self):
        all_samples_df = pd.DataFrame(
            {
                "sample_num": [1, 2],
                "indent_num": [1, 1],
                "displacement_offset": [False, False],
                "other_offset": [True, False],
            }
        )
        anomaly_types = ["displacement_offset", "other_offset"]

        tmpdir = tempfile.mkdtemp()
        all_samples_path = os.path.join(tmpdir, "all_samples_labelled.csv")
        all_samples_df.to_csv(all_samples_path)

        with pytest.raises(
            ValueError,
            match='Can not generate full_data.py because all_samples_df does not have "path" column',
        ):
            IndentManager.IndentManager(tmpdir, anomaly_types)

        rmtree(tmpdir)

    def test_correct_usage_generation(self):
        all_samples_df = pd.DataFrame(
            {
                "sample_num": [1, 2],
                "indent_num": [1, 1],
                "displacement_offset": [False, False],
                "other_offset": [True, False],
                "path": ["i1.csv", "i2.csv"],
            }
        )
        anomaly_types = ["displacement_offset", "other_offset"]

        tmpdir = tempfile.mkdtemp()
        all_samples_path = os.path.join(tmpdir, "all_samples_labelled.csv")
        all_samples_df.to_csv(all_samples_path)

        indent = pd.DataFrame(
            {
                "depth_nm": np.ones(10),
                "load_micro_N": np.ones(10),
                "time_s": np.ones(10),
            }
        )

        nanoindent_data_path = os.path.join(tmpdir, "Nanoindent_data")
        os.mkdir(nanoindent_data_path)
        indent.to_csv(os.path.join(nanoindent_data_path, "i1.csv"))
        indent.to_csv(os.path.join(nanoindent_data_path, "i2.csv"))

        IndentManager.IndentManager(tmpdir, anomaly_types)

        rmtree(tmpdir)

    def test_correct_usage_full_data(self):
        all_samples_df = pd.DataFrame(
            {
                "sample_num": [1, 2],
                "indent_num": [1, 1],
                "displacement_offset": [False, False],
                "other_offset": [True, False],
            }
        )
        full_data = np.ones((2, 5, 3))
        anomaly_types = ["displacement_offset", "other_offset"]

        tmpdir = tempfile.mkdtemp()
        all_samples_path = os.path.join(tmpdir, "all_samples_labelled.csv")
        all_samples_df.to_csv(all_samples_path)
        full_data_path = os.path.join(tmpdir, "full_data.npy")
        np.save(full_data_path, full_data)

        IndentManager.IndentManager(tmpdir, anomaly_types)

        rmtree(tmpdir)


class TestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        all_samples_df = pd.DataFrame(
            {
                "sample_num": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
                "indent_num": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
                "displacement_offset": [False] * 12,
                "other_offset": [True] * 12,
            }
        )
        full_data = np.ones((12, 5, 3))
        anomaly_types = ["displacement_offset", "other_offset"]

        cls.tmpdir = tempfile.mkdtemp()
        all_samples_path = os.path.join(cls.tmpdir, "all_samples_labelled.csv")
        all_samples_df.to_csv(all_samples_path)
        full_data_path = os.path.join(cls.tmpdir, "full_data.npy")
        np.save(full_data_path, full_data)

        cls.im = IndentManager.IndentManager(cls.tmpdir, anomaly_types)

    @classmethod
    def tearDownClass(cls):
        rmtree(cls.tmpdir)


class TestSetUp(TestBase):

    def test_has_attributes(self):
        assert hasattr(self.im, "anomaly_types")
        assert hasattr(self.im, "all_samples_df")
        assert hasattr(self.im, "good_samples_df")
        assert hasattr(self.im, "all_sample_nums")
        assert hasattr(self.im, "full_data")
        assert hasattr(self.im, "indents")
        assert hasattr(self.im, "indent_infos")
        assert hasattr(self.im, "sample_size")

    def test_type_of_anomaly_types(self):
        assert isinstance(self.im.anomaly_types, list)
        assert all(
            isinstance(anomaly_type, str) for anomaly_type in self.im.anomaly_types
        )

    def test_type_of_all_samples_df(self):
        assert isinstance(self.im.all_samples_df, pd.DataFrame)

    def test_sample_num_and_indent_num_columns_in_all_samples_df(self):
        assert {"sample_num", "indent_num"}.issubset(self.im.all_samples_df)

    def test_type_of_good_samples_df(self):
        assert isinstance(self.im.good_samples_df, pd.DataFrame)

    def test_sample_num_and_indent_num_columns_in_good_samples_df(self):
        assert {"sample_num", "indent_num"}.issubset(self.im.good_samples_df)

    def test_type_of_all_sample_nums(self):
        assert isinstance(self.im.all_sample_nums, np.ndarray)

    def test_type_all_sample_nums_are_valid(self):
        assert set(self.im.all_sample_nums).issubset(
            self.im.all_samples_df["sample_num"]
        )

    def test_type_of_full_data(self):
        assert isinstance(self.im.full_data, np.ndarray)

    def test_type_of_indents(self):
        assert isinstance(self.im.indents, IndentDict)

    def test_type_of_indent_infos(self):
        assert isinstance(self.im.indent_infos, IndentDict)

    def test_type_of_sample_size(self):
        assert isinstance(self.im.sample_size, (int, np.integer))


class TestGetExtraSampleDF(TestBase):

    def test_wrong_argument_type(self):
        with pytest.raises(TypeError):
            self.im.get_extra_sample_df(0)

    def test_wrong_columns(self):
        empty_df = pd.DataFrame()
        with pytest.raises(KeyError):
            self.im.get_extra_sample_df(empty_df)

    def test_empty_columns(self):
        empty_columns_df = pd.DataFrame(columns=["sample_num", "indent_num"])
        with pytest.raises(ValueError):
            self.im.get_extra_sample_df(empty_columns_df)

    def test_exceeding_num_ex_samples(self):
        subset_df = pd.DataFrame({"sample_num": [1], "indent_num": [3]})
        with pytest.raises(
            ValueError,
            match="num_ex_samples must not exceed the number of samples not needing reverification",
        ):
            self.im.get_extra_sample_df(subset_df, 20)

    def test_negative_num_ex_samples(self):
        subset_df = pd.DataFrame({"sample_num": [1], "indent_num": [3]})
        with pytest.raises(
            ValueError, match="num_ex_samples needs to be a positive number"
        ):
            self.im.get_extra_sample_df(subset_df, -1)

    def test_correct_usage(self):
        subset_df = pd.DataFrame({"sample_num": [1], "indent_num": [3]})
        self.im.get_extra_sample_df(subset_df, 1)


class TestGetNonAnomalousAverage(TestBase):

    def test_wrong_argument_type(self):
        with pytest.raises(TypeError):
            self.im.get_non_anomalous_average("test")
        with pytest.raises(TypeError):
            self.im.get_non_anomalous_average(1, 1)

    def test_non_existent_sample_num(self):
        with pytest.raises(ValueError):
            self.im.get_non_anomalous_average(-1)

    def test_correct_usage(self):
        self.im.get_non_anomalous_average(1)


class TestGetIndentClassifications(TestBase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.sample_num = 1
        cls.anomaly_type = "displacement_offset"
        cls.indent_range = range(3)

    def test_wrong_argument_type(self):
        with pytest.raises(TypeError):
            self.im.get_indent_classifications(
                "test", self.anomaly_type, self.indent_range
            )
        with pytest.raises(TypeError):
            self.im.get_indent_classifications(self.sample_num, 0, self.indent_range)
        with pytest.raises(TypeError):
            self.im.get_indent_classifications(self.sample_num, self.anomaly_type, 0)

    def test_correct_usage(self):
        self.im.get_indent_classifications(
            self.sample_num, self.anomaly_type, self.indent_range
        )

    def test_non_existent_sample_num(self):
        with pytest.raises(ValueError):
            self.im.get_indent_classifications(-1, self.anomaly_type, self.indent_range)

    def test_non_existent_anomaly_type(self):
        with pytest.raises(ValueError):
            self.im.get_indent_classifications(
                self.sample_num, "TEST", self.indent_range
            )

    def test_negative_indent_range(self):
        with pytest.raises(ValueError):
            self.im.get_indent_classifications(
                self.sample_num, self.anomaly_type, range(-5, 0)
            )

    def test_exceeding_indent_range(self):
        with pytest.raises(ValueError):
            self.im.get_indent_classifications(
                self.sample_num, self.anomaly_type, range(10)
            )


class TestGetOffsetIndent(TestBase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.sample_num = 1
        cls.indent_num = 1
        cls.indent_range = range(3)

    def test_wrong_argument_type(self):
        with pytest.raises(TypeError):
            self.im.get_offset_indent("test", self.indent_num, self.indent_range)
        with pytest.raises(TypeError):
            self.im.get_offset_indent(self.sample_num, "test", self.indent_range)
        with pytest.raises(TypeError):
            self.im.get_offset_indent(self.sample_num, self.indent_num, 0)

    def test_correct_usage(self):
        self.im.get_offset_indent(self.sample_num, self.indent_num, self.indent_range)

    def test_non_existent_sample_num(self):
        with pytest.raises(ValueError):
            self.im.get_offset_indent(-1, self.indent_num, self.indent_range)

    def test_non_existent_indent_num(self):
        with pytest.raises(ValueError):
            self.im.get_offset_indent(self.sample_num, -1, self.indent_range)

    def test_negative_indent_range(self):
        with pytest.raises(ValueError):
            self.im.get_offset_indent(self.sample_num, self.indent_num, range(-5, 0))

    def test_exceeding_indent_range(self):
        with pytest.raises(ValueError):
            self.im.get_offset_indent(self.sample_num, self.indent_num, range(10))


class TestGetIndentData(TestBase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.sample_num = 1
        cls.indent_num = 1
        cls.anomaly_type = "displacement_offset"
        cls.include_any = True

    def test_wrong_argument_type(self):
        with pytest.raises(TypeError):
            self.im.get_indent_data(
                "test", self.indent_num, self.anomaly_type, self.include_any
            )
        with pytest.raises(TypeError):
            self.im.get_indent_data(
                self.sample_num, "test", self.anomaly_type, self.include_any
            )
        with pytest.raises(TypeError):
            self.im.get_indent_data(
                self.sample_num, self.indent_num, 0, self.include_any
            )
        with pytest.raises(TypeError):
            self.im.get_indent_data(
                self.sample_num, self.indent_num, self.anomaly_type, 0
            )

    def test_correct_usage(self):
        self.im.get_indent_data(
            self.sample_num, self.indent_num, self.anomaly_type, self.include_any
        )

    def test_non_existent_sample_num(self):
        with pytest.raises(ValueError):
            self.im.get_indent_data(
                -1, self.indent_num, self.anomaly_type, self.include_any
            )

    def test_non_existent_indent_num(self):
        with pytest.raises(ValueError):
            self.im.get_indent_data(
                self.sample_num, -1, self.anomaly_type, self.include_any
            )

    def test_non_existent_anomaly_type(self):
        with pytest.raises(ValueError):
            self.im.get_indent_data(
                self.sample_num, self.indent_num, "TEST", self.include_any
            )


class TestExportResults(TestBase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.reverification_pairs = pd.DataFrame(
            {"sample_num": [1, 1], "indent_num": [1, 2]}
        )
        cls.export_filename = "test.csv"
        cls.resultsdir = tempfile.mkdtemp()
        cls.num_ex_samples = 2

    def test_wrong_argument_type(self):
        with pytest.raises(TypeError):
            self.im.export_results(
                0, self.export_filename, self.resultsdir, self.num_ex_samples
            )
        with pytest.raises(TypeError):
            self.im.export_results(
                self.reverification_pairs, 0, self.resultsdir, self.num_ex_samples
            )
        with pytest.raises(TypeError):
            self.im.export_results(
                self.reverification_pairs, self.export_filename, 0, self.num_ex_samples
            )

    def test_correct_usage(self):
        self.im.export_results(
            self.reverification_pairs,
            self.export_filename,
            self.resultsdir,
            self.num_ex_samples,
        )

    def test_wrong_columns(self):
        empty_df = pd.DataFrame()
        with pytest.raises(KeyError):
            self.im.export_results(
                empty_df, self.export_filename, self.resultsdir, self.num_ex_samples
            )
