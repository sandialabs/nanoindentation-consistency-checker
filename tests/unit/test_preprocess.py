#  ___________________________________________________________________________
#  Copyright (c) 2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________


"""
Unit tests for utils/preprocess.py
"""

import unittest

import numpy as np
import pandas as pd
import pytest

import src.utils.preprocess as preprocess


class TestCreateLoadedIndents(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.all_samples_df = pd.DataFrame(
            {
                "sample_num": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
                "indent_num": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
                "displacement_offset": [False] * 12,
                "other_offset": [True] * 12,
            }
        )
        cls.full_data = np.ones((12, 5, 2))
        cls.anomaly_types = ["displacement_offset", "other_offset"]

    def test_wrong_argument_type(self):
        with pytest.raises(TypeError, match="all_samples_df must be pd.DataFrame"):
            preprocess.create_loaded_indents(0, self.full_data, self.anomaly_types)
        with pytest.raises(TypeError, match="full_data must be np.ndarray"):
            preprocess.create_loaded_indents(self.all_samples_df, 0, self.anomaly_types)
        with pytest.raises(TypeError, match="anomaly_types must be list"):
            preprocess.create_loaded_indents(self.all_samples_df, self.full_data, 0)

    def test_correct_usage(self):
        preprocess.create_loaded_indents(
            self.all_samples_df, self.full_data, self.anomaly_types
        )

    def test_missing_indents_from_all_samples_df(self):
        missing_indents_samples_df = pd.DataFrame(
            {
                "sample_num": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4],
                "indent_num": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2],
                "displacement_offset": [False] * 11,
                "other_offset": [True] * 11,
            }
        )
        with pytest.raises(
            ValueError, match="All samples should have same amount of indents"
        ):
            preprocess.create_loaded_indents(
                missing_indents_samples_df, self.full_data, self.anomaly_types
            )

    def test_missing_indents_from_full_data(self):
        missing_data = np.ones((11, 5, 2))
        with pytest.raises(
            ValueError,
            match="Full data should have same amount of total indents as all_samples_df",
        ):
            preprocess.create_loaded_indents(
                self.all_samples_df, missing_data, self.anomaly_types
            )

    def test_wrong_anomaly_types_list_of_str(self):
        wrong_types = ["displacement_offset", 1]
        with pytest.raises(ValueError, match="anomaly_types must be list of str"):
            preprocess.create_loaded_indents(
                self.all_samples_df, self.full_data, wrong_types
            )

    def test_wrong_anomaly_types(self):
        missing_types = ["displacement_offset", "random_anomaly"]
        with pytest.raises(
            ValueError,
            match="anomaly_types should be a subset of all_samples_df.columns",
        ):
            preprocess.create_loaded_indents(
                self.all_samples_df, self.full_data, missing_types
            )
