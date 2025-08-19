#  ___________________________________________________________________________
#  Copyright (c) 2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________


"""
Unit tests for main_reverification_labels.py
"""

import os
import tempfile
import unittest
from shutil import rmtree

import pytest

import src.main.reverification_labels as reverify_labels


class TestVerifyFunctions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp()
        cls.all_samples_path = os.path.join(cls.tmpdir, "all_samples_labelled.csv")
        cls.full_data_path = os.path.join(cls.tmpdir, "full_data.npy")

    @classmethod
    def tearDownClass(cls):
        rmtree(cls.tmpdir)

    def test_verify_datapath_fakepath(self):
        bogus_path = "/path/is/totally/bogus"
        with pytest.raises(NotADirectoryError):
            reverify_labels.verify_datapath(bogus_path)

    def test_verify_datapath_no_labelled_file(self):
        with pytest.raises(FileNotFoundError):
            reverify_labels.verify_datapath(self.tmpdir)

    def test_verify_datapath_no_full_data_file(self):
        with open(self.all_samples_path, "w") as f:
            f.write("Hello, world!")
        with pytest.raises(FileNotFoundError):
            reverify_labels.verify_datapath(self.tmpdir)
        os.remove(self.all_samples_path)

    def test_csv_npy_files_exist(self):
        with open(self.all_samples_path, "w") as f:
            f.write("Hello, world!")
        with open(self.full_data_path, "w") as f:
            f.write("Hello, world!")
        reverify_labels.verify_datapath(self.tmpdir)
        os.remove(self.all_samples_path)
        os.remove(self.full_data_path)


class TestMain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp()
        cls.all_samples_path = os.path.join(cls.tmpdir, "all_samples_labelled.csv")
        cls.full_data_path = os.path.join(cls.tmpdir, "full_data.npy")
        with open(cls.all_samples_path, "w") as f:
            f.write("Hello, world!")
        with open(cls.full_data_path, "w") as f:
            f.write("Hello, world!")

    @classmethod
    def tearDownClass(cls):
        rmtree(cls.tmpdir)

    def test_correct_args_given(self):
        reverify_labels.verify_argv(["main.py", "displacement_offset", self.tmpdir])

    def test_too_few_args_given(self):
        with pytest.raises(SystemExit):
            reverify_labels.verify_argv(["main.py"])
        with pytest.raises(SystemExit):
            reverify_labels.verify_argv(["main.py", "displacement_offset"])

    def test_too_many_args_given(self):
        with pytest.raises(SystemExit):
            reverify_labels.verify_argv(
                ["main.py", "displacement_offset", self.tmpdir, "extra"]
            )

    def test_incorrect_anomaly_type(self):
        with pytest.raises(ValueError):
            reverify_labels.verify_argv(["main.py", "NONE", self.tmpdir])
