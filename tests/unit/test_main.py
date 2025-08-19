#  ___________________________________________________________________________
#  Copyright (c) 2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pytest
import argparse
import os
from unittest.mock import patch, MagicMock
import tempfile
import pandas as pd

from src.main.main import (
    argparse_verify_anomaly_type,
    argparse_verify_datapath,
    get_datapath_from_config,
    build_parser,
)


def test_argparse_verify_datapath_valid_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        valid_path = os.path.join(tmpdir, "valid_datapath")
        valid_data_dir = os.path.join(valid_path, "Nanoindent_data")
        os.mkdir(valid_path)
        os.mkdir(valid_data_dir)
        csv_file_path = os.path.join(valid_path, "all_samples_labelled.csv")
        pd.DataFrame().to_csv(csv_file_path, index=False)
        assert argparse_verify_datapath(valid_path) == valid_path


def test_argparse_verify_datapath_invalid_path():
    invalid_path = "/bogus/path"
    with pytest.raises(argparse.ArgumentTypeError):
        argparse_verify_datapath(invalid_path)


def test_argparse_verify_anomaly_type_valid_type():
    valid_type = "force_offset"
    assert argparse_verify_anomaly_type(valid_type) == valid_type


def test_argparse_verify_anomaly_type_invalid_type():
    invalid_type = "fake_anomaly"
    with pytest.raises(argparse.ArgumentTypeError):
        argparse_verify_anomaly_type(invalid_type)


def test_get_datapath_from_config():
    with patch("os.path.join") as mock_join:
        mock_join.return_value = "/path/to/config.py"
        with patch("importlib.util.spec_from_file_location") as mock_spec:
            mock_spec.return_value = MagicMock()
            with patch("importlib.util.module_from_spec") as mock_module:
                mock_module.return_value = MagicMock(
                    USER_DATAPATH="/path/to/user/datapath"
                )
                with patch("os.path.expanduser") as mock_expanduser:
                    mock_expanduser.return_value = "/path/to/expanded/user/datapath"
                    with patch("os.path.abspath") as mock_abspath:
                        mock_abspath.return_value = "/path/to/absolute/user/datapath"
                        assert (
                            get_datapath_from_config()
                            == "/path/to/absolute/user/datapath"
                        )


def test_build_parser():
    parser = build_parser()
    assert isinstance(parser, argparse.ArgumentParser)
