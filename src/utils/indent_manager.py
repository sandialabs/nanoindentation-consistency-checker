#  ___________________________________________________________________________
#  Copyright (c) 2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________


"""
IndentManager provides a common object to store dataframe like "all_samples_df" and indent
functionality
"""

import os

import numpy as np
import pandas as pd

from src.utils.preprocess import create_loaded_indents
from src.utils.setup_data import CustomFileError, normalize


class IndentManager:

    def __init__(self, datapath: str, anomaly_types, progress=None):
        self.progress = progress
        self.anomaly_types = anomaly_types

        (
            self.all_samples_df,
            self.good_samples_df,
            self.all_sample_nums,
            self.full_data,
            self.indents,
            self.indent_infos,
        ) = self.load_data(datapath)

        self.sample_size = self.all_samples_df["sample_num"].value_counts().unique()[0]

    def load_data(self, datapath: str):
        """
        Loads data from datapath

        Args:
            datapath : path to data that contains "all_samples_labelled.csv" and "full_data.npy" (or
            "Nanoindent_data/" to generate "full_data.npy")

        Returns:
            all_samples_df : dataframe of all samples
            good_samples_df : dataframe of all samples excluding fully anomalous indents
            all_sample_nums : array of all "good" sample numbers
            indents : Access with (sample_num, indent_num). All the points in an indent
            indent_infos : Access with (sample_num, indent_num). Values are the anomalies of an
            indent
        """
        # Load data
        all_samples_path = os.path.join(datapath, "all_samples_labelled.csv")
        all_samples_df = pd.read_csv(all_samples_path)

        if "verify_zero_offset_type" not in all_samples_df.columns:
            all_samples_df["verify_zero_offset_type"] = 0

        good_samples_df = all_samples_df
        if "fully_anomalous_sample" in all_samples_df.columns:
            good_samples_df = all_samples_df[
                (all_samples_df["fully_anomalous_sample"] == 0)
            ]
        all_sample_nums = good_samples_df["sample_num"].unique()

        # Load all of the data
        full_data_path = os.path.join(datapath, "full_data.npy")
        if os.path.exists(full_data_path):
            full_data = np.load(full_data_path)
        else:
            print("Generating full_data.npy")
            if "path" not in all_samples_df.columns:
                raise ValueError(
                    'Can not generate full_data.py because all_samples_df does not have "path" column'
                )
            full_data = []
            if self.progress:
                task = self.progress.add_task(
                    f"[green]Creating {full_data_path}...",
                    total=len(all_samples_df.index),
                )
            for i in all_samples_df.index:
                fpath = all_samples_df.loc[i, "path"]
                fpath = os.path.join(datapath, "Nanoindent_data", fpath)
                ld_df = pd.read_csv(fpath)
                full_data.append(ld_df[["depth_nm", "load_micro_N", "time_s"]])
                if self.progress:
                    self.progress.update(task, advance=1)
            full_data = np.array(full_data)
            print(f"Generated {full_data_path}")
            np.save(full_data_path, full_data)

        indents, indent_infos = create_loaded_indents(
            all_samples_df, full_data, self.anomaly_types
        )

        return (
            all_samples_df,
            good_samples_df,
            all_sample_nums,
            full_data,
            indents,
            indent_infos,
        )

    def get_extra_sample_df(self, reverification_df: pd.DataFrame, num_ex_samples=20):
        """
        Returns an extra sample df that contains random samples to consider for reverification

        Args:
            reverification_df : dataframe of samples needing reverification
            num_ex_samples : number of extra samples to include

        Returns:
            extra_sample_df : dataframe of random samples not needing reverification
        """
        if not isinstance(reverification_df, pd.DataFrame):
            raise TypeError("reverification_df must be pd.DataFrame")
        if not {"sample_num", "indent_num"}.issubset(reverification_df):
            raise KeyError(
                "sample_num, indent_num are required keys of reverification_df"
            )
        if num_ex_samples <= 0:
            raise ValueError("num_ex_samples needs to be a positive number")

        # Get all the sample/indent sets
        all_index = set(
            self.all_samples_df.reset_index()
            .set_index(["sample_num", "indent_num"])
            .index
        )

        # Get all the "reverify index" set of sample and indent numbers
        reverify_index = set(
            reverification_df.set_index(["sample_num", "indent_num"]).index
        )

        # Get the samples that don't need to be reverified
        non_flagged = list(all_index - reverify_index)
        if num_ex_samples > len(non_flagged):
            raise ValueError(
                "num_ex_samples must not exceed the number of samples not needing reverification"
            )

        # Pick num_ex_samples random extra samples that don't need to reverified as a control
        np.random.seed(1)
        indices = [
            non_flagged[i]
            for i in np.random.choice(
                np.arange(len(non_flagged), dtype=int), num_ex_samples, replace=False
            )
        ]

        # Extract the label information
        extra_sample_df = (
            self.all_samples_df.set_index(["sample_num", "indent_num"])
            .loc[indices]
            .reset_index()
        )

        # Flag whether or not a consistency check was needed
        extra_sample_df["consistency_flag"] = 0

        return extra_sample_df

    def get_non_anomalous_average(self, sample_num: int, modification=lambda x: x):
        """
        Returns non-anomalous average within a sample

        Args:
            sample_num : Sample number
            modification : Function to modify each indent (default is nothing)

        Returns:
            average of non-anomalous indents
        """
        if not isinstance(sample_num, (int, np.integer)):
            raise TypeError("sample_num must be int")
        if not callable(modification):
            raise TypeError("modification must be function")
        if sample_num not in self.all_samples_df["sample_num"]:
            raise ValueError("sample_num must exist in all_samples_df")

        modified_indents = []
        for indent_num in self.all_samples_df.loc[
            self.all_samples_df["sample_num"] == sample_num, "indent_num"
        ]:
            indent_info = self.indent_infos[(sample_num, indent_num)]
            if indent_info.any():
                continue

            indent = self.indents[(sample_num, indent_num)][:, :2]
            indent = modification(indent)
            modified_indents.append(indent)

        modified_indents = np.array(modified_indents)
        return modified_indents.mean(axis=0)

    def get_indent_classifications(
        self, sample_num: int, anomaly_type: str, indent_range: range
    ):
        """
        Return offsetted non anomalous and anomalous indents within a sample

        Args:
            sample_num (int) : Sample number
            anomaly_type (String) : Specific anomaly type
            indent_range (List(int)) : Range to look at for indents.
            Usually either: range(0, LOADING_PORTION_END) (loading curve),
            range(LOADING_PORTION_END, UNLOADING_PORTION_START) (holding),
            range(UNLOADING_PORTION_END, INDENT_END) (unloading curve)

        Returns:
            non_anomalous_indents, anomalous_indents, non_anomalous_indent_nums,
            anomalous_indent_nums
        """
        if not isinstance(sample_num, (int, np.integer)):
            raise TypeError("sample_num must be int")
        if not isinstance(anomaly_type, str):
            raise TypeError("anomaly_type must be str")
        if not isinstance(indent_range, range):
            raise TypeError("indent_range must be range")

        if sample_num not in self.all_samples_df["sample_num"]:
            raise ValueError("sample_num must exist in all_samples_df")
        if anomaly_type not in self.all_samples_df.columns:
            raise ValueError("anomaly_type must be a column in all_samples_df")

        if indent_range.start < 0:
            raise ValueError("indent_range must start at a positive number")
        if indent_range.stop > self.full_data.shape[1]:
            raise ValueError(
                "indent_range end must be within number of points in an indent"
            )

        non_anomalous_examples = self.all_samples_df.loc[
            (self.all_samples_df["sample_num"] == sample_num)
            & (self.all_samples_df[anomaly_type] == 0),
            ["sample_num", "indent_num"],
        ].values
        non_anomalous_indent_nums = non_anomalous_examples[:, 1]

        non_anomalous_indents = [
            self.indents[(sample_num, indent_num)][indent_range, :2]
            for sample_num, indent_num in non_anomalous_examples
        ]
        non_anomalous_indents = [
            indent - indent.min(axis=0) for indent in non_anomalous_indents
        ]
        non_anomalous_indents = np.array(non_anomalous_indents)

        anomalous_examples = self.all_samples_df.loc[
            (self.all_samples_df["sample_num"] == sample_num)
            & (self.all_samples_df[anomaly_type] != 0),
            ["sample_num", "indent_num"],
        ].values
        anomalous_indent_nums = anomalous_examples[:, 1]

        anomalous_indents = [
            self.indents[(sample_num, indent_num)][indent_range, :2]
            for sample_num, indent_num in anomalous_examples
        ]
        anomalous_indents = [
            indent - indent.min(axis=0) for indent in anomalous_indents
        ]
        anomalous_indents = np.array(anomalous_indents)

        return (
            non_anomalous_indents,
            anomalous_indents,
            non_anomalous_indent_nums,
            anomalous_indent_nums,
        )

    def get_offset_indent(self, sample_num: int, indent_num: int, indent_range: range):
        """
        Return an indent that is offset such that the minimum is at (0, 0).
        Also return normalized version

        Args:
            sample_num (int) : Sample number
            indent_num (int) : Indent number
            indent_range (List(int)) : Range to look at for indents.
            Usually either: range(0, LOADING_PORTION_END) (loading curve),
            range(LOADING_PORTION_END, UNLOADING_PORTION_START) (holding),
            range(UNLOADING_PORTION_END, INDENT_END) (unloading curve)

        Returns:
            indent, normalized_indent
        """
        if not isinstance(sample_num, (int, np.integer)):
            raise TypeError("sample_num must be int")
        if not isinstance(indent_num, (int, np.integer)):
            raise TypeError("indent_num must be int")
        if not isinstance(indent_range, range):
            raise TypeError("indent_range must be range")

        if sample_num not in self.all_samples_df["sample_num"]:
            raise ValueError("sample_num must exist in all_samples_df")
        if (
            indent_num
            not in self.all_samples_df.loc[
                self.all_samples_df["sample_num"] == sample_num, "indent_num"
            ].values
        ):
            raise ValueError(
                "(sample_num, indent_num) pair must exist in all_samples_df"
            )

        if indent_range.start < 0:
            raise ValueError("indent_range must start at a positive number")
        if indent_range.stop > self.full_data.shape[1]:
            raise ValueError(
                "indent_range end must be within number of points in an indent"
            )

        indent = self.indents[(sample_num, indent_num)][indent_range, :2]
        normalized_indent = normalize(indent)
        indent[:, 0] -= indent[:, 0].min()
        indent[:, 1] -= indent[:, 1].min()

        return indent, normalized_indent

    def get_indent_data(
        self, sample_num: int, indent_num: int, anomaly_type: str, include_any: bool
    ):
        """
        Returns initial indent data dictionary

        Args:
            sample_num (int) : Sample number
            indent_num (int) : Indent number
            anomaly_type (String) : Specific anomaly type
            include_any (bool) : Should indent_data include "any" (has any anomaly)

        Returns:
            dictionary
        """
        if not isinstance(sample_num, (int, np.integer)):
            raise TypeError("sample_num must be int")
        if not isinstance(indent_num, (int, np.integer)):
            raise TypeError("indent_num must be int")
        if not isinstance(anomaly_type, str):
            raise TypeError("anomaly_type must be str")
        if not isinstance(include_any, bool):
            raise TypeError("include_any must be bool")

        if sample_num not in self.all_samples_df["sample_num"]:
            raise ValueError("sample_num must exist in all_samples_df")
        if (
            indent_num
            not in self.all_samples_df.loc[
                self.all_samples_df["sample_num"] == sample_num, "indent_num"
            ].values
        ):
            raise ValueError(
                "(sample_num, indent_num) pair must exist in all_samples_df"
            )

        if anomaly_type not in self.all_samples_df.columns:
            raise ValueError("anomaly_type must be a column in all_samples_df")

        indent_info = self.indent_infos[(sample_num, indent_num)]
        indent_data = {
            "sample_num": sample_num,
            "indent_num": indent_num,
            anomaly_type: indent_info[anomaly_type],
        }

        if include_any:
            indent_data["any"] = indent_info.any()

        return indent_data

    def export_results(
        self,
        reverification_pairs: pd.DataFrame,
        export_filename,
        resultsdir="results",
        num_ex_samples=20,
    ):
        """
        Exports the results from the reverification_list into export_filename. 20 additional random
        pairs will be added at the end.

        Args:
            reverification_pairs : Dataframe of (sample_num, indent_num) pairs for reverification
            export_filename : Filename to create

        Returns:
            None
        """
        if not isinstance(reverification_pairs, pd.DataFrame):
            raise TypeError("reverification_pairs must be pd.DataFrame")

        if not {"sample_num", "indent_num"}.issubset(reverification_pairs.columns):
            raise KeyError(
                "sample_num, indent_num are required keys of reverification_pairs"
            )

        if not os.path.exists(resultsdir):
            os.mkdir(resultsdir)

        destination = os.path.join(resultsdir, export_filename)

        reverification_df = pd.merge(
            reverification_pairs,
            self.all_samples_df,
            how="left",
            on=["sample_num", "indent_num"],
        )
        reverification_df["consistency_flag"] = 1

        if num_ex_samples > 0:
            extra_sample_df = self.get_extra_sample_df(
                reverification_df, num_ex_samples
            )

            test_set = pd.concat([reverification_df, extra_sample_df]).reset_index(
                drop=True
            )
        else:
            test_set = reverification_df

        try:
            test_set.to_csv(destination, index=False)
        except FileNotFoundError as e:
            raise CustomFileError(e, destination)
