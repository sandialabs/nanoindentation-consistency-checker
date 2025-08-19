#  ___________________________________________________________________________
#  Copyright (c) 2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________


"""
Main program for creating anomaly reverification csvs
"""

import sys

from src.anomaly_scripts.displacement_offset_reverification_labels import (
    create_displacement_offset_test_labels_csv,
)
from src.anomaly_scripts.force_offset_reverification_labels import (
    create_force_offset_test_labels_csv,
)
from src.anomaly_scripts.tip_displacement_reverification_labels import (
    create_tip_displacement_test_labels_csv,
)
from src.anomaly_scripts.too_deep_reverification_labels import (
    create_too_deep_test_labels_csv,
)
from src.anomaly_scripts.unusual_loading_curvature_reverification_labels import (
    create_unusual_loading_curvature_test_labels_csv,
)
from src.anomaly_scripts.unusual_unloading_curvature_reverification_labels import (
    create_unusual_unloading_curvature_test_labels_csv,
)
from src.utils.setup_data import verify_anomaly_type, verify_datapath


def verify_argv(argv: list):
    if len(argv) != 3:
        print(f"Usage: {argv[0]} anomaly_type datapath")
        print("   anomaly_type : anomaly type to create a reverification csv for")
        print(
            '   datapath : path to data that contains "all_samples_labelled.csv" and '
            '"full_data.npy" (or "Nanoindent_data/" to generate "full_data.npy")'
        )
        sys.exit(1)

    anomaly_type = argv[1]
    datapath = argv[2]
    verify_anomaly_type(anomaly_type)
    verify_datapath(datapath)


def reverify_labels(args=None, progress=None):
    task = progress.add_task("[cyan]Reverifying labels...", start=False)

    progress.start_task(task)
    if not args:
        verify_argv(sys.argv)

        anomaly_type = sys.argv[1]
        datapath = sys.argv[2]
    else:
        anomaly_type = args.anomaly
        datapath = args.datapath

    if anomaly_type == "displacement_offset":
        create_displacement_offset_test_labels_csv(datapath, progress)
    elif anomaly_type == "force_offset":
        create_force_offset_test_labels_csv(datapath, progress)
    elif anomaly_type == "tip_displacement":
        create_tip_displacement_test_labels_csv(datapath, progress)
    elif anomaly_type == "too_deep":
        create_too_deep_test_labels_csv(datapath, progress)
    elif anomaly_type == "unusual_loading_curvature":
        create_unusual_loading_curvature_test_labels_csv(datapath, progress)
    elif anomaly_type == "unusual_unloading_curvature":
        create_unusual_unloading_curvature_test_labels_csv(datapath, progress)
    progress.update(task, completed=100)


if __name__ == "__main__":
    reverify_labels()
