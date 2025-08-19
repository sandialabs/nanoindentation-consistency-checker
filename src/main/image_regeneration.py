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

import os
import sys

import pandas as pd
from matplotlib import pyplot as plt

from src.main.config import INDENT_END, LOADING_PORTION_END, UNLOADING_PORTION_START
from src.utils.indent_manager import IndentManager
from src.utils.preprocess import ANOMALY_TYPES
from src.utils.setup_data import CustomFileError, verify_anomaly_type, verify_datapath


def get_column_name(column_number):
    if column_number == 0:
        return r"Depth nm"
    elif column_number == 1:
        return r"Load $\mu N$"
    elif column_number == 2:
        return r"Timestep"
    else:
        raise ValueError("column_number must be 0, 1, or 2")


def generate_reverification_labels(
    progress, datapath, anomaly_type, indent_range, column1, column2
):
    # Read in labels
    im = IndentManager(datapath, ANOMALY_TYPES)
    if anomaly_type == "all":
        test_set = im.all_samples_df.copy()
    else:
        test_set = pd.read_csv(
            os.path.join("results", f"{anomaly_type}_test_labels.csv")
        )

    # Set index by sample
    test_set.set_index("sample_num", inplace=True)

    IMAGE_FOLDER_PATH = "images"

    # Generate folder for all of the images
    os.makedirs(IMAGE_FOLDER_PATH, exist_ok=True)

    sample_nums = test_set.index.unique()
    task1 = progress.add_task("[green]Rendering samples...", total=len(sample_nums))
    # For each sample, render all indents in grey and overlay a black indent for
    # each individual indent per sample
    fig, ax = plt.subplots()
    for sample_num in sample_nums:
        for indent_num in range(im.sample_size):
            indent = im.indents[(sample_num, indent_num)]
            ax.plot(
                indent[indent_range, column1],
                indent[indent_range, column2],
                color="silver",
                linewidth=0.75,
            )

        # Set x and y labels for load displacement curves
        ax.set_xlabel(get_column_name(column1))
        ax.set_ylabel(get_column_name(column2))

        (highlight,) = ax.plot(
            indent[indent_range, column1],
            indent[indent_range, column2],
            color="black",
            linewidth=1.5,
        )

        indents = range(test_set.loc[[sample_num]].shape[0])
        task2 = progress.add_task(
            f"[green]Rendering indents for sample {sample_num}...", total=len(indents)
        )
        # Generate images for indents
        for i in indents:
            indent_num = test_set.loc[[sample_num], "indent_num"].iloc[i]
            filename = os.path.join(
                IMAGE_FOLDER_PATH, f"{anomaly_type}_{sample_num}_{indent_num}.png"
            )
            if os.path.exists(filename):
                progress.update(task2, advance=1)
                continue

            indent = im.indents[(sample_num, indent_num)]
            highlight.set_xdata(indent[indent_range, column1])
            highlight.set_ydata(indent[indent_range, column2])
            ax.set_title(f"Sample {sample_num}, indent {indent_num}")
            fig.canvas.draw()

            try:
                plt.savefig(filename)
            except FileNotFoundError as e:
                raise CustomFileError(e, filename)

            progress.update(task2, advance=1)

        ax.clear()
        progress.remove_task(task2)
        progress.update(task1, advance=1)

    plt.close()


def verify_argv(argv: list):
    if len(argv) != 3:
        print(f"Usage: {argv[0]} anomaly_type datapath")
        print("   anomaly_type : anomaly type to create a reverification images for")
        print(
            '   datapath : path to data that contains "all_samples_labelled.csv" and '
            '"full_data.npy" (or "Nanoindent_data/" to generate "full_data.npy")'
        )
        sys.exit(1)

    anomaly_type = argv[1]
    datapath = argv[2]
    verify_anomaly_type(anomaly_type, True)
    verify_datapath(datapath)


def generate_images(args=None, progress=None):
    if not args:
        verify_argv(sys.argv)

        anomaly_type = sys.argv[1]
        datapath = sys.argv[2]
    else:
        anomaly_type = args.anomaly
        datapath = args.datapath

    task = progress.add_task(f"[cyan]Generating {anomaly_type} images...", start=False)

    progress.start_task(task)
    if anomaly_type == "all":
        generate_reverification_labels(
            progress, datapath, "all", range(0, INDENT_END), 0, 1
        )
    elif anomaly_type == "displacement_offset":
        generate_reverification_labels(
            progress,
            datapath,
            "displacement_offset",
            range(0, LOADING_PORTION_END // 2),
            2,
            0,
        )
    elif anomaly_type == "force_offset":
        generate_reverification_labels(
            progress, datapath, "force_offset", range(0, LOADING_PORTION_END // 2), 2, 1
        )
    elif anomaly_type == "tip_displacement":
        generate_reverification_labels(
            progress,
            datapath,
            "tip_displacement",
            range(LOADING_PORTION_END, UNLOADING_PORTION_START),
            2,
            0,
        )
    elif anomaly_type == "too_deep":
        generate_reverification_labels(
            progress, datapath, "too_deep", range(0, LOADING_PORTION_END), 0, 1
        )
    elif anomaly_type == "unusual_loading_curvature":
        generate_reverification_labels(
            progress,
            datapath,
            "unusual_loading_curvature",
            range(0, LOADING_PORTION_END),
            0,
            1,
        )
    elif anomaly_type == "unusual_unloading_curvature":
        generate_reverification_labels(
            progress,
            datapath,
            "unusual_unloading_curvature",
            range(UNLOADING_PORTION_START, INDENT_END),
            0,
            1,
        )
    progress.update(task, completed=100)


if __name__ == "__main__":
    generate_images()
