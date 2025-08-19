#  ___________________________________________________________________________
#  Copyright (c) 2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import argparse
import importlib.util
import os
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
)

from src.utils.setup_data import verify_anomaly_type, verify_datapath
from src.main.reverification_labels import reverify_labels
from src.main.image_regeneration import generate_images
from src.main.relabel_gui import activate_gui


def argparse_verify_datapath(datapath: str):
    try:
        verify_datapath(datapath)
        return datapath
    except Exception as e:
        raise argparse.ArgumentTypeError(str(e))


def argparse_verify_anomaly_type(anomaly_type: str, include_all=False):
    try:
        verify_anomaly_type(anomaly_type, include_all)
        return anomaly_type
    except Exception as e:
        raise argparse.ArgumentTypeError(str(e))


def get_datapath_from_config():
    config_file_path = os.path.join(os.path.dirname(__file__), 'config.py')
    spec = importlib.util.spec_from_file_location("config", config_file_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    # Expand and fully resolve the path to avoid oddities
    expanded_path = os.path.expanduser(config.USER_DATAPATH)
    full_path = os.path.abspath(expanded_path)

    return full_path


def build_parser():
    # The top level parser.
    parser = argparse.ArgumentParser(
        description="The main function for nanoindentation utilities. \
            Type 'nanoindent COMMAND --help' to get information \
            about a particular command."
    )

    # Holds the parsers for all the individual commands.
    subparsers = parser.add_subparsers(title="commands", dest="command")

    subparser_reverify = subparsers.add_parser(
        'reverify-labels', description="Create a csv for an anomaly type."
    )
    subparser_reverify.add_argument(
        '--anomaly',
        type=lambda x: argparse_verify_anomaly_type(x, False),
        default=None,
        required=True,
        help="Anomaly type for which to create a reverification csv.",
    )
    subparser_reverify.add_argument(
        '--datapath',
        type=argparse_verify_datapath,
        default=None,
        required=False,
        help="Path to data that contains 'all_samples_labelled.csv' and "
        "'full_data.npy' (or 'Nanoindent_data/' to generate 'full_data.npy'). "
        "Not required if data path is set in config.py.",
    )

    subparser_images = subparsers.add_parser(
        'generate-images',
        description="Generate images for all indents and specific anomaly types.",
    )
    subparser_images.add_argument(
        '--anomaly',
        type=lambda x: argparse_verify_anomaly_type(x, True),
        default='all',
        help="Select `all` to images for all indents or select a specific anomaly type. "
        "This relies on the existence of the `results/{anomaly_type}_test_labels.csv` file. "
        "Default is `all`.",
    )
    subparser_images.add_argument(
        '--datapath',
        type=argparse_verify_datapath,
        default=None,
        required=False,
        help="Path to data that contains 'all_samples_labelled.csv' and "
        "'full_data.npy' (or 'Nanoindent_data/' to generate 'full_data.npy'). "
        "Not required if data path is set in config.py.",
    )

    subparser_gui = subparsers.add_parser(
        'activate-gui',
        description="View and interact with reverification labels from an anomaly type. "
        "The `reverify-labels` and `generate-images` commands must be completed "
        "in order for this command to work.",
    )
    subparser_gui.add_argument(
        '--anomaly',
        type=lambda x: argparse_verify_anomaly_type(x, False),
        default=None,
        required=True,
        help="Anomaly type for which to open an interactive window.",
    )

    return parser


def runner():
    parser = build_parser()
    arguments = parser.parse_args()

    # Get datapath from config.py if not provided in command line
    if hasattr(arguments, 'datapath') and arguments.datapath is None:
        datapath = get_datapath_from_config()
        if datapath == '':
            parser.error(
                "Datapath must be provided either in the command "
                f"line or in main/config.py for the '{arguments.command}' command "
            )
        arguments.datapath = datapath

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
    ) as progress:
        if arguments.command == "reverify-labels":
            reverify_labels(arguments, progress)
        elif arguments.command == "generate-images":
            generate_images(arguments, progress)
        elif arguments.command == "activate-gui":
            activate_gui(arguments)
        else:
            print("Please choose one of the options:\n")
            parser.print_help()


if __name__ == "__main__":
    runner()
