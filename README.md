# Nanoindentation Label Consistency Checker

Nanoindentation anomaly labels aren't readily available, which could lead newly
labelled datasets to be inconsistent with how labels are applied. This
repository aims to mitigate inconsistency in anomaly labels through
machine learning models and statistical approaches. This is accomplished
through three parts:

1. **Label reverification**: The dataset is examined in a certain way for an
   anomaly type to create a csv for which indents are predicted to be labelled
   inconsistently with respect to other labelled indents.
2. **Image generation**: Given a label reverification csv, images are created
   for a certain anomaly type based on which aspects of an indent are
   related to that anomaly type.
3. **Interactive consistency check**: Given the images and label
   reverification csv, a GUI can be interacted with to reverify inconsistently
   labelled indents.

# Installation and Usage

These three parts can be ran through programs under the `src` folder.
Before usage, please follow the directions below to properly set up and install
all requirements. Information on how to run each part is in the accompanying
[README](./src/README.md).

## Getting Started

### Python Virtual Environment

Run the following command to create a clean virtual environment:

```
# Replace <name-of-env> with whatever name you prefer
$ python -m venv <name-of-env>
```

This command will create a new directory named `name-of-env` in your current
working directory. To activate the virtual environment, use the following
command:

```
# MacOS/Linux
$ source <name-of-env>/bin/activate
# Windows
$ <name-of-env>\Scripts\activate
```


After activation, you should see the name of your environment prefixed in your
terminal prompt, like this:

```
(<name-of-env>) $
```

To deactivate it, simply type `deactivate` and hit enter.

### Installation

Change your directory to where the package code is located:

```
$ cd /path/to/nanindentation-consistency-checker/
```

 Run the following command to install the package:

 ```
 $ pip install -e .
 ```

Installing in "editable" mode allows any changes you make to the code to be
reflected immediately in your environment without needing to reinstall
the package.

## File Structure

    code/
    ├── src/               # Main source code
    ├── tests/             # Tests
    └── README.md

## Data

### Required Data

Create a folder that contains all of your nanoindentation data:

    data/
    ├── Nanoindent_data            # Folder containing each indent's load-displacement data
    └── all_samples_labelled.csv   # csv for each indent's metadata


#### Nanoindent_data

This folder needs to contain a csv for each indent. Each row in
`all_samples_labelled.csv` will point to their respective csv in this folder.
Each csv needs to adhere to the following requirements:

- 3 columns: `depth_nm`, `load_micro_N`, `time_s`
- All csvs must have the same number of rows
- All values are of type float


#### all_samples_labelled.csv

This csv needs to contain the following columns with restrictions:

- **sample_num**: Integer with equal amount of repeat numbers
- **indent_num**: Integer that is unique within same sample_num but is repeated for  sample_num
- **path**: Relative path in `Nanoindent_data/` that points to a csv
- **verify_zero_offset_type**: Leftover from our dataset where we assume `verify_zero_offset_type`
  is `force_offset`. 0 to represent indent doesn't have this anomaly, any other number to represent indent does have this anomaly
- **fully_anomalous_sample**: 0 to represent indent doesn't have this anomaly, any other number to represent indent does have this anomaly
- **force_offset**: 0 to represent indent doesn't have this anomaly, any other number to represent indent does have this anomaly
- **displacement_offset**: 0 to represent indent doesn't have this anomaly, any other number to represent indent does have this anomaly
- **unusual_unloading_curvature**: 0 to represent indent doesn't have this anomaly, any other number to represent indent does have this anomaly
- **unusual_loading_curvature**: 0 to represent indent doesn't have this anomaly, any other number to represent indent does have this anomaly
- **tip_displacement_decreases_at_holding**: 0 to represent indent doesn't have this anomaly, any other number to represent indent does have this anomaly
- **too_deep**: 0 to represent indent doesn't have this anomaly, any other number to represent indent does have this anomaly


### Configuring Program to Work with Your Dataset

Before running the programs in the `src` folder, `src/main/config.py`
needs to be modified for a specific dataset. Currently, `src/main/config.py`
is set to values working with our dataset. 

- **LOADING_PORTION_END**: Index of the end of the loading portion/start of holding portion
- **UNLOADING_PORTION_START**: Index of the start of the unloading portion/end of holding portion
- **INDENT_END**: Index of last point in indent

You can also optionally set the path to your data folder to avoid needing
to supply the path while running the various steps:

- **USER_DATAPATH**: Full path to required data

-------------

Legal Disclaimer
----------------

By contributing to this software project, you are agreeing to the
following terms and conditions for your contributions:

1. You agree your contributions are submitted under the BSD license.
2. You represent you are authorized to make the contributions and grant
   the license. If your employer has rights to intellectual property that
   includes your contributions, you represent that you have received
   permission to make contributions and grant the required license on
   behalf of that employer.

