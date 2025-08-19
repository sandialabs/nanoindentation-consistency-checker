# Consistency Checker

Reverification of nanoindentation is split into three parts:

- Detecting inconsistencies
- Visualizing indents based on anomaly type
- Manual reverification of detected inconsistencies

There are 6 supported anomaly types in this repository with their own characteristics.

## Anomaly Types

- `displacement_offset`: Specific type of a zero offset anomaly. Characterized by
  depth being less than 0 or a "jump" in depth during the beginning of the loading portion.
- `force_offset`: Specific type of a zero offset anomaly. Characterized by a "jump"
  load at the start of the loading portion.
- `tip_displacement`: This anomaly type is actually called "tip_displacement_decreases_at_holding",
  but tip_displacement is used when running commands. Characterized by depth
  decreasing during the holding portion.
- `too_deep`: Characterized by sufficiently larger depth during the loading
  portion compared to normal indents.
- `unusual_loading_curvature`: Characterized by a sufficiently different loading
  portion compared to normal indents.
- `unusual_unloading_curvature`: Characterized by a sufficiently different unloading
  portion compared to normal indents.

In all steps below, users will need to replace `{anomaly_type}` with an anomaly
type from the above list.

## Usage

The main script to run all functionality is in `main/main.py`.
Its usage is listed below:

```
$ python src/main/main.py -h
usage: main.py [-h] {reverify-labels,generate-images,activate-gui} ...

The main function for nanoindentation utilities. Type 'nanoindent COMMAND --help' to get information about a particular command.

options:
  -h, --help            show this help message and exit

commands:
  {reverify-labels,generate-images,activate-gui}
```

## Create Reverification CSV

Run the following command to create a csv for an anomaly type:

```
$ python src/main/main.py reverify-labels -h
usage: main.py reverify-labels [-h] --anomaly ANOMALY [--datapath DATAPATH]

Create a csv for an anomaly type.

options:
  -h, --help           show this help message and exit
  --anomaly ANOMALY    Anomaly type for which to create a reverification csv.
  --datapath DATAPATH  Path to data that contains 'all_samples_labelled.csv' and 'full_data.npy' (or
                       'Nanoindent_data/' to generate 'full_data.npy'). Not required if data path is set in config.py.
```

An example command:

```
python src/main/main.py reverify-labels --datapath ~/Data/my_data --anomaly force_offset
```

This will create `results/{anomaly_type}_test_labels.csv`

## Create Images

Run the following command:

```
$ python src/main/main.py generate-images -h
usage: main.py generate-images [-h] [--anomaly ANOMALY] [--datapath DATAPATH]

Generate images for all indents and specific anomaly types.

options:
  -h, --help           show this help message and exit
  --anomaly ANOMALY    Select `all` to images for all indents or select a specific anomaly type. This relies on the
                       existence of the `results/{anomaly_type}_test_labels.csv` file. Default is `all`.
  --datapath DATAPATH  Path to data that contains 'all_samples_labelled.csv' and 'full_data.npy' (or
                       'Nanoindent_data/' to generate 'full_data.npy'). Not required if data path is set in config.py.
```

### Create images for all indents

To generate images for all indents, run the following command:

```
python src/main/main.py generate-images --datapath ~/Data/my_data
```

This command only needs to be run once. If all indent images are already created,
there is no need to redo this step as it takes a long time (around 30 minutes).

This will generate `images/all_images/`.


### Generate reverification images

An example command:

```
python src/main/main.py generate-images --anomaly force_offset --datapath ~/Data/my_data
```

This step needs to be completed per anomaly type. It relies on
`results/{anomaly_type}_test_labels.csv`

This will generate `images/{anomaly_type}_images/`.

If this program is terminated while running, rerunning this command will
skip already generated images.

To regenerate images, delete `images/{anomaly_type}_images/` and run the
command again.


## Interact with GUI

To view and interact with reverification labels from an anomaly type, the
previous steps must be completed and the following files/folders must be present:

- `images/all_images/`
- `images/{anomaly_type}_images/`
- `results/{anomaly_type}_test_labels.csv`

Run the following command to open the GUI:

```
$ python src/main/main.py activate-gui -h
usage: main.py activate-gui [-h] --anomaly ANOMALY

View and interact with reverification labels from an anomaly type. The
`reverify-labels` and `generate-images` commands must be
completed in order for this command to work.

options:
  -h, --help         show this help message and exit
  --anomaly ANOMALY  Anomaly type for which to open an interactive window.
```

An example command:

```
python src/main/main.py activate-gui --anomaly force_offset
```

After reverifying all test labels, this will create
`new_labels/{anomaly_type}/new_{anomaly_type}_test_labels_[timestamp].csv` with
the following columns:

- **sample_num**: Sample number
- **indent_num**: Indent number
- **old_{anomaly_type}**: Original label for {anomaly_type}
- **new_{anomaly_type}**: New label for {anomaly_type} based on interaction with GUI
- **consistency_flag**: 0 if not marked for reverification, 1 if marked for reverification

For traceability, we create a file with a unique timestamp each time the
program is ran to ensure accidental changes or quitting the program can be saved or reversed.


## File Structure
    .
    ├── anomaly_scripts/                # Scripts for each anomaly type to identify inconsistencies
    ├── images/                         # Generated folder for indent visualization
    ├── main/                           # Folder with the main user scripts
    ├── new_labels/                     # Generated folder for user selected new labels based on detected inconsistencies
    ├── results/]                       # Generated folder for identified inconsistencies
    ├── utils/                          # Utility functions
    └── README.md
