#  ___________________________________________________________________________
#  Copyright (c) 2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________


"""
Interactive GUI based on
    - "images/all_images"
    - "images/{anomaly_type}_images"
    - "results/{anomaly_type}_test_labels.csv"
Creates "new_labels/new_{anomaly_type}_test_labels.csv"
"""

import os
import sys
import tkinter as tk
from datetime import datetime
from tkinter import messagebox, simpledialog

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from src.utils.setup_data import CustomFileError, verify_anomaly_type


def get_sample_indent_nums(filename):
    sample_num = filename.split("_")[-2]
    indent_num = filename.split("_")[-1].replace(".png", "")

    return sample_num, indent_num


def verify_same_pairs(df1, df2):
    df1_set = set(map(tuple, df1[["sample_num", "indent_num"]].values.astype(int)))
    df2_set = set(map(tuple, df2[["sample_num", "indent_num"]].values.astype(int)))

    return df1_set == df2_set


class ReevaluateIndents:
    """App class"""

    def __init__(self, master, anomaly_type):
        self.master = master
        self.master.title("Plot Viewer")

        self.anomaly_type = anomaly_type
        if anomaly_type == "tip_displacement":
            anomaly_type = "tip_displacement_decreases_at_holding"

        self.abnormal_text = " ".join(self.anomaly_type.split("_")).capitalize()

        self.new_label_df = pd.read_csv(
            os.path.join("results", f"{self.anomaly_type}_test_labels.csv")
        )[["sample_num", "indent_num", anomaly_type, "consistency_flag"]]

        IMAGE_FOLDER_PATH = "images"

        self.anomaly_type_imlist = []
        for sample_num, sample_df in self.new_label_df.groupby("sample_num"):
            indent_nums = sample_df["indent_num"].values
            for indent_num in indent_nums:
                filename = os.path.join(
                    IMAGE_FOLDER_PATH,
                    f"{self.anomaly_type}_{sample_num}_{indent_num}.png",
                )
                self.anomaly_type_imlist.append(filename)

        has_same_pairs = verify_same_pairs(
            self.new_label_df,
            pd.DataFrame(
                [
                    {"sample_num": sample_num, "indent_num": indent_num}
                    for filename in self.anomaly_type_imlist
                    for sample_num, indent_num in [get_sample_indent_nums(filename)]
                ]
            ),
        )
        if not has_same_pairs:
            raise ValueError(
                f"{os.path.join('results', self.anomaly_type)}_test_labels.csv and {os.path.join('images', self.anomaly_type)}_images must have the same (sample_num, indent_num) pairs."
            )

        self.new_label_df.set_index(["sample_num", "indent_num"], inplace=True)
        self.new_label_df.rename(
            columns={anomaly_type: f"old_{self.anomaly_type}"}, inplace=True
        )
        self.new_label_df[f"new_{self.anomaly_type}"] = ""
        self.new_label_df["consistency_flag"] = self.new_label_df.pop(
            "consistency_flag"
        )

        self.current_index = 0

        # Create figure and canvas
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(14, 5))

        self.canvas = FigureCanvasTkAgg(self.fig, master=master)
        self.canvas.draw()
        # self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.canvas.get_tk_widget().grid(row=0, column=0, columnspan=28)

        self.initialize_buttons()
        self.master.protocol("WM_DELETE_WINDOW", self.quit)

        # Display the first plot
        self.display_plot()

    def initialize_buttons(self):
        """
        This function is used to initialize the buttons in the GUI:
            - normal_button: clicking this indicates that the indentation curve would be considered
                normal
            - abnormal_button: clicking this indicates that there is an abnormal anomaly_type, and
                generates the abnormality buttons to specify a level (1=small, 2=average, 3=large).
            - other_button: clicking this allows the user to write a message for the indent rather
                than clicking a classification.
        """

        self.normal_button = tk.Button(
            self.master, text="Normal", command=self.record_normal
        )
        # self.normal_button.pack(side=tk.TOP, padx=20)
        self.normal_button.grid(row=1, column=10, padx=10, pady=5)

        self.abnormal_button = tk.Button(
            self.master, text=self.abnormal_text, command=self.record_abnormal
        )
        # self.abnormal_button.pack(side=tk.TOP, padx=20, pady=5)
        self.abnormal_button.grid(row=1, column=11, padx=10, pady=5)

        self.other_button = tk.Button(
            self.master, text="Other", command=self.record_other
        )
        # self.other_button.pack(side=tk.TOP, padx=20, pady=5)
        self.other_button.grid(row=1, column=12, padx=10, pady=5)

        self.back_button = tk.Button(
            self.master, text="Back", command=self.previous_plot
        )
        # self.back_button.pack(side=tk.LEFT, padx=20, pady=5)
        self.back_button.grid(row=1, column=20, padx=5, pady=5)

        # Additional buttons for abnormality levels (initially hidden)
        self.abnormality_buttons_frame = tk.Frame(self.master)
        self.abnormality_buttons = []
        for level in range(1, 4):
            button = tk.Button(
                self.abnormality_buttons_frame,
                text=f"Level {level}",
                command=lambda l=level: self.record_abnormality(l),
            )
            # button.pack(side=tk.LEFT, padx=5)
            button.grid(row=2, column=level + 11, pady=5)
            self.abnormality_buttons.append(button)

    def display_plot(self):
        """Displays current plot"""
        if self.current_index < len(self.anomaly_type_imlist):
            self.ax1.clear()
            img1 = mpimg.imread(self.anomaly_type_imlist[self.current_index])
            fpath2 = self.anomaly_type_imlist[self.current_index].replace(
                os.path.join("images", self.anomaly_type), os.path.join("images", "all")
            )
            img2 = mpimg.imread(fpath2)
            self.ax1.imshow(img1)
            self.ax2.imshow(img2)

            self.ax1.axis("off")
            self.ax2.axis("off")

            self.ax1.set_title("Initial Curve")
            self.ax2.set_title("Full Load Displacement Curve")

            self.canvas.draw()
        else:
            messagebox.showinfo("Info", "No more plots to display.")
            self.quit()

    def record(self, record_value):
        """Records response"""
        sample_num, indent_num = get_sample_indent_nums(
            self.anomaly_type_imlist[self.current_index]
        )
        self.new_label_df.loc[
            (int(sample_num), int(indent_num)), f"new_{self.anomaly_type}"
        ] = record_value

    def record_normal(self):
        """Records normal response"""
        self.record(0)
        self.next_plot()

    def record_abnormal(self):
        """Records abnormal response"""
        self.abnormality_buttons_frame.grid(
            row=2, column=11
        )  # Show abnormality level buttons

    def record_abnormality(self, level):
        """Records abnormal level response"""
        self.record(level)
        self.abnormality_buttons_frame.grid_forget()
        self.next_plot()

    def record_other(self):
        """Records other response"""
        response = simpledialog.askstring("Input", "Please describe the abnormality:")
        if response:
            self.record(response)
            self.next_plot()

    def previous_plot(self):
        """Shows previous plot"""
        if self.current_index > 0:
            self.current_index -= 1
            self.abnormality_buttons_frame.grid_forget()  # Hide abnormality level buttons
            self.display_plot()
        else:
            messagebox.showinfo("Info", "This is the first plot.")

    def next_plot(self):
        """Shows next plot"""
        self.current_index += 1
        self.display_plot()

    def quit(self):
        self.master.quit()
        self.master.destroy()


def verify_argv(argv: list):
    if len(argv) != 2:
        print(f"Usage: {argv[0]} anomaly_type")
        print("   anomaly_type : anomaly type to open a interactive GUI for")
        sys.exit(1)

    anomaly_type = argv[1]
    verify_anomaly_type(anomaly_type)


def activate_gui(args=None):
    if not args:
        verify_argv(sys.argv)

        anomaly_type = sys.argv[1]
    else:
        anomaly_type = args.anomaly

    root = tk.Tk()
    app = ReevaluateIndents(root, anomaly_type)
    root.mainloop()

    NEW_LABELS_PATH = "new_labels"
    os.makedirs(NEW_LABELS_PATH, exist_ok=True)

    filename = (
        f"{anomaly_type}_test_labels_{datetime.now().strftime(r'%y-%m-%d_%H-%M')}.csv"
    )
    destination = os.path.join(NEW_LABELS_PATH, filename)

    try:
        app.new_label_df.to_csv(destination, index=True)
    except FileNotFoundError as e:
        raise CustomFileError(e, destination)


if __name__ == "__main__":
    activate_gui()
