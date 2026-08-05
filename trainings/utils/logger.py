import json
import os
import shutil
import warnings
from datetime import datetime
from pathlib import Path

import git
import matplotlib.pyplot as plt
import numpy as np


class Logger:
    def __init__(self, log_folder, path_yaml, overwrite=False, plot_frequency=None):
        self.log_folder = log_folder
        self.path_yaml = Path(path_yaml)
        self.overwrite = overwrite
        self.plot_frequency = plot_frequency

        self.logs = {}
        self.log_keys = []

        self._create_log_folder()
        self._init_log_folder()
        self._init_plots()

    def _create_log_folder(self):
        if os.path.exists(self.log_folder):
            if self.overwrite:
                shutil.rmtree(self.log_folder)
            else:
                raise Exception(f"Folder {self.log_folder} already exists.")

        os.mkdir(self.log_folder)

    def _init_log_folder(self):
        self.print(f"Logs from file : {self.log_folder}")
        self.print(f"Starting at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        shutil.copy(self.path_yaml, Path(self.log_folder, "params.yaml"))

        self._register_git_commit()

    def _init_plots(self):
        self.current_iter = 0
        os.mkdir(Path(self.log_folder, "plots"))

    def print(self, *args):
        text = "".join(map(str, args))
        with open(self.log_folder + "/training.log", "a") as f:
            f.write(text + "\n")
        print(text)

    def _register_git_commit(self):
        # optional; runs off git-less worktrees (rsync'd copies, S3 checkouts) also log fine
        try:
            repo = git.Repo(".")
        except git.InvalidGitRepositoryError:
            self.print("git commit : (not a git repo)")
            return
        git_commit = repo.git.rev_parse("HEAD")
        self.print(f"git commit : {git_commit}")

    def log(self, key, iteration, value):
        if type(value) == np.ndarray:
            value = value.item()

        if key not in self.log_keys:
            self.log_keys.append(key)
            self.logs[key] = {}

        self.logs[key][iteration] = value

        # plot every plot_frequency
        self.current_iter += 1
        if self.plot_frequency is not None and self.plot_frequency == self.current_iter:
            self.plot_logs(key)
            self.current_iter = 0

    def plot_logs(self, key):
        plt.plot(list(self.logs[key].keys()), list(self.logs[key].values()))
        plt.grid(True)
        plt.title(key)
        plt.xlabel("Epochs")
        plt.ylabel(key)
        plt.savefig(Path(self.log_folder, f"plots/{key}.png"))
        plt.yscale("log")
        plt.savefig(Path(self.log_folder, f"plots/{key}_log.png"))
        plt.close()

    def register_metadata(self, metadata):
        """Adds metadata entries from a dict

        Args:
            metadata (dict): dict specifying the metadata entries to add
        """
        for key in metadata.keys():
            self.metadata[key] = metadata[key]
        if self.log_dir is not None:
            self._save_metadata()

    def save(self):
        """Saves log data to log folder"""
        if self.log_dir is not None:
            results_filename = self.log_folder + "/results.json"
            with open(results_filename, "w") as file:
                json.dump(self.logs, file)

        else:
            warnings.warn("The logger does not have a log_dir so was not saved to disk")

    def log_array_comparison(self, title, array1, array2):
        line1 = "  ".join([f"{x:8.2f}" for x in array1])
        line2 = "  ".join([f"{x:8.2f}" for x in array2])

        with open(self.log_folder + "/predictions.txt", "a") as f:
            f.write(title + ": \n" + line1 + "\n" + line2 + "\n\n")

    def log_model(self, model):
        with open(self.log_folder + "/model_architecture.txt", "w") as f:
            f.write(str(model))

        self.print("Model architecture saved to model_architecture.txt")

    def log_matrix(self, filename, matrix_np, fmt="%10.4f"):
        # log a numpy matrix as a text file (to debug)
        # filename must be .txt
        # format fmt : float '%2f', binary '%d'
        np.savetxt(self.log_folder + "/" + filename, matrix_np, fmt=fmt, delimiter=" ")
