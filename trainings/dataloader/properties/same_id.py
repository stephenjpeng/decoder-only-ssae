import json
import os
from enum import Enum
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np

from trainings.dataloader.properties.properties import Properties
from trainings.utils.logger import Logger


def get_cid_never_same(
    properties_same_json: Dict[str, str], category_to_cid: Dict[str, int]
) -> tuple[List[str], List[int]]:
    categories_never_same = [k for (k, v) in properties_same_json.items() if not v]
    cid_never_same = [category_to_cid[cns] for cns in categories_never_same]
    return categories_never_same, cid_never_same


def get_pid_never_same(
    cid_never_same: List[str], cid_to_pids: Dict[int, np.array]
) -> tuple[np.array, int]:
    # categories can have different numbers of properties, so this must stay
    # a flat list comprehension rather than np.array(list_of_lists) -- the
    # latter requires a rectangular shape and raises on ragged input.
    pid_never_same = np.array(
        [pid for cns in cid_never_same for pid in cid_to_pids[cns]]
    )
    return pid_never_same, len(pid_never_same)


def get_cid_to_pid_same(
    properties_same_json: Dict[str, str],
    category_to_cid: Dict[str, int],
    property_to_pid: Dict[str, int],
) -> Dict[int, np.array]:
    # mapping properties_same_json to ids

    cid_to_pid_same = {}
    for category, properties in properties_same_json.items():
        if properties:  # False means it is never same
            cid_to_pid_same[category_to_cid[category]] = [
                property_to_pid[property] for property in properties
            ]

    return cid_to_pid_same


def get_tid_same(
    tid_to_pids: Dict[int, np.array],
    cid_to_pid_same: Dict[int, np.array],
    cid_never_same: List[int],
) -> List[int]:
    tid_same = []
    for tid, pids in tid_to_pids.items():
        is_same = True
        for cid, pid in enumerate(pids):
            if cid not in cid_never_same and pid not in cid_to_pid_same[cid]:
                is_same = False

        if is_same:
            tid_same.append(tid)

    return tid_same


class SameId:
    def __init__(self, folder_path, properties: Properties, logger: Logger = None):
        self.folder_path = folder_path
        self.properties = properties
        self.logger = logger
        self.log_print = print if self.logger is None else logger.print

        self.log_print("Initializing SameId class...")

        self.filename_categories_with_properties = Path(
            folder_path, "properties_same.json"
        )

        self.reader()
        self.check_inputs()

        self.get_mappings_nerver_same()
        self.get_mapping_tid_same()

        self.log_print("categories_never_same : ", self.categories_never_same)
        self.log_print("n_pid_never_same : ", self.n_pid_never_same)
        self.log_print("cid_never_same : ", self.cid_never_same)
        self.log_print("pid_never_same : ", self.pid_never_same)
        self.log_print("cid_to_pid_same : ", self.cid_to_pid_same)
        self.log_print("tid_same : ", self.tid_same)

        self.log_print(
            f"{len(self.tid_same)} prompts are the same id, over a total of {len(self.properties.tid_to_pids)} prompts."
        )

        self.log_print("SameId class initialized.")

    def reader(self):
        with open(self.folder_path + "/properties_same.json", "r") as f:
            self.properties_same_json = json.load(f)

    def check_inputs(self) -> None:
        # check that properties_same_json has the same categories as category_to_cid

        if not sorted(list(self.properties.category_to_cid.keys())) == sorted(
            list(self.properties_same_json.keys())
        ):
            raise Exception(
                "Using same identity: properties_same.json must have the same keys as properties.json"
            )

    def get_mappings_nerver_same(self) -> None:
        self.categories_never_same, self.cid_never_same = get_cid_never_same(
            self.properties_same_json, self.properties.category_to_cid
        )

        self.pid_never_same, self.n_pid_never_same = get_pid_never_same(
            self.cid_never_same, self.properties.cid_to_pids
        )

    def get_mapping_tid_same(self) -> None:
        self.cid_to_pid_same = get_cid_to_pid_same(
            self.properties_same_json,
            self.properties.category_to_cid,
            self.properties.property_to_pid,
        )

        self.tid_same = get_tid_same(
            self.properties.tid_to_pids, self.cid_to_pid_same, self.cid_never_same
        )
