from datetime import datetime
import getpass
import pandas as pd
import pickle
import os

import quick_pp.las_handler as las


class Project:
    def __init__(self, name="", description="", user=getpass.getuser(), time=datetime.now()):
        self.name = name or f"New_Project_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.description = description
        self.modified_by = user
        self.modified_date = time
        self.data = {}
        self.history = []

    def add_data(self, data, group_by="WELL_NAME"):
        for well_name, well_data in data.groupby(group_by):
            well = Well(well_name, f"Well {well_name}")
            well.add_data(well_data)
            self.data.update({well_name: well})
        self.update_history(action=f"Added data to project {self.name}")

    def add_well(self, well):
        self.data.update({well.name: well})
        self.update_history(action=f"Added well {well.name} to project {self.name}")

    def read_las(self, path: list):
        for file in path:
            well = Well("", "")
            well.read_las(file)
            header_df = well.header.T
            well.name = header_df[header_df['mnemonic'] == 'WELL']['value'].values[0]
            well.description = f"Well {well.name}"
            self.add_well(well)
        self.update_history(action=f"Read LAS file for project {self.name}")

    def save(self, name="", folder="data/04_project", ext=".qpp"):
        if not os.path.exists(folder):
            os.makedirs(folder)
        path = os.path.join(folder, f"{name}{ext}" or f"{self.name}.qpp")
        with open(path, "wb") as f:
            pickle.dump(self, f)
        self.update_history(action=f"Saved project to {path}")

    def load(self, path: str):
        with open(path, "rb") as f:
            project = pickle.load(f)
        return project

    def update_history(self, user=getpass.getuser(), time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'), action=""):
        self.history.append({"user": user, "time": time, "action": action})

    def __str__(self):
        return f"Project: {self.name} - {self.description}"


class Well:
    def __init__(self, name, description, user=getpass.getuser(), time=datetime.now()):
        self.name = name or f"New_Well_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.description = description
        self.modified_by = user
        self.modified_date = time
        self.header = {}
        self.data = pd.DataFrame()
        self.history = []

    def add_data(self, data):
        self.data = data
        self.update_history(action=f"Added data to well {self.name}")

    def read_las(self, path: str):
        with open(path, "rb") as f:
            data, header = las.read_las_files([f])
        self.data = data
        self.header = header
        self.update_history(action=f"Read LAS file {path}")

    def update_history(self, user=getpass.getuser(), time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'), action=""):
        self.history.append({"user": user, "time": time, "action": action})

    def __str__(self):
        return f"Well: {self.name} - {self.description}"
