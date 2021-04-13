"""
This is the configuration module of fenicsadapter
"""

import json
import os
import sys


class Config:
    """
    Handles reading of config. parameters of the fenicsadapter based on JSON
    configuration file. Initializer calls read_json() method. Instance attributes
    can be accessed by provided getter functions.
    """

    def __init__(self, adapter_config_filename):

        self._config_file_name = None
        self._participant_name = None
        self._coupling_mesh_name = None
        self._read_data_name = None
        self._write_data_name = None

        self.read_json(adapter_config_filename)

    def read_json(self, adapter_config_filename):
        """
        Reads JSON adapter configuration file and saves the data to the respective instance attributes.

        Parameters
        ----------
        adapter_config_filename : string
            Name of the JSON configuration file
        """
        folder = os.path.dirname(os.path.join(os.getcwd(), os.path.dirname(sys.argv[0]), adapter_config_filename))
        path = os.path.join(folder, os.path.basename(adapter_config_filename))
        read_file = open(path, "r")
        data = json.load(read_file)
        self._config_file_name = os.path.join(folder, data["config_file_name"])
        self._participant_name = data["participant_name"]
        self._coupling_mesh_name = data["interface"]["coupling_mesh_name"]

        try:
            self._write_data_name = data["interface"]["write_data_name"]
        except KeyError:
            self._write_data_name = None  # not required for one-way coupling, if this participant reads data

        try:
            self._read_data_name = data["interface"]["read_data_name"]
        except KeyError:
            self._read_data_name = None  # not required for one-way coupling, if this participant writes data

        read_file.close()

    def get_config_file_name(self):
        return self._config_file_name

    def get_participant_name(self):
        return self._participant_name

    def get_coupling_mesh_name(self):
        return self._coupling_mesh_name

    def get_read_data_name(self):
        return self._read_data_name

    def get_write_data_name(self):
        return self._write_data_name
