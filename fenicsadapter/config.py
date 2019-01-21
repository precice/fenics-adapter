"""This is configuration module of fenicsadapter."""

import os, sys, json

class Config(object):
    """This class handles the configuration of the fenicsadapter based on JSON
    configuration file. Initializer calls readJSON() method.

    :ivar _config_file_name: name of the configuration file
    :ivar _solver_name : name of the solver
    :ivar _coupling_mesh_name: name of mesh as defined in preCICE config
    :ivar _read_data_name: name of read data as defined in preCICE config
    :ivar _write_data_name: name of write data as defined in preCICE config
    :func readJSON(): method accessing data in JSON file
    """

    def __init__(self):
        self.readJSON()

        self._config_file_name
        self._solver_name
        self._coupling_mesh_name
        self._read_data_name
        self._write_data_name


    def readJSON(self):
        """ This method reads JSON configuration file in "r" mode and saves the data to
        the respective instance attributes. Config file name must be:
        precice-adapter-config.json

        :var path: stores path to the JSON config file
        :var data: data decoded from JSON files
        :var read_file: stores file path
        :raise IOError: if open fails
        """

        path =  os.getcwd() + "/" + os.path.dirname(sys.argv[0]) + "/precice-adapter-config.json"
        print(path)

        try:
            read_file = open(path, "r")
        except IOError:
            print("An error occured trying to open the file.")
            raise IOError

        data = json.load(read_file)

        self._config_file_name = data["config_file_name"]
        self._solver_name = data["solver_name"]
        self._coupling_mesh_name = data["interface"]["coupling_mesh_name"]
        self._write_data_name = data["interface"]["write_data_name"]
        self._read_data_name = data["interface"]["read_data_name"]

        read_file.close()
