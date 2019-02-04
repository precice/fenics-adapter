from unittest.mock import MagicMock
import numpy as np

PyActionReadIterationCheckpoint = MagicMock(return_value=1)
PyActionWriteIterationCheckpoint = MagicMock(return_value=2)
PyActionWriteInitialData = MagicMock()


class PySolverInterface:

    def __init__(self, name, rank, procs):
        pass

    def readBlockScalarData(self, read_data_id, n_vertices, vertex_ids, read_data):
        raise Exception("not implemented")

    def writeBlockScalarData(self, write_data_id, n_vertices, vertex_ids, write_data):
        raise Exception("not implemented")

    def getDataID(self, foo, bar):
        raise Exception("not implemented")

    def getMeshID(self, foo):
        raise Exception("not implemented")

    def advance(self, foo):
        raise Exception("not implemented")

    def isActionRequired(self, py_action):
        raise Exception("not implemented")

    def isCouplingOngoing(self):
        raise Exception("not implemented")

    def configure(self, foo):
        raise Exception("not implemented")

    def getDimensions(self):
        raise Exception("not implemented")
