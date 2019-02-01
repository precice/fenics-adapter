from unittest.mock import MagicMock
import numpy as np

PyActionReadIterationCheckpoint = MagicMock(return_value=1)
PyActionWriteIterationCheckpoint = MagicMock(return_value=2)
PyActionWriteInitialData = MagicMock()


class PySolverInterface:

    readCheckpointReturn = False
    writeCheckpointReturn = False

    def __init__(self):
        pass

    def __new__(cls, name, rank, procs):
        return super().__new__(cls)

    def readBlockScalarData(self, read_data_id, n_vertices, vertex_ids, read_data):
        assert(type(read_data) == np.ndarray)
        read_data += 1
        pass

    def writeBlockScalarData(self, write_data_id, n_vertices, vertex_ids, write_data):
        assert(type(write_data) == np.ndarray)
        write_data += 2
        pass

    def getDataID(self, foo, bar):
        return None

    def advance(self, foo):
        return 1.0

    def isActionRequired(self, py_action):
        if py_action == PyActionReadIterationCheckpoint():
            return self.readCheckpointReturn
        elif py_action == PyActionWriteIterationCheckpoint():
            return self.writeCheckpointReturn
