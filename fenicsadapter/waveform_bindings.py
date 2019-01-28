try:
    import PySolverInterface
    from PySolverInterface import PyActionReadIterationCheckpoint, PyActionWriteInitialData, PyActionWriteIterationCheckpoint
except ImportError:
    import os
    import sys
    # check if PRECICE_ROOT is defined
    if not os.getenv('PRECICE_ROOT'):
       raise Exception("ERROR: PRECICE_ROOT not defined!")

    precice_root = os.getenv('PRECICE_ROOT')
    precice_python_adapter_root = precice_root+"/src/precice/bindings/python"
    sys.path.insert(0, precice_python_adapter_root)
    import PySolverInterface


class WaveformBindings(PySolverInterface.PySolverInterface):
    def __init__(self, name, rank, procs):
        print("INIT CALLED!")
        super().__init__()


    def __new__(cls, name, rank, procs):
        print("NEW CALLED!")
        return super().__new__(cls, name, rank, procs)

