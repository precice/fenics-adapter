try:
    import precice
    from precice import action_read_iteration_checkpoint, action_write_initial_data, action_write_iteration_checkpoint
except ImportError:
    import os
    import sys
    # check if PRECICE_ROOT is defined
    if not os.getenv('PRECICE_ROOT'):
       raise Exception("ERROR: PRECICE_ROOT not defined!")

    precice_root = os.getenv('PRECICE_ROOT')
    precice_python_adapter_root = precice_root+"/src/precice/bindings/python"
    sys.path.insert(0, precice_python_adapter_root)
    import precice
    from precice import action_read_iteration_checkpoint, action_write_initial_data, action_write_iteration_checkpoint


class WaveformBindings(precice.Interface):
    pass
