class Participant:
    """
    Mock representation of preCICE to be used in all mock tests. Dummy implementation of all functions below are
    to be used where the preCICE API calls via the python bindings are done in the FEniCS Adapter
    """

    def __init__(self, name, config_file, rank, procs):
        pass

    def read_data(self, read_mesh_name, read_data_name, vertex_ids, dt):
        raise Exception("not implemented")

    def write_data(self, write_mesh_name, write_data_name, vertex_ids, write_data):
        raise Exception("not implemented")

    def initialize():
        raise Exception("not implemented")

    def advance(self, foo):
        raise Exception("not implemented")

    def finalize():
        raise Exception("not implemented")

    def requires_initial_data(self):
        raise Exception("not implemented")

    def requires_reading_checkpoint(self):
        raise Exception("not implemented")

    def requires_writing_checkpoint(self):
        raise Exception("not implemented")

    def is_coupling_ongoing(self):
        raise Exception("not implemented")

    def mark_action_fulfilled(self, action):
        raise Exception("not implemented")

    def get_dimensions(self):
        raise Exception("not implemented")

    def get_max_time_step_size(self):
        raise Exception("not implemented")

    def is_time_window_complete(self):
        raise Exception("not implemented")
