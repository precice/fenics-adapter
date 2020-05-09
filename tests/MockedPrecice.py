from unittest.mock import MagicMock

action_read_iteration_checkpoint = MagicMock(return_value=1)
action_write_iteration_checkpoint = MagicMock(return_value=2)
action_write_initial_data = MagicMock()


class Interface:
    def __init__(self, name, config_file, rank, procs):
        pass

    def read_block_scalar_data(self, read_data_id, vertex_ids):
        raise Exception("not implemented")

    def read_block_vector_data(self, read_data_id, vertex_ids):
        raise Exception("not implemented")

    def write_block_scalar_data(self, write_data_id, vertex_ids, write_data):
        raise Exception("not implemented")

    def write_block_vector_data(self, write_data_id, vertex_ids, write_data):
        raise Exception("not implemented")

    def get_data_id(self, foo, bar):
        raise Exception("not implemented")

    def get_mesh_id(self, foo):
        raise Exception("not implemented")

    def initialize_data(self):
        raise Exception("not implemented")

    def advance(self, foo):
        raise Exception("not implemented")

    def is_action_required(self, action):
        raise Exception("not implemented")

    def is_coupling_ongoing(self):
        raise Exception("not implemented")

    def mark_action_fulfilled(self, action):
        raise Exception("not implemented")

    def get_dimensions(self):
        raise Exception("not implemented")

    def is_time_window_complete(self):
        raise Exception("not implemented")
