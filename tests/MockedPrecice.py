from unittest.mock import MagicMock
import numpy as np

action_read_iteration_checkpoint = MagicMock(return_value=1)
action_write_iteration_checkpoint = MagicMock(return_value=2)
action_write_initial_data = MagicMock()


class Interface:

    def __init__(self, name, rank, procs):
        pass

    def read_block_scalar_data(self, read_data_id, n_vertices, vertex_ids, read_data):
        raise Exception("not implemented")

    def write_block_scalar_data(self, write_data_id, n_vertices, vertex_ids, write_data):
        raise Exception("not implemented")

    def get_data_id(self, foo, bar):
        raise Exception("not implemented")

    def get_mesh_id(self, foo):
        raise Exception("not implemented")

    def advance(self, foo):
        raise Exception("not implemented")

    def is_action_required(self, py_action):
        raise Exception("not implemented")

    def is_coupling_ongoing(self):
        raise Exception("not implemented")

    def configure(self, foo):
        raise Exception("not implemented")

    def get_dimensions(self):
        raise Exception("not implemented")
