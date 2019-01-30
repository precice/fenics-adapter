# comments on test layout: https://docs.pytest.org/en/latest/goodpractices.html
# first install package python setup.py install
# then run tests with python setup.py test -s tests.test_waveform_bindings

from unittest.mock import MagicMock, patch, Mock
from unittest import TestCase
import warnings

fake_dolfin = MagicMock()
fake_PySolverInterface = MagicMock()


@patch.dict('sys.modules', **{'PySolverInterface': fake_PySolverInterface, 'dolfin': fake_dolfin})
class TestWaveformBindings(TestCase):
    def setUp(self):
        fake_PySolverInterface.PyActionReadIterationCheckpoint = MagicMock(return_value=1)
        fake_PySolverInterface.PyActionWriteIterationCheckpoint = MagicMock(return_value=2)
        warnings.simplefilter('ignore', category=ImportWarning)

    def test_import(self):
        from fenicsadapter.waveform_bindings import WaveformBindings
        self.assertTrue(True)

    def test_init(self):
        from fenicsadapter.waveform_bindings import WaveformBindings
        WaveformBindings()
