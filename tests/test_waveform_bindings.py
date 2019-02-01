# comments on test layout: https://docs.pytest.org/en/latest/goodpractices.html
# first install package python setup.py install
# then run tests with python setup.py test -s tests.test_waveform_bindings

from unittest.mock import MagicMock, patch, Mock
from unittest import TestCase
import warnings


class TestWaveformBindings(TestCase):
    def setUp(self):
        warnings.simplefilter('ignore', category=ImportWarning)

    def test_import(self):
        from fenicsadapter.waveform_bindings import WaveformBindings
        self.assertTrue(True)

    def test_init(self):
        from fenicsadapter.waveform_bindings import WaveformBindings
        with patch("PySolverInterface.PySolverInterface") as MagicMock:
            WaveformBindings("Dummy", 0, 1)

