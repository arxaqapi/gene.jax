import unittest
import time

from gene.timer import Timer


class TestTimer(unittest.TestCase):
    def test_timer_dt(self):
        timer = Timer()
        timer.start()

        time.sleep(2)

        dt = timer.stop()
        self.assertIsNone(timer.reset())
