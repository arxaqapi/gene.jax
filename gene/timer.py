import time
from math import modf


class Timer:
    def __init__(self) -> None:
        self.reset()

    def start(self):
        # self.start_time = time.process_time()
        # self.start_time = time.perf_counter()
        self.start_time = time.time()

    def stop(self) -> float:
        """Stops the timer and returns the elapsed time"""
        self.stop_time = time.time()
        return self.stop_time - self.start_time

    def reset(self):
        self.start_time = 0.0
        self.stop_time = 0.0

    def __str__(self) -> str:
        dt = self.stop_time - self.start_time
        frac, minutes = modf(dt / 60)
        return f"{minutes} min. {frac * 60} sec. elapsed"
