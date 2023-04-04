import numpy as np
from scipy.interpolate import BPoly
from findiff import FinDiff
import matplotlib.pyplot as plt


class GalvoScan:

    def __init__(self):
        self.sample_rate = 100000
        self.dt = 1 / self.sample_rate
        self.v_max = 1.
        self.a_max = 1.
        self.t_acceleration = (self.v_max / self.a_max) / dt
        self.acceleration_samples = int(self.t_acceleration) * self.sample_rate

        self.initial_position = 0.
        self.initial_wait = 40e-5
        self.start_position = 4.
        self.scan_steps = 40
        self.step_size = 1.
        self.end_position = self.start_position + (self.step_size * self.scan_steps)
        self.dwell_time = 10e-5
        self.step_interval = 10e-5
        self.return_time = 40e-5

    def generate_one_axis_scan(self):
        initial_samples = int(self.sample_rate * self.initial_wait)
        dwell_samples = int(self.sample_rate * self.dwell_time)
        interval_samples = int(self.sample_rate * self.step_interval)

        _bp = BPoly.from_derivatives([0, 1], [[0., 0., 0.], [1., 0., 0.]])
        interval_curve = _bp(np.linspace(0, 1, interval_samples))
        dwell_curve = self.np.ones(dwell_samples)
        single_step = self.step_size * np.append(interval_curve, dwell_curve)

        _bp = BPoly.from_derivatives([0, 1], [[1., 0., 0.], [0., 0., 0.]])
        one_axis_scan = self.initial_position + (self.start_position - self.initial_position) * _bp(
            np.linspace(0, 1, int(initial_samples / 2)))
        one_axis_scan = np.append(one_axis_scan, self.start_position * np.ones(int(initial_samples / 2)))
        for i in range(self.scan_steps):
            one_axis_scan = np.append(one_axis_scan, self.start_position + i * self.step_size + single_step)
        return one_axis_scan

    def _plot_curve(self, y):
        fig, ax = plt.subplots()
        ax.plot(y)
        plt.show()

    def _first_order_derivative(self, x, y):
        dx = x[1] - x[0]
        d_dx = FinDiff(0, dx, 1)
        return d_dx(y)

    def _second_order_derivative(self, x, y):
        dx = x[1] - x[0]
        d2_dx2 = FinDiff(0, dx, 2)
        return d2_dx2(y)
