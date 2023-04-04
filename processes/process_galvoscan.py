import numpy as np
from scipy.interpolate import BPoly
from findiff import FinDiff
import matplotlib.pyplot as plt


class GalvoScan:

    def __init__(self):
        self.sample_rate = 100000
        self.dt = 1 / self.sample_rate
        self.v_max = 0.1  # µm/µs
        self.a_max = 1.  # 1e-4 µm/µs^2
        self.conversion_factor = 17  # µm / V
        self.v_max = self.v_max / self.conversion_factor
        self.a_max = self.a_max / self.conversion_factor
        self.voltage_range = 10.  # V

    def _acceleration_curve(self):
        t, d = self._acceleration(self.a_max, 0., self.v_max)
        acceleration_samples = int(np.ceil(np.ceil(t) / self.dt))
        acceleration_distance = np.ceil(d)
        _bp = BPoly.from_derivatives([0, acceleration_samples],
                                     [[0, 0., self.a_max],
                                      [acceleration_distance, self.v_max, 0.]])
        self.acceleration_curve = _bp(np.linspace(0, acceleration_distance, acceleration_samples))

    def map_scan(self):
        self.start_position = 4.
        self.scan_steps = 5
        self.dwell_time = 10e-5
        self.step_interval = 10e-5
        self.return_time = 40e-5
        self.scan_speed = 1.
        self.dwell_speed = 0.4
        self.step_size = 0.5
        self.end_position = self.start_position + self.step_size * self.scan_steps

    def generate_one_axis_scan(self):
        one_axis_scan = np.array([])
        initial_samples = int(self.sample_rate * self.initial_wait)
        step_samples = int(self.sample_rate * (self.dwell_time + self.step_interval))
        return_samples = int(self.sample_rate * self.return_time)
        _bp = BPoly.from_derivatives([0, initial_samples], [[0., 0., 0.], [self.start_position, self.scan_speed, 0.]])
        initial_curve = _bp(np.linspace(0, initial_samples, initial_samples))
        _bp = BPoly.from_derivatives(
            [0, 0.2 * step_samples, 0.5 * step_samples, 0.8 * step_samples, step_samples],
            [[0., self.scan_speed, 0.], [0.5 * self.step_size, self.dwell_speed, 0.],
             [1. * self.step_size, self.dwell_speed, 0.], [1.5 * self.step_size, self.dwell_speed, 0.],
             [2. * self.step_size, self.scan_speed, 0.]])
        _curve = _bp(np.linspace(0, self.step_size, step_samples))
        for i in range(self.scan_steps):
            one_axis_scan = np.append(one_axis_scan, self.start_position + i * self.step_size + _curve)
        # _bp = BPoly.from_derivatives([0, interval_samples], [[0., self.scan_speed, 0.], [self.step_size, self.dwell_speed, 0.]])
        # interval_curve = _bp(np.linspace(0, interval_samples, interval_samples))
        # _bp = BPoly.from_derivatives([0, dwell_samples], [[0., self.dwell_speed, 0.], [self.dwell_speed * self.dwell_time, self.scan_speed, 0.]])
        # dwell_curve = _bp(np.linspace(0, dwell_samples, dwell_samples))
        # single_step = np.append(interval_curve, dwell_curve)
        #

        # one_axis_scan = np.append(initial_curve, one_axis_scan)
        # end_position = one_axis_scan[-1]
        # _bp = BPoly.from_derivatives([0, 1], [[end_position, self.scan_speed, 0.], [0., 0., 0.]])
        # return_curve = _bp(np.linspace(end_position, 0, int(return_samples)))
        # one_axis_scan = np.append(one_axis_scan, return_curve)
        return one_axis_scan

    def _plot_curve(self, y):
        fig, ax = plt.subplots()
        ax.plot(y)
        plt.show()

    def _acceleration(self, a, v0, vt):
        t = (vt - v0) / a
        s = v0 * t + 0.5 * a * t ** 2
        return t, s

    def _end_velocity(self, a, v0, d):
        return np.sqrt(v0 ^ 2 + 2 * a * d)

    def _end_position(self, a, v0, t):
        return v0 * t + 0.5 * a * t ^ 2

    def _first_order_derivative(self, x, y):
        dx = x[1] - x[0]
        d_dx = FinDiff(0, dx, 1)
        return d_dx(y)

    def _second_order_derivative(self, x, y):
        dx = x[1] - x[0]
        d2_dx2 = FinDiff(0, dx, 2)
        return d2_dx2(y)


if __name__ == '__main__':
    g = GalvoScan()
    g.map_the_scan(-5.)
    g._plot_curve(g.initial_curve)
    # y = g.generate_one_axis_scan()
    # g._plot_curve(y)
    # dy = g._first_order_derivative(np.arange(len(y)), y)
    # g._plot_curve(dy)
    # d2y = g._second_order_derivative(np.arange(len(y)), y)
    # g._plot_curve(d2y)
