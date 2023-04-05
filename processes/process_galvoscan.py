import numpy as np
from scipy.interpolate import BPoly
from findiff import FinDiff
import matplotlib.pyplot as plt


class GalvoScan:

    def __init__(self):
        self.sample_rate = 100000
        self.dt = 1 / self.sample_rate
        self.v_max = 0.1  # µm/µs
        self.a_max = 1e-4  # µm/µs^2
        self.conversion_factor = 17  # µm / V
        self.v_max = 1e6 * self.v_max / self.conversion_factor
        self.a_max = 1e12 * self.a_max / self.conversion_factor
        self.voltage_range = 10.  # V


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

    def generate_scan(self):
        ac, vc, pc = self._acceleration_curve(self.a_max, 0, self.v_max, self.dt)


    def generate_one_axis_scan(self):
        one_axis_scan = np.array([])
        p, v, a = g._acceleration_curve(g.a_max, 0, g.v_max, g.dt)
        one_axis_scan = np.append(one_axis_scan, p) - 10
        t = (np.ceil(p[-1]) - p[-1]) / g.v_max
        p, v = g._scan_curve(g.v_max, t, g.dt)
        one_axis_scan = np.append(one_axis_scan, p + one_axis_scan[-1])
        # initial_samples = int(self.sample_rate * self.initial_wait)
        # step_samples = int(self.sample_rate * (self.dwell_time + self.step_interval))
        # return_samples = int(self.sample_rate * self.return_time)
        # _bp = BPoly.from_derivatives([0, initial_samples], [[0., 0., 0.], [self.start_position, self.scan_speed, 0.]])
        # initial_curve = _bp(np.linspace(0, initial_samples, initial_samples))
        # _bp = BPoly.from_derivatives(
        #     [0, 0.2 * step_samples, 0.5 * step_samples, 0.8 * step_samples, step_samples],
        #     [[0., self.scan_speed, 0.], [0.5 * self.step_size, self.dwell_speed, 0.],
        #      [1. * self.step_size, self.dwell_speed, 0.], [1.5 * self.step_size, self.dwell_speed, 0.],
        #      [2. * self.step_size, self.scan_speed, 0.]])
        # _curve = _bp(np.linspace(0, self.step_size, step_samples))
        # for i in range(self.scan_steps):
        #     one_axis_scan = np.append(one_axis_scan, self.start_position + i * self.step_size + _curve)
        # _bp = BPoly.from_derivatives([0, interval_samples], [[0., self.scan_speed, 0.], [self.step_size, self.dwell_speed, 0.]])
        # interval_curve = _bp(np.linspace(0, interval_samples, interval_samples))
        # _bp = BPoly.from_derivatives([0, dwell_samples], [[0., self.dwell_speed, 0.], [self.dwell_speed * self.dwell_time, self.scan_speed, 0.]])
        # dwell_curve = _bp(np.linspace(0, dwell_samples, dwell_samples))
        # single_step = np.append(interval_curve, dwell_curve)

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

    def _acceleration_curve(self, a, v0, vt, dt):
        t = (vt - v0) / a
        d = v0 * t + 0.5 * a * t ** 2
        acceleration_samples = int(np.ceil(t / dt)) + 1
        xt = np.arange(acceleration_samples) * dt
        acceleration_curve = np.ones(acceleration_samples) * a
        velocity_curve = v0 + xt * acceleration_curve
        position_curve = v0 * xt + 0.5 * a * xt ** 2
        return position_curve, velocity_curve, acceleration_curve

    def _scan_curve(self, v, t, dt):
        scan_samples = int(np.ceil(t / dt))
        xt = (np.arange(scan_samples - 1) + 1) * dt
        velocity_curve = np.ones(scan_samples) * v
        position_curve = v * xt
        return position_curve, velocity_curve


if __name__ == '__main__':
    g = GalvoScan()
    c = g.generate_one_axis_scan()
    g._plot_curve(c)
    # p, v, a = g._acceleration_curve(g.a_max, 0, g.v_max, g.dt)
    # g._plot_curve(a)
    # g._plot_curve(v)
    # g._plot_curve(p)
    # print(p[-1])
    # t = (np.ceil(p[-1]) - p[-1]) / g.v_max
    # p, v = g._scan_curve(g.v_max, t, g.dt)
    # g._plot_curve(v)
    # g._plot_curve(p)
    # print(p[-1])
    # p, v = g._scan_curve(g.v_max, g.dt)
    # p, v, a = g._acceleration_curve(-g.a_max, g.v_max, -g.v_max, g.dt)
    # g._plot_curve(a)
    # g._plot_curve(v)
    # g._plot_curve(p)
    # print(p[-1])

