import numpy as np
from scipy.interpolate import BPoly


class GalvoScan:

    def __init__(self):
        self.sample_rate = 100000  # Hz
        self.dt = 1 / self.sample_rate  # s
        self.v_max = 4e2  # V/s
        self.a_max = 2.4e7  # V/s^2
        self.voltage_range = 10.  # V

        self.gv_start = -1.0
        self.gv_stop = 1.0
        self.laser_start = 8
        self.laser_interval = 16

    def map_scan(self, gv_start=None, gv_stop=None, laser_start=None, laser_interval=None):
        if gv_start is not None:
            self.gv_start = gv_start
        if gv_stop is not None:
            self.gv_stop = gv_stop
        if laser_start is not None:
            self.laser_start = laser_start
        if laser_interval is not None:
            self.laser_interval = laser_interval
        t_scan = (self.gv_stop - self.gv_start) / self.v_max
        s_scan = int(np.ceil(t_scan / self.dt)) + 1
        seq_scan = np.linspace(-1, 1, s_scan)
        pos = seq_scan[np.arange(self.laser_start, s_scan, self.laser_interval)]
        t_acc = self.v_max / self.a_max
        s_acc = int(np.ceil(t_acc / self.dt))
        points = np.array([0, t_acc, t_acc + self.dt * (s_acc / 2)])
        derivatives = np.array([[0, 0, self.a_max], [0.5 * self.a_max * t_acc ** 2, self.v_max, self.a_max],
                                [0.5 * self.a_max * t_acc ** 2 + self.v_max * self.dt * (s_acc / 2), self.v_max, 0]])
        bp = BPoly.from_derivatives(points, derivatives)
        seq_acc = bp(np.linspace(0, (s_acc + 1) * self.dt, int(s_scan * 0.01)))
        x_axis_scan = np.append(seq_acc - seq_acc.max() - self.dt * self.v_max + self.gv_start, seq_scan)
        seq_deacc = self.gv_stop - np.flip(seq_acc) + seq_acc.max() + self.v_max * 0.75 * self.dt
        x_axis_scan = np.append(x_axis_scan, seq_deacc)
        x_axis_scan = np.pad(x_axis_scan, (int(self.laser_interval / 2), int(self.laser_interval / 2)), 'constant',
                             constant_values=(x_axis_scan[0], x_axis_scan[-1]))
        x_axis_scan = np.append(x_axis_scan, np.flip(x_axis_scan)) + 0.5 * self.v_max * self.dt
        x_axis_scan = np.tile(x_axis_scan, int(pos.shape[0] / 2))
        laser_trigger = np.zeros(s_scan)
        ids = [item for sublist in [(i - 1, i) for i in range(self.laser_start, s_scan, self.laser_interval)] for item
               in sublist]
        laser_trigger[ids] = 1
        laser_trigger = np.pad(laser_trigger, (
            int(self.laser_interval / 2) + int(s_scan * 0.01), int(self.laser_interval / 2) + int(s_scan * 0.01)),
                               'constant', constant_values=(0, 0))
        laser_trigger = np.append(laser_trigger, laser_trigger)
        laser_trigger = np.tile(laser_trigger, int(pos.shape[0] / 2))
        y_axis_scan = pos[0] * np.ones(s_scan + int(s_scan * 0.01) + int(self.laser_interval / 2))
        for n in range(int(pos.shape[0] / 2) * 2 - 1):
            acc = pos[n] + seq_acc
            deacc = pos[n + 1] - np.flip(seq_acc)
            inter = np.linspace(acc[-1], deacc[0], self.laser_interval + 2)
            temp = np.append(acc, inter[1:-1])
            temp = np.append(temp, deacc)
            y_axis_scan = np.append(y_axis_scan, temp)
            y_axis_scan = np.append(y_axis_scan, pos[n + 1] * np.ones(s_scan))
        y_axis_scan = np.append(y_axis_scan, pos[int(pos.shape[0] / 2) * 2 - 1] * np.ones(
            int(s_scan * 0.01) + int(self.laser_interval / 2)))
        camera_trigger = np.zeros(laser_trigger.shape[0])
        camera_trigger[ids[0]:ids[-1]] = 1
        return x_axis_scan, y_axis_scan, laser_trigger, camera_trigger, pos
