import math

import numpy as np
from scipy.interpolate import BPoly


class TriggerSequence:

    def __init__(self):
        self.sequence_time = 0.04
        self.sample_rate = 100000
        self.dt = 1 / self.sample_rate
        # piezo scanner
        self.piezo_starts = [0., 0., 0.]
        self.piezo_step_sizes = [0.03, 0.03, 0.08]
        self.piezo_ranges = [0.6, 0.6, 0.08]
        self.piezo_return_time = 0.002
        self.piezo_conv_factors = [10., 10., 10.]
        self.piezo_analog_start = 0.03
        # galvo scanner
        self.v_max = 4e2  # V/s
        self.a_max = 2.4e7  # V/s^2
        self.voltage_range = 10.  # V
        self.galvo_start = -1.0
        self.galvo_stop = 1.0
        self.galvo_laser_start = 8
        self.galvo_laser_interval = 16
        # digital triggers
        self.digital_starts = [0.002, 0.007, 0.007, 0.012, 0.012, 0.012]
        self.digital_ends = [0.004, 0.01, 0.01, 0.015, 0.015, 0.015]

    def update_piezo_scan_parameters(self, piezo_ranges, piezo_step_sizes, piezo_starts, piezo_analog_start):
        if all(i >= 0 for i in piezo_ranges):
            self.piezo_ranges = piezo_ranges
        self.piezo_step_sizes = piezo_step_sizes
        if all(i >= 0 for i in piezo_starts):
            self.piezo_starts = piezo_starts
        self.piezo_analog_start = piezo_analog_start

    def update_galvo_scan_parameters(self, gv_start=None, gv_stop=None, laser_start=None, laser_interval=None):
        if gv_start is not None:
            self.galvo_start = gv_start
        if gv_stop is not None:
            self.galvo_stop = gv_stop
        if laser_start is not None:
            self.galvo_laser_start = laser_start
        if laser_interval is not None:
            self.galvo_laser_interval = laser_interval

    def update_digital_parameters(self, sequence_time, digital_starts, digital_ends):
        self.sequence_time = sequence_time
        self.digital_starts = digital_starts
        self.digital_ends = digital_ends

    def generate_digital_triggers_sw(self, lasers, camera):
        cycle_samples = self.sequence_time * self.sample_rate
        cycle_samples = int(np.ceil(cycle_samples))
        digital_trigger = np.zeros((len(self.digital_starts), cycle_samples))
        for laser in lasers:
            startSamp = int(np.round(self.digital_starts[laser] * self.sample_rate))
            endSamp = int(np.round(self.digital_ends[laser] * self.sample_rate))
            digital_trigger[laser, startSamp:endSamp] = 1
        startSamp = int(np.round(self.digital_starts[camera + 4] * self.sample_rate))
        endSamp = int(np.round(self.digital_ends[camera + 4] * self.sample_rate))
        digital_trigger[camera + 4, startSamp:endSamp] = 1
        return digital_trigger

    def generate_trigger_sequence_gs(self, lasers, camera):
        return_samples = 128
        t_scan = (self.galvo_stop - self.galvo_start) / self.v_max
        s_scan = int(np.ceil(t_scan / self.dt)) + 1
        seq_scan = np.linspace(-1, 1, s_scan)
        pos = seq_scan[np.arange(self.galvo_laser_start, s_scan, self.galvo_laser_interval)]
        print('dot number:', pos.shape[0])
        t_acc = self.v_max / self.a_max
        s_acc = int(np.ceil(t_acc / self.dt))
        points = np.array([0, t_acc, t_acc + self.dt * (s_acc / 2)])
        derivatives = np.array([[0, 0, self.a_max], [0.5 * self.a_max * t_acc ** 2, self.v_max, self.a_max],
                                [0.5 * self.a_max * t_acc ** 2 + self.v_max * self.dt * (s_acc / 2), self.v_max, 0]])
        bp = BPoly.from_derivatives(points, derivatives)
        seq_acc = bp(np.linspace(0, (s_acc + 1) * self.dt, int(s_scan * 0.01)))
        x_axis_scan = np.append(seq_acc - seq_acc.max() - self.dt * self.v_max + self.galvo_start, seq_scan)
        seq_deacc = self.galvo_stop - np.flip(seq_acc) + seq_acc.max() + self.v_max * 0.75 * self.dt
        x_axis_scan = np.append(x_axis_scan, seq_deacc)
        x_axis_scan = np.pad(x_axis_scan, (int(self.galvo_laser_interval), return_samples),
                             'constant', constant_values=(x_axis_scan[0], x_axis_scan[0]))
        x_axis_scan = np.append(x_axis_scan, x_axis_scan)
        x_axis_scan = np.tile(x_axis_scan, int(pos.shape[0] / 2))
        laser_trigger = np.zeros(s_scan)
        ids = [item for sublist in
               [(i - 1, i) for i in range(self.galvo_laser_start, s_scan, self.galvo_laser_interval)] for item in
               sublist]
        laser_trigger[ids] = 1
        laser_trigger = np.pad(laser_trigger, (
            int(self.galvo_laser_interval) + int(s_scan * 0.01),
            return_samples + int(s_scan * 0.01)),
                               'constant', constant_values=(0, 0))
        laser_trigger = np.append(laser_trigger, laser_trigger)
        laser_trigger = np.tile(laser_trigger, int(pos.shape[0] / 2))
        y_axis_scan = pos[0] * np.ones(s_scan + 2 * int(s_scan * 0.01) + int(self.galvo_laser_interval))
        for n in range(int(pos.shape[0] / 2) * 2 - 1):
            acc = pos[n] + seq_acc
            deacc = pos[n + 1] - np.flip(seq_acc)
            inter = np.linspace(acc[-1], deacc[0], self.galvo_laser_interval + 2)
            temp = np.append(acc, inter[1:-1])
            temp = np.append(temp, deacc)
            y_axis_scan = np.append(y_axis_scan, temp)
            y_axis_scan = np.append(y_axis_scan, pos[n + 1] * np.ones(s_scan + return_samples))
        y_axis_scan = np.append(y_axis_scan,
                                pos[int(pos.shape[0] / 2) * 2 - 1] * np.ones(return_samples))
        analog_trigger = np.zeros((2, x_axis_scan.shape[0]))
        analog_trigger[0] = x_axis_scan
        analog_trigger[1] = y_axis_scan
        camera_trigger = np.ones(laser_trigger.shape[0])
        camera_trigger[:int(self.galvo_laser_interval) + int(s_scan * 0.01) - 1] = 0
        camera_trigger[- int(self.galvo_laser_interval) - int(s_scan * 0.01) + 1:] = 0
        digital_trigger = np.zeros((len(self.digital_starts), camera_trigger.shape[0]))
        for laser in lasers:
            digital_trigger[laser] = laser_trigger
        digital_trigger[camera + 4] = camera_trigger
        return analog_trigger, digital_trigger, pos

    def generate_galvo_scanning(self, lasers, camera):
        freq = 128
        w = 2 * np.pi * freq
        T = 1 / freq
        scan_samples = int(np.ceil(T / self.dt))
        print(scan_samples)
        n_pos = 16
        amp = 1.2
        single_scan_x = amp * np.sin(w * self.dt * np.arange(scan_samples) - 0.5 * np.pi)
        dot_pos = np.linspace(self.galvo_start, self.galvo_stop, n_pos)
        scan_times = np.arcsin(dot_pos / amp)
        sample_points = (scan_times + 0.5 * np.pi) / (w * self.dt)
        laser_sequence = []
        for s in sample_points:
            if math.ceil(s) != math.floor(s):
                laser_sequence.append(math.floor(s))
                laser_sequence.append(math.ceil(s))
            else:
                laser_sequence.append(int(s) - 1)
        laser_scan = np.zeros(scan_samples)
        laser_scan[laser_sequence] = 1
        laser_scan = laser_scan + np.flip(laser_scan)
        y_point_0 = math.ceil((np.arcsin(-1.1 / amp) + 0.5 * np.pi) / (w * self.dt))
        y_point_1 = math.floor((np.arcsin(1.1 / amp) + 0.5 * np.pi) / (w * self.dt))
        scanning = np.abs(y_point_1 - y_point_0)
        single_scan_y = np.ones(y_point_1) * dot_pos[0]
        single_scan_y = np.append(single_scan_y, np.linspace(dot_pos[0], dot_pos[1], scan_samples - 2 * y_point_1))
        single_scan_y = np.append(single_scan_y, np.ones(scanning) * dot_pos[1])
        single_scan_y = np.append(single_scan_y, np.linspace(dot_pos[1], (dot_pos[1] + dot_pos[2]) / 2,
                                                             int(scan_samples / 2 - y_point_1)))
        for i in range(1, int(n_pos / 2)):
            a = 2 * i - 1
            b = 2 * i
            c = 2 * i + 1
            d = min(n_pos - 1, 2 * i + 2)
            single_scan_y = np.append(single_scan_y, np.linspace((dot_pos[a] + dot_pos[b]) / 2, dot_pos[b],
                                                                 int(scan_samples / 2 - y_point_1)))
            single_scan_y = np.append(single_scan_y, dot_pos[b] * np.ones(scanning))
            single_scan_y = np.append(single_scan_y, np.linspace(dot_pos[b], dot_pos[c], scan_samples - 2 * y_point_1))
            single_scan_y = np.append(single_scan_y, dot_pos[c] * np.ones(scanning))
            single_scan_y = np.append(single_scan_y, np.linspace(dot_pos[c], (dot_pos[c] + dot_pos[d]) / 2,
                                                                 int(scan_samples / 2 - y_point_1)))
        analog_trigger = np.zeros((2, int(n_pos / 2) * scan_samples))
        analog_trigger[0] = np.tile(single_scan_x, int(n_pos / 2))
        analog_trigger[1] = single_scan_y
        digital_trigger = np.zeros((len(self.digital_starts), int(n_pos / 2) * scan_samples))
        for laser in lasers:
            digital_trigger[laser] = np.tile(laser_scan, int(n_pos / 2))
        camera_trigger = np.ones(int(n_pos / 2) * scan_samples)
        camera_trigger[:y_point_0] = 0
        camera_trigger[-y_point_0:] = 0
        digital_trigger[camera + 4] = camera_trigger
        return analog_trigger, digital_trigger

    def generate_trigger_sequence_2d(self):
        digital_trigger_sequences = []
        analog_trigger_sequences = []
        cycle_samples = self.sequence_time * self.sample_rate
        cycle_samples = int(np.ceil(cycle_samples))
        return_samples = self.piezo_return_time * self.sample_rate
        return_samples = int(np.ceil(return_samples))
        [fast_axis_size, middle_axis_size] = [(self.piezo_ranges[i] / self.piezo_conv_factors[i]) for i in range(2)]
        [fast_axis_step_size, middle_axis_step_size] = [(self.piezo_step_sizes[i] / self.piezo_conv_factors[i]) for i in
                                                        range(2)]
        fast_axis_positions = 1 + int(np.ceil(fast_axis_size / fast_axis_step_size))
        middle_axis_positions = 1 + int(np.ceil(middle_axis_size / middle_axis_step_size))
        positions = fast_axis_positions * middle_axis_positions

        for i, start in enumerate(self.digital_starts):
            temp = np.zeros(cycle_samples)
            startSamp = int(np.round(start * self.sample_rate))
            endSamp = int(np.round(self.digital_ends[i] * self.sample_rate))
            temp[startSamp:endSamp] = 1
            digital_trigger_sequences.append(np.tile(temp, fast_axis_positions))
            digital_trigger_sequences[i] = np.append(digital_trigger_sequences[i], np.zeros(return_samples))
            digital_trigger_sequences[i] = np.tile(digital_trigger_sequences[i], middle_axis_positions)

        cycle = np.zeros(cycle_samples)
        startSamp = int(np.round(self.piezo_analog_start * self.sample_rate))
        cycle[startSamp:] = self.bp_increase(np.linspace(0, 1, int(cycle_samples - startSamp)))
        # cycle[startSamp:] = np.linspace(0, 1, int(cycle_samples-startSamp))
        temp = cycle * fast_axis_step_size
        for j in range(fast_axis_positions - 2):
            j = j + 1
            temp = np.append(temp, cycle * fast_axis_step_size + j * fast_axis_step_size)
        cycle = np.ones(startSamp) * fast_axis_step_size * (fast_axis_positions - 1)
        temp = np.append(temp, cycle)
        temp = np.append(temp, self.bp_decrease(
            np.linspace(0, 1, int(cycle_samples - startSamp) + return_samples)) * fast_axis_step_size * (
                                 fast_axis_positions - 1))
        analog_trigger_sequences.append(np.tile(temp, middle_axis_positions))

        cycle = np.zeros((cycle_samples * fast_axis_positions + return_samples))
        cycle[cycle_samples * fast_axis_positions:] = self.bp_increase(np.linspace(0, 1, return_samples))
        temp = cycle * middle_axis_step_size
        for j in range(middle_axis_positions - 2):
            j = j + 1
            temp = np.append(temp, cycle * middle_axis_step_size + j * middle_axis_step_size)
        cycle = np.ones((cycle_samples * fast_axis_positions + return_samples)) * middle_axis_step_size * (
                middle_axis_positions - 1)
        analog_trigger_sequences.append(np.append(temp, cycle))

        return np.asarray(analog_trigger_sequences), np.asarray(digital_trigger_sequences), positions

    def generate_trigger_sequence_3d(self):
        digital_trigger_sequences = []
        analog_trigger_sequences = []
        cycle_samples = self.sequence_time * self.sample_rate
        cycle_samples = int(np.ceil(cycle_samples))
        return_samples = self.piezo_return_time * self.sample_rate
        return_samples = int(np.ceil(return_samples))
        [fast_axis_size, middle_axis_size, slow_axis_size] = [(self.piezo_ranges[i] / self.piezo_conv_factors[i]) for i
                                                              in
                                                              range(3)]
        [fast_axis_step_size, middle_axis_step_size, slow_axis_step_size] = [
            (self.piezo_step_sizes[i] / self.piezo_conv_factors[i])
            for i in range(3)]
        [fast_axis_start, middle_axis_start, slow_axis_start] = [(self.piezo_starts[i] / self.piezo_conv_factors[i]) for
                                                                 i in
                                                                 range(3)]
        fast_axis_positions = 1 + int(np.ceil(fast_axis_size / fast_axis_step_size))
        middle_axis_positions = 1 + int(np.ceil(middle_axis_size / middle_axis_step_size))
        slow_axis_positions = 1 + int(np.ceil(slow_axis_size / slow_axis_step_size))
        positions = fast_axis_positions * middle_axis_positions * slow_axis_positions

        for i, start in enumerate(self.digital_starts):
            temp = np.zeros(cycle_samples)
            startSamp = int(np.round(start * self.sample_rate))
            endSamp = int(np.round(self.digital_ends[i] * self.sample_rate))
            temp[startSamp:endSamp] = 1
            digital_trigger_sequences.append(np.tile(temp, fast_axis_positions))
            digital_trigger_sequences[i] = np.append(digital_trigger_sequences[i], np.zeros(return_samples))
            digital_trigger_sequences[i] = np.tile(digital_trigger_sequences[i],
                                                   middle_axis_positions * slow_axis_positions)

        cycle = np.zeros(cycle_samples)
        startSamp = int(np.round(self.piezo_analog_start * self.sample_rate))
        cycle[startSamp:] = np.linspace(0, 1, int(cycle_samples - startSamp))
        temp = cycle * fast_axis_step_size
        for j in range(fast_axis_positions - 2):
            j = j + 1
            temp = np.append(temp, cycle * fast_axis_step_size + j * fast_axis_step_size)
        cycle = np.ones(startSamp) * fast_axis_step_size * (fast_axis_positions - 1)
        temp = np.append(temp, cycle)
        temp = np.append(temp,
                         np.linspace(1, 0, int(cycle_samples - startSamp) + return_samples) * fast_axis_step_size * (
                                 fast_axis_positions - 1))
        analog_trigger_sequences.append(np.tile(temp, middle_axis_positions * slow_axis_positions))

        cycle = np.zeros((cycle_samples * fast_axis_positions + return_samples))
        cycle[cycle_samples * fast_axis_positions:] = 1
        temp = cycle * middle_axis_step_size
        for j in range(middle_axis_positions - 2):
            j = j + 1
            temp = np.append(temp, cycle * middle_axis_step_size + j * middle_axis_step_size)
        cycle = np.ones((cycle_samples * fast_axis_positions + return_samples)) * middle_axis_step_size * (
                middle_axis_positions - 1)
        temp = np.append(temp, cycle)
        analog_trigger_sequences.append(np.tile(temp, slow_axis_positions))

        cycle = np.zeros(((cycle_samples * fast_axis_positions + return_samples) * middle_axis_positions))
        cycle[(cycle_samples * fast_axis_positions + return_samples) * middle_axis_positions - return_samples:] = 1
        temp = cycle * slow_axis_step_size
        for j in range(slow_axis_positions - 2):
            j = j + 1
            temp = np.append(temp, cycle * slow_axis_step_size + j * slow_axis_step_size)
        cycle = np.ones(
            ((cycle_samples * fast_axis_positions + return_samples) * middle_axis_positions)) * slow_axis_step_size * (
                        slow_axis_positions - 1)
        temp = np.append(temp, cycle)
        analog_trigger_sequences.append(temp)

        return np.asarray(analog_trigger_sequences), np.asarray(digital_trigger_sequences), positions

    def generate_trigger_sequence_beadscan_2d(self, laser, camera):
        digital_trigger_sequences = []
        analog_trigger_sequences = []
        cycle_samples = self.sequence_time * self.sample_rate
        cycle_samples = int(np.ceil(cycle_samples))
        return_samples = self.piezo_return_time * self.sample_rate
        return_samples = int(np.ceil(return_samples))
        [fast_axis_size, middle_axis_size] = [(self.piezo_ranges[i] / self.piezo_conv_factors[i]) for i in range(2)]
        [fast_axis_step_size, middle_axis_step_size] = [(self.piezo_step_sizes[i] / self.piezo_conv_factors[i]) for i in
                                                        range(2)]
        [fast_axis_start, middle_axis_start] = [(self.piezo_starts[i] / self.piezo_conv_factors[i]) for i in range(2)]
        fast_axis_positions = 1 + int(np.ceil(fast_axis_size / fast_axis_step_size))
        middle_axis_positions = 1 + int(np.ceil(middle_axis_size / middle_axis_step_size))
        positions = fast_axis_positions * middle_axis_positions
        # total_samples = ((cycle_samples * fast_axis_positions) + return_samples) * middle_axis_positions

        for i, start in enumerate(self.digital_starts):
            temp = np.zeros(cycle_samples)
            startSamp = int(np.round(start * self.sample_rate))
            endSamp = int(np.round(self.digital_ends[i] * self.sample_rate))
            temp[startSamp:endSamp] = 1
            digital_trigger_sequences.append(np.tile(temp, fast_axis_positions))
            digital_trigger_sequences[i] = np.append(digital_trigger_sequences[i], np.zeros(return_samples))
            digital_trigger_sequences[i] = np.tile(digital_trigger_sequences[i], middle_axis_positions)
        digital_trigger_sequences[0].fill(0)
        digital_trigger_sequences[1].fill(0)
        digital_trigger_sequences[2].fill(0)
        digital_trigger_sequences[laser] = digital_trigger_sequences[3]

        cycle = np.zeros(cycle_samples)
        startSamp = int(np.round(self.piezo_analog_start * self.sample_rate))
        cycle[startSamp:] = np.linspace(0, 1, int(cycle_samples - startSamp))
        temp = cycle * fast_axis_step_size
        for j in range(fast_axis_positions - 2):
            j = j + 1
            temp = np.append(temp, cycle * fast_axis_step_size + j * fast_axis_step_size)
        cycle = np.ones(startSamp) * fast_axis_step_size * (fast_axis_positions - 1)
        temp = np.append(temp, cycle)
        temp = np.append(temp,
                         np.linspace(1, 0, int(cycle_samples - startSamp) + return_samples) * fast_axis_step_size * (
                                 fast_axis_positions - 1))
        analog_trigger_sequences.append(np.tile(temp, middle_axis_positions))

        cycle = np.zeros((cycle_samples * fast_axis_positions + return_samples))
        cycle[cycle_samples * fast_axis_positions:] = 1
        temp = cycle * middle_axis_step_size
        for j in range(middle_axis_positions - 2):
            j = j + 1
            temp = np.append(temp, cycle * middle_axis_step_size + j * middle_axis_step_size)
        cycle = np.ones((cycle_samples * fast_axis_positions + return_samples)) * middle_axis_step_size * (
                middle_axis_positions - 1)
        analog_trigger_sequences.append(np.append(temp, cycle))

        # analog_trigger_sequences.append(np.ones(total_samples)*self.piezo_starts[2])

        return np.asarray(analog_trigger_sequences), np.asarray(digital_trigger_sequences), positions

    def generate_trigger_sequence_beadscan_3d(self, l):
        digital_trigger_sequences = []
        analog_trigger_sequences = []
        cycle_samples = self.sequence_time * self.sample_rate
        cycle_samples = int(np.ceil(cycle_samples))
        return_samples = self.piezo_return_time * self.sample_rate
        return_samples = int(np.ceil(return_samples))
        [fast_axis_size, middle_axis_size, slow_axis_size] = [(self.piezo_ranges[i] / self.piezo_conv_factors[i]) for i
                                                              in
                                                              range(3)]
        [fast_axis_step_size, middle_axis_step_size, slow_axis_step_size] = [
            (self.piezo_step_sizes[i] / self.piezo_conv_factors[i])
            for i in range(3)]
        [fast_axis_start, middle_axis_start, slow_axis_start] = [(self.piezo_starts[i] / self.piezo_conv_factors[i]) for
                                                                 i in
                                                                 range(3)]
        fast_axis_positions = 1 + int(np.ceil(fast_axis_size / fast_axis_step_size))
        middle_axis_positions = 1 + int(np.ceil(middle_axis_size / middle_axis_step_size))
        slow_axis_positions = 1 + int(np.ceil(slow_axis_size / slow_axis_step_size))
        positions = fast_axis_positions * middle_axis_positions * slow_axis_positions

        for i, start in enumerate(self.digital_starts):
            temp = np.zeros(cycle_samples)
            startSamp = int(np.round(start * self.sample_rate))
            endSamp = int(np.round(self.digital_ends[i] * self.sample_rate))
            temp[startSamp:endSamp] = 1
            digital_trigger_sequences.append(np.tile(temp, fast_axis_positions))
            digital_trigger_sequences[i] = np.append(digital_trigger_sequences[i], np.zeros(return_samples))
            digital_trigger_sequences[i] = np.tile(digital_trigger_sequences[i],
                                                   middle_axis_positions * slow_axis_positions)
        digital_trigger_sequences[0].fill(0)
        digital_trigger_sequences[1].fill(0)
        digital_trigger_sequences[2].fill(0)
        digital_trigger_sequences[l] = digital_trigger_sequences[3]

        cycle = np.zeros(cycle_samples)
        startSamp = int(np.round(self.piezo_analog_start * self.sample_rate))
        cycle[startSamp:] = np.linspace(0, 1, int(cycle_samples - startSamp))
        temp = cycle * fast_axis_step_size
        for j in range(fast_axis_positions - 2):
            j = j + 1
            temp = np.append(temp, cycle * fast_axis_step_size + j * fast_axis_step_size)
        cycle = np.ones(startSamp) * fast_axis_step_size * (fast_axis_positions - 1)
        temp = np.append(temp, cycle)
        temp = np.append(temp,
                         np.linspace(1, 0, int(cycle_samples - startSamp) + return_samples) * fast_axis_step_size * (
                                 fast_axis_positions - 1))
        analog_trigger_sequences.append(np.tile(temp, middle_axis_positions * slow_axis_positions))

        cycle = np.zeros((cycle_samples * fast_axis_positions + return_samples))
        cycle[cycle_samples * fast_axis_positions:] = 1
        temp = cycle * middle_axis_step_size
        for j in range(middle_axis_positions - 2):
            j = j + 1
            temp = np.append(temp, cycle * middle_axis_step_size + j * middle_axis_step_size)
        cycle = np.ones((cycle_samples * fast_axis_positions + return_samples)) * middle_axis_step_size * (
                middle_axis_positions - 1)
        temp = np.append(temp, cycle)
        analog_trigger_sequences.append(np.tile(temp, slow_axis_positions))

        cycle = np.zeros(((cycle_samples * fast_axis_positions + return_samples) * middle_axis_positions))
        cycle[(cycle_samples * fast_axis_positions + return_samples) * middle_axis_positions - return_samples:] = 1
        temp = cycle * slow_axis_step_size
        for j in range(slow_axis_positions - 2):
            j = j + 1
            temp = np.append(temp, cycle * slow_axis_step_size + j * slow_axis_step_size)
        cycle = np.ones(
            ((cycle_samples * fast_axis_positions + return_samples) * middle_axis_positions)) * slow_axis_step_size * (
                        slow_axis_positions - 1)
        temp = np.append(temp, cycle)
        analog_trigger_sequences.append(temp)

        return np.asarray(analog_trigger_sequences), np.asarray(digital_trigger_sequences), positions
