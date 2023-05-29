import numpy as np
from scipy.interpolate import BPoly


class TriggerSequence:

    def __init__(self):
        self.sequence_time = 0.04
        self.sample_rate = 100000
        self.piezo_starts = [0., 0., 0.]
        self.piezo_step_sizes = [0.03, 0.03, 0.08]
        self.piezo_ranges = [0.6, 0.6, 0.08]
        self.piezo_return_time = 0.002
        self.piezo_conv_factors = [10., 10., 10.]
        self.piezo_analog_start = 0.03
        self.galvo_starts = [-1.0, -1.0]
        self.galvo_stops = [1.0, 1.0]
        self.galvo_step_sizes = [0.04, 0.04]
        self.galvo_prep = 0.4
        self.digital_starts = [0.002, 0.007, 0.007, 0.012, 0.012, 0.012]
        self.digital_ends = [0.004, 0.01, 0.01, 0.015, 0.015, 0.015]
        self.bp_increase = BPoly.from_derivatives([0, 1], [[0., 0., 0.], [1., 0., 0.]])
        self.bp_decrease = BPoly.from_derivatives([0, 1], [[1., 0., 0.], [0., 0., 0.]])

    def update_piezo_scan_parameters(self, piezo_ranges, piezo_step_sizes, piezo_starts, piezo_analog_start):
        self.piezo_ranges = piezo_ranges
        self.piezo_step_sizes = piezo_step_sizes
        self.piezo_starts = piezo_starts
        self.piezo_analog_start = piezo_analog_start

    def update_galvo_scan_parameters(self, galvo_starts, galvo_stops, galvo_step_sizes):
        self.galvo_starts = galvo_starts
        self.galvo_stops = galvo_stops
        self.galvo_step_sizes = galvo_step_sizes

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
        galvo_steps_x = int(
            np.ceil(1 + 0.5 * np.abs(self.galvo_starts[0] - self.galvo_stops[0]) / self.galvo_step_sizes[0]))
        galvo_steps_y = int(
            np.ceil(1 + 0.5 * np.abs(self.galvo_starts[1] - self.galvo_stops[1]) / self.galvo_step_sizes[1]))
        scan_x_p = np.arange(self.galvo_starts[0], self.galvo_stops[0] + self.galvo_step_sizes[0],
                             self.galvo_step_sizes[0])
        scan_x_temp = np.append(scan_x_p, np.flip(scan_x_p))
        scan_y_temp = np.ones(scan_x_temp.shape[0]) * self.galvo_starts[1]
        scan_y_temp[scan_x_p.shape[0]:] += self.galvo_step_sizes[1]
        scan_x_h = scan_x_temp
        for i in range(galvo_steps_x - 1):
            scan_x_h = np.append(scan_x_h, scan_x_temp)
        scan_y_h = scan_y_temp
        for i in range(galvo_steps_y - 1):
            scan_y_h = np.append(scan_y_h, scan_y_temp + (2 * i + 2) * self.galvo_step_sizes[1])
        scan_x = np.append(scan_x_h, scan_y_h)
        scan_y = np.append(scan_y_h, -scan_x_h)
        analog_trigger = np.zeros((len(self.galvo_starts), scan_x.shape[0] + 32))
        analog_trigger[0] = np.pad(scan_x, (16, 16), 'constant', constant_values=(0, 0))
        analog_trigger[1] = np.pad(scan_y, (16, 16), 'constant', constant_values=(0, 0))
        digital_trigger = np.zeros((len(self.digital_starts), analog_trigger[0].shape[0]))
        laser_trigger = np.ones(scan_x.shape[0])
        laser_trigger[1::2] = 0
        for laser in lasers:
            digital_trigger[laser] = np.pad(laser_trigger, (16, 16), 'constant', constant_values=(0, 0))
        camera_trigger = np.ones(scan_x.shape[0])
        digital_trigger[camera + 4] = np.pad(camera_trigger, (16, 16), 'constant', constant_values=(0, 0))
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
