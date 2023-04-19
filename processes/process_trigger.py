import numpy as np
from scipy.interpolate import BPoly

class TriggerSequence:

    def __init__(self):
        self.sequence_time = 0.04
        self.sample_rate = 100000
        self.axis_lengths = [0.6, 0.6, 0.08]
        self.step_sizes = [0.03, 0.03, 0.08]
        self.axis_start_pos = [0., 0., 0.]
        self.return_time = 0.002
        self.convFactors = [10., 10., 10.]
        self.analog_start = 0.03
        self.digital_starts = [0.002, 0.007, 0.007, 0.012, 0.012, 0.012]
        self.digital_ends = [0.004, 0.01, 0.01, 0.015, 0.015, 0.015]
        self.bp_increase = BPoly.from_derivatives([0, 1], [[0., 0., 0.], [1., 0., 0.]])
        self.bp_decrease = BPoly.from_derivatives([0, 1], [[1., 0., 0.], [0., 0., 0.]])

    def updata_parameters(self, sequence_time, sample_rate, axis_lengths, step_sizes, axis_start_pos, return_time,
                          convFactors, analog_start, digital_starts, digital_ends):
        self.sequence_time = sequence_time
        self.sample_rate = sample_rate
        self.axis_lengths = axis_lengths
        self.step_sizes = step_sizes
        self.axis_start_pos = axis_start_pos
        self.return_time = return_time
        self.convFactors = convFactors
        self.analog_start = analog_start
        self.digital_starts = digital_starts
        self.digital_ends = digital_ends

    def generate_digital_triggers(self, laser, camera):
        cycle_samples = self.sequence_time * self.sample_rate
        cycle_samples = int(np.ceil(cycle_samples))
        digital_trigger = np.zeros((len(self.digital_starts), cycle_samples))
        startSamp = int(np.round(self.digital_starts[2] * self.sample_rate))
        endSamp = int(np.round(self.digital_ends[2] * self.sample_rate))
        digital_trigger[laser, startSamp:endSamp] = 1
        digital_trigger[camera + 4, startSamp:endSamp] = 1
        return digital_trigger

    def generate_digital_triggers_sw(self, laser, camera):
        cycle_samples = self.sequence_time * self.sample_rate
        cycle_samples = int(np.ceil(cycle_samples))
        digital_trigger = np.zeros((len(self.digital_starts), cycle_samples))
        startSamp = int(np.round(self.digital_starts[0] * self.sample_rate))
        endSamp = int(np.round(self.digital_ends[0] * self.sample_rate))
        digital_trigger[0, startSamp:endSamp] = 1
        startSamp = int(np.round(self.digital_starts[2] * self.sample_rate))
        endSamp = int(np.round(self.digital_ends[2] * self.sample_rate))
        digital_trigger[laser, startSamp:endSamp] = 1
        digital_trigger[camera + 4, startSamp:endSamp] = 1
        return digital_trigger

    def generate_trigger_sequence_2d(self):
        digital_trigger_sequences = []
        analog_trigger_sequences = []
        cycle_samples = self.sequence_time * self.sample_rate
        cycle_samples = int(np.ceil(cycle_samples))
        return_samples = self.return_time * self.sample_rate
        return_samples = int(np.ceil(return_samples))
        [fast_axis_size, middle_axis_size] = [(self.axis_lengths[i] / self.convFactors[i]) for i in range(2)]
        [fast_axis_step_size, middle_axis_step_size] = [(self.step_sizes[i] / self.convFactors[i]) for i in range(2)]
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
        startSamp = int(np.round(self.analog_start * self.sample_rate))
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
        return_samples = self.return_time * self.sample_rate
        return_samples = int(np.ceil(return_samples))
        [fast_axis_size, middle_axis_size, slow_axis_size] = [(self.axis_lengths[i] / self.convFactors[i]) for i in
                                                              range(3)]
        [fast_axis_step_size, middle_axis_step_size, slow_axis_step_size] = [(self.step_sizes[i] / self.convFactors[i])
                                                                             for i in range(3)]
        [fast_axis_start, middle_axis_start, slow_axis_start] = [(self.axis_start_pos[i] / self.convFactors[i]) for i in
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
        startSamp = int(np.round(self.analog_start * self.sample_rate))
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
        return_samples = self.return_time * self.sample_rate
        return_samples = int(np.ceil(return_samples))
        [fast_axis_size, middle_axis_size] = [(self.axis_lengths[i] / self.convFactors[i]) for i in range(2)]
        [fast_axis_step_size, middle_axis_step_size] = [(self.step_sizes[i] / self.convFactors[i]) for i in range(2)]
        [fast_axis_start, middle_axis_start] = [(self.axis_start_pos[i] / self.convFactors[i]) for i in range(2)]
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
        startSamp = int(np.round(self.analog_start * self.sample_rate))
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

        # analog_trigger_sequences.append(np.ones(total_samples)*self.axis_start_pos[2])

        return np.asarray(analog_trigger_sequences), np.asarray(digital_trigger_sequences), positions

    def generate_trigger_sequence_beadscan_3d(self, l):
        digital_trigger_sequences = []
        analog_trigger_sequences = []
        cycle_samples = self.sequence_time * self.sample_rate
        cycle_samples = int(np.ceil(cycle_samples))
        return_samples = self.return_time * self.sample_rate
        return_samples = int(np.ceil(return_samples))
        [fast_axis_size, middle_axis_size, slow_axis_size] = [(self.axis_lengths[i] / self.convFactors[i]) for i in
                                                              range(3)]
        [fast_axis_step_size, middle_axis_step_size, slow_axis_step_size] = [(self.step_sizes[i] / self.convFactors[i])
                                                                             for i in range(3)]
        [fast_axis_start, middle_axis_start, slow_axis_start] = [(self.axis_start_pos[i] / self.convFactors[i]) for i in
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
        startSamp = int(np.round(self.analog_start * self.sample_rate))
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
