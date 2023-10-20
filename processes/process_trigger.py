import numpy as np
from scipy.interpolate import BPoly


class TriggerSequence:
    class TriggerParameters:
        def __init__(self):
            self.sample_rate = 100000  # Hz

            self.cycle_time = [0.05, 0.05]
            self.initial_time = [0.008, 0.008]
            self.standby_time = [0.04, 0.04]
            # piezo scanner
            self.piezo_conv_factors = [10., 10., 10.]
            self.piezo_steps = [0.032, 0.032, 0.0]
            self.piezo_ranges = [0.64, 0.64, 0.0]
            self.piezo_positions = [50., 50., 50.]
            self.piezo_return_time = 0.016
            # galvo scanner
            self.v_max = 4e2  # V/s
            self.a_max = 2.4e7  # V/s^2
            self.voltage_range = 10.  # V
            self.galvo_start = -1.6  # V
            self.galvo_stop = 1.6  # V
            self.galvo_laser_start = 8
            self.galvo_laser_interval = 16
            # dot array
            self.dot_start = -1.0  # V
            self.dot_range = 2.0  # V
            self.dot_step = 0.05  # V
            # sawtooth wave
            self.frequency = 200  # Hz
            # square wave
            self.samples_high = 2
            self.samples_low = 4
            # digital triggers
            self.digital_starts = [0.002, 0.007, 0.007, 0.012, 0.012, 0.012]
            self.digital_ends = [0.004, 0.010, 0.010, 0.015, 0.015, 0.015]
            self._get_parameters()

        def _get_parameters(self):
            self.dt = 1 / self.sample_rate
            self.piezo_steps = [step_size / conv_factor for step_size, conv_factor in
                                zip(self.piezo_steps, self.piezo_conv_factors)]
            self.piezo_ranges = [move_range / conv_factor for move_range, conv_factor in
                                 zip(self.piezo_ranges, self.piezo_conv_factors)]
            self.piezo_positions = [position / conv_factor for position, conv_factor in
                                    zip(self.piezo_positions, self.piezo_conv_factors)]
            self.piezo_starts = [i - j for i, j in zip(self.piezo_positions, [k / 2 for k in self.piezo_ranges])]
            self.piezo_scan_pos = [1 + int(np.ceil(safe_divide(scan_range, scan_step))) for scan_range, scan_step in
                                   zip(self.piezo_ranges, self.piezo_steps)]
            self.dot_pos = np.arange(self.dot_start, self.dot_start + self.dot_range + self.dot_step, self.dot_step)
            self.duration = self.dot_pos.size / self.frequency
            self.samples_period = int(self.sample_rate / self.frequency)
            self.samples_delay = int(np.floor(np.abs(self.dot_start - self.galvo_start) / (
                    np.abs(self.galvo_stop - self.galvo_start) / self.samples_period)))
            self.samples_offset = int(
                self.samples_period - self.samples_delay - (self.samples_high + self.samples_low) * self.dot_pos.shape[
                    0])

    def __init__(self, logg=None):
        self.logg = logg or self.setup_logging()
        self._parameters = self.TriggerParameters()

    def __getattr__(self, item):
        if hasattr(self._parameters, item):
            return getattr(self._parameters, item)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{item}'")

    @staticmethod
    def setup_logging():
        import logging
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
        return logging

    def update_piezo_scan_parameters(self, piezo_ranges=None, piezo_step_sizes=None, piezo_positions=None):
        if piezo_ranges is not None:
            self.piezo_ranges = piezo_ranges
        if piezo_step_sizes is not None:
            self.piezo_step_sizes = piezo_step_sizes
        if piezo_positions is not None:
            self.piezo_positions = piezo_positions
        self._parameters._get_parameters()

    def update_galvo_scan_parameters(self, gv_start=None, gv_stop=None, laser_start=None, laser_interval=None,
                                     acceleration=None, velocity=None):
        if gv_start is not None:
            self.galvo_start = gv_start
        if gv_stop is not None:
            self.galvo_stop = gv_stop
        if laser_start is not None:
            self.galvo_laser_start = laser_start
        if laser_interval is not None:
            self.galvo_laser_interval = laser_interval
        if acceleration is not None:
            self.a_max = acceleration
        if velocity is not None:
            self.v_max = velocity
        self._parameters._get_parameters()

    def update_digital_parameters(self, digital_starts=None, digital_ends=None):
        if digital_starts is not None:
            self.digital_starts = digital_starts
        if digital_ends is not None:
            self.digital_ends = digital_ends
        self._parameters._get_parameters()

    def update_camera_parameters(self, initial_time=None, standby_time=None, cycle_time=None):
        if initial_time is not None:
            self.initial_time = initial_time
        if standby_time is not None:
            self.standby_time = standby_time
        if self.cycle_time is not None:
            self.cycle_time = cycle_time
        self._parameters._get_parameters()

    def generate_digital_triggers(self, lasers, camera):
        _starts = (np.array(self.digital_starts) * self.sample_rate).astype(int)
        _ends = (np.array(self.digital_ends) * self.sample_rate).astype(int)
        initial_samples = int(np.ceil(self.initial_time[camera] * self.sample_rate))
        standby_samples = int(np.ceil(self.standby_time[camera] * self.sample_rate))
        cam_ind = camera + 4
        offset = max(0, initial_samples - _starts[cam_ind])
        _starts += offset
        _ends += offset
        cycle_samples = _ends[cam_ind] + standby_samples
        digital_trigger = np.zeros((len(self.digital_starts), cycle_samples), dtype=int)
        digital_trigger[cam_ind, _starts[cam_ind]:_ends[cam_ind]] = 1
        for laser in lasers:
            digital_trigger[laser, _starts[laser]:_ends[laser]] = 1
        return digital_trigger

    def generate_wfs_triggers(self, lasers, camera):
        _starts = [int(digital_start * self.sample_rate) for digital_start in self.digital_starts]
        _ends = [int(digital_end * self.sample_rate) for digital_end in self.digital_ends]
        cycle_samples = int(np.ceil(self.cycle_time[camera] * self.sample_rate))
        digital_trigger = np.zeros((len(self.digital_starts), cycle_samples))
        for laser in lasers:
            digital_start = _starts[laser]
            digital_end = _ends[laser]
            digital_trigger[laser, digital_start:digital_end] = 1
        cam_ind = camera + 4
        digital_start = _starts[cam_ind]
        digital_end = _ends[cam_ind]
        digital_trigger[cam_ind, digital_start:digital_end] = 1
        return digital_trigger

    def generate_confocal_triggers(self, lasers, camera):
        """
        analog to galvo x and y
        digital to laser and camera
        """
        t_scan = (self.galvo_stop - self.galvo_start) / self.v_max
        s_scan = int(np.ceil(t_scan / self.dt)) + 1
        seq_scan = np.linspace(-1, 1, s_scan)
        pos = seq_scan[np.arange(self.galvo_laser_start, s_scan, self.galvo_laser_interval)]
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
        x_axis_scan = np.pad(x_axis_scan, (0, int(self.galvo_laser_interval / 2)), 'constant',
                             constant_values=(0, x_axis_scan[-1]))
        x_axis_scan = np.append(x_axis_scan, np.flip(x_axis_scan))
        x_axis_scan = np.pad(x_axis_scan, (0, self.galvo_laser_interval), 'constant',
                             constant_values=(0, x_axis_scan[-1]))
        x_axis_scan = np.tile(x_axis_scan, int(pos.shape[0] / 2))
        y_axis_scan = pos[0] * np.ones(s_scan + int(s_scan * 0.01))
        for n in range(pos.shape[0] - 2):
            acc = pos[n] + seq_acc
            deacc = pos[n + 1] - np.flip(seq_acc)
            t_scan_y = (deacc[0] - acc[-1]) / self.v_max
            s_scan_y = int(np.ceil(t_scan_y / self.dt)) + 1
            seq_scan = np.linspace(acc[-1], deacc[0], s_scan_y)
            temp = np.append(acc, seq_scan[1:-1])
            temp = np.append(temp, deacc)
            temp = np.append(temp, pos[n + 1] * np.ones(s_scan + self.galvo_laser_interval - seq_scan.shape[0] + 2))
            y_axis_scan = np.append(y_axis_scan, temp)
        y_axis_scan = np.append(y_axis_scan, y_axis_scan[-1] - seq_acc)
        t_return = (y_axis_scan[-1] - (pos[0] + seq_acc[-1])) / self.v_max
        s_return = int(np.ceil(t_return / self.dt)) + 1
        seq_return = np.linspace(y_axis_scan[-1], pos[0] + seq_acc[-1], s_return)
        y_axis_scan = np.append(y_axis_scan, seq_return[1:-1])
        y_axis_scan = np.append(y_axis_scan, pos[0] + np.flip(seq_acc))
        x_axis_scan = np.append(x_axis_scan, x_axis_scan[-1] * np.ones(y_axis_scan.shape[0] - x_axis_scan.shape[0]))
        laser_trigger = np.zeros(2 * s_scan + 4 * int(s_scan * 0.01) + 2 * self.galvo_laser_interval)
        laser_trigger[int(s_scan * 0.01):s_scan + int(s_scan * 0.01)] = 1
        laser_trigger[3 * int(s_scan * 0.01) + s_scan + self.galvo_laser_interval:3 * int(
            s_scan * 0.01) + s_scan + self.galvo_laser_interval + s_scan] = 1
        laser_trigger = np.tile(laser_trigger, int(pos.shape[0] / 2))
        camera_trigger = np.ones(laser_trigger.shape[0])
        camera_trigger[:int(s_scan * 0.01)] = 0
        camera_trigger[- int(self.galvo_laser_interval) - int(s_scan * 0.01):] = 0
        laser_trigger = np.append(laser_trigger, np.zeros(x_axis_scan.shape[0] - laser_trigger.shape[0]))
        camera_trigger = np.append(camera_trigger, np.zeros(x_axis_scan.shape[0] - camera_trigger.shape[0]))
        n = len(lasers)
        analog_trigger = [np.tile(x_axis_scan, n), np.tile(y_axis_scan, n)]
        digital_trigger = np.zeros((len(self.digital_starts), n * camera_trigger.shape[0]))
        if n == 1:
            digital_trigger[lasers[0]] = laser_trigger
            digital_trigger[camera + 4] = camera_trigger
        elif n == 2:
            temp = np.zeros_like(laser_trigger)
            digital_trigger[lasers[0]] = np.append(laser_trigger, temp)
            digital_trigger[lasers[1]] = np.append(temp, laser_trigger)
            digital_trigger[camera + 4] = np.append(temp, camera_trigger)
        return np.asarray(analog_trigger), digital_trigger, pos

    def generate_galvo_scanning(self, lasers, camera):
        """
        analog to galvo x and y
        digital to laser and camera
        """
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

    def generate_resolft_sequence(self):
        """
        analog to piezo x, y, and z
        digital to laser and camera
        """
        digital_trigger_sequences = []
        analog_trigger_sequences = []
        cycle_samples = self.cycle_time * self.sample_rate
        cycle_samples = int(np.ceil(cycle_samples))
        return_samples = self.piezo_return_time * self.sample_rate
        return_samples = int(np.ceil(return_samples))
        [fast_axis_range, middle_axis_range, slow_axis_range] = [(self.piezo_ranges[i] / self.piezo_conv_factors[i]) for
                                                                 i
                                                                 in range(3)]
        [fast_axis_step, middle_axis_step, slow_axis_step] = [
            (self.piezo_step_sizes[i] / self.piezo_conv_factors[i]) for i in range(3)]
        [fast_axis_start, middle_axis_start, slow_axis_start] = [(self.piezo_starts[i] / self.piezo_conv_factors[i]) for
                                                                 i in range(3)]
        fast_axis_positions = 1 + int(np.ceil(safe_divide(fast_axis_range, fast_axis_step)))
        middle_axis_positions = 1 + int(np.ceil(safe_divide(middle_axis_range, middle_axis_step)))
        slow_axis_positions = 1 + int(np.ceil(safe_divide(slow_axis_range, slow_axis_step)))
        positions = fast_axis_positions * middle_axis_positions * slow_axis_positions

        for i, start in enumerate(self.digital_starts):
            temp = np.zeros(cycle_samples)
            digital_start = int(np.round(start * self.sample_rate))
            digital_end = int(np.round(self.digital_ends[i] * self.sample_rate))
            temp[digital_start:digital_end] = 1
            digital_trigger_sequences.append(np.tile(temp, fast_axis_positions))
            digital_trigger_sequences[i] = np.append(digital_trigger_sequences[i], np.zeros(return_samples))
            digital_trigger_sequences[i] = np.tile(digital_trigger_sequences[i],
                                                   middle_axis_positions * slow_axis_positions)

        cycle = np.zeros(cycle_samples)
        digital_start = int(np.round(self.piezo_analog_start * self.sample_rate))
        cycle[digital_start:] = np.linspace(0, 1, int(cycle_samples - digital_start))
        temp = cycle * fast_axis_step
        for j in range(fast_axis_positions - 2):
            j = j + 1
            temp = np.append(temp, cycle * fast_axis_step + j * fast_axis_step)
        cycle = np.ones(digital_start) * fast_axis_step * (fast_axis_positions - 1)
        temp = np.append(temp, cycle)
        temp = np.append(temp,
                         np.linspace(1, 0, int(cycle_samples - digital_start) + return_samples) * fast_axis_step * (
                                 fast_axis_positions - 1))
        analog_trigger_sequences.append(np.tile(temp, middle_axis_positions * slow_axis_positions) + fast_axis_start)

        cycle = np.zeros((cycle_samples * fast_axis_positions + return_samples))
        cycle[cycle_samples * fast_axis_positions:] = 1
        temp = cycle * middle_axis_step
        for j in range(middle_axis_positions - 2):
            j = j + 1
            temp = np.append(temp, cycle * middle_axis_step + j * middle_axis_step)
        cycle = np.ones((cycle_samples * fast_axis_positions + return_samples)) * middle_axis_step * (
                middle_axis_positions - 1)
        temp = np.append(temp, cycle)
        analog_trigger_sequences.append(np.tile(temp, slow_axis_positions) + middle_axis_start)

        cycle = np.zeros(((cycle_samples * fast_axis_positions + return_samples) * middle_axis_positions))
        if slow_axis_positions > 1:
            cycle[(cycle_samples * fast_axis_positions + return_samples) * middle_axis_positions - return_samples:] = 1
            temp = cycle * slow_axis_step
            for j in range(slow_axis_positions - 2):
                j = j + 1
                temp = np.append(temp, cycle * slow_axis_step + j * slow_axis_step)
            cycle = np.ones(
                ((cycle_samples * fast_axis_positions + return_samples) * middle_axis_positions)) * slow_axis_step * (
                            slow_axis_positions - 1)
            temp = np.append(temp, cycle)
        else:
            temp = cycle
        analog_trigger_sequences.append(temp + slow_axis_start)

        return np.asarray(analog_trigger_sequences), np.asarray(digital_trigger_sequences), positions

    def generate_confocal_resolft_2d(self, camera=0):
        interval_samples = 16
        digital_sequences = [np.empty((0,)) for _ in range(len(self.digital_starts))]
        galvo_sequences = [np.empty((0,)) for _ in range(2)]
        piezo_sequences = [np.empty((0,)) for _ in range(2)]
        _starts = [int(digital_start * self.sample_rate) for digital_start in self.digital_starts]
        _ends = [int(digital_end * self.sample_rate) for digital_end in self.digital_ends]
        _sth = np.linspace(0, self.duration, int(self.sample_rate * self.duration), endpoint=False)
        fast_axis_galvo = np.abs(self.galvo_stop - self.galvo_start) * np.mod(self.frequency * _sth,
                                                                              1.0) + self.galvo_start
        slow_axis_galvo = self.dot_start + self.dot_step * np.floor(self.frequency * _sth).astype(int)
        square_wave = np.pad(np.ones((self.samples_high + self.samples_low) * self.dot_pos.size),
                             (self.samples_delay, self.samples_offset), 'constant', constant_values=(0, 0))
        laser_trigger = np.tile(square_wave, self.dot_pos.size)
        # ON
        galvo_sequences[0] = np.pad(fast_axis_galvo, (interval_samples, 0), 'constant',
                                    constant_values=(self.galvo_start, self.galvo_start))
        galvo_sequences[1] = np.pad(slow_axis_galvo, (interval_samples, 0), 'constant',
                                    constant_values=(self.dot_start, self.dot_start))
        digital_sequences[0] = np.pad(laser_trigger, (interval_samples, 0), 'constant', constant_values=(0, 0))
        for i in range(5):
            digital_sequences[i + 1] = np.zeros(digital_sequences[0].size)
        # OFF
        off_samples = _ends[1] - _starts[1]
        galvo_sequences[0] = np.concatenate(
            (galvo_sequences[0], self.galvo_start * np.ones(off_samples + interval_samples)))
        galvo_sequences[1] = np.concatenate(
            (galvo_sequences[1], self.dot_start * np.ones(off_samples + interval_samples)))
        digital_sequences[0] = np.concatenate((digital_sequences[0], np.zeros(off_samples + interval_samples)))
        digital_sequences[1] = np.concatenate(
            (digital_sequences[1], np.concatenate((np.zeros(interval_samples), np.ones(off_samples)))))
        digital_sequences[2] = digital_sequences[1]
        for i in range(3):
            digital_sequences[i + 3] = np.concatenate(
                (digital_sequences[i + 3], np.zeros(off_samples + interval_samples)))
        # Read
        galvo_sequences[0] = np.concatenate(
            (galvo_sequences[0], np.concatenate((self.galvo_start * np.ones(interval_samples), fast_axis_galvo))))
        galvo_sequences[1] = np.concatenate(
            (galvo_sequences[1], np.concatenate((self.dot_start * np.ones(interval_samples), slow_axis_galvo))))
        for i in range(3):
            digital_sequences[i] = np.concatenate(
                (digital_sequences[i], np.zeros(laser_trigger.size + interval_samples)))
        digital_sequences[3] = np.concatenate(
            (digital_sequences[3], np.concatenate((np.zeros(interval_samples), laser_trigger))))
        temp = np.pad(np.ones(laser_trigger.size - int(self.samples_delay / 2) - int(self.samples_offset / 2)),
                      (interval_samples + int(self.samples_delay / 2), int(self.samples_offset / 2)), 'constant',
                      constant_values=(0, 0))
        digital_sequences[4] = np.concatenate((digital_sequences[4], temp))
        digital_sequences[5] = np.concatenate((digital_sequences[5], np.zeros(laser_trigger.size + interval_samples)))
        # Piezo Fast Axis
        standby_samples = int(np.ceil(self.standby_time[camera] * self.sample_rate))
        return_samples = int(np.ceil(self.piezo_return_time * self.sample_rate))
        _temp = np.tile(np.zeros(digital_sequences[0].size + standby_samples), self.piezo_scan_pos[0])
        _temp[-standby_samples:-standby_samples + return_samples] = self.piezo_steps[1] * np.linspace(0, 1,
                                                                                                      return_samples)
        _temp[-standby_samples + return_samples:] = self.piezo_steps[1]
        piezo_sequences[1] = _temp + self.piezo_starts[1]
        for i in range(self.piezo_scan_pos[1] - 1):
            temp = _temp + self.piezo_starts[1] + (i + 1) * self.piezo_steps[1]
            piezo_sequences[1] = np.concatenate((piezo_sequences[1], temp))
        piezo_sequences[0] = np.zeros(digital_sequences[0].size) + self.piezo_starts[0]
        _temp = np.concatenate((np.linspace(0, 1, standby_samples), np.ones(digital_sequences[0].size)))
        for i in range(self.piezo_scan_pos[0] - 1):
            temp = _temp * self.piezo_steps[0] + self.piezo_starts[0] + i * self.piezo_steps[0]
            piezo_sequences[0] = np.concatenate((piezo_sequences[0], temp))
        piezo_sequences[0] = np.concatenate(
            (piezo_sequences[0], np.linspace(piezo_sequences[0][-1], piezo_sequences[0][0], return_samples)))
        piezo_sequences[0] = np.concatenate(
            (piezo_sequences[0], piezo_sequences[0][-1] * np.ones(standby_samples - return_samples)))
        piezo_sequences[0] = np.tile(piezo_sequences[0], self.piezo_scan_pos[1])
        for i in range(2):
            piezo_sequences[i][-interval_samples:] = np.linspace(piezo_sequences[i][-1], self.piezo_positions[i],
                                                                 interval_samples)
        galvo_sequences[0] = np.concatenate((galvo_sequences[0], self.galvo_start * np.ones(standby_samples)))
        galvo_sequences[1] = np.concatenate((galvo_sequences[1], self.dot_start * np.ones(standby_samples)))
        for i in range(2):
            galvo_sequences[i] = np.tile(galvo_sequences[i], self.piezo_scan_pos[0])
            galvo_sequences[i] = np.tile(galvo_sequences[i], self.piezo_scan_pos[1])
            galvo_sequences[i][-interval_samples:] = np.linspace(galvo_sequences[i][-1], 0., interval_samples)
        for i in range(6):
            digital_sequences[i] = np.concatenate((digital_sequences[i], np.zeros(standby_samples)))
            digital_sequences[i] = np.tile(digital_sequences[i], self.piezo_scan_pos[0])
            digital_sequences[i] = np.tile(digital_sequences[i], self.piezo_scan_pos[1])
        return np.asarray(galvo_sequences), np.asarray(piezo_sequences), np.asarray(digital_sequences), sum(
            self.piezo_scan_pos)

    def generate_galvo_resolft_2d(self, camera=0):
        interval_samples = 16
        digital_sequences = [np.empty((0,)) for _ in range(len(self.digital_starts))]
        galvo_sequences = [np.empty((0,)) for _ in range(2)]
        piezo_sequences = [np.empty((0,)) for _ in range(2)]
        _starts = [int(digital_start * self.sample_rate) for digital_start in self.digital_starts]
        _ends = [int(digital_end * self.sample_rate) for digital_end in self.digital_ends]
        _sth = np.linspace(0, self.duration, int(self.sample_rate * self.duration), endpoint=False)
        fast_axis_galvo = np.abs(self.galvo_stop - self.galvo_start) * np.mod(self.frequency * _sth,
                                                                              1.0) + self.galvo_start
        slow_axis_galvo = self.dot_start + self.dot_step * np.floor(self.frequency * _sth).astype(int)
        one_period = np.concatenate((np.ones(self.samples_high), np.zeros(self.samples_low)))
        square_wave = np.pad(np.tile(one_period, self.dot_pos.size), (self.samples_delay, self.samples_offset),
                             'constant', constant_values=(0, 0))
        laser_trigger = np.tile(square_wave, self.dot_pos.size)
        # ON
        galvo_sequences[0] = np.pad(fast_axis_galvo, (interval_samples, 0), 'constant',
                                    constant_values=(self.galvo_start, self.galvo_start))
        galvo_sequences[1] = np.pad(slow_axis_galvo, (interval_samples, 0), 'constant',
                                    constant_values=(self.dot_start, self.dot_start))
        digital_sequences[0] = np.pad(laser_trigger, (interval_samples, 0), 'constant', constant_values=(0, 0))
        for i in range(5):
            digital_sequences[i + 1] = np.zeros(digital_sequences[0].size)
        # OFF
        off_samples = _ends[1] - _starts[1]
        galvo_sequences[0] = np.concatenate(
            (galvo_sequences[0], self.galvo_start * np.ones(off_samples + interval_samples)))
        galvo_sequences[1] = np.concatenate(
            (galvo_sequences[1], self.dot_start * np.ones(off_samples + interval_samples)))
        digital_sequences[0] = np.concatenate((digital_sequences[0], np.zeros(off_samples + interval_samples)))
        digital_sequences[1] = np.concatenate(
            (digital_sequences[1], np.concatenate((np.zeros(interval_samples), np.ones(off_samples)))))
        digital_sequences[2] = digital_sequences[1]
        for i in range(3):
            digital_sequences[i + 3] = np.concatenate(
                (digital_sequences[i + 3], np.zeros(off_samples + interval_samples)))
        # Read
        galvo_sequences[0] = np.concatenate(
            (galvo_sequences[0], np.concatenate((self.galvo_start * np.ones(interval_samples), fast_axis_galvo))))
        galvo_sequences[1] = np.concatenate(
            (galvo_sequences[1], np.concatenate((self.dot_start * np.ones(interval_samples), slow_axis_galvo))))
        for i in range(3):
            digital_sequences[i] = np.concatenate(
                (digital_sequences[i], np.zeros(laser_trigger.size + interval_samples)))
        digital_sequences[3] = np.concatenate(
            (digital_sequences[3], np.concatenate((np.zeros(interval_samples), laser_trigger))))
        temp = np.pad(np.ones(laser_trigger.size - int(self.samples_delay / 2) - int(self.samples_offset / 2)),
                      (interval_samples + int(self.samples_delay / 2), int(self.samples_offset / 2)), 'constant',
                      constant_values=(0, 0))
        digital_sequences[4] = np.concatenate((digital_sequences[4], temp))
        digital_sequences[5] = np.concatenate((digital_sequences[5], np.zeros(laser_trigger.size + interval_samples)))
        # Piezo Fast Axis
        standby_samples = int(np.ceil(self.standby_time[camera] * self.sample_rate))
        return_samples = int(np.ceil(self.piezo_return_time * self.sample_rate))
        _temp = np.tile(np.zeros(digital_sequences[0].size + standby_samples), self.piezo_scan_pos[0])
        _temp[-standby_samples:-standby_samples + return_samples] = self.piezo_steps[1] * np.linspace(0, 1,
                                                                                                      return_samples)
        _temp[-standby_samples + return_samples:] = self.piezo_steps[1]
        piezo_sequences[1] = _temp + self.piezo_starts[1]
        for i in range(self.piezo_scan_pos[1] - 1):
            temp = _temp + self.piezo_starts[1] + (i + 1) * self.piezo_steps[1]
            piezo_sequences[1] = np.concatenate((piezo_sequences[1], temp))
        piezo_sequences[0] = np.zeros(digital_sequences[0].size) + self.piezo_starts[0]
        _temp = np.concatenate((np.linspace(0, 1, standby_samples), np.ones(digital_sequences[0].size)))
        for i in range(self.piezo_scan_pos[0] - 1):
            temp = _temp * self.piezo_steps[0] + self.piezo_starts[0] + i * self.piezo_steps[0]
            piezo_sequences[0] = np.concatenate((piezo_sequences[0], temp))
        piezo_sequences[0] = np.concatenate(
            (piezo_sequences[0], np.linspace(piezo_sequences[0][-1], piezo_sequences[0][0], return_samples)))
        piezo_sequences[0] = np.concatenate(
            (piezo_sequences[0], piezo_sequences[0][-1] * np.ones(standby_samples - return_samples)))
        piezo_sequences[0] = np.tile(piezo_sequences[0], self.piezo_scan_pos[1])
        for i in range(2):
            piezo_sequences[i][-interval_samples:] = np.linspace(piezo_sequences[i][-1], self.piezo_positions[i],
                                                                 interval_samples)
        galvo_sequences[0] = np.concatenate((galvo_sequences[0], self.galvo_start * np.ones(standby_samples)))
        galvo_sequences[1] = np.concatenate((galvo_sequences[1], self.dot_start * np.ones(standby_samples)))
        for i in range(2):
            galvo_sequences[i] = np.tile(galvo_sequences[i], self.piezo_scan_pos[0])
            galvo_sequences[i] = np.tile(galvo_sequences[i], self.piezo_scan_pos[1])
            galvo_sequences[i][-interval_samples:] = np.linspace(galvo_sequences[i][-1], 0., interval_samples)
        for i in range(6):
            digital_sequences[i] = np.concatenate((digital_sequences[i], np.zeros(standby_samples)))
            digital_sequences[i] = np.tile(digital_sequences[i], self.piezo_scan_pos[0])
            digital_sequences[i] = np.tile(digital_sequences[i], self.piezo_scan_pos[1])
        return np.asarray(galvo_sequences), np.asarray(piezo_sequences), np.asarray(digital_sequences), sum(
            self.piezo_scan_pos)

    def generate_bead_scan_2d(self, camera=0):
        cam_ind = camera + 4
        digital_trigger_sequences = []
        analog_trigger_sequences = []
        initial_samples = int(np.ceil(self.initial_time[camera] * self.sample_rate))
        standby_samples = int(np.ceil(self.standby_time[camera] * self.sample_rate))
        return_samples = int(np.ceil(self.piezo_return_time * self.sample_rate))
        _starts = [int(digital_start * self.sample_rate) for digital_start in self.digital_starts]
        _ends = [int(digital_end * self.sample_rate) for digital_end in self.digital_ends]

        if _starts[cam_ind] <= initial_samples:
            temp = initial_samples - _starts[cam_ind]
            _starts = [(_start + temp) for _start in _starts]
            _ends = [(_end + temp) for _end in _ends]
        cycle_samples = _ends[cam_ind] + standby_samples
        print(cycle_samples / self.sample_rate)
        [fast_axis_size, middle_axis_size] = [(self.piezo_ranges[i] / self.piezo_conv_factors[i]) for i in range(2)]
        [fast_axis_step_size, middle_axis_step_size] = [(self.piezo_step_sizes[i] / self.piezo_conv_factors[i]) for i in
                                                        range(2)]
        [fast_axis_start, middle_axis_start] = [(self.piezo_starts[i] / self.piezo_conv_factors[i]) for i in range(2)]
        fast_axis_positions = 1 + int(np.ceil(fast_axis_size / fast_axis_step_size))
        middle_axis_positions = 1 + int(np.ceil(middle_axis_size / middle_axis_step_size))
        positions = fast_axis_positions * middle_axis_positions
        # total_samples = ((cycle_samples * fast_axis_positions) + return_samples) * middle_axis_positions

        cycle = np.zeros(cycle_samples)
        digital_start = _ends[cam_ind]
        cycle[digital_start:] = np.linspace(0, 1, int(cycle_samples - digital_start))
        temp = cycle * fast_axis_step_size
        for j in range(fast_axis_positions - 2):
            j = j + 1
            temp = np.append(temp, cycle * fast_axis_step_size + j * fast_axis_step_size)
        cycle = np.ones(digital_start) * fast_axis_step_size * (fast_axis_positions - 1)
        temp = np.append(temp, cycle)
        temp = np.append(temp,
                         np.linspace(1, 0,
                                     int(cycle_samples - digital_start) + return_samples) * fast_axis_step_size * (
                                 fast_axis_positions - 1))
        analog_trigger_sequences.append(np.tile(temp, middle_axis_positions) + fast_axis_start)

        cycle = np.zeros((cycle_samples * fast_axis_positions + return_samples))
        cycle[cycle_samples * fast_axis_positions:] = 1
        temp = cycle * middle_axis_step_size
        for j in range(middle_axis_positions - 2):
            j = j + 1
            temp = np.append(temp, cycle * middle_axis_step_size + j * middle_axis_step_size)
        cycle = np.ones((cycle_samples * fast_axis_positions + return_samples)) * middle_axis_step_size * (
                middle_axis_positions - 1)
        analog_trigger_sequences.append(np.append(temp, cycle) + middle_axis_start)

        for i, start in enumerate(_starts):
            temp = np.zeros(cycle_samples)
            end = _ends[i]
            temp[start:end] = 1
            digital_trigger_sequences.append(np.tile(temp, fast_axis_positions))
            digital_trigger_sequences[i] = np.append(digital_trigger_sequences[i], np.zeros(return_samples))
            digital_trigger_sequences[i] = np.tile(digital_trigger_sequences[i], middle_axis_positions)
        # digital_trigger_sequences[0].fill(0)
        # digital_trigger_sequences[1].fill(0)
        # digital_trigger_sequences[2].fill(0)
        # digital_trigger_sequences[cam_ind] = digital_trigger_sequences[3]

        return np.asarray(analog_trigger_sequences), np.asarray(digital_trigger_sequences), positions


def safe_divide(numerator, denominator):
    try:
        return numerator / denominator
    except ZeroDivisionError:
        return 0
