import numpy as np


class TriggerSequence:
    class TriggerParameters:
        def __init__(self):
            # daq
            self.sample_rate = 100000  # Hz
            self.dt = 1 / self.sample_rate
            # camera
            self.cycle_time = 0.05
            self.initial_time = 0.008
            self.standby_time = 0.04
            # piezo scanner
            self.piezo_conv_factors = [10., 10., 10.]
            self.piezo_steps = [0.032, 0.032, 0.128]
            self.piezo_ranges = [0.0, 0.0, 0.0]
            self.piezo_positions = [50., 50., 50.]
            self.piezo_return_time = 0.016
            self.piezo_steps = [step_size / conv_factor for step_size, conv_factor in
                                zip(self.piezo_steps, self.piezo_conv_factors)]
            self.piezo_ranges = [move_range / conv_factor for move_range, conv_factor in
                                 zip(self.piezo_ranges, self.piezo_conv_factors)]
            self.piezo_positions = [position / conv_factor for position, conv_factor in
                                    zip(self.piezo_positions, self.piezo_conv_factors)]
            self.piezo_starts = [i - j for i, j in zip(self.piezo_positions, [k / 2 for k in self.piezo_ranges])]
            self.piezo_scan_pos = [1 + int(np.ceil(safe_divide(scan_range, scan_step))) for scan_range, scan_step in
                                   zip(self.piezo_ranges, self.piezo_steps)]
            self.piezo_scan_positions = [np.arange(start, start + range_ + step, step) for start, range_, step in
                                         zip(self.piezo_starts, self.piezo_ranges, self.piezo_steps)]
            # galvo scanner
            self.galvo_start = -1.2  # V
            self.galvo_stop = 1.2  # V
            self.galvo_return = 64  # ~640 us
            # sawtooth wave
            self.frequency = 250  # Hz
            self.samples_period = int(self.sample_rate / self.frequency)
            # dot array
            self.dot_start = -0.8  # V
            self.dot_range = 1.6  # V
            self.dot_offset = 0.0  # V
            self.dot_step_s = 4  # samples
            self.dot_step_v = (self.dot_step_s / self.sample_rate) * (
                    np.abs(self.galvo_stop - self.galvo_start) / (1 / self.frequency))
            self.dot_pos = np.arange(self.dot_start, self.dot_start + self.dot_range + self.dot_step_v, self.dot_step_v)
            self.duration = self.dot_pos.size * self.samples_period
            # square wave
            self.samples_high = 1
            self.samples_low = self.dot_step_s - self.samples_high
            self.samples_delay = int(np.abs(self.dot_start - self.galvo_start) / (
                    np.abs(self.galvo_stop - self.galvo_start) / self.samples_period))
            self.samples_offset = self.samples_period - self.samples_delay - self.dot_step_s * self.dot_pos.size
            # digital triggers
            self.digital_starts = [0.002, 0.007, 0.007, 0.012, 0.012, 0.012, 0.012]
            self.digital_ends = [0.004, 0.010, 0.010, 0.015, 0.015, 0.015, 0.015]
            self.digital_starts = [int(digital_start * self.sample_rate) for digital_start in self.digital_starts]
            self.digital_ends = [int(digital_end * self.sample_rate) for digital_end in self.digital_ends]

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

    def update_piezo_scan_parameters(self, piezo_ranges=None, piezo_steps=None, piezo_positions=None):
        original_values = {
            "piezo_ranges": self.piezo_ranges,
            "piezo_steps": self.piezo_steps,
            "piezo_positions": self.piezo_positions
        }
        try:
            if piezo_ranges is not None:
                self.piezo_ranges = piezo_ranges
            if piezo_steps is not None:
                self.piezo_steps = piezo_steps
            if piezo_positions is not None:
                self.piezo_positions = piezo_positions
            self.piezo_steps = [step_size / conv_factor for step_size, conv_factor in
                                zip(self.piezo_steps, self.piezo_conv_factors)]
            self.piezo_ranges = [move_range / conv_factor for move_range, conv_factor in
                                 zip(self.piezo_ranges, self.piezo_conv_factors)]
            self.piezo_positions = [position / conv_factor for position, conv_factor in
                                    zip(self.piezo_positions, self.piezo_conv_factors)]
            self.piezo_starts = [i - j for i, j in zip(self.piezo_positions, [k / 2 for k in self.piezo_ranges])]
            self.piezo_scan_pos = [1 + int(np.ceil(safe_divide(scan_range, scan_step))) for scan_range, scan_step in
                                   zip(self.piezo_ranges, self.piezo_steps)]
            self.piezo_scan_positions = [np.arange(start, start + range_ + step, step) for start, range_, step in
                                         zip(self.piezo_starts, self.piezo_ranges, self.piezo_steps)]
            if any(np.any(np.logical_or(pos < 0., pos > 10.)) for pos in self.piezo_scan_positions):
                self.logg.error("Invalid parameter combination.")
                raise ValueError("Invalid Piezo scanning parameters.")
        except ValueError:
            for attr, value in original_values.items():
                setattr(self, attr, value)
            self.logg.info("Piezo scanning parameters reverted to original values.")
            return

    def update_galvo_scan_parameters(self, galvo_start=None, galvo_stop=None, dot_start=None, dot_range=None,
                                     dot_offset=None, dot_step_s=None, frequency=None, samples_delay=None,
                                     samples_high=None):
        original_values = {
            "galvo_start": self.galvo_start,
            "galvo_stop": self.galvo_stop,
            "dot_start": self.dot_start,
            "dot_range": self.dot_range,
            "dot_offset": self.dot_offset,
            "dot_step_s": self.dot_step_s,
            "frequency": self.frequency,
            "samples_period": self.samples_period,
            "samples_high": self.samples_high,
            "samples_delay": self.samples_delay,
        }
        try:
            if galvo_start is not None:
                self.galvo_start = galvo_start
            if galvo_stop is not None:
                self.galvo_stop = galvo_stop
            if dot_start is not None:
                self.dot_start = dot_start
            if dot_range is not None:
                self.dot_range = dot_range
            if dot_offset is not None:
                self.dot_offset = dot_offset
            if frequency is not None:
                self.frequency = frequency
                self.samples_period = int(self.sample_rate / self.frequency)
            if dot_step_s is not None:
                self.dot_step_s = dot_step_s
                self.dot_step_v = (self.dot_step_s / self.sample_rate) * (
                        np.abs(self.galvo_stop - self.galvo_start) / (1 / self.frequency))
            if samples_high is not None:
                self.samples_high = samples_high
                self.samples_low = self.dot_step_s - self.samples_high
                self.dot_pos = np.arange(self.dot_start, self.dot_start + self.dot_range + self.dot_step_v,
                                         self.dot_step_v)
                self.duration = self.dot_pos.size * self.samples_period
            if samples_delay is not None:
                self.samples_delay = samples_delay + int(np.floor(np.abs(self.dot_start - self.galvo_start) / (
                        np.abs(self.galvo_stop - self.galvo_start) / self.samples_period)))
            else:
                self.samples_delay = int(np.floor(np.abs(self.dot_start - self.galvo_start) / (
                        np.abs(self.galvo_stop - self.galvo_start) / self.samples_period)))
            self.samples_offset = self.samples_period - self.samples_delay - self.dot_step_s * self.dot_pos.size
            if self.samples_offset < 0:
                self.logg.error("Invalid parameter combination leading to negative samples_offset.")
                raise ValueError("Invalid Galvo scanning parameters.")
        except ValueError:
            for attr, value in original_values.items():
                setattr(self, attr, value)
            self.logg.info("Galvo scanning parameters reverted to original values.")
            return

    def update_digital_parameters(self, digital_starts=None, digital_ends=None):
        if digital_starts is not None:
            self.digital_starts = digital_starts
        if digital_ends is not None:
            self.digital_ends = digital_ends
        self.digital_starts = [int(digital_start * self.sample_rate) for digital_start in self.digital_starts]
        self.digital_ends = [int(digital_end * self.sample_rate) for digital_end in self.digital_ends]

    def update_camera_parameters(self, initial_time=None, standby_time=None, cycle_time=None):
        if initial_time is not None:
            self.initial_time = initial_time
        if standby_time is not None:
            self.standby_time = standby_time
        if self.cycle_time is not None:
            self.cycle_time = cycle_time

    def generate_digital_triggers(self, lasers, camera):
        initial_samples = int(np.ceil(self.initial_time * self.sample_rate))
        standby_samples = int(np.ceil(self.standby_time * self.sample_rate))
        cam_ind = camera + 4
        offset = max(0, initial_samples - self.digital_starts[cam_ind])
        self.digital_starts = [(_start + offset) for _start in self.digital_starts]
        self.digital_ends = [(_end + offset) for _end in self.digital_ends]
        cycle_samples = self.digital_ends[cam_ind] + standby_samples
        digital_trigger = np.zeros((len(self.digital_starts), cycle_samples), dtype=int)
        digital_trigger[cam_ind, self.digital_starts[cam_ind]:self.digital_ends[cam_ind]] = 1
        for laser in lasers:
            digital_trigger[laser, self.digital_starts[laser]:self.digital_ends[laser]] = 1
        return digital_trigger

    def generate_linescan_resolft_2d(self, camera=0):
        interval_samples = 16
        digital_sequences = [np.empty((0,)) for _ in range(len(self.digital_starts))]
        galvo_sequences = [np.empty((0,)) for _ in range(2)]
        piezo_sequences = [np.empty((0,)) for _ in range(2)]
        _sth = np.linspace(0, self.duration, self.duration, endpoint=False)
        fast_axis_galvo = np.abs(self.galvo_stop - self.galvo_start) * (
                np.mod(_sth, self.samples_period) / self.samples_period) + self.galvo_start
        slow_axis_galvo = self.dot_offset + self.dot_start + self.dot_step_v * (_sth // self.samples_period)
        square_wave = np.pad(np.ones((self.samples_period - 2 * self.samples_delay)),
                             (self.samples_delay, self.samples_delay), 'constant', constant_values=(0, 0))
        laser_trigger = np.tile(square_wave, self.dot_pos.size)
        # ON
        galvo_sequences[0] = np.pad(fast_axis_galvo, (interval_samples, 0), 'constant',
                                    constant_values=(self.galvo_start, self.galvo_start))
        galvo_sequences[1] = np.pad(slow_axis_galvo, (interval_samples, 0), 'constant',
                                    constant_values=(self.dot_start, self.dot_start))
        digital_sequences[0] = np.pad(laser_trigger, (interval_samples, 0), 'constant', constant_values=(0, 0))
        for i in range(6):
            digital_sequences[i + 1] = np.zeros(digital_sequences[0].size)
        # OFF
        off_samples = self.digital_ends[1] - self.digital_starts[1]
        galvo_sequences[0] = np.concatenate(
            (galvo_sequences[0], self.galvo_start * np.ones(off_samples + interval_samples)))
        galvo_sequences[1] = np.concatenate(
            (galvo_sequences[1], self.dot_start * np.ones(off_samples + interval_samples)))
        digital_sequences[0] = np.concatenate((digital_sequences[0], np.zeros(off_samples + interval_samples)))
        digital_sequences[1] = np.concatenate(
            (digital_sequences[1], np.concatenate((np.zeros(interval_samples), np.ones(off_samples)))))
        digital_sequences[2] = digital_sequences[1]
        for i in range(4):
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
        temp = np.pad(np.ones(laser_trigger.size - 2 * self.samples_delay),
                      (interval_samples + self.samples_delay, self.samples_delay), 'constant', constant_values=(0, 0))
        for i in range(3):
            if i == camera:
                digital_sequences[i + 4] = np.concatenate((digital_sequences[i + 4], temp))
            else:
                digital_sequences[i + 4] = np.concatenate(
                    (digital_sequences[i + 4], np.zeros(laser_trigger.size + interval_samples)))
        # Piezo Fast Axis
        standby_samples = int(np.ceil(self.standby_time * self.sample_rate))
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
        for i in range(7):
            digital_sequences[i] = np.concatenate((digital_sequences[i], np.zeros(standby_samples)))
            digital_sequences[i] = np.tile(digital_sequences[i], self.piezo_scan_pos[0])
            digital_sequences[i] = np.tile(digital_sequences[i], self.piezo_scan_pos[1])
        scan_pos = 1
        for num in self.piezo_scan_pos:
            scan_pos *= num
        self.logg.info("\nGalvo start, and stop: {}\n"
                       "Dot start, step, range, and numbers: {}\n"
                       "Piezo starts: {}\n"
                       "Piezo steps: {}\n"
                       "Piezo ranges: {}\n"
                       "Piezo positions: {}".format([self.galvo_start, self.galvo_stop],
                                                    [self.dot_start, self.dot_step_v, self.dot_range,
                                                     self.dot_pos.size],
                                                    self.piezo_starts, self.piezo_steps, self.piezo_ranges, scan_pos))
        return np.asarray(galvo_sequences), np.asarray(piezo_sequences), np.asarray(digital_sequences), scan_pos

    def generate_dotscan_resolft_2d(self, camera=0):
        interval_samples = 16
        digital_sequences = [np.empty((0,)) for _ in range(len(self.digital_starts))]
        galvo_sequences = [np.empty((0,)) for _ in range(2)]
        piezo_sequences = [np.empty((0,)) for _ in range(2)]
        _sth = np.linspace(0, self.duration, self.duration, endpoint=False)
        fast_axis_galvo = np.abs(self.galvo_stop - self.galvo_start) * (
                np.mod(_sth, self.samples_period) / self.samples_period) + self.galvo_start
        slow_axis_galvo = self.dot_offset + self.dot_start + self.dot_step_v * (_sth // self.samples_period)
        _sqr = np.concatenate((np.ones(self.samples_high), np.zeros(self.samples_low)))
        square_wave = np.pad(np.tile(_sqr, self.dot_pos.size), (self.samples_delay, self.samples_offset),
                             'constant', constant_values=(0, 0))
        laser_trigger = np.tile(square_wave, self.dot_pos.size)
        # ON
        galvo_sequences[0] = np.pad(fast_axis_galvo, (interval_samples, 0), 'constant',
                                    constant_values=(self.galvo_start, self.galvo_start))
        galvo_sequences[1] = np.pad(slow_axis_galvo, (interval_samples, 0), 'constant',
                                    constant_values=(self.dot_start, self.dot_start))
        digital_sequences[0] = np.pad(laser_trigger, (interval_samples, 0), 'constant', constant_values=(0, 0))
        for i in range(6):
            digital_sequences[i + 1] = np.zeros(digital_sequences[0].size)
        # OFF
        off_samples = self.digital_ends[1] - self.digital_starts[1]
        galvo_sequences[0] = np.concatenate(
            (galvo_sequences[0], self.galvo_start * np.ones(off_samples + interval_samples)))
        galvo_sequences[1] = np.concatenate(
            (galvo_sequences[1], self.dot_start * np.ones(off_samples + interval_samples)))
        digital_sequences[0] = np.concatenate((digital_sequences[0], np.zeros(off_samples + interval_samples)))
        digital_sequences[1] = np.concatenate(
            (digital_sequences[1], np.concatenate((np.zeros(interval_samples), np.ones(off_samples)))))
        digital_sequences[2] = digital_sequences[1]
        for i in range(4):
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
        temp = np.pad(np.ones(laser_trigger.size - 2 * self.samples_delay),
                      (interval_samples + self.samples_delay, self.samples_delay), 'constant', constant_values=(0, 0))
        for i in range(3):
            if i == camera:
                digital_sequences[i + 4] = np.concatenate((digital_sequences[i + 4], temp))
            else:
                digital_sequences[i + 4] = np.concatenate(
                    (digital_sequences[i + 4], np.zeros(laser_trigger.size + interval_samples)))
        # Piezo Fast Axis
        standby_samples = int(np.ceil(self.standby_time * self.sample_rate))
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
        for i in range(7):
            digital_sequences[i] = np.concatenate((digital_sequences[i], np.zeros(standby_samples)))
            digital_sequences[i] = np.tile(digital_sequences[i], self.piezo_scan_pos[0])
            digital_sequences[i] = np.tile(digital_sequences[i], self.piezo_scan_pos[1])
        scan_pos = 1
        for num in self.piezo_scan_pos:
            scan_pos *= num
        self.logg.info("\nGalvo start, and stop: {} \n"
                       "Dot start, step, range, and numbers: {} \n"
                       "Piezo starts: {} \n"
                       "Piezo steps: {} \n"
                       "Piezo ranges: {} \n"
                       "Piezo positions: {}".format([self.galvo_start, self.galvo_stop],
                                                    [self.dot_start, self.dot_step_v, self.dot_range,
                                                     self.dot_pos.size],
                                                    self.piezo_starts, self.piezo_steps, self.piezo_ranges, scan_pos))
        return np.asarray(galvo_sequences), np.asarray(piezo_sequences), np.asarray(digital_sequences), scan_pos

    def generate_bead_scan_2d(self, camera=0):
        cam_ind = camera + 4
        digital_trigger_sequences = []
        analog_trigger_sequences = []
        initial_samples = int(np.ceil(self.initial_time * self.sample_rate))
        standby_samples = int(np.ceil(self.standby_time * self.sample_rate))
        return_samples = int(np.ceil(self.piezo_return_time * self.sample_rate))

        offset = max(0, initial_samples - self.digital_starts[cam_ind])
        self.digital_starts = [(_start + offset) for _start in self.digital_starts]
        self.digital_ends = [(_end + offset) for _end in self.digital_ends]
        cycle_samples = self.digital_ends[cam_ind] + standby_samples

        cycle = np.zeros(cycle_samples)
        digital_start = self.digital_ends[cam_ind]
        cycle[digital_start:] = np.linspace(0, 1, int(cycle_samples - digital_start))
        temp = cycle * self.piezo_steps[0]
        for j in range(self.piezo_scan_pos[0] - 2):
            j = j + 1
            temp = np.append(temp, cycle * self.piezo_steps[0] + j * self.piezo_steps[0])
        cycle = np.ones(digital_start) * self.piezo_steps[0] * (self.piezo_scan_pos[0] - 1)
        temp = np.append(temp, cycle)
        temp = np.append(temp,
                         np.linspace(1, 0,
                                     int(cycle_samples - digital_start) + return_samples) * self.piezo_steps[0] * (
                                 self.piezo_scan_pos[0] - 1))
        analog_trigger_sequences.append(np.tile(temp, self.piezo_scan_pos[1]) + self.piezo_starts[0])

        cycle = np.zeros((cycle_samples * self.piezo_scan_pos[0] + return_samples))
        cycle[cycle_samples * self.piezo_scan_pos[0]:] = 1
        temp = cycle * self.piezo_steps[1]
        for j in range(self.piezo_scan_pos[1] - 2):
            j = j + 1
            temp = np.append(temp, cycle * self.piezo_steps[1] + j * self.piezo_steps[1])
        cycle = np.ones((cycle_samples * self.piezo_scan_pos[0] + return_samples)) * self.piezo_steps[1] * (
                self.piezo_scan_pos[1] - 1)
        analog_trigger_sequences.append(np.append(temp, cycle) + self.piezo_starts[1])

        for i, start in enumerate(self.digital_starts):
            temp = np.zeros(cycle_samples)
            end = self.digital_ends[i]
            temp[start:end] = 1
            digital_trigger_sequences.append(np.tile(temp, self.piezo_scan_pos[0]))
            digital_trigger_sequences[i] = np.append(digital_trigger_sequences[i], np.zeros(return_samples))
            digital_trigger_sequences[i] = np.tile(digital_trigger_sequences[i], self.piezo_scan_pos[1])
        # digital_trigger_sequences[0].fill(0)
        # digital_trigger_sequences[1].fill(0)
        # digital_trigger_sequences[2].fill(0)
        # digital_trigger_sequences[cam_ind] = digital_trigger_sequences[3]

        return np.asarray(analog_trigger_sequences), np.asarray(digital_trigger_sequences), sum(self.piezo_scan_pos)


def safe_divide(numerator, denominator):
    try:
        return numerator / denominator
    except ZeroDivisionError:
        return 0
