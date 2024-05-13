import numpy as np


class TriggerSequence:
    class TriggerParameters:
        def __init__(self, sample_rate=250000):
            # daq
            self.sample_rate = sample_rate  # Hz
            # camera
            self.cycle_time = 0.05
            self.initial_time = 0.24
            self.standby_time = 0.05
            self.exposure_samples = 0.
            self.exposure_time = self.exposure_samples / self.sample_rate
            # piezo scanner
            self.piezo_conv_factors = [10., 10., 10.]
            self.piezo_steps = [0.03, 0.03, 0.128]
            self.piezo_ranges = [0.09, 0.09, 0.0]
            self.piezo_positions = [50., 50., 50.]
            self.piezo_return_time = 0.08
            self.piezo_steps = [step_size / conv_factor for step_size, conv_factor in
                                zip(self.piezo_steps, self.piezo_conv_factors)]
            self.piezo_ranges = [move_range / conv_factor for move_range, conv_factor in
                                 zip(self.piezo_ranges, self.piezo_conv_factors)]
            self.piezo_positions = [position / conv_factor for position, conv_factor in
                                    zip(self.piezo_positions, self.piezo_conv_factors)]
            self.piezo_starts = [i - j for i, j in zip(self.piezo_positions, [k / 2 for k in self.piezo_ranges])]
            self.piezo_scan_pos = [1 + int(np.ceil(safe_divide(scan_range, scan_step))) for scan_range, scan_step in
                                   zip(self.piezo_ranges, self.piezo_steps)]
            self.piezo_scan_positions = [np.arange(start, start + range_, step) for start, range_, step in
                                         zip(self.piezo_starts, self.piezo_ranges, self.piezo_steps)]
            # galvo scanner
            self.galvo_origins = [0.0, 0.0]  # V
            self.galvo_ranges = [0.5, 0.5]  # V
            self.galvo_starts = [o_ - r_ / 2 for (o_, r_) in zip(self.galvo_origins, self.galvo_ranges)]
            self.galvo_stops = [o_ + r_ / 2 for (o_, r_) in zip(self.galvo_origins, self.galvo_ranges)]
            self.galvo_return = int(8e-4 * self.sample_rate)  # ~800 us
            # dot array
            self.dot_ranges = [0.2, 0.2]  # V
            self.dot_starts = [o_ - r_ / 2 for (o_, r_) in zip(self.galvo_origins, self.dot_ranges)]
            self.dot_step_s = 80  # samples
            self.dot_step_v = 0.0173  # volts
            self.dot_step_y = 0.0173  # volts
            self.up_rate = self.dot_step_v / self.dot_step_s
            self.dot_pos = np.arange(self.dot_starts[0], self.dot_starts[0] + self.dot_ranges[0] + self.dot_step_v,
                                     self.dot_step_v)
            # sawtooth wave
            self.ramp_up = np.arange(self.galvo_starts[0], self.galvo_stops[0], self.up_rate)
            self.ramp_up_samples = self.ramp_up.size
            self.ramp_down_fraction = 0.016
            self.ramp_down_samples = int(np.ceil(self.ramp_up_samples * self.ramp_down_fraction))
            self.frequency = int(self.sample_rate / self.ramp_up_samples)  # Hz
            # square wave
            self.samples_high = 1
            self.samples_low = self.dot_step_s - self.samples_high
            self.samples_delay = int(np.abs(self.dot_starts[0] - self.galvo_starts[0]) / self.up_rate)
            self.samples_offset = self.ramp_up_samples - self.samples_delay - self.dot_step_s * self.dot_pos.size
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

    def update_nidaq_parameters(self, sample_rate=None):
        if sample_rate is not None:
            self._parameters = self.TriggerParameters(sample_rate)

    def update_piezo_scan_parameters(self, piezo_ranges=None, piezo_steps=None, piezo_positions=None,
                                     piezo_return_time=None):
        original_values = {
            "piezo_ranges": self.piezo_ranges,
            "piezo_steps": self.piezo_steps,
            "piezo_positions": self.piezo_positions,
            "piezo_return_time": self.piezo_return_time
        }
        try:
            if piezo_ranges is not None:
                self.piezo_ranges = piezo_ranges
            if piezo_steps is not None:
                self.piezo_steps = piezo_steps
            if piezo_positions is not None:
                self.piezo_positions = piezo_positions
            if piezo_return_time is not None:
                self.piezo_return_time = piezo_return_time
            self.piezo_steps = [step_size / conv_factor for step_size, conv_factor in
                                zip(self.piezo_steps, self.piezo_conv_factors)]
            self.piezo_ranges = [move_range / conv_factor for move_range, conv_factor in
                                 zip(self.piezo_ranges, self.piezo_conv_factors)]
            self.piezo_positions = [position / conv_factor for position, conv_factor in
                                    zip(self.piezo_positions, self.piezo_conv_factors)]
            self.piezo_starts = [i - j for i, j in zip(self.piezo_positions, [k / 2 for k in self.piezo_ranges])]
            self.piezo_scan_pos = [1 + int(np.ceil(safe_divide(scan_range, scan_step))) for scan_range, scan_step in
                                   zip(self.piezo_ranges, self.piezo_steps)]
            self.piezo_scan_positions = [np.arange(start, start + range_, step) for start, range_, step in
                                         zip(self.piezo_starts, self.piezo_ranges, self.piezo_steps)]
            if any(np.any(np.logical_or(pos < 0., pos > 10.)) for pos in self.piezo_scan_positions):
                self.logg.error("Invalid parameter combination.")
                raise ValueError("Invalid Piezo scanning parameters.")
        except ValueError:
            for attr, value in original_values.items():
                setattr(self, attr, value)
            self.logg.info("Piezo scanning parameters reverted to original values.")
            return

    def update_galvo_scan_parameters(self, origins=None, ranges=None, foci=None):
        original_values = {
            "frequency": self.frequency,
            "galvo_origins": self.galvo_origins,
            "galvo_ranges": self.galvo_ranges,
            "galvo_starts": self.galvo_starts,
            "galvo_stops": self.galvo_stops,
            "dot_ranges": self.dot_ranges,
            "dot_starts": self.dot_starts,
            "dot_step_v": self.dot_step_v,
            "dot_step_s": self.dot_step_s,
            "dot_step_y": self.dot_step_y,
            "dot_pos": self.dot_pos,
            "samples_low": self.samples_low,
            "samples_delay": self.samples_delay,
            "samples_offset": self.samples_offset
        }
        try:
            if origins is not None:
                self.galvo_origins = origins
            if ranges is not None:
                self.galvo_ranges, self.dot_ranges = ranges
            if foci is not None:
                [self.dot_step_s, self.dot_step_v, self.dot_step_y] = foci
            self.galvo_starts = [o_ - r_ / 2 for (o_, r_) in zip(self.galvo_origins, self.galvo_ranges)]
            self.galvo_stops = [o_ + r_ / 2 for (o_, r_) in zip(self.galvo_origins, self.galvo_ranges)]
            self.dot_starts = [o_ - r_ / 2 for (o_, r_) in zip(self.galvo_origins, self.dot_ranges)]
            self.dot_pos = np.arange(self.dot_starts[0], self.dot_starts[0] + self.dot_ranges[0] + self.dot_step_v,
                                     self.dot_step_v)
            self.up_rate = self.dot_step_v / self.dot_step_s
            self.samples_low = self.dot_step_s - self.samples_high
            self.ramp_up = np.arange(self.galvo_starts[0], self.galvo_stops[0], self.up_rate)
            self.ramp_up_samples = self.ramp_up.size
            self.ramp_down_samples = int(np.ceil(self.ramp_up_samples * self.ramp_down_fraction))
            self.frequency = int(self.sample_rate / self.ramp_up_samples)  # Hz
            self.samples_delay = int(np.abs(self.dot_starts[0] - self.galvo_starts[0]) / self.up_rate)
            self.samples_offset = self.ramp_up_samples - self.samples_delay - self.dot_step_s * self.dot_pos.size
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
        self.exposure_samples = self.digital_ends[cam_ind] - self.digital_starts[cam_ind]
        self.exposure_time = self.exposure_samples / self.sample_rate
        digital_trigger[cam_ind, self.digital_starts[cam_ind]:self.digital_ends[cam_ind]] = 1
        for laser in lasers:
            digital_trigger[laser, self.digital_starts[laser]:self.digital_ends[laser]] = 1
        return digital_trigger

    def generate_dotscan_resolft_2d(self, camera=0):
        interval_samples = self.galvo_return
        initial_samples = int(np.ceil(self.initial_time * self.sample_rate))
        digital_sequences = [np.empty((0,)) for _ in range(len(self.digital_starts))]
        galvo_sequences = [np.empty((0,)) for _ in range(2)]
        piezo_sequences = [np.empty((0,)) for _ in range(2)]
        ramp_down = np.linspace(self.ramp_up[-1], self.ramp_up[0], num=self.ramp_down_samples, endpoint=True)
        extended_cycle = np.concatenate((self.ramp_up, ramp_down))
        fast_axis_galvo = np.tile(extended_cycle, self.dot_pos.size)
        slow_axis_galvo = np.zeros_like(fast_axis_galvo)
        indices = np.arange(self.ramp_up_samples + 1, len(fast_axis_galvo), extended_cycle.size)
        slow_axis_galvo[indices] = 1
        slow_axis_galvo = np.cumsum(slow_axis_galvo) * self.dot_step_y + self.dot_starts[1]
        slow_axis_galvo[-self.ramp_down_samples:] = np.linspace(slow_axis_galvo[-self.ramp_down_samples],
                                                                self.dot_starts[1], self.ramp_down_samples)
        _sqr = np.pad(np.ones(self.samples_high), (0, self.samples_low), 'constant', constant_values=(0, 0))
        square_wave = np.pad(np.tile(_sqr, self.dot_pos.size),
                             (self.samples_delay, self.samples_offset + self.ramp_down_samples), 'constant',
                             constant_values=(0, 0))
        laser_trigger = np.tile(square_wave, self.dot_pos.size)
        # ON
        galvo_sequences[0] = np.pad(fast_axis_galvo, (interval_samples, 0), 'constant',
                                    constant_values=(self.galvo_starts[0], self.galvo_starts[0]))
        galvo_sequences[1] = np.pad(slow_axis_galvo, (interval_samples, 0), 'constant',
                                    constant_values=(self.dot_starts[1], self.dot_starts[1]))
        digital_sequences[0] = np.pad(laser_trigger, (interval_samples, 0), 'constant', constant_values=(0, 0))
        for i in range(6):
            digital_sequences[i + 1] = np.zeros(digital_sequences[0].size)
        # OFF
        off_samples = self.digital_ends[1] - self.digital_starts[1]
        galvo_sequences[0] = np.concatenate(
            (galvo_sequences[0], self.galvo_starts[0] * np.ones(off_samples + initial_samples + interval_samples)))
        galvo_sequences[1] = np.concatenate(
            (galvo_sequences[1], self.dot_starts[1] * np.ones(off_samples + initial_samples + interval_samples)))
        digital_sequences[0] = np.concatenate(
            (digital_sequences[0], np.zeros(off_samples + initial_samples + interval_samples)))
        digital_sequences[1] = np.concatenate(
            (digital_sequences[1], np.concatenate((np.zeros(interval_samples), np.ones(off_samples)))))
        digital_sequences[1] = np.concatenate(
            (digital_sequences[1], np.zeros(initial_samples)))
        for i in range(5):
            digital_sequences[i + 2] = np.concatenate(
                (digital_sequences[i + 2], np.zeros(off_samples + initial_samples + interval_samples)))
        # Read
        galvo_sequences[0] = np.concatenate(
            (galvo_sequences[0], np.concatenate((self.galvo_starts[0] * np.ones(interval_samples), fast_axis_galvo))))
        galvo_sequences[1] = np.concatenate(
            (galvo_sequences[1], np.concatenate((self.dot_starts[1] * np.ones(interval_samples), slow_axis_galvo))))
        for i in range(3):
            digital_sequences[i] = np.concatenate(
                (digital_sequences[i], np.zeros(laser_trigger.size + interval_samples)))
        digital_sequences[3] = np.concatenate(
            (digital_sequences[3], np.concatenate((np.zeros(interval_samples), laser_trigger))))
        self.exposure_samples = laser_trigger.size - 2 * self.samples_delay + self.dot_step_s
        self.exposure_time = self.exposure_samples / self.sample_rate
        temp = np.pad(np.ones(self.exposure_samples),
                      (interval_samples + self.samples_delay, self.samples_delay - self.dot_step_s), 'constant',
                      constant_values=(0, 0))
        for i in range(3):
            if i == camera:
                digital_sequences[i + 4] = np.concatenate((digital_sequences[i + 4], temp))
            else:
                digital_sequences[i + 4] = np.concatenate(
                    (digital_sequences[i + 4], np.zeros(laser_trigger.size + interval_samples)))
        # Piezo Fast Axis
        standby_samples = int(np.ceil(self.standby_time * self.sample_rate))
        return_samples = int(np.ceil(self.piezo_return_time * self.sample_rate))
        if standby_samples > return_samples:
            cycle_samples = digital_sequences[0].size + standby_samples
            _temp = np.zeros(cycle_samples * self.piezo_scan_pos[0])
            _temp[-standby_samples:-standby_samples + return_samples] = self.piezo_steps[1] * smooth_ramp(0, 1,
                                                                                                          return_samples)
            _temp[-standby_samples + return_samples:] = self.piezo_steps[1]
            piezo_sequences[1] = _temp + self.piezo_starts[1]
            for i in range(self.piezo_scan_pos[1] - 1):
                temp = _temp + self.piezo_starts[1] + (i + 1) * self.piezo_steps[1]
                piezo_sequences[1] = np.concatenate((piezo_sequences[1], temp))
            piezo_sequences[1][-standby_samples:-standby_samples + return_samples] = smooth_ramp(
                piezo_sequences[1][-standby_samples], self.piezo_positions[1], return_samples)
            piezo_sequences[1][-standby_samples + return_samples:] = self.piezo_positions[1]
            piezo_sequences[0] = np.zeros(digital_sequences[0].size) + self.piezo_starts[0]
            _temp = np.concatenate((smooth_ramp(0, 1, standby_samples), np.ones(digital_sequences[0].size)))
            for i in range(self.piezo_scan_pos[0] - 1):
                temp = _temp * self.piezo_steps[0] + self.piezo_starts[0] + i * self.piezo_steps[0]
                piezo_sequences[0] = np.concatenate((piezo_sequences[0], temp))
            piezo_sequences[0] = np.concatenate((piezo_sequences[0], smooth_ramp(
                piezo_sequences[0][-1], piezo_sequences[0][0], return_samples)))
            piezo_sequences[0] = np.concatenate(
                (piezo_sequences[0], piezo_sequences[0][-1] * np.ones(standby_samples - return_samples)))
            piezo_sequences[0] = np.tile(piezo_sequences[0], self.piezo_scan_pos[0])
            piezo_sequences[0][-standby_samples:-standby_samples + return_samples] = smooth_ramp(
                piezo_sequences[0][-standby_samples], self.piezo_positions[0], return_samples)
            piezo_sequences[0][-standby_samples + return_samples:] = self.piezo_positions[0]
            galvo_sequences[0] = np.concatenate((galvo_sequences[0], self.galvo_starts[0] * np.ones(standby_samples)))
            galvo_sequences[1] = np.concatenate((galvo_sequences[1], self.dot_starts[1] * np.ones(standby_samples)))
            for i in range(7):
                digital_sequences[i] = np.concatenate((digital_sequences[i], np.zeros(standby_samples)))
                digital_sequences[i] = np.tile(digital_sequences[i], self.piezo_scan_pos[0])
                digital_sequences[i] = np.tile(digital_sequences[i], self.piezo_scan_pos[1])
        else:
            _temp = np.tile(np.zeros(digital_sequences[0].size + return_samples), self.piezo_scan_pos[0])
            _temp[-return_samples:] = self.piezo_steps[1] * smooth_ramp(0, 1, return_samples)
            piezo_sequences[1] = _temp + self.piezo_starts[1]
            for i in range(self.piezo_scan_pos[1] - 1):
                temp = _temp + self.piezo_starts[1] + (i + 1) * self.piezo_steps[1]
                piezo_sequences[1] = np.concatenate((piezo_sequences[1], temp))
            piezo_sequences[1][-return_samples:] = smooth_ramp(piezo_sequences[1][-return_samples],
                                                               self.piezo_positions[1], return_samples)
            piezo_sequences[0] = np.zeros(digital_sequences[0].size) + self.piezo_starts[0]
            _temp = np.concatenate((smooth_ramp(0, 1, return_samples), np.ones(digital_sequences[0].size)))
            for i in range(self.piezo_scan_pos[0] - 1):
                temp = _temp * self.piezo_steps[0] + self.piezo_starts[0] + i * self.piezo_steps[0]
                piezo_sequences[0] = np.concatenate((piezo_sequences[0], temp))
            piezo_sequences[0] = np.concatenate(
                (piezo_sequences[0], smooth_ramp(piezo_sequences[0][-1], piezo_sequences[0][0], return_samples)))
            piezo_sequences[0] = np.tile(piezo_sequences[0], self.piezo_scan_pos[1])
            piezo_sequences[0][-return_samples:] = smooth_ramp(piezo_sequences[0][-return_samples],
                                                               self.piezo_positions[0], return_samples)
            galvo_sequences[0] = np.concatenate((galvo_sequences[0], self.galvo_starts[0] * np.ones(return_samples)))
            galvo_sequences[1] = np.concatenate((galvo_sequences[1], self.dot_starts[1] * np.ones(return_samples)))
            for i in range(7):
                digital_sequences[i] = np.concatenate((digital_sequences[i], np.zeros(return_samples)))
                digital_sequences[i] = np.tile(digital_sequences[i], self.piezo_scan_pos[0])
                digital_sequences[i] = np.tile(digital_sequences[i], self.piezo_scan_pos[1])
        for i in range(2):
            temp = smooth_ramp(self.piezo_positions[i], piezo_sequences[i][0], return_samples)
            piezo_sequences[i] = np.concatenate((temp, piezo_sequences[i]))
        for i in range(2):
            galvo_sequences[i] = np.tile(galvo_sequences[i], self.piezo_scan_pos[0])
            galvo_sequences[i] = np.tile(galvo_sequences[i], self.piezo_scan_pos[1])
            galvo_sequences[i][-interval_samples:] = smooth_ramp(galvo_sequences[i][-1], 0., interval_samples)
        for i in range(2):
            galvo_sequences[i] = np.concatenate((np.ones(return_samples) * galvo_sequences[i][0], galvo_sequences[i]))
        for i in range(7):
            digital_sequences[i] = np.concatenate((np.zeros(return_samples), digital_sequences[i]))
        scan_pos = 1
        for num in self.piezo_scan_pos:
            scan_pos *= num
        self.logg.info("\nGalvo start, and stop: {} \n"
                       "Dot start, step, range, and numbers: {} \n"
                       "Piezo starts: {} \n"
                       "Piezo steps: {} \n"
                       "Piezo ranges: {} \n"
                       "Piezo positions: {}".format([self.galvo_starts, self.galvo_stops],
                                                    [self.dot_starts, [self.dot_step_v, self.dot_step_y],
                                                     self.dot_ranges, self.dot_pos.size],
                                                    self.piezo_starts, self.piezo_steps, self.piezo_ranges, scan_pos))
        return np.asarray(galvo_sequences), np.asarray(piezo_sequences), np.asarray(digital_sequences), scan_pos

    def generate_point_scan_2d(self, camera=0):
        interval_samples = self.galvo_return
        initial_samples = int(np.ceil(self.initial_time * self.sample_rate))
        digital_sequences = [np.empty((0,)) for _ in range(len(self.digital_starts))]
        galvo_sequences = [np.empty((0,)) for _ in range(2)]
        ramp_down_samples = int(self.samples_period * self.ramp_down_fraction)
        ramp_up = np.linspace(self.galvo_starts[0], self.galvo_stops[0], num=self.samples_period, endpoint=False)
        ramp_down = np.linspace(self.galvo_stops[0], self.galvo_starts[0], num=ramp_down_samples, endpoint=True)
        extended_cycle = np.concatenate((ramp_up, ramp_down))
        fast_axis_galvo = np.tile(extended_cycle, self.dot_pos.size)
        cycle_length = self.samples_period + ramp_down_samples
        slow_axis_galvo = np.zeros_like(fast_axis_galvo)
        indices = np.arange(self.samples_period + 1, len(fast_axis_galvo), cycle_length)
        slow_axis_galvo[indices] = 1
        slow_axis_galvo = np.cumsum(slow_axis_galvo) * self.dot_step_v + self.dot_starts[1]
        slow_axis_galvo[-ramp_down_samples:] = np.linspace(slow_axis_galvo[-ramp_down_samples], self.dot_starts[1],
                                                           ramp_down_samples)
        _sqr = np.pad(np.ones(self.samples_high), (0, self.samples_low), 'constant', constant_values=(0, 0))
        square_wave = np.pad(np.tile(_sqr, self.dot_pos.size),
                             (self.samples_delay, self.samples_offset + ramp_down_samples), 'constant',
                             constant_values=(0, 0))
        laser_trigger = np.tile(square_wave, self.dot_pos.size)
        # ON
        galvo_sequences[0] = np.pad(fast_axis_galvo, (interval_samples, 0), 'constant',
                                    constant_values=(self.galvo_starts[0], self.galvo_starts[0]))
        galvo_sequences[1] = np.pad(slow_axis_galvo, (interval_samples, 0), 'constant',
                                    constant_values=(self.dot_starts[1], self.dot_starts[1]))
        digital_sequences[0] = np.pad(laser_trigger, (interval_samples, 0), 'constant', constant_values=(0, 0))
        for i in range(6):
            digital_sequences[i + 1] = np.zeros(digital_sequences[0].size)
        # OFF
        off_samples = self.digital_ends[1] - self.digital_starts[1]
        galvo_sequences[0] = np.concatenate(
            (galvo_sequences[0], self.galvo_starts[0] * np.ones(off_samples + initial_samples + interval_samples)))
        galvo_sequences[1] = np.concatenate(
            (galvo_sequences[1], self.dot_starts[1] * np.ones(off_samples + initial_samples + interval_samples)))
        digital_sequences[0] = np.concatenate(
            (digital_sequences[0], np.zeros(off_samples + initial_samples + interval_samples)))
        digital_sequences[1] = np.concatenate(
            (digital_sequences[1], np.concatenate((np.zeros(interval_samples), np.ones(off_samples)))))
        digital_sequences[1] = np.concatenate(
            (digital_sequences[1], np.zeros(initial_samples)))
        for i in range(5):
            digital_sequences[i + 2] = np.concatenate(
                (digital_sequences[i + 2], np.zeros(off_samples + initial_samples + interval_samples)))
        # Read
        galvo_sequences[0] = np.concatenate(
            (galvo_sequences[0], np.concatenate((self.galvo_starts[0] * np.ones(interval_samples), fast_axis_galvo))))
        galvo_sequences[1] = np.concatenate(
            (galvo_sequences[1], np.concatenate((self.dot_starts[1] * np.ones(interval_samples), slow_axis_galvo))))
        for i in range(3):
            digital_sequences[i] = np.concatenate(
                (digital_sequences[i], np.zeros(laser_trigger.size + interval_samples)))
        digital_sequences[3] = np.concatenate(
            (digital_sequences[3], np.concatenate((np.zeros(interval_samples), laser_trigger))))
        temp = np.pad(np.ones(laser_trigger.size - 2 * self.samples_delay + self.dot_step_s),
                      (interval_samples + self.samples_delay, self.samples_delay - self.dot_step_s), 'constant',
                      constant_values=(0, 0))
        for i in range(3):
            if i == camera:
                digital_sequences[i + 4] = np.concatenate((digital_sequences[i + 4], temp))
            else:
                digital_sequences[i + 4] = np.concatenate(
                    (digital_sequences[i + 4], np.zeros(laser_trigger.size + interval_samples)))
        for i in range(2):
            galvo_sequences[i][-interval_samples:] = smooth_ramp(galvo_sequences[i][-1], 0., interval_samples)
        for i in range(2):
            galvo_sequences[i] = np.concatenate((np.ones(interval_samples) * galvo_sequences[i][0], galvo_sequences[i]))
        for i in range(7):
            digital_sequences[i] = np.concatenate((np.zeros(interval_samples), digital_sequences[i]))
        self.logg.info("\nGalvo start, and stop: {} \n"
                       "Dot start, step, range, and numbers: {} \n".format([self.galvo_starts, self.galvo_stops],
                                                                           [self.dot_starts, self.dot_step_v,
                                                                            self.dot_ranges, self.dot_pos.size]))
        return np.asarray(galvo_sequences), np.asarray(digital_sequences)

    def generate_monalisa_scan_2d(self, camera=0):
        cam_ind = camera + 4
        digital_sequences = [np.empty((0,)) for _ in range(len(self.digital_starts))]
        piezo_sequences = [np.empty((0,)) for _ in range(2)]
        initial_samples = int(np.ceil(self.initial_time * self.sample_rate))
        offset = max(0, initial_samples - self.digital_starts[cam_ind])
        self.digital_starts = [(_start + offset) for _start in self.digital_starts]
        self.digital_ends = [(_end + offset) for _end in self.digital_ends]
        standby_samples = int(np.ceil(self.standby_time * self.sample_rate))
        return_samples = int(np.ceil(self.piezo_return_time * self.sample_rate))
        if standby_samples > return_samples:
            cycle_samples = self.digital_ends[cam_ind] + standby_samples
            _temp = np.zeros((cycle_samples * self.piezo_scan_pos[0]))
            _temp[-standby_samples:-standby_samples + return_samples] = self.piezo_steps[1] * smooth_ramp(0, 1,
                                                                                                          return_samples)
            _temp[-standby_samples + return_samples:] = self.piezo_steps[1]
            piezo_sequences[1] = _temp + self.piezo_starts[1]
            for i in range(self.piezo_scan_pos[1] - 1):
                temp = _temp + self.piezo_starts[1] + (i + 1) * self.piezo_steps[1]
                piezo_sequences[1] = np.concatenate((piezo_sequences[1], temp))
            piezo_sequences[1][-standby_samples:-standby_samples + return_samples] = smooth_ramp(
                piezo_sequences[1][-standby_samples], self.piezo_positions[1], return_samples)
            piezo_sequences[1][-standby_samples + return_samples:] = self.piezo_positions[1]
            piezo_sequences[0] = np.concatenate(
                (np.zeros(self.digital_ends[cam_ind]), smooth_ramp(0, self.piezo_steps[0], standby_samples))) + \
                                 self.piezo_starts[0]
            _temp = self.piezo_starts[0] + self.piezo_steps[0] * np.concatenate(
                (np.zeros(self.digital_ends[cam_ind]), smooth_ramp(0, 1, standby_samples)))
            for i in range(self.piezo_scan_pos[0] - 1):
                temp = _temp + (i + 1) * self.piezo_steps[0]
                piezo_sequences[0] = np.concatenate((piezo_sequences[0], temp))
            piezo_sequences[0][-standby_samples:-standby_samples + return_samples] = smooth_ramp(
                piezo_sequences[0][-standby_samples], piezo_sequences[0][0], return_samples)
            piezo_sequences[0][-standby_samples + return_samples:] = piezo_sequences[0][0]
            piezo_sequences[0] = np.tile(piezo_sequences[0], self.piezo_scan_pos[0])
            piezo_sequences[0][-standby_samples:-standby_samples + return_samples] = smooth_ramp(
                piezo_sequences[0][-standby_samples], self.piezo_positions[0], return_samples)
            piezo_sequences[0][-standby_samples + return_samples:] = self.piezo_positions[0]
        else:
            cycle_samples = self.digital_ends[cam_ind] + standby_samples
            _temp = np.zeros((cycle_samples * self.piezo_scan_pos[0] + return_samples - standby_samples))
            _temp[-return_samples:] = self.piezo_steps[1] * smooth_ramp(0, 1, return_samples)
            piezo_sequences[1] = _temp + self.piezo_starts[1]
            for i in range(self.piezo_scan_pos[1] - 1):
                temp = _temp + self.piezo_starts[1] + (i + 1) * self.piezo_steps[1]
                piezo_sequences[1] = np.concatenate((piezo_sequences[1], temp))
            piezo_sequences[1][-return_samples:] = smooth_ramp(piezo_sequences[1][-return_samples],
                                                               self.piezo_positions[1], return_samples)
            piezo_sequences[0] = np.concatenate(
                (np.zeros(self.digital_ends[cam_ind]), smooth_ramp(0, self.piezo_steps[0], standby_samples))) + \
                                 self.piezo_starts[0]
            _temp = self.piezo_starts[0] + self.piezo_steps[0] * np.concatenate(
                (np.zeros(self.digital_ends[cam_ind]), smooth_ramp(0, 1, standby_samples)))
            for i in range(self.piezo_scan_pos[0] - 1):
                temp = _temp + (i + 1) * self.piezo_steps[0]
                piezo_sequences[0] = np.concatenate((piezo_sequences[0], temp))
            piezo_sequences[0] = np.concatenate(
                (piezo_sequences[0], piezo_sequences[0][-1] * np.ones(return_samples - standby_samples)))
            piezo_sequences[0][-return_samples:] = smooth_ramp(piezo_sequences[0][-return_samples],
                                                               piezo_sequences[0][0], return_samples)
            piezo_sequences[0] = np.tile(piezo_sequences[0], self.piezo_scan_pos[0])
            piezo_sequences[0][-return_samples:] = smooth_ramp(piezo_sequences[0][-return_samples],
                                                               self.piezo_positions[0], return_samples)
        for i in range(2):
            temp = smooth_ramp(self.piezo_positions[i], piezo_sequences[i][0], return_samples)
            piezo_sequences[i] = np.concatenate((temp, piezo_sequences[i]))
        scan_pos = 1
        for num in self.piezo_scan_pos:
            scan_pos *= num
        for i in range(4):
            temp = np.zeros(cycle_samples)
            temp[self.digital_starts[i]:self.digital_ends[i]] = 1
            temp = np.tile(temp, self.piezo_scan_pos[0])
            if standby_samples < return_samples:
                temp = np.concatenate((temp, np.zeros(return_samples - standby_samples)))
            temp = np.tile(temp, self.piezo_scan_pos[1])
            temp = np.concatenate((np.zeros(return_samples), temp))
            digital_sequences[i] = temp
        for i in range(3):
            if i == camera:
                self.exposure_samples = self.digital_ends[i + 4] - self.digital_starts[i + 4]
                self.exposure_time = self.exposure_samples / self.sample_rate
                temp = np.zeros(cycle_samples)
                temp[self.digital_starts[i + 4]:self.digital_ends[i + 4]] = 1
                temp = np.tile(temp, self.piezo_scan_pos[0])
                if standby_samples < return_samples:
                    temp = np.concatenate((temp, np.zeros(return_samples - standby_samples)))
                temp = np.tile(temp, self.piezo_scan_pos[1])
                temp = np.concatenate((np.zeros(return_samples), temp))
                digital_sequences[i + 4] = temp
            else:
                digital_sequences[i + 4] = np.zeros(digital_sequences[0].shape[0])
        return np.asarray(piezo_sequences), np.asarray(digital_sequences), scan_pos


def smooth_ramp(start, end, samples, curve_half=0.02):
    n = int(curve_half * samples)
    x = np.linspace(0, np.pi / 2, n, endpoint=True)
    signal_first_half = np.sin(x) * (end - start) / np.sin(np.pi / 2) + start
    signal_second_half = np.full(samples - n, end)
    return np.concatenate((signal_first_half, signal_second_half))


def safe_divide(numerator, denominator):
    try:
        return numerator / denominator
    except ZeroDivisionError:
        return 0
