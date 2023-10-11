import warnings

import nidaqmx
from nidaqmx.constants import Edge, AcquisitionType, LineGrouping, FrequencyUnits, Level, WAIT_INFINITELY
from nidaqmx.error_codes import DAQmxWarnings
from nidaqmx.system import System

warnings.filterwarnings("error", category=nidaqmx.DaqWarning)


class NIDAQ:
    class NIDAQSettings:
        def __init__(self):
            self.sample_rate = 100000
            self.duty_cycle = 0.5
            self.piezo_channels = "Dev1/ao0:1"
            self.galvo_channels = "Dev1/ao2:3"
            self.digital_channels = "Dev1/port0/line0:5"
            self.clock_channel = "Dev1/ctr0"
            self.clock = None
            self.mode = None

    def __init__(self, logg=None):
        self.logg = logg or self.setup_logging()
        self.device = self._initialize()
        self._settings = self.NIDAQSettings()
        self.tasks = {}
        self._active = {}
        self._running = {}
        self.tasks, self._active, self._running, = self._configure()

    def __del__(self):
        pass

    def __getattr__(self, item):
        if hasattr(self._settings, item):
            return getattr(self._settings, item)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{item}'")

    def close(self):
        self.device.reset_device()

    @staticmethod
    def setup_logging():
        import logging
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
        return logging

    def _initialize(self):
        try:
            local_system = System.local()
            driver_version = local_system.driver_version
            print("DAQmx {0}.{1}.{2}".format(driver_version.major_version, driver_version.minor_version,
                                             driver_version.update_version))
            return local_system.devices[0]
        except Exception as e:
            self.logg.error(f"Error initializing NIDAQ: {e}")

    def _configure(self):
        try:
            tasks = {"piezo": None, "galvo": None, "piezo_pos": None, "digital": None, "clock": None}
            _active = {key: False for key in self.tasks.keys()}
            _running = {key: False for key in self.tasks.keys()}
            return tasks, _active, _running
        except nidaqmx.DaqWarning as e:
            self.logg.warning("DaqWarning caught as exception: %s", e)
            try:
                assert e.error_code == DAQmxWarnings.STOPPED_BEFORE_DONE, "Unexpected error code: {}".format(
                    e.error_code)
            except AssertionError as ae:
                self.logg.error("Assertion Error: %s", ae)

    def set_piezo_position(self, pos_x, pos_y):
        try:
            self.tasks["piezo"] = nidaqmx.Task("piezo")
            self.tasks["piezo"].ao_channels.add_ao_voltage_chan(self.piezo_channels, min_val=0., max_val=10.)
            self.tasks["piezo"].timing.cfg_samp_clk_timing(self.sample_rate, source="100kHzTimebase",
                                                           active_edge=Edge.RISING,
                                                           sample_mode=AcquisitionType.FINITE,
                                                           samps_per_chan=1)
            self.tasks["piezo"].write([pos_x, pos_y], auto_start=True)
            self.tasks["piezo"].wait_until_done(WAIT_INFINITELY)
            self.tasks["piezo"].stop()
            self.tasks["piezo"].close()
        except nidaqmx.DaqWarning as e:
            self.logg.warning("DaqWarning caught as exception: %s", e)
            try:
                assert e.error_code == DAQmxWarnings.STOPPED_BEFORE_DONE, "Unexpected error code: {}".format(
                    e.error_code)
            except AssertionError as ae:
                self.logg.error("Assertion Error: %s", ae)

    def get_piezo_position(self):
        if not self.tasks["piezo_pos"].is_task_done():
            self.tasks["piezo_pos"].stop()
        try:
            pos = self.tasks["piezo_pos"].read(number_of_samples_per_channel=1)
            return pos
        except nidaqmx.DaqWarning as e:
            self.logg.warning("DaqWarning caught as exception: %s", e)
            try:
                assert e.error_code == DAQmxWarnings.STOPPED_BEFORE_DONE, "Unexpected error code: {}".format(
                    e.error_code)
            except AssertionError as ae:
                self.logg.error("Assertion Error: %s", ae)

    def set_galvo_position(self, pos_x, pos_y):
        try:
            self.tasks["galvo"] = nidaqmx.Task("galvo")
            self.tasks["galvo"].ao_channels.add_ao_voltage_chan(self.galvo_channels, min_val=-10., max_val=10.)
            self.tasks["galvo"].timing.cfg_samp_clk_timing(self.sample_rate, source="100kHzTimebase",
                                                           active_edge=Edge.RISING,
                                                           sample_mode=AcquisitionType.FINITE,
                                                           samps_per_chan=1)
            self.tasks["galvo"].write([[pos_x], [pos_y]], auto_start=True)
            self.tasks["galvo"].wait_until_done(WAIT_INFINITELY)
            self.tasks["galvo"].stop()
            self.tasks["galvo"].close()
        except nidaqmx.DaqWarning as e:
            self.logg.warning("DaqWarning caught as exception: %s", e)
            try:
                assert e.error_code == DAQmxWarnings.STOPPED_BEFORE_DONE, "Unexpected error code: {}".format(
                    e.error_code)
            except AssertionError as ae:
                self.logg.error("Assertion Error: %s", ae)

    def write_digital_sequences(self, digital_sequences, clock_source="100kHzTimebase", mode="finite"):
        self.clock = clock_source
        if mode == "continuous":
            self.mode = AcquisitionType.CONTINUOUS
        else:
            self.mode = AcquisitionType.FINITE
        try:
            self.tasks["digital"] = nidaqmx.Task("digital")
            self.tasks["digital"].do_channels.add_do_chan(self.digital_channels,
                                                          line_grouping=LineGrouping.CHAN_PER_LINE)
            _channels, _samples = digital_sequences.shape
            self.tasks["digital"].timing.cfg_samp_clk_timing(self.sample_rate, source=self.clock,
                                                             active_edge=Edge.RISING,
                                                             sample_mode=self.mode,
                                                             samps_per_chan=_samples)
            self.tasks["digital"].write(digital_sequences == 1.0, auto_start=False)
            self._active["digital"] = True
            self.logg.info("Successfully Write Digital Channels")
            if self.clock == "Ctr0InternalOutput":
                self.tasks["clock"] = nidaqmx.Task("clock")
                self.tasks["clock"].co_channels.add_co_pulse_chan_freq(self.clock_channel, units=FrequencyUnits.HZ,
                                                                       idle_state=Level.LOW, initial_delay=0.0,
                                                                       freq=self.sample_rate,
                                                                       duty_cycle=self.duty_cycle)
                self.tasks["clock"].timing.cfg_implicit_timing(sample_mode=AcquisitionType.CONTINUOUS)
                self._active["clock"] = True
                self.logg.info("Successfully Write Clock Channel")
        except nidaqmx.DaqWarning as e:
            self.logg.warning("DaqWarning caught as exception: %s", e)
            try:
                assert e.error_code == DAQmxWarnings.STOPPED_BEFORE_DONE, "Unexpected error code: {}".format(
                    e.error_code)
            except AssertionError as ae:
                self.logg.error("Assertion Error: %s", ae)

    def run_digital_trigger(self):
        try:
            if self.clock == "100kHzTimebase":
                self._running["digital"] = True
                self.tasks["digital"].start()
            if self.clock == "Ctr0InternalOutput":
                self._running["digital"] = True
                self._running["clock"] = True
                self.tasks["digital"].start()
                self.tasks["clock"].start()
            if self.mode == AcquisitionType.FINITE:
                self.tasks["digital"].wait_until_done(WAIT_INFINITELY)
        except nidaqmx.DaqWarning as e:
            self.logg.warning("DaqWarning caught as exception: %s", e)
            try:
                assert e.error_code == DAQmxWarnings.STOPPED_BEFORE_DONE, "Unexpected error code: {}".format(
                    e.error_code)
            except AssertionError as ae:
                self.logg.error("Assertion Error: %s", ae)

    def write_piezo_scan(self, piezo_sequence):
        try:
            self.tasks["piezo"] = nidaqmx.Task("piezo")
            self.tasks["piezo"].ao_channels.add_ao_voltage_chan(self.piezo_channels, min_val=0., max_val=10.)
            _channels, _samples = piezo_sequence.shape
            self.tasks["piezo"].timing.cfg_samp_clk_timing(self.sample_rate, source="Ctr0InternalOutput",
                                                           active_edge=Edge.RISING,
                                                           sample_mode=AcquisitionType.FINITE,
                                                           samps_per_chan=_samples)
            self.tasks["piezo"].write(piezo_sequence, auto_start=False)
            self._active["piezo"] = True
            self.logg.info("Successfully Write Piezo Scanning Channels")
        except nidaqmx.DaqWarning as e:
            self.logg.warning("DaqWarning caught as exception: %s", e)
            try:
                assert e.error_code == DAQmxWarnings.STOPPED_BEFORE_DONE, "Unexpected error code: {}".format(
                    e.error_code)
            except AssertionError as ae:
                self.logg.error("Assertion Error: %s", ae)

    def write_galvo_scan(self, galvo_sequence):
        try:
            self.tasks["galvo"] = nidaqmx.Task("galvo")
            self.tasks["galvo"].ao_channels.add_ao_voltage_chan(self.galvo_channels, min_val=-10., max_val=10.)
            _channels, _samples = galvo_sequence.shape
            self.tasks["galvo"].timing.cfg_samp_clk_timing(self.sample_rate, source="Ctr0InternalOutput",
                                                           active_edge=Edge.RISING,
                                                           sample_mode=AcquisitionType.FINITE,
                                                           samps_per_chan=_samples)
            self.tasks["galvo"].write(galvo_sequence, auto_start=False)
            self._active["galvo"] = True
            self.logg.info("Successfully Write Galvo Scanning Channels")
        except nidaqmx.DaqWarning as e:
            self.logg.warning("DaqWarning caught as exception: %s", e)
            try:
                assert e.error_code == DAQmxWarnings.STOPPED_BEFORE_DONE, "Unexpected error code: {}".format(
                    e.error_code)
            except AssertionError as ae:
                self.logg.error("Assertion Error: %s", ae)

    def write_triggers(self, piezo_sequence=None, galvo_sequence=None, digital_sequences=None):
        try:
            if piezo_sequence is not None:
                self.write_piezo_scan(piezo_sequence)
                self.tasks["piezo"].start()
                self._running["piezo"] = True
            if galvo_sequence is not None:
                self.write_galvo_scan(galvo_sequence)
                self.tasks["galvo"].start()
                self._running["galvo"] = True
            if digital_sequences is not None:
                self.write_digital_sequences(digital_sequences, "Ctr0InternalOutput", "finite")
                self.tasks["digital"].start()
                self._running["digital"] = True
                self._running["clock"] = True
            self.logg.info("Successfully Write All Trigger Channels")
        except nidaqmx.DaqWarning as e:
            self.logg.warning("DaqWarning caught as exception: %s", e)
            try:
                assert e.error_code == DAQmxWarnings.STOPPED_BEFORE_DONE, "Unexpected error code: {}".format(
                    e.error_code)
            except AssertionError as ae:
                self.logg.error("Assertion Error: %s", ae)

    def run_triggers(self):
        try:
            self.tasks["clock"].start()
            self.tasks["digital"].wait_until_done(WAIT_INFINITELY)
        except nidaqmx.DaqWarning as e:
            self.logg.warning("DaqWarning caught as exception: %s", e)
            try:
                assert e.error_code == DAQmxWarnings.STOPPED_BEFORE_DONE, "Unexpected error code: {}".format(
                    e.error_code)
            except AssertionError as ae:
                self.logg.error("Assertion Error: %s", ae)

    def stop_triggers(self, _close=True):
        for key, _task in self.tasks.items():
            if self._running.get(key, False):
                _task.stop()
        self._running = {key: False for key in self._running}
        if _close:
            self.close_triggers()
        
    def close_triggers(self):
        for key, _task in self.tasks.items():
            if self._active.get(key, False):
                _task.close()
        self._active = {key: False for key in self._active}        
        