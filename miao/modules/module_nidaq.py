import warnings

import nidaqmx
import numpy as np
from nidaqmx.constants import Edge, AcquisitionType, LineGrouping, FrequencyUnits, Level, WAIT_INFINITELY
from nidaqmx.error_codes import DAQmxWarnings
from nidaqmx.stream_readers import AnalogSingleChannelReader  # , AnalogMultiChannelReader
from nidaqmx.stream_writers import AnalogSingleChannelWriter  # , AnalogMultiChannelWriter
from nidaqmx.system import System

warnings.filterwarnings("error", category=nidaqmx.DaqWarning)


class NIDAQ:
    class NIDAQSettings:

        def __init__(self):
            self.sample_rate = 100000
            self.duty_cycle = 0.5
            self.piezo_channels = "Dev1/ao0:1"
            self.galvo_channels = "Dev1/ao2:3"
            self.analog_channels = "Dev1/ao0:3"
            self.digital_channels = "Dev1/port0/line0:6"
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
            self.logg.info("DAQmx {0}.{1}.{2}".format(driver_version.major_version, driver_version.minor_version,
                                                      driver_version.update_version))
            return local_system.devices[0]
        except Exception as e:
            self.logg.error(f"Error initializing NIDAQ: {e}")

    def _configure(self):
        try:
            tasks = {"piezo": None, "galvo": None, "piezo_pos": None, "analog": None, "digital": None, "clock": None}
            _active = {key: False for key in tasks.keys()}
            _running = {key: False for key in tasks.keys()}
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
            with nidaqmx.Task() as task:
                task.ao_channels.add_ao_voltage_chan(self.piezo_channels, min_val=0., max_val=10.)
                task.timing.cfg_samp_clk_timing(rate=2000000, sample_mode=AcquisitionType.FINITE, samps_per_chan=1,
                                                active_edge=Edge.RISING)
                task.write([pos_x, pos_y], auto_start=True)
                task.wait_until_done(WAIT_INFINITELY)
                self.logg.info("Channels " + self.piezo_channels + " Write Successfully")
                task.stop()
        except nidaqmx.DaqWarning as e:
            self.logg.warning("DaqWarning caught as exception: %s", e)
            try:
                assert e.error_code == DAQmxWarnings.STOPPED_BEFORE_DONE, "Unexpected error code: {}".format(
                    e.error_code)
            except AssertionError as ae:
                self.logg.error("Assertion Error: %s", ae)

    def get_piezo_position(self):
        try:
            with nidaqmx.Task() as task:
                task.ai_channels.add_ai_voltage_chan("Dev1/ai0:1", min_val=-10.0, max_val=10.0)
                task.timing.cfg_samp_clk_timing(rate=500000, sample_mode=AcquisitionType.FINITE, samps_per_chan=10,
                                                active_edge=Edge.RISING)
                pos = task.read(number_of_samples_per_channel=10)
            return [sum(p) / len(p) for p in pos]
        except nidaqmx.DaqWarning as e:
            self.logg.warning("DaqWarning caught as exception: %s", e)
            try:
                assert e.error_code == DAQmxWarnings.STOPPED_BEFORE_DONE, "Unexpected error code: {}".format(
                    e.error_code)
            except AssertionError as ae:
                self.logg.error("Assertion Error: %s", ae)

    def set_galvo_position(self, pos_x, pos_y):
        try:
            with nidaqmx.Task() as task:
                task.ao_channels.add_ao_voltage_chan(self.galvo_channels, min_val=-10., max_val=10.)
                task.timing.cfg_samp_clk_timing(rate=2000000, sample_mode=AcquisitionType.FINITE, samps_per_chan=1,
                                                active_edge=Edge.RISING)
                task.write([[pos_x], [pos_y]], auto_start=True)
                task.wait_until_done(WAIT_INFINITELY)
                self.logg.info("Channels " + self.galvo_channels + " Write Successfully")
                task.stop()
        except nidaqmx.DaqWarning as e:
            self.logg.warning("DaqWarning caught as exception: %s", e)
            try:
                assert e.error_code == DAQmxWarnings.STOPPED_BEFORE_DONE, "Unexpected error code: {}".format(
                    e.error_code)
            except AssertionError as ae:
                self.logg.error("Assertion Error: %s", ae)

    def get_galvo_position(self):
        try:
            with nidaqmx.Task() as task:
                task.ai_channels.add_ai_voltage_chan("Dev1/ai4:5", min_val=-10.0, max_val=10.0)
                task.timing.cfg_samp_clk_timing(rate=500000, sample_mode=AcquisitionType.FINITE, samps_per_chan=10,
                                                active_edge=Edge.RISING)
                pos = task.read(number_of_samples_per_channel=10)
            return [sum(p) / len(p) for p in pos]
        except nidaqmx.DaqWarning as e:
            self.logg.warning("DaqWarning caught as exception: %s", e)
            try:
                assert e.error_code == DAQmxWarnings.STOPPED_BEFORE_DONE, "Unexpected error code: {}".format(
                    e.error_code)
            except AssertionError as ae:
                self.logg.error("Assertion Error: %s", ae)

    def write_clock_channel(self):
        try:
            self.tasks["clock"] = nidaqmx.Task("clock")
            self.tasks["clock"].co_channels.add_co_pulse_chan_freq(self.clock_channel, units=FrequencyUnits.HZ,
                                                                   idle_state=Level.LOW, initial_delay=0.0,
                                                                   freq=self.sample_rate, duty_cycle=self.duty_cycle)
            self.tasks["clock"].timing.cfg_implicit_timing(sample_mode=AcquisitionType.CONTINUOUS)
            self._active["clock"] = True
            self.logg.info("Channel " + self.clock_channel + " Writes Successfully")
        except nidaqmx.DaqWarning as e:
            self.logg.warning("DaqWarning caught as exception: %s", e)
            try:
                assert e.error_code == DAQmxWarnings.STOPPED_BEFORE_DONE, "Unexpected error code: {}".format(
                    e.error_code)
            except AssertionError as ae:
                self.logg.error("Assertion Error: %s", ae)

    def write_digital_sequences(self, digital_sequences, base_clock=False, finite=True):
        if base_clock:
            self.clock = "20MHzTimebase"
        else:
            self.clock = "Ctr0InternalOutput"
            self.write_clock_channel()
        if finite:
            self.mode = AcquisitionType.FINITE
        else:
            self.mode = AcquisitionType.CONTINUOUS
        try:
            self.tasks["digital"] = nidaqmx.Task("digital")
            self.tasks["digital"].do_channels.add_do_chan(self.digital_channels,
                                                          line_grouping=LineGrouping.CHAN_PER_LINE)
            _channels, _samples = digital_sequences.shape
            self.tasks["digital"].timing.cfg_samp_clk_timing(self.sample_rate, source=self.clock,
                                                             active_edge=Edge.RISING, sample_mode=self.mode,
                                                             samps_per_chan=_samples)
            self.tasks["digital"].write(digital_sequences == 1.0, auto_start=False)
            self._active["digital"] = True
            self.logg.info("Channels " + self.digital_channels + " Write Successfully")
        except nidaqmx.DaqWarning as e:
            self.logg.warning("DaqWarning caught as exception: %s", e)
            try:
                assert e.error_code == DAQmxWarnings.STOPPED_BEFORE_DONE, "Unexpected error code: {}".format(
                    e.error_code)
            except AssertionError as ae:
                self.logg.error("Assertion Error: %s", ae)

    def write_piezo_scan(self, piezo_sequences, finite=True):
        if finite:
            mode = AcquisitionType.FINITE
        else:
            mode = AcquisitionType.CONTINUOUS
        try:
            self.tasks["piezo"] = nidaqmx.Task("piezo")
            self.tasks["piezo"].ao_channels.add_ao_voltage_chan(self.piezo_channels, min_val=0., max_val=10.)
            _channels, _samples = piezo_sequences.shape
            self.tasks["piezo"].timing.cfg_samp_clk_timing(self.sample_rate, source="Ctr0InternalOutput",
                                                           active_edge=Edge.RISING, sample_mode=mode,
                                                           samps_per_chan=_samples)
            self.tasks["piezo"].write(piezo_sequences, auto_start=False)
            self._active["piezo"] = True

            self.logg.info("Channels " + self.piezo_channels + " Write Successfully")
        except nidaqmx.DaqWarning as e:
            self.logg.warning("DaqWarning caught as exception: %s", e)
            try:
                assert e.error_code == DAQmxWarnings.STOPPED_BEFORE_DONE, "Unexpected error code: {}".format(
                    e.error_code)
            except AssertionError as ae:
                self.logg.error("Assertion Error: %s", ae)

    def write_galvo_scan(self, galvo_sequences, finite=True):
        if finite:
            mode = AcquisitionType.FINITE
        else:
            mode = AcquisitionType.CONTINUOUS
        try:
            self.tasks["galvo"] = nidaqmx.Task("galvo")
            self.tasks["galvo"].ao_channels.add_ao_voltage_chan(self.galvo_channels, min_val=-10., max_val=10.)
            _channels, _samples = galvo_sequences.shape
            self.tasks["galvo"].timing.cfg_samp_clk_timing(self.sample_rate, source="Ctr0InternalOutput",
                                                           active_edge=Edge.RISING, sample_mode=mode,
                                                           samps_per_chan=_samples)
            self.tasks["galvo"].write(galvo_sequences, auto_start=False)
            self._active["galvo"] = True
            self.logg.info("Channels " + self.galvo_channels + " Write Successfully")
        except nidaqmx.DaqWarning as e:
            self.logg.warning("DaqWarning caught as exception: %s", e)
            try:
                assert e.error_code == DAQmxWarnings.STOPPED_BEFORE_DONE, "Unexpected error code: {}".format(
                    e.error_code)
            except AssertionError as ae:
                self.logg.error("Assertion Error: %s", ae)

    def write_analog_sequences(self, analog_sequences=None, finite=True):
        if finite:
            mode = AcquisitionType.FINITE
        else:
            mode = AcquisitionType.CONTINUOUS
        try:
            self.tasks["analog"] = nidaqmx.Task("analog")
            self.tasks["analog"].ao_channels.add_ao_voltage_chan(self.analog_channels, min_val=-10., max_val=10.)
            _channels, _samples = analog_sequences.shape
            self.tasks["analog"].timing.cfg_samp_clk_timing(self.sample_rate, source="Ctr0InternalOutput",
                                                            active_edge=Edge.RISING, sample_mode=mode,
                                                            samps_per_chan=_samples)
            self.tasks["analog"].write(analog_sequences, auto_start=False)
            self._active["analog"] = True
            self.logg.info("Channels " + self.analog_channels + " Write Successfully")
        except nidaqmx.DaqWarning as e:
            self.logg.warning("DaqWarning caught as exception: %s", e)
            try:
                assert e.error_code == DAQmxWarnings.STOPPED_BEFORE_DONE, "Unexpected error code: {}".format(
                    e.error_code)
            except AssertionError as ae:
                self.logg.error("Assertion Error: %s", ae)

    def write_triggers(self, piezo_sequences=None, galvo_sequences=None, digital_sequences=None, finite=True):
        try:
            if digital_sequences is not None:
                self.write_digital_sequences(digital_sequences=digital_sequences, finite=finite)
            if piezo_sequences is not None and galvo_sequences is not None:
                self.write_analog_sequences(np.concatenate((piezo_sequences, galvo_sequences)), finite)
            else:
                if piezo_sequences is not None:
                    self.write_piezo_scan(piezo_sequences, finite)
                elif galvo_sequences is not None:
                    self.write_galvo_scan(galvo_sequences, finite)
        except nidaqmx.DaqWarning as e:
            self.logg.warning("DaqWarning caught as exception: %s", e)
            try:
                assert e.error_code == DAQmxWarnings.STOPPED_BEFORE_DONE, "Unexpected error code: {}".format(
                    e.error_code)
            except AssertionError as ae:
                self.logg.error("Assertion Error: %s", ae)

    def run_digital_trigger(self):
        try:
            self._running["digital"] = True
            self.tasks["digital"].start()
            if self.clock == "Ctr0InternalOutput":
                self._running["clock"] = True
                self.tasks["clock"].start()
            if self.mode == AcquisitionType.FINITE:
                self.tasks["digital"].wait_until_done(WAIT_INFINITELY)
        except nidaqmx.DaqWarning as e:
            self.logg.warning(f"DaqWarning caught as exception: {e}")
            try:
                assert e.error_code == DAQmxWarnings.STOPPED_BEFORE_DONE, "Unexpected error code: {}".format(
                    e.error_code)
            except AssertionError as ae:
                self.logg.error("Assertion Error: %s", ae)

    def start_triggers(self):
        try:
            for key, _task in self.tasks.items():
                if key != "clock":
                    if self._active.get(key, False):
                        if not self._running[key]:
                            _task.start()
                            self._running[key] = True
        except nidaqmx.DaqWarning as e:
            self.logg.warning("DaqWarning caught as exception: %s", e)

    def run_triggers(self):
        try:
            self.start_triggers()
            if self.clock == "Ctr0InternalOutput":
                self._running["clock"] = True
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

    def stop_triggers(self, _close=True):
        for key, _task in self.tasks.items():
            if self._active.get(key, False):
                if self._running.get(key, False):
                    _task.stop()
        self._running = {key: False for key in self._running}
        if _close:
            self.close_triggers()

    def close_triggers(self):
        for key, _task in self.tasks.items():
            if self._active.get(key, False):
                _task.close()
                _task = None
        self._active = {key: False for key in self._active}

    def measure_io(self, output_channel, input_channel, data):
        num_samples = data.shape[0]
        acquired_data = np.zeros(num_samples)
        with nidaqmx.Task() as output_task:
            output_task.ao_channels.add_ao_voltage_chan(output_channel, min_val=-10., max_val=10.)
            output_task.timing.cfg_samp_clk_timing(rate=self.sample_rate,
                                                   sample_mode=AcquisitionType.FINITE,
                                                   samps_per_chan=num_samples)
            with nidaqmx.Task() as input_task:
                input_task.ai_channels.add_ai_voltage_chan(input_channel, min_val=-10., max_val=10.)
                input_task.timing.cfg_samp_clk_timing(rate=self.sample_rate,
                                                      sample_mode=AcquisitionType.FINITE,
                                                      samps_per_chan=num_samples,
                                                      source=f'/Dev1/ao/SampleClock')
                writer = AnalogSingleChannelWriter(output_task.out_stream)
                reader = AnalogSingleChannelReader(input_task.in_stream)
                writer.write_many_sample(data)
                input_task.start()
                output_task.start()
                output_task.wait_until_done()
                input_task.wait_until_done()
                reader.read_many_sample(data=acquired_data, number_of_samples_per_channel=num_samples)
        return acquired_data
