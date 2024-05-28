import time
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
    def __init__(self, logg=None, config=None):
        self.logg = logg or self.setup_logging()
        self.config = config or self.load_configs()
        self.channels = self.config.configs["Triggers"]["NIDAQ"]["Channels"]
        self.analog_output_channels = {"piezo_x": "Dev1/ao0",
                                       "piezo_y": "Dev1/ao1",
                                       "piezo_z": "Dev1/ao2",
                                       "galvo_x": "Dev2/ao0",
                                       "galvo_y": "Dev2/ao1",
                                       "galvo_s": "Dev2/ao2"}
        self.digital_output_channels = {"laser_405": "Dev1/port0/line0",
                                        "laser_488_0": "Dev1/port0/line1",
                                        "laser_488_1": "Dev1/port0/line2",
                                        "laser_488_2": "Dev1/port0/line3",
                                        "andor ccd": "Dev1/port0/line4",
                                        "hamamatsu scmos": "Dev1/port0/line5",
                                        "thorlabs cmos": "Dev1/port0/line6",
                                        "tis cmos": "Dev1/port0/line7"}
        self.analog_input_channels = {"piezo_x": "Dev1/ai0",
                                      "piezo_y": "Dev1/ai1",
                                      "piezo_z": "Dev1/ai2",
                                      "galvo_x": "Dev2/ai0",
                                      "galvo_y": "Dev2/ai1",
                                      "galvo_s": "Dev2/ai2"}
        self.sample_rate = 250000
        self.clock_channel = ["Dev1/ctr0", "Dev1/ctr0"]
        self.clock_rate = 2000000
        self.duty_cycle = 0.5
        self.clock = "Ctr0InternalOutput"
        self.mode = None
        self.device = self._initialize()
        self._settings = self.NIDAQSettings()
        self.tasks = {}
        self._active = {}
        self._running = {}
        self.tasks, self._active, self._running, = self._configure()

    def close(self):
        self.device.reset_device()

    @staticmethod
    def setup_logging():
        import logging
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
        return logging

    @staticmethod
    def load_configs():
        import json
        config_file = input("Enter configuration file directory: ")
        with open(config_file, 'r') as f:
            cfg = json.load(f)
        return cfg

    def _initialize(self):
        try:
            local_system = System.local()
            driver_version = local_system.driver_version
            self.logg.info("DAQmx {0}.{1}.{2}".format(driver_version.major_version, driver_version.minor_version,
                                                      driver_version.update_version))
            return local_system.devices
        except Exception as e:
            self.logg.error(f"Error initializing NIDAQ: {e}")

    def _configure(self):
        try:
            tasks = {"analog": None, "digital": None, "clock": None}
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

    def set_piezo_position(self, pos_x, pos_y, pos_z):
        try:
            with nidaqmx.Task() as task:
                task.ao_channels.add_ao_voltage_chan(self.analog_output_channels["piezo_x"], min_val=0., max_val=10.)
                task.ao_channels.add_ao_voltage_chan(self.analog_output_channels["piezo_y"], min_val=0., max_val=10.)
                task.ao_channels.add_ao_voltage_chan(self.analog_output_channels["piezo_z"], min_val=0., max_val=10.)
                task.timing.cfg_samp_clk_timing(rate=2000000, sample_mode=AcquisitionType.FINITE, samps_per_chan=1,
                                                active_edge=Edge.RISING)
                task.write([pos_x, pos_y, pos_z], auto_start=True)
                task.wait_until_done(WAIT_INFINITELY)
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
                task.ai_channels.add_ai_voltage_chan(self.analog_input_channels["piezo_x"], min_val=-10.0, max_val=10.0)
                task.ai_channels.add_ai_voltage_chan(self.analog_input_channels["piezo_y"], min_val=-10.0, max_val=10.0)
                task.ai_channels.add_ai_voltage_chan(self.analog_input_channels["piezo_z"], min_val=-10.0, max_val=10.0)
                task.timing.cfg_samp_clk_timing(rate=500000, sample_mode=AcquisitionType.FINITE, samps_per_chan=16,
                                                active_edge=Edge.RISING)
                pos = task.read(number_of_samples_per_channel=16)
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
                task.ao_channels.add_ao_voltage_chan(self.analog_output_channels["galvo_x"], min_val=-10., max_val=10.)
                task.ao_channels.add_ao_voltage_chan(self.analog_output_channels["galvo_y"], min_val=-10., max_val=10.)
                task.timing.cfg_samp_clk_timing(rate=2000000, sample_mode=AcquisitionType.FINITE, samps_per_chan=1,
                                                active_edge=Edge.RISING)
                task.write([pos_x, pos_y], auto_start=True)
                task.wait_until_done(WAIT_INFINITELY)
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
                task.ai_channels.add_ai_voltage_chan(self.analog_input_channels["galvo_x"], min_val=-10.0, max_val=10.0)
                task.ai_channels.add_ai_voltage_chan(self.analog_input_channels["galvo_y"], min_val=-10.0, max_val=10.0)
                task.timing.cfg_samp_clk_timing(rate=500000, sample_mode=AcquisitionType.FINITE, samps_per_chan=16,
                                                active_edge=Edge.RISING)
                pos = task.read(number_of_samples_per_channel=16)
            return [sum(p) / len(p) for p in pos]
        except nidaqmx.DaqWarning as e:
            self.logg.warning("DaqWarning caught as exception: %s", e)
            try:
                assert e.error_code == DAQmxWarnings.STOPPED_BEFORE_DONE, "Unexpected error code: {}".format(
                    e.error_code)
            except AssertionError as ae:
                self.logg.error("Assertion Error: %s", ae)

    def set_galvo_switcher(self, pos):
        try:
            with nidaqmx.Task() as task:
                task.ao_channels.add_ao_voltage_chan(self.analog_output_channels["galvo_s"], min_val=-10., max_val=10.)
                task.timing.cfg_samp_clk_timing(rate=2000000, sample_mode=AcquisitionType.FINITE, samps_per_chan=1,
                                                active_edge=Edge.RISING)
                task.write([pos], auto_start=True)
                task.wait_until_done(WAIT_INFINITELY)
                task.stop()
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
            self.tasks["clock"].co_channels.add_co_pulse_chan_freq(self.clock_channel[0], units=FrequencyUnits.HZ,
                                                                   idle_state=Level.LOW, initial_delay=0.0,
                                                                   freq=self.sample_rate, duty_cycle=self.duty_cycle)
            self.tasks["clock"].co_pulse_freq_timebase_src = '20MHzTimebase'
            self.tasks["clock"].co_pulse_freq_timebase_rate = self.clock_rate
            self.tasks["clock"].timing.cfg_implicit_timing(sample_mode=AcquisitionType.CONTINUOUS)
            self.tasks["clock"].export_signals.export_signal(ExportSignal.COUNTER_OUTPUT_EVENT, "Dev1/PFI0")
            self._active["clock"] = True
            self.logg.info("Clock is Written to " + self.clock_channel[0] + " and Exported to " + self.clock_channel[1])
        except nidaqmx.DaqWarning as e:
            self.logg.warning("DaqWarning caught as exception: %s", e)
            try:
                assert e.error_code == DAQmxWarnings.STOPPED_BEFORE_DONE, "Unexpected error code: {}".format(
                    e.error_code)
            except AssertionError as ae:
                self.logg.error("Assertion Error: %s", ae)

    def write_digital_sequences(self, digital_sequences):
        try:
            self.tasks["digital"] = nidaqmx.Task("digital")
            self.tasks["digital"].do_channels.add_do_chan(self.digital_channels,
                                                          line_grouping=LineGrouping.CHAN_PER_LINE)
            _channels, _samples = digital_sequences.shape
            self.tasks["digital"].timing.cfg_samp_clk_timing(rate=self.sample_rate, source=self.clock,
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

    def write_analog_sequences(self, analog_sequences=None):
        try:
            self.tasks["analog"] = nidaqmx.Task("analog")
            self.tasks["analog"].ao_channels.add_ao_voltage_chan(self.analog_channels, min_val=-10., max_val=10.)
            _channels, _samples = analog_sequences.shape
            self.tasks["analog"].timing.cfg_samp_clk_timing(rate=self.sample_rate, source=self.clock,
                                                            active_edge=Edge.RISING, sample_mode=self.mode,
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
        self.write_clock_channel()
        if finite:
            self.mode = AcquisitionType.FINITE
        else:
            self.mode = AcquisitionType.CONTINUOUS
        try:
            if digital_sequences is not None:
                self.write_digital_sequences(digital_sequences=digital_sequences)
            if piezo_sequences is not None and galvo_sequences is not None:
                self.write_analog_sequences(np.concatenate((piezo_sequences, galvo_sequences)))
            else:
                if piezo_sequences is not None:
                    self.write_piezo_scan(piezo_sequences)
                elif galvo_sequences is not None:
                    self.write_galvo_scan(galvo_sequences)
        except nidaqmx.DaqWarning as e:
            self.logg.warning("DaqWarning caught as exception: %s", e)
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

    def measure_ao(self, output_channel, input_channel, data):
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

    def measure_do(self, output_channel, input_channel, data):
        num_samples = data.shape[0]
        with nidaqmx.Task() as task_do, nidaqmx.Task() as task_ai, nidaqmx.Task() as task_clock:
            task_clock.co_channels.add_co_pulse_chan_freq(self.clock_channel, units=FrequencyUnits.HZ,
                                                          idle_state=Level.LOW, initial_delay=0.0,
                                                          freq=self.sample_rate, duty_cycle=self.duty_cycle)
            task_clock.co_pulse_freq_timebase_src = '20MHzTimebase'
            task_clock.co_pulse_freq_timebase_rate = self.clock_rate
            task_clock.timing.cfg_implicit_timing(sample_mode=AcquisitionType.CONTINUOUS)
            # Configure DO as before
            task_do.do_channels.add_do_chan(output_channel, line_grouping=LineGrouping.CHAN_PER_LINE)
            task_do.timing.cfg_samp_clk_timing(rate=self.sample_rate, source=self.clock,
                                               active_edge=Edge.RISING, sample_mode=AcquisitionType.FINITE,
                                               samps_per_chan=num_samples)
            task_do.write(data == 1, auto_start=False)
            task_ai.ai_channels.add_ai_voltage_chan(input_channel)
            task_ai.timing.cfg_samp_clk_timing(rate=self.sample_rate, source=self.clock,
                                               active_edge=Edge.RISING, sample_mode=AcquisitionType.FINITE,
                                               samps_per_chan=num_samples)
            # Start AI first but it waits for the trigger
            task_ai.start()
            # Trigger by writing to DO
            task_do.start()
            task_clock.start()
            task_do.wait_until_done()
            # Read the analog input response
            acquired_data = task_ai.read(number_of_samples_per_channel=num_samples, timeout=10)
            # Stop AI task
            task_clock.stop()
            task_do.stop()
            task_ai.stop()
        return acquired_data

    def check_task_status(self, task):
        try:
            if task.is_task_done():
                return True
            else:
                return False
        except nidaqmx.DaqError as e:
            self.logg.error(f"Error checking task status: {e}")
            return True
