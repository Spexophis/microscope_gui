import warnings

import nidaqmx
from nidaqmx.constants import Edge, AcquisitionType, LineGrouping, FrequencyUnits, Level
from nidaqmx.error_codes import DAQmxWarnings
from nidaqmx.system import System

warnings.filterwarnings("error", category=nidaqmx.DaqWarning)


class NIDAQ:

    def __init__(self, frequency=100000, duty_cycle=0.5, logg=None):
        super().__init__()
        self.frequency = frequency
        self.duty_cycle = duty_cycle
        local_system = System.local()
        driver_version = local_system.driver_version
        print("DAQmx {0}.{1}.{2}".format(driver_version.major_version, driver_version.minor_version,
                                         driver_version.update_version))
        self.device = local_system.devices[0]
        self.tasks = {"piezo": nidaqmx.Task("piezo"),
                      "galvo": nidaqmx.Task("galvo"),
                      "piezo_pos": nidaqmx.Task("piezo_pos"),
                      "digital": nidaqmx.Task("digital"),
                      "clock": nidaqmx.Task("clock")}
        self._runtask = {key: False for key in self.tasks.keys()}
        try:
            self.tasks["piezo_pos"].ai_channels.add_ai_voltage_chan("Dev1/ai0:2", min_val=0., max_val=10.)
            self.tasks["digital"].do_channels.add_do_chan("Dev1/port0/line0:5",
                                                          line_grouping=LineGrouping.CHAN_PER_LINE)
            self.tasks["clock"].co_channels.add_co_pulse_chan_freq("Dev1/ctr0", units=FrequencyUnits.HZ,
                                                                   idle_state=Level.LOW, initial_delay=0.0,
                                                                   freq=self.frequency, duty_cycle=self.duty_cycle)
            self.tasks["clock"].timing.cfg_implicit_timing(sample_mode=AcquisitionType.CONTINUOUS)
        except nidaqmx.DaqWarning as e:
            print("DaqWarning caught as exception: {0}\n".format(e))
            assert e.error_code == DAQmxWarnings.STOPPED_BEFORE_DONE

    def __del__(self):
        pass

    def close(self):
        self.device.reset_device()

    def set_piezo_position(self, pos_x, pos_y):
        if self.tasks["piezo"].is_task_done():
            self.tasks["piezo"].close()
        else:
            self.tasks["piezo"].stop()
            self.tasks["piezo"].close()
        try:
            self.tasks["piezo"] = nidaqmx.Task("piezo")
            self.tasks["piezo"].ao_channels.add_ao_voltage_chan("Dev1/ao0:1", min_val=0., max_val=10.)
            self.tasks["piezo"].timing.cfg_samp_clk_timing(self.frequency, source="100kHzTimebase",
                                                           active_edge=Edge.RISING,
                                                           sample_mode=AcquisitionType.FINITE,
                                                           samps_per_chan=1)
            self.tasks["piezo"].write([pos_x, pos_y], auto_start=True)
            self.tasks["piezo"].wait_until_done()
            self.tasks["piezo"].stop()
        except nidaqmx.DaqWarning as e:
            print("DaqWarning caught as exception: {0}\n".format(e))
            assert e.error_code == DAQmxWarnings.STOPPED_BEFORE_DONE

    def get_piezo_position(self):
        if not self.tasks["piezo_pos"].is_task_done():
            self.tasks["piezo_pos"].stop()
        try:
            pos = self.tasks["piezo_pos"].read(number_of_samples_per_channel=1)
            return pos
        except nidaqmx.DaqWarning as e:
            print("DaqWarning caught as exception: {0}\n".format(e))
            assert e.error_code == DAQmxWarnings.STOPPED_BEFORE_DONE

    def piezo_scan(self, piezo_sequence):
        if self.tasks["piezo"].is_task_done():
            self.tasks["piezo"].close()
        else:
            self.tasks["piezo"].stop()
            self.tasks["piezo"].close()
        try:
            self.tasks["piezo"] = nidaqmx.Task("piezo")
            self.tasks["piezo"].ao_channels.add_ao_voltage_chan("Dev1/ao0:1", min_val=0., max_val=10.)
            _channels, _samples = piezo_sequence.shape
            self.tasks["piezo"].timing.cfg_samp_clk_timing(self.frequency, source="Ctr0InternalOutput",
                                                           active_edge=Edge.RISING,
                                                           sample_mode=AcquisitionType.FINITE,
                                                           samps_per_chan=_samples)
            self.tasks["piezo"].write(piezo_sequence, auto_start=False)
        except nidaqmx.DaqWarning as e:
            print("DaqWarning caught as exception: {0}\n".format(e))
            assert e.error_code == DAQmxWarnings.STOPPED_BEFORE_DONE

    def set_galvo_position(self, pos_x, pos_y):
        if self.tasks["galvo"].is_task_done():
            self.tasks["galvo"].close()
        else:
            self.tasks["galvo"].stop()
            self.tasks["galvo"].close()
        try:
            self.tasks["galvo"] = nidaqmx.Task("piezo")
            self.tasks["galvo"].ao_channels.add_ao_voltage_chan("Dev1/ao2:3", min_val=-10., max_val=10.)
            self.tasks["galvo"].timing.cfg_samp_clk_timing(self.frequency, source="100kHzTimebase",
                                                           active_edge=Edge.RISING,
                                                           sample_mode=AcquisitionType.FINITE,
                                                           samps_per_chan=1)
            self.tasks["galvo"].write([[pos_x], [pos_y]], auto_start=True)
            self.tasks["galvo"].wait_until_done()
            self.tasks["galvo"].stop()
        except nidaqmx.DaqWarning as e:
            print("DaqWarning caught as exception: {0}\n".format(e))
            assert e.error_code == DAQmxWarnings.STOPPED_BEFORE_DONE

    def galvo_scan(self, galvo_sequence):
        if self.tasks["galvo"].is_task_done():
            self.tasks["galvo"].close()
        else:
            self.tasks["galvo"].stop()
            self.tasks["galvo"].close()
        try:
            self.tasks["galvo"] = nidaqmx.Task("piezo")
            self.tasks["galvo"].ao_channels.add_ao_voltage_chan("Dev1/ao2:3", min_val=-10., max_val=10.)
            _channels, _samples = galvo_sequence.shape
            self.tasks["galvo"].timing.cfg_samp_clk_timing(self.frequency, source="Ctr0InternalOutput",
                                                           active_edge=Edge.RISING,
                                                           sample_mode=AcquisitionType.FINITE,
                                                           samps_per_chan=_samples)
            self.tasks["galvo"].write(galvo_sequence, auto_start=False)
        except nidaqmx.DaqWarning as e:
            print("DaqWarning caught as exception: {0}\n".format(e))
            assert e.error_code == DAQmxWarnings.STOPPED_BEFORE_DONE

    def write_digital_sequences(self, digital_sequences, clock_source="100kHzTimebase", mode=AcquisitionType.FINITE):
        if not self.tasks["digital"].is_task_done():
            self.tasks["digital"].stop()
        try:
            _channels, _samples = digital_sequences.shape
            self.tasks["digital"].timing.cfg_samp_clk_timing(self.frequency, source=clock_source,
                                                             active_edge=Edge.RISING,
                                                             sample_mode=mode,
                                                             samps_per_chan=_samples)
            self.tasks["digital"].write(digital_sequences == 1.0, auto_start=False)
        except nidaqmx.DaqWarning as e:
            print("DaqWarning caught as exception: {0}\n".format(e))
            assert e.error_code == DAQmxWarnings.STOPPED_BEFORE_DONE

    def run_digital_trigger(self, digital_sequences, clock_source="100kHzTimebase", mode="continuous"):
        if mode == "continuous":
            _mode = AcquisitionType.CONTINUOUS
        else:
            _mode = AcquisitionType.FINITE
        if clock_source == "Ctr0InternalOutput":
            self.write_digital_sequences(digital_sequences, clock_source, _mode)
            self.tasks["digital"].start()
            self.tasks["clock"].start()
            self._runtask["digital"] = True
            self._runtask["clock"] = True
            if mode != "continuous":
                self.tasks["digital"].wait_until_done()
        elif clock_source == "100kHzTimebase":
            self.write_digital_sequences(digital_sequences, clock_source, _mode)
            self.tasks["digital"].start()
            self._runtask["digital"] = True
            if mode != "continuous":
                self.tasks["digital"].wait_until_done()

    def run_digital_triggers(self, n):
        for i in range(n):
            try:
                self.tasks["digital"].start()
                self._runtask["digital"] = True
                self.tasks["digital"].wait_until_done()
                self.tasks["digital"].stop()
                self._runtask["digital"] = False
            except nidaqmx.DaqWarning as e:
                print("DaqWarning caught as exception: {0}\n".format(e))
                assert e.error_code == DAQmxWarnings.STOPPED_BEFORE_DONE

    def run_triggers(self, piezo_sequence=None, galvo_sequence=None, digital_sequences=None):
        if piezo_sequence is not None:
            self.piezo_scan(piezo_sequence)
            self.tasks["piezo"].start()
            self._runtask["piezo"] = True
        if galvo_sequence is not None:
            self.galvo_scan(galvo_sequence)
            self.tasks["galvo"].start()
            self._runtask["galvo"] = True
        if digital_sequences is not None:
            if piezo_sequence is not None or galvo_sequence is not None:
                clock_source = "Ctr0InternalOutput"
                mode = AcquisitionType.FINITE
                self.write_digital_sequences(digital_sequences, clock_source, mode)
                self.tasks["digital"].start()
                self.tasks["clock"].start()
                self._runtask["digital"] = True
                self._runtask["clock"] = True
                self.tasks["digital"].wait_until_done()
            else:
                clock_source = "100kHzTimebase"
                mode = AcquisitionType.CONTINUOUS
                self.write_digital_sequences(digital_sequences, clock_source, mode)
                self.tasks["digital"].start()
                self._runtask["digital"] = True

    def stop_triggers(self):
        for key, _task in self.tasks.items():
            if self._runtask.get(key, False):
                _task.stop()
        self._runtask = {key: False for key in self._runtask}
