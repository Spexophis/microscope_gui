import PyDAQmx
import numpy as np
from PyDAQmx import DAQmxConstants
from PyDAQmx import DAQmxTypes


class NIDAQ:

    def __init__(self, frequency=100000, duty_cycle=0.5):
        super().__init__()

        self.frequency = frequency
        self.duty_cycle = duty_cycle

        try:
            self.counterHandle = PyDAQmx.TaskHandle(0)
            PyDAQmx.DAQmxCreateTask("", DAQmxTypes.byref(self.counterHandle))
            PyDAQmx.DAQmxCreateCOPulseChanFreq(self.counterHandle, 'Dev1/ctr0', '', DAQmxConstants.DAQmx_Val_Hz,
                                               DAQmxConstants.DAQmx_Val_Low, 0.0,
                                               self.frequency, self.duty_cycle)

            self.doHandle = PyDAQmx.TaskHandle(0)
            PyDAQmx.DAQmxCreateTask("", DAQmxTypes.byref(self.doHandle))
            PyDAQmx.DAQmxCreateDOChan(self.doHandle, 'Dev1/port0/line0:5', '', DAQmxConstants.DAQmx_Val_ChanPerLine)

            self.piezoHandle = PyDAQmx.TaskHandle(0)
            PyDAQmx.DAQmxCreateTask("", DAQmxTypes.byref(self.piezoHandle))
            PyDAQmx.DAQmxCreateAOVoltageChan(self.piezoHandle, 'Dev1/ao0:1', '', 0.0, 10.0,
                                             DAQmxConstants.DAQmx_Val_Volts, None)

            self.galvoHandle = PyDAQmx.TaskHandle(0)
            PyDAQmx.DAQmxCreateTask("", DAQmxTypes.byref(self.galvoHandle))
            PyDAQmx.DAQmxCreateAOVoltageChan(self.galvoHandle, 'Dev1/ao2:3', '', -10.0, 10.0,
                                             DAQmxConstants.DAQmx_Val_Volts, None)

            self.rpHandle = PyDAQmx.TaskHandle(0)
            PyDAQmx.DAQmxCreateTask("", DAQmxTypes.byref(self.rpHandle))
            PyDAQmx.DAQmxCreateAIVoltageChan(self.rpHandle, 'Dev1/ai0:2', '', DAQmxConstants.DAQmx_Val_RSE, 0.0, 10.0,
                                             DAQmxConstants.DAQmx_Val_Volts, None)

        except:
            errBuff = DAQmxTypes.create_string_buffer(b"", 2048)
            PyDAQmx.DAQmxGetExtendedErrorInfo(errBuff, 2048)
            print(errBuff.value)

    def __del__(self):
        self.reset_daq()
        del self.counterHandle, self.doHandle, self.piezoHandle, self.galvoHandle, self.rpHandle

    def reset_daq(self):
        try:
            PyDAQmx.DAQmxResetDevice("Dev1")
            print('DAQ board reset')
        except:
            errBuff = DAQmxTypes.create_string_buffer(b"", 2048)
            PyDAQmx.DAQmxGetExtendedErrorInfo(errBuff, 2048)
            print(errBuff.value)

    def set_xyz(self, pos_x, pos_y, pos_z):
        try:
            PyDAQmx.DAQmxWriteAnalogF64(self.piezoHandle, 1, True, 10.0, DAQmxConstants.DAQmx_Val_GroupByChannel,
                                        np.array([pos_x, pos_y], dtype=np.float64), None, None)
            PyDAQmx.DAQmxStartTask(self.piezoHandle)
            PyDAQmx.DAQmxStopTask(self.piezoHandle)
        except:
            errBuff = DAQmxTypes.create_string_buffer(b"", 2048)
            PyDAQmx.DAQmxGetExtendedErrorInfo(errBuff, 2048)
            print(errBuff.value)

    def get_xyz(self):
        try:
            data = np.zeros((3,), dtype=np.float64)
            read = DAQmxTypes.int32()
            PyDAQmx.DAQmxStartTask(self.rpHandle)
            PyDAQmx.DAQmxReadAnalogF64(self.rpHandle, 1, 10.0, DAQmxConstants.DAQmx_Val_GroupByChannel, data, 3,
                                       DAQmxTypes.byref(read), None)
            PyDAQmx.DAQmxStopTask(self.rpHandle)
            return data
        except:
            errBuff = DAQmxTypes.create_string_buffer(b"", 2048)
            PyDAQmx.DAQmxGetExtendedErrorInfo(errBuff, 2048)
            print(errBuff.value)

    def set_galvo(self, axis_x, axis_y):
        try:
            PyDAQmx.DAQmxWriteAnalogF64(self.galvoHandle, 1, True, 10.0, DAQmxConstants.DAQmx_Val_GroupByChannel,
                                        np.array([axis_x, axis_y], dtype=np.float64), None, None)
            PyDAQmx.DAQmxStartTask(self.galvoHandle)
            PyDAQmx.DAQmxStopTask(self.galvoHandle)
        except:
            errBuff = DAQmxTypes.create_string_buffer(b"", 2048)
            PyDAQmx.DAQmxGetExtendedErrorInfo(errBuff, 2048)
            print(errBuff.value)

    def scan_galvo(self, galvo_xy):
        channels, samples = galvo_xy.shape
        try:
            PyDAQmx.DAQmxCfgSampClkTiming(self.galvoHandle, r'100kHzTimebase', self.frequency,
                                          DAQmxConstants.DAQmx_Val_Rising,
                                          DAQmxConstants.DAQmx_Val_FiniteSamps, samples)
            PyDAQmx.DAQmxWriteAnalogF64(self.galvoHandle, samples, False, -1, DAQmxConstants.DAQmx_Val_GroupByChannel,
                                        galvo_xy.astype(np.float64), None, None)
            PyDAQmx.DAQmxStartTask(self.galvoHandle)
        except:
            errBuff = DAQmxTypes.create_string_buffer(b"", 2048)
            PyDAQmx.DAQmxGetExtendedErrorInfo(errBuff, 2048)
            print(errBuff.value)

    def stop_galvo(self):
        try:
            PyDAQmx.DAQmxStopTask(self.galvoHandle)
        except:
            errBuff = DAQmxTypes.create_string_buffer(b"", 2048)
            PyDAQmx.DAQmxGetExtendedErrorInfo(errBuff, 2048)
            print(errBuff.value)

    def trig_open(self, do_sequences):
        do_channels, do_samples = do_sequences.shape
        try:
            PyDAQmx.DAQmxCfgSampClkTiming(self.doHandle, r'100kHzTimebase', self.frequency,
                                          DAQmxConstants.DAQmx_Val_Rising,
                                          DAQmxConstants.DAQmx_Val_ContSamps, do_samples)
            PyDAQmx.DAQmxWriteDigitalLines(self.doHandle, do_samples, False, -1,
                                           DAQmxConstants.DAQmx_Val_GroupByChannel,
                                           do_sequences.astype(np.uint8), None, None)
        except:
            errBuff = DAQmxTypes.create_string_buffer(b"", 2048)
            PyDAQmx.DAQmxGetExtendedErrorInfo(errBuff, 2048)
            print(errBuff.value)

    def trig_run(self):
        try:
            PyDAQmx.DAQmxStartTask(self.doHandle)
        except:
            errBuff = DAQmxTypes.create_string_buffer(b"", 2048)
            PyDAQmx.DAQmxGetExtendedErrorInfo(errBuff, 2048)
            print(errBuff.value)

    def trig_stop(self):
        try:
            PyDAQmx.DAQmxStopTask(self.doHandle)
        except:
            errBuff = DAQmxTypes.create_string_buffer(b"", 2048)
            PyDAQmx.DAQmxGetExtendedErrorInfo(errBuff, 2048)
            print(errBuff.value)

    def trig_open_ao(self, do_sequences):
        do_channels, do_samples = do_sequences.shape
        try:
            PyDAQmx.DAQmxCfgSampClkTiming(self.doHandle, r'100kHzTimebase', self.frequency,
                                          DAQmxConstants.DAQmx_Val_Rising,
                                          DAQmxConstants.DAQmx_Val_FiniteSamps, do_samples)
            PyDAQmx.DAQmxWriteDigitalLines(self.doHandle, do_samples, False, -1,
                                           DAQmxConstants.DAQmx_Val_GroupByChannel,
                                           do_sequences.astype(np.uint8), None, None)
        except:
            errBuff = DAQmxTypes.create_string_buffer(b"", 2048)
            PyDAQmx.DAQmxGetExtendedErrorInfo(errBuff, 2048)
            print(errBuff.value)

    def trigger_sequence(self, ao_sequences, do_sequences):
        try:
            ao_channels, ao_samples = ao_sequences.shape
            do_channels, do_samples = do_sequences.shape
            PyDAQmx.DAQmxCfgImplicitTiming(self.counterHandle, DAQmxConstants.DAQmx_Val_ContSamps, do_samples)
            PyDAQmx.DAQmxCfgSampClkTiming(self.piezoHandle, r'Ctr0InternalOutput', self.frequency,
                                          DAQmxConstants.DAQmx_Val_Rising,
                                          DAQmxConstants.DAQmx_Val_FiniteSamps, ao_samples)
            PyDAQmx.DAQmxCfgSampClkTiming(self.doHandle, r'Ctr0InternalOutput', self.frequency,
                                          DAQmxConstants.DAQmx_Val_Rising,
                                          DAQmxConstants.DAQmx_Val_FiniteSamps, do_samples)
            PyDAQmx.DAQmxWriteAnalogF64(self.piezoHandle, ao_samples, False, -1,
                                        DAQmxConstants.DAQmx_Val_GroupByChannel,
                                        ao_sequences.astype(np.float64), None, None)
            PyDAQmx.DAQmxWriteDigitalLines(self.doHandle, do_samples, False, -1,
                                           DAQmxConstants.DAQmx_Val_GroupByChannel,
                                           do_sequences.astype(np.uint8), None, None)
        except:
            errBuff = DAQmxTypes.create_string_buffer(b"", 2048)
            PyDAQmx.DAQmxGetExtendedErrorInfo(errBuff, 2048)
            print(errBuff.value)

    def run_sequence(self):
        try:
            PyDAQmx.DAQmxStartTask(self.doHandle)
            PyDAQmx.DAQmxStartTask(self.piezoHandle)
            PyDAQmx.DAQmxStartTask(self.counterHandle)
            PyDAQmx.DAQmxWaitUntilTaskDone(self.doHandle, -1)
            PyDAQmx.DAQmxStopTask(self.counterHandle)
            PyDAQmx.DAQmxStopTask(self.doHandle)
            PyDAQmx.DAQmxStopTask(self.piezoHandle)
        except:
            errBuff = DAQmxTypes.create_string_buffer(b"", 2048)
            PyDAQmx.DAQmxGetExtendedErrorInfo(errBuff, 2048)
            print(errBuff.value)

    def trigger_scan(self, ao_sequences, do_sequences):
        try:
            ao_channels, ao_samples = ao_sequences.shape
            do_channels, do_samples = do_sequences.shape
            PyDAQmx.DAQmxCfgImplicitTiming(self.counterHandle, DAQmxConstants.DAQmx_Val_ContSamps, do_samples)
            PyDAQmx.DAQmxCfgSampClkTiming(self.galvoHandle, r'Ctr0InternalOutput', self.frequency,
                                          DAQmxConstants.DAQmx_Val_Rising,
                                          DAQmxConstants.DAQmx_Val_FiniteSamps, ao_samples)
            PyDAQmx.DAQmxCfgSampClkTiming(self.doHandle, r'Ctr0InternalOutput', self.frequency,
                                          DAQmxConstants.DAQmx_Val_Rising,
                                          DAQmxConstants.DAQmx_Val_FiniteSamps, do_samples)
            PyDAQmx.DAQmxWriteAnalogF64(self.galvoHandle, ao_samples, False, -1,
                                        DAQmxConstants.DAQmx_Val_GroupByChannel,
                                        ao_sequences.astype(np.float64), None, None)
            PyDAQmx.DAQmxWriteDigitalLines(self.doHandle, do_samples, False, -1,
                                           DAQmxConstants.DAQmx_Val_GroupByChannel,
                                           do_sequences.astype(np.uint8), None, None)
        except:
            errBuff = DAQmxTypes.create_string_buffer(b"", 2048)
            PyDAQmx.DAQmxGetExtendedErrorInfo(errBuff, 2048)
            print(errBuff.value)

    def run_scan(self):
        try:
            PyDAQmx.DAQmxStartTask(self.doHandle)
            PyDAQmx.DAQmxStartTask(self.galvoHandle)
            PyDAQmx.DAQmxStartTask(self.counterHandle)
            PyDAQmx.DAQmxWaitUntilTaskDone(self.doHandle, -1)
            PyDAQmx.DAQmxStopTask(self.counterHandle)
            PyDAQmx.DAQmxStopTask(self.doHandle)
            PyDAQmx.DAQmxStopTask(self.galvoHandle)
        except:
            errBuff = DAQmxTypes.create_string_buffer(b"", 2048)
            PyDAQmx.DAQmxGetExtendedErrorInfo(errBuff, 2048)
            print(errBuff.value)
