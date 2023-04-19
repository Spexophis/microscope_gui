import numpy as np
import PyDAQmx
from PyDAQmx.DAQmxFunctions import *
from PyDAQmx.DAQmxConstants import *
from PyDAQmx.DAQmxTypes import *


class NIDAQ:

    def __init__(self, frequency=100000, duty_cycle=0.5):
        super().__init__()

        self.frequency = frequency
        self.duty_cycle = duty_cycle

        try:
            self.counterHandle = TaskHandle(0)
            DAQmxCreateTask("", byref(self.counterHandle))
            DAQmxCreateCOPulseChanFreq(self.counterHandle, 'Dev1/ctr0', '', DAQmx_Val_Hz, DAQmx_Val_Low, 0.0,
                                       self.frequency, self.duty_cycle)

            self.doHandle = TaskHandle(0)
            DAQmxCreateTask("", byref(self.doHandle))
            DAQmxCreateDOChan(self.doHandle, 'Dev1/port0/line0:5', '', DAQmx_Val_ChanPerLine)

            self.piezoHandle = TaskHandle(0)
            DAQmxCreateTask("", byref(self.piezoHandle))
            DAQmxCreateAOVoltageChan(self.piezoHandle, 'Dev1/ao0:1', '', 0.0, 10.0, DAQmx_Val_Volts, None)

            self.galvoHandle = TaskHandle(0)
            DAQmxCreateTask("", byref(self.galvoHandle))
            DAQmxCreateAOVoltageChan(self.galvoHandle, 'Dev1/ao2:3', '', -10.0, 10.0, DAQmx_Val_Volts, None)

            self.rpHandle = TaskHandle(0)
            DAQmxCreateTask("", byref(self.rpHandle))
            DAQmxCreateAIVoltageChan(self.rpHandle, 'Dev1/ai0:2', '', DAQmx_Val_RSE, 0.0, 10.0, DAQmx_Val_Volts, None)

        except:
            errBuff = create_string_buffer(b"", 2048)
            PyDAQmx.DAQmxGetExtendedErrorInfo(errBuff, 2048)
            print(errBuff.value)

    def __del__(self):
        self.Reset_daq()
        del self.counterHandle, self.doHandle, self.piezoHandle, self.galvoHandle, self.rpHandle

    def Reset_daq(self):
        try:
            PyDAQmx.DAQmxResetDevice("Dev1")
            print('DAQ board reset')
        except:
            errBuff = create_string_buffer(b"", 2048)
            PyDAQmx.DAQmxGetExtendedErrorInfo(errBuff, 2048)
            print(errBuff.value)

    def set_xyz(self, pos_x, pos_y, pos_z):
        try:
            DAQmxWriteAnalogF64(self.piezoHandle, 1, True, 10.0, DAQmx_Val_GroupByChannel,
                                np.array([pos_x, pos_y], dtype=np.float64), None, None)
            DAQmxStartTask(self.piezoHandle)
            DAQmxStopTask(self.piezoHandle)
        except:
            errBuff = create_string_buffer(b"", 2048)
            PyDAQmx.DAQmxGetExtendedErrorInfo(errBuff, 2048)
            print(errBuff.value)

    def get_xyz(self):
        try:
            data = np.zeros((3,), dtype=np.float64)
            read = int32()
            DAQmxStartTask(self.rpHandle)
            DAQmxReadAnalogF64(self.rpHandle, 1, 10.0, DAQmx_Val_GroupByChannel, data, 3, byref(read), None)
            DAQmxStopTask(self.rpHandle)
            return data
        except:
            errBuff = create_string_buffer(b"", 2048)
            PyDAQmx.DAQmxGetExtendedErrorInfo(errBuff, 2048)
            print(errBuff.value)

    def set_galvo(self, axis_x, axis_y):
        try:
            DAQmxWriteAnalogF64(self.galvoHandle, 1, True, 10.0, DAQmx_Val_GroupByChannel,
                                np.array([axis_x, axis_y], dtype=np.float64), None, None)
            DAQmxStartTask(self.galvoHandle)
            DAQmxStopTask(self.galvoHandle)
        except:
            errBuff = create_string_buffer(b"", 2048)
            PyDAQmx.DAQmxGetExtendedErrorInfo(errBuff, 2048)
            print(errBuff.value)

    def scan_galvo(self, galvo_xy):
        channels, samples = galvo_xy.shape
        try:
            DAQmxCfgSampClkTiming(self.galvoHandle, r'100kHzTimebase', self.frequency, DAQmx_Val_Rising,
                                  DAQmx_Val_FiniteSamps, samples)
            DAQmxWriteAnalogF64(self.galvoHandle, samples, False, -1, DAQmx_Val_GroupByChannel,
                                galvo_xy.astype(np.float64), None, None)
            DAQmxStartTask(self.galvoHandle)
        except:
            errBuff = create_string_buffer(b"", 2048)
            PyDAQmx.DAQmxGetExtendedErrorInfo(errBuff, 2048)
            print(errBuff.value)

    def stop_galvo(self):
        try:
            DAQmxStopTask(self.galvoHandle)
        except:
            errBuff = create_string_buffer(b"", 2048)
            PyDAQmx.DAQmxGetExtendedErrorInfo(errBuff, 2048)
            print(errBuff.value)

    def Trig_open(self, do_sequences):
        do_channels, do_samples = do_sequences.shape
        try:
            DAQmxCfgSampClkTiming(self.doHandle, r'100kHzTimebase', self.frequency, DAQmx_Val_Rising,
                                  DAQmx_Val_ContSamps, do_samples)
            DAQmxWriteDigitalLines(self.doHandle, do_samples, False, -1, DAQmx_Val_GroupByChannel,
                                   do_sequences.astype(np.uint8), None, None)
        except:
            errBuff = create_string_buffer(b"", 2048)
            PyDAQmx.DAQmxGetExtendedErrorInfo(errBuff, 2048)
            print(errBuff.value)

    def Trig_run(self):
        try:
            DAQmxStartTask(self.doHandle)
        except:
            errBuff = create_string_buffer(b"", 2048)
            PyDAQmx.DAQmxGetExtendedErrorInfo(errBuff, 2048)
            print(errBuff.value)

    def Trig_stop(self):
        try:
            DAQmxStopTask(self.doHandle)
        except:
            errBuff = create_string_buffer(b"", 2048)
            PyDAQmx.DAQmxGetExtendedErrorInfo(errBuff, 2048)
            print(errBuff.value)

    def Trig_open_ao(self, do_sequences):
        do_channels, do_samples = do_sequences.shape
        try:
            DAQmxCfgSampClkTiming(self.doHandle, r'100kHzTimebase', self.frequency, DAQmx_Val_Rising,
                                  DAQmx_Val_FiniteSamps, do_samples)
            DAQmxWriteDigitalLines(self.doHandle, do_samples, False, -1, DAQmx_Val_GroupByChannel,
                                   do_sequences.astype(np.uint8), None, None)
        except:
            errBuff = create_string_buffer(b"", 2048)
            PyDAQmx.DAQmxGetExtendedErrorInfo(errBuff, 2048)
            print(errBuff.value)

    def Trigger_sequence(self, ao_sequences, do_sequences):
        try:
            ao_channels, ao_samples = ao_sequences.shape
            do_channels, do_samples = do_sequences.shape
            DAQmxCfgImplicitTiming(self.counterHandle, DAQmx_Val_ContSamps, do_samples)
            DAQmxCfgSampClkTiming(self.piezoHandle, r'Ctr0InternalOutput', self.frequency, DAQmx_Val_Rising,
                                  DAQmx_Val_FiniteSamps, ao_samples)
            DAQmxCfgSampClkTiming(self.doHandle, r'Ctr0InternalOutput', self.frequency, DAQmx_Val_Rising,
                                  DAQmx_Val_FiniteSamps, do_samples)
            DAQmxWriteAnalogF64(self.piezoHandle, ao_samples, False, -1, DAQmx_Val_GroupByChannel,
                                ao_sequences.astype(np.float64), None, None)
            DAQmxWriteDigitalLines(self.doHandle, do_samples, False, -1, DAQmx_Val_GroupByChannel,
                                   do_sequences.astype(np.uint8), None, None)
        except:
            errBuff = create_string_buffer(b"", 2048)
            PyDAQmx.DAQmxGetExtendedErrorInfo(errBuff, 2048)
            print(errBuff.value)

    def Run_sequence(self):
        try:
            DAQmxStartTask(self.doHandle)
            DAQmxStartTask(self.piezoHandle)
            DAQmxStartTask(self.counterHandle)
            DAQmxWaitUntilTaskDone(self.doHandle, -1)
            DAQmxStopTask(self.counterHandle)
            DAQmxStopTask(self.doHandle)
            DAQmxStopTask(self.piezoHandle)
        except:
            errBuff = create_string_buffer(b"", 2048)
            PyDAQmx.DAQmxGetExtendedErrorInfo(errBuff, 2048)
            print(errBuff.value)
