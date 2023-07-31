import ctypes as ct
import sys

sys.path.append(r'C:\Program Files\Mad City Labs\MicroDrive')

micro_dll = r'C:\Program Files\Mad City Labs\MicroDrive\MicroDrive.dll'


class MCLMicroDrive:

    def __init__(self):
        super().__init__()

        self.errorDictionary = {0: 'MCL_SUCCESS',
                                -1: 'MCL_GENERAL_ERROR',
                                -2: 'MCL_DEV_ERROR',
                                -3: 'MCL_DEV_NOT_ATTACHED',
                                -4: 'MCL_USAGE_ERROR',
                                -5: 'MCL_DEV_NOT_READY',
                                -6: 'MCL_ARGUMENT_ERROR',
                                -7: 'MCL_INVALID_AXIS',
                                -8: 'MCL_INVALID_HANDLE'}

        # Dictionary to know the axis limit returrns. Dicitionary saves [axis, forward (1) or backward (-1), description]
        self.motorLimits = [[1, -1, 'Axis 1 reverse limit'],  # 126 <-> '1111110' <-> position 0
                            [1, 1, 'Axis 1 forward limit'],  # 125 <-> '1111101' <-> position 1
                            [2, -1, 'Axis 2 reverse limit'],  # 123 <-> '1111011' <-> position 2
                            [2, 1, 'Axis 2 forward limit'],  # 119 <-> '1110111' <-> position 3
                            [3, -1, 'Axis 3 reverse limit'],  # 111 <-> '1101111' <-> position 4
                            [3, 1, 'Axis 3 forward limit']]  # 095 <-> '1011111' <-> position 5

        # Load the DLL
        self.mcldeck = ct.cdll.LoadLibrary(micro_dll)
        # Release existing handles
        self.mcldeck.MCL_ReleaseAllHandles()
        # Connect to the instrument and create a handle
        self.handle = self.mcldeck.MCL_InitHandle()  # Handle number is assigned, which is a positive integer
        # Check if connection was successful
        if self.handle > 0:
            print(
                'Connected to MadDeck SN: ' + str(self.mcldeck.MCL_GetSerialNumber(self.handle)) + '\nWith handle: ' + str(
                    self.handle))
            encoderResolution_temp = ct.pointer(ct.c_double())
            stepSize_temp = ct.pointer(ct.c_double())
            maxVelocity_temp = ct.pointer(ct.c_double())
            maxVelocityTwoAxis_temp = ct.pointer(ct.c_double())
            maxVelocityThreeAxis_temp = ct.pointer(ct.c_double())
            minVelocity_temp = ct.pointer(ct.c_double())
            self.mcldeck.MCL_MDInformation(encoderResolution_temp, stepSize_temp, maxVelocity_temp, maxVelocityTwoAxis_temp,
                                       maxVelocityThreeAxis_temp, minVelocity_temp, self.handle)
            self.encoderResolution = encoderResolution_temp.contents.value
            self.stepSize = stepSize_temp.contents.value
            self.maxVelocity = maxVelocity_temp.contents.value
            self.maxVelocityTwoAxis = maxVelocityTwoAxis_temp.contents.value
            self.maxVelocityThreeAxis = maxVelocityThreeAxis_temp.contents.value
            self.minVelocity = minVelocity_temp.contents.value
            del encoderResolution_temp
            del stepSize_temp
            del maxVelocity_temp
            del maxVelocityTwoAxis_temp
            del maxVelocityThreeAxis_temp
            del minVelocity_temp
            # Set standard minimum and maximum velocity
            self.velocityMin = self.minVelocity  # mm/s
            self.velocityMax = self.maxVelocity  # mm/s
            self.totalScanRange = 23  # mm
        else:
            print('MadDeck Connection failed.')

    def __del__(self):
        pass

    def close(self):
        """
        Closes the connection by releasing the handle.
        """
        self.stopMoving()
        self.mcldeck.MCL_ReleaseHandle(self.handle)
        print('Deck Handle released.')

    def getInfo(self):
        """
        Returns info about the motors:
            encoderResolution = 0.05
            stepSize =  9.??e-5??
            maxVelocity = 4
            maxVelocityTwoAxis = ??
            maxVelocityThreeAxis = 3
            minVelocity = 0.019??
        """
        if self.handle > 0:
            # Device attached
            print('Device attached: ' + str(self.mcldeck.MCL_DeviceAttached(ct.c_uint(500), self.handle)))
            # Serial number
            print('SN: ' + str(self.mcldeck.MCL_GetSerialNumber(self.handle)))
            # Product ID:
            PID = ct.pointer(ct.c_ushort())
            self.mcldeck.MCL_GetProductID(PID, self.handle)
            print('PID: ' + str(PID.contents.value))
            # Encoder, StepSize and Velocities
            encoderResolution = ct.pointer(ct.c_double())
            stepSize = ct.pointer(ct.c_double())
            maxVelocity = ct.pointer(ct.c_double())
            maxVelocityTwoAxis = ct.pointer(ct.c_double())
            maxVelocityThreeAxis = ct.pointer(ct.c_double())
            minVelocity = ct.pointer(ct.c_double())
            self.mcldeck.MCL_MDInformation(encoderResolution, stepSize, maxVelocity, maxVelocityTwoAxis,
                                       maxVelocityThreeAxis, minVelocity, self.handle)
            print('encoderResolution: ' + str(encoderResolution.contents.value))
            print('stepSize: ' + str(stepSize.contents.value))
            print('maxVelocity: ' + str(maxVelocity.contents.value))
            print('maxVelocityTwoAxis: ' + str(maxVelocityTwoAxis.contents.value))
            print('maxVelocityThreeAxis: ' + str(maxVelocityThreeAxis.contents.value))
            print('minVelocity: ' + str(minVelocity.contents.value))
        else:
            print('Invalid handle. No device is connncted.')

    def _getStatus(self):  # Internal function to get the error number
        status_temp = ct.pointer(ct.c_ushort())
        self.mcldeck.MCL_MDStatus(status_temp, self.handle)
        result_temp = status_temp.contents.value
        del status_temp
        return result_temp

    def getStatus(self):
        """
        Returns a list of motors that are out of bounds (reverse of forward limit)
        [axis, forward (1) / reverse (-1), description]
        [1,-1,'Axis 1 reverse limit']
        [1, 1,'Axis 1 forward limit']
        [2,-1,'Axis 2 reverse limit']
        [2, 1,'Axis 2 forward limit']
        [3,-1,'Axis 3 reverse limit']
        [3, 1,'Axis 3 forward limit']
        """
        status = self._getStatus()
        errorsLimit = []
        for i, b in enumerate(bin(status)[:1:-1]):
            if b == '0':
                errorsLimit.append(self.motorLimits[i])
        if errorsLimit == []:  # If no limit is detected, we add the All ok line
            errorsLimit.append([0, 0, 'All ok'])
        return errorsLimit

    def wait(self):
        """
        This function takes approximately 10ms if the motors are not moving.
        """
        errorNumber = self.mcldeck.MCL_MicroDriveWait(self.handle)
        if errorNumber != 0:
            print('Error while waiting: ' + self.errorDictionary[errorNumber])

    # Start: Internal move functions that have no error handling and should be used with caution and only if one is familiar with the motors

    def _moveRelativeAxis(self, axis, distance, velocity=1.5):
        errorCode = self.mcldeck.MCL_MDMove(ct.c_uint(axis), ct.c_double(velocity), ct.c_double(distance), self.handle)
        # self.wait()
        return errorCode

    # End: Internal move functions that have no error handling and should be used with caution and only if one is familiar with the motors

    def moveRelativeAxis(self, axis, distance, velocity=1.5):
        """
        Moves a single axis by distance with velocity.
        """
        # Check the given velocity
        if velocity > self.velocityMax:
            print('Given velocity is too high. Velocity is set to maximum value.')
            velocity = self.velocityMax
        elif velocity < self.velocityMin:
            print('Given velocity is too high. Velocity is set to minimum value.')
            velocity = self.velocityMin
        # Move the stage    
        errorNumber = self._moveRelativeAxis(axis, distance, velocity)
        # Check for error
        if errorNumber != 0:
            print('Error while moving axis ' + str(axis) + ': ' + self.errorDictionary[errorNumber])
        # Check if motors moved out of bounds
        # status = self.getStatus()
        # if status[0] != [0,0,'All ok']:
        #     print('Motor moved out of bounds: ' + str([temp[2] for temp in status]))

    def isMoving(self):
        """
        Checks if motors are moving.
        This function takes approximately 20ms.
        """
        isMoving = ct.pointer(ct.c_int())
        self.mcldeck.MCL_MicroDriveMoveStatus(isMoving, self.handle)
        result_temp = isMoving.contents.value
        del isMoving
        return result_temp

    def stopMoving(self):
        """
        Stops motors from moving.
        """
        status = ct.pointer(ct.c_ushort())
        errorNumber = self.mcldeck.MCL_MDStop(status, self.handle)
        del status
        if errorNumber != 0:
            print('Error while stopping device: ' + self.errorDictionary[errorNumber])

    def getPositionStepsTakenAxis(self, axis):
        microSteps = ct.pointer(ct.c_int())
        errorNumber = self.mcldeck.MCL_MDCurrentPositionM(ct.c_int(axis), microSteps, self.handle)
        # Check for error
        if errorNumber != 0:
            print('Error reading the position of axis' + str(axis) + ': ' + self.errorDictionary[errorNumber])
        position_temp = microSteps.contents.value * self.stepSize
        del microSteps
        return position_temp
