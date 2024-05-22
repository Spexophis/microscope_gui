import time


class PID:
    """PID Controller
    Originally from IvPID. Author: Caner Durmusoglu
    """

    def __init__(self, P=0.2, I=0.0, D=0.0, current_time=None):
        self.Kp = P
        self.Ki = I
        self.Kd = D
        self.sample_time = 0.00
        self.current_time = current_time if current_time is not None else time.time()
        self.last_time = self.current_time
        self.clear()

    def clear(self):
        """Clears PID computations and coefficients"""
        self.SetPoint = 0.0
        self.PTerm = 0.0
        self.ITerm = 0.0
        self.DTerm = 0.0
        self.last_error = 0.0
        self.int_error = 0.0  # Windup Guard
        self.windup_guard = 20.0
        self.output = 0.0

    def update(self, feedback_value, current_time=None):
        """Calculates PID value for given reference feedback"""
        error = self.SetPoint - feedback_value
        self.current_time = current_time if current_time is not None else time.time()
        delta_time = self.current_time - self.last_time
        delta_error = error - self.last_error

        if delta_time >= self.sample_time:
            self.PTerm = self.Kp * error
            self.ITerm += error * delta_time

            # Anti-windup: Clamp ITerm to windup_guard
            self.ITerm = max(min(self.ITerm, self.windup_guard), -self.windup_guard)

            self.DTerm = delta_error / delta_time if delta_time > 0 else 0.0

            self.last_time = self.current_time
            self.last_error = error

            self.output = self.PTerm + (self.Ki * self.ITerm) + (self.Kd * self.DTerm)

    @property
    def kp(self):
        return self.Kp

    @kp.setter
    def kp(self, value):
        self.Kp = value

    @property
    def ki(self):
        return self.Ki

    @ki.setter
    def ki(self, value):
        self.Ki = value

    @property
    def kd(self):
        return self.Kd

    @kd.setter
    def kd(self, value):
        self.Kd = value

    @property
    def windup(self):
        return self.windup_guard

    @windup.setter
    def windup(self, value):
        self.windup_guard = value

    @property
    def sample_time(self):
        return self._sample_time

    @sample_time.setter
    def sample_time(self, value):
        self._sample_time = value


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import numpy as np


    class SimulatedSystem:
        def __init__(self, initial_value=0.0):
            self.value = initial_value

        def update(self, ctrl_sig):
            # Simple linear model with noise: the system's value changes based on the control signal
            self.value += np.random.normal(ctrl_sig, 0.01)

    # Initialize the system and PID controller
    system = SimulatedSystem(initial_value=np.random.normal(50., 0.01))
    pid = PID(P=0.8, I=0.6, D=0.0)
    pid.SetPoint = 50.0

    # Lists to store values for plotting
    time_values = []
    position_values = []
    position_signal = []
    control_values = []

    # Simulate for 100 time steps
    for t in range(1, 100):
        current_position = system.value
        pid.update(current_position)

        control_signal = pid.output
        system.update(control_signal)

        # Store values for plotting
        time_values.append(t)
        position_values.append(current_position)
        position_signal.append(system.value)
        control_values.append(control_signal)

        # Simulate a time delay (for real-time systems)
        time.sleep(0.1)

    # Plot the results
    plt.figure(figsize=(12, 6))

    # Plot temperature
    plt.subplot(3, 1, 1)
    plt.plot(time_values, position_values, label="Position")
    plt.axhline(y=pid.SetPoint, color='r', linestyle='--', label="Set Point")
    plt.xlabel("Time")
    plt.ylabel("Position")
    plt.legend()
    plt.title("Position Control using PID")

    # Plot position signal
    plt.subplot(3, 1, 2)
    plt.plot(time_values, position_signal, label="Position Signal")
    plt.xlabel("Time")
    plt.ylabel("Position Signal")
    plt.legend()

    # Plot control signal
    plt.subplot(3, 1, 3)
    plt.plot(time_values, control_values, label="Control Signal")
    plt.xlabel("Time")
    plt.ylabel("Control Signal")
    plt.legend()

    plt.tight_layout()
    plt.show()
