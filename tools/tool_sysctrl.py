"""
Linear quadratic Gaussian (LQG) control based on Kalman filter
"""

import numpy as np
import tifffile as tf
from scipy.linalg import solve_discrete_are


class DynamicControl:

    def __init__(self, variable_number, logg=None):
        self.logg = logg or self.setup_logging()
        self.v_n = variable_number
        self._setup_control()

    @staticmethod
    def setup_logging():
        import logging
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
        return logging

    def _setup_control(self):
        # System parameters for a multi-variable system
        self.A = np.eye(self.v_n)  # System dynamics matrix
        self.B = tf.imread(
            r"C:\Users\ruizhe.lin\Documents\data\dm_files\bax513\influence_function_zonal_20240219.tif")  # Control input matrix
        self.H = np.eye(self.v_n)  # Measurement matrix
        self.Q = np.eye(self.v_n) * 0.001  # Process noise covariance
        self.R = np.eye(self.v_n) * 0.01  # Measurement noise covariance

        # LQR parameters for multi-variable system
        self.Q_lqr = np.eye(self.v_n) * 0.5  # State weight in the cost function
        self.R_lqr = np.eye(97) * 0.5  # Control weight in the cost function

        # Compute LQR gain K using the solution of Riccati equation for multi-variable system
        self.P = solve_discrete_are(self.A, self.B, self.Q_lqr, self.R_lqr)
        self.K = np.linalg.inv(self.B.T @ self.P @ self.B + self.R_lqr) @ (self.B.T @ self.P @ self.A)

    def kalman_filter_multi(self, est_x, est_p, u, z):
        """
        Kalman Filter function for multi-variable system
        u: control
        z: measurement
        """
        # Prediction
        prd_x = self.A @ est_x + self.B @ u
        prd_p = self.A @ est_p @ self.A.T + self.Q
        # Measurement Update
        k_gain = prd_p @ self.H.T @ np.linalg.inv(self.H @ prd_p @ self.H.T + self.R)  # Kalman gain
        est_new_x = prd_x + k_gain @ (z - self.H @ prd_x)
        est_new_p = (np.eye(self.v_n) - k_gain @ self.H) @ prd_p
        return est_new_x, est_new_p


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Time simulation parameters
    time_steps = 512  # Total number of time steps to simulate
    disturbance_interval = 64  # Apply a disturbance every 'disturbance_interval' steps

    v_n = 16
    dc = DynamicControl(v_n)
    # Initialize arrays to store simulation data
    x_est_history = np.zeros((v_n, time_steps))
    control_history = np.zeros((v_n, time_steps))

    # Initial state and covariance for multi-variable system
    x_est = np.zeros((v_n, 1))  # Initial estimate of state
    p_est = np.eye(v_n)  # Initial estimate of error covariance

    # Simulation loop
    for t in range(time_steps):
        # Apply disturbance at specified intervals
        if t > 0 and t % disturbance_interval == 0:
            disturbance = np.random.normal(0, 1, (v_n, 1))  # Random disturbance
        else:
            disturbance = np.zeros((v_n, 1))  # No disturbance

        # Simulate measurement with disturbance and noise
        measurement = x_est + disturbance + np.random.normal(0, np.sqrt(dc.R[0, 0]), (v_n, 1))

        # Calculate control based on LQR and current state estimate
        control = -dc.K @ x_est

        # Update state estimate with Kalman Filter
        x_est, p_est = dc.kalman_filter_multi(x_est, p_est, control, measurement)

        # Store history for plotting
        x_est_history[:, t] = x_est.ravel()
        control_history[:, t] = control.ravel()

    # Plotting
    time = np.arange(time_steps)
    fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    for i in range(4):
        axs[i].plot(time, x_est_history[i, :], label=f'State {i + 1}')
        axs[i].set_ylabel(f'State {i + 1}')
        axs[i].legend(loc="upper right")
        axs[i].grid(True)

    axs[-1].set_xlabel('Time Step')
    plt.tight_layout()
    plt.show()
