"""
Linear quadratic Gaussian (LQG) control based on Kalman filter
"""

import numpy as np
import tifffile as tf
from scipy.linalg import solve_continuous_are, svd


class DynamicControl:

    def __init__(self, n_states=16, n_inputs=97, n_outputs=16, logg=None):
        self.logg = logg or self.setup_logging()
        self.n_states = n_states
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self._initialize_control()

    @staticmethod
    def setup_logging():
        import logging
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
        return logging

    def _initialize_control(self):
        # Define the state-space model with appropriate dimensions
        self.A = 0.1 * np.random.randn(self.n_states, self.n_states)  # State transition matrix
        # Control matrix
        self.B = 0.1 * np.random.randn(self.n_states, self.n_inputs)
        self.B = self.B.swapaxes(0, 1)
        self.C = 0.1 * np.random.randn(self.n_outputs, self.n_states)  # Measurement matrix
        self.D = np.zeros((self.n_outputs, self.n_inputs))  # Direct transmission matrix
        # Initial state and covariance
        self.x = 0.1 * np.random.randn(self.n_states, 1)
        self.p = np.eye(self.n_states)
        # Desired state
        self.x_desired = np.zeros((self.n_states, 1))
        # Initial noise covariance matrices
        self.Q = np.eye(self.n_states)  # Process noise covariance
        self.R = np.eye(self.n_outputs)  # Measurement noise covariance
        # Initial cost matrices for LQR
        self.Q_lqr = np.eye(self.n_states)
        self.R_lqr = np.eye(self.n_inputs)
        self.K = self.lqr()
        self.step = 0.02
        self.residuals = []

    def compute_control(self, measurement, adaptive=True):
        # Run the Kalman filter for the current measurement
        self.x, self.p, y = self.kalman_filter(self.x, self.p, measurement)
        self.residuals.append(y.flatten())
        # Compute control input based on the current state estimate
        u = -self.K @ (self.x - self.x_desired)
        if adaptive:
            # Update Q and R
            self.update_covariances()
        return self.x, u

    # Kalman Filter Implementation
    def kalman_filter(self, x0, p0, measurement):
        # Prediction
        x_pre = self.A @ x0
        p_pre = self.A @ p0 @ self.A.T + self.Q
        # Update
        y = measurement - self.C @ x_pre
        s = self.C @ p_pre @ self.C.T + self.R
        k = p_pre @ self.C.T @ np.linalg.inv(s)
        x = x_pre + k @ y
        p = (np.eye(len(self.A)) - k @ self.C) @ p_pre
        return x, p, y

    # Linear Quadratic Regulator (LQR) Implementation
    def lqr(self):
        # Solve the Riccati equation
        x_ = solve_continuous_are(self.A, self.B, self.Q_lqr, self.R_lqr)
        # Compute the LQR gain
        k_ = np.linalg.inv(self.R_lqr) @ self.B.T @ x_
        return k_

    # Function to update Q and R
    def update_covariances(self, window_size=10, alpha=0.1):
        if len(self.residuals) >= window_size:
            residuals_window = np.array(self.residuals[-window_size:])
            sample_cov = np.cov(residuals_window, rowvar=False)
            self.R = sample_cov
            # Update Q using a heuristic
            self.Q = alpha * np.eye(self.Q.shape[0]) + (1 - alpha) * self.Q

    def get_simulated_measurement(self):
        return 0.1 * np.random.randn(self.n_outputs, 1)

    def subspace_identification(self, inputs, outputs):
        U = inputs.T  # Input matrix (n_inputs x N_steps)
        Y = outputs.T  # Output matrix (n_outputs x N_steps)
        N_steps = U.shape[1]
        # Form data matrices
        L = 2 * self.n_states
        H0 = np.hstack([Y[:, i:N_steps - L + i + 1] for i in range(L)])  # Hankel matrix of outputs
        H1 = np.hstack([U[:, i:N_steps - L + i + 1] for i in range(L)])  # Hankel matrix of inputs
        H = np.vstack((H0, H1))
        # SVD decomposition
        U, s, Vh = svd(H, full_matrices=False)
        U1 = U[:self.n_states, :]
        # Form state sequence
        X = U1 @ np.diag(np.sqrt(s))
        # Estimate A, B, C, D matrices
        X1 = X[:, :-1]
        X2 = X[:, 1:]
        Y1 = Y[:, L:N_steps]
        U1 = U[:, L:N_steps]
        # Solve for system matrices
        AB = np.linalg.lstsq(np.vstack((X1, U1)).T, X2.T, rcond=None)[0].T
        A = AB[:, :self.n_states]
        B = AB[:, self.n_states:]
        CD = np.linalg.lstsq(np.vstack((X1, U1)).T, Y1.T, rcond=None)[0].T
        C = CD[:, :self.n_states]
        D = CD[:, self.n_states:]
        return A, B, C, D


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Run the control loop
    N_steps = 128
    x_estimates = []
    measurements = []
    control_inputs = []

    ctrl = DynamicControl()

    for step in range(N_steps):
        z = ctrl.get_simulated_measurement()
        measurements.append(z)
        x_est, u = ctrl.compute_control(z, False)
        x_estimates.append(x_est)
        control_inputs.append(u)

    # Plot results
    plt.figure(figsize=(12, 8))
    i = 0
    plt.plot([m[i] for m in measurements], label=f'Measure {i + 1}')
    plt.plot([x[i, 0] for x in x_estimates], label=f'State {i + 1}')
    plt.plot([u[i] for u in control_inputs], label=f'Control {i + 1}')
    plt.legend()
    plt.show()
