"""
Linear quadratic Gaussian (LQG) control based on Kalman filter
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve_discrete_are

# System parameters for a 4-variable system
A = np.eye(4)  # System dynamics matrix
B = np.eye(4)  # Control input matrix
H = np.eye(4)  # Measurement matrix
Q = np.eye(4) * 0.01  # Process noise covariance
R = np.eye(4) * 0.01  # Measurement noise covariance

# LQR parameters for 4-variable system
Q_lqr = np.eye(4)  # State weight in the cost function
R_lqr = np.eye(4) * 0.01  # Control weight in the cost function

# Compute LQR gain K using the solution of Riccati equation for 4-variable system
P = solve_discrete_are(A, B, Q_lqr, R_lqr)
K = np.linalg.inv(B.T @ P @ B + R_lqr) @ (B.T @ P @ A)


# Adjust Kalman Filter function for 4-variable system
def kalman_filter_multi(x_est, P_est, u, z):
    # Prediction
    x_pred = A @ x_est + B @ u
    P_pred = A @ P_est @ A.T + Q

    # Measurement Update
    K_gain = P_pred @ H.T @ np.linalg.inv(H @ P_pred @ H.T + R)  # Kalman gain
    x_est_new = x_pred + K_gain @ (z - H @ x_pred)
    P_est_new = (np.eye(4) - K_gain @ H) @ P_pred

    return x_est_new, P_est_new


# Time simulation parameters
time_steps = 1024  # Total number of time steps to simulate
disturbance_interval = 64  # Apply a disturbance every 'disturbance_interval' steps

# Initialize arrays to store simulation data
x_est_history = np.zeros((4, time_steps))
control_history = np.zeros((4, time_steps))

# Initial state and covariance for 4-variable system
x_est = np.zeros((4, 1))  # Initial estimate of state (4-dimensional)
P_est = np.eye(4)  # Initial estimate of error covariance (4x4 matrix)
print(P_est)

# Simulation loop
for t in range(time_steps):
    # Apply disturbance at specified intervals
    if t > 0 and t % disturbance_interval == 0:
        disturbance = np.random.normal(0, 2, (4, 1))  # Random disturbance
    else:
        disturbance = np.zeros((4, 1))  # No disturbance

    # Simulate measurement with disturbance and noise
    measurement = x_est + disturbance + np.random.normal(0, np.sqrt(R[0, 0]), (4, 1))

    # Calculate control based on LQR and current state estimate
    control = -K @ x_est

    # Update state estimate with Kalman Filter
    x_est, P_est = kalman_filter_multi(x_est, P_est, control, measurement)

    # Store history for plotting
    x_est_history[:, t] = x_est.ravel()
    control_history[:, t] = control.ravel()

print(P_est)

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
