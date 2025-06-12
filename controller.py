import numpy as np

# === PID gains for each axis: x, y, z, and yaw ===
# Parameters come from the models of reinforcement learning
# Runing pid_rl/run_best.py can change the parameters to the best in models
PID = {
    'Kp': np.array([1.9379, 1.3628, 3.6606, 2.8345]),
    'Ki': np.array([0.0536, 0.0376, 0.0992, 0.0292]),
    'Kd': np.array([0.5225, 0.3717, 0.0000, 0.0000])
}

# === Controller state ===
initialized = False
previous_target = None
I = np.zeros(4)      # Integral terms for x, y, z, yaw
P_pre = np.zeros(4)  # Previous proportional errors
t = 0                # Timer for periodic reset

# === Wrap yaw angle to [-π, π] ===
def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

# === Convert global velocity to body frame ===
def to_body_frame(v, yaw):
    cos_yaw = np.cos(-yaw)
    sin_yaw = np.sin(-yaw)
    x_body = cos_yaw * v[0] - sin_yaw * v[1]
    y_body = sin_yaw * v[0] + cos_yaw * v[1]
    return np.array([x_body, y_body, v[2]])

# === Main PID controller ===
def controller(state, target_pos, dt):
    global initialized, previous_target, PID, I, P_pre, t

    # === Reset controller state when first called or when target changes ===
    if not initialized or previous_target is None or not np.allclose(previous_target, target_pos):
        initialized = True
        previous_target = target_pos
        I[:] = 0.0
        P_pre[:] = 0.0
        t = 0

    # === Clear integral term every 3 seconds to reduce steady-state overshoot ===
    t += dt
    if t >= 3:
        t = 0
        I[:] = 0.0
        #print('erase')  # For debugging: indicate integral reset

    # === Get current position and yaw from state ===
    current_status = np.hstack((state[:3], state[5]))

    # === Compute control error ===
    error = target_pos - current_status
    error[3] = wrap_angle(error[3])  # Wrap yaw error to [-π, π]

    # === PID computation ===
    I += error * dt
    D = (error - P_pre) / dt
    P_pre[:] = error

    # Combine P, I, D terms
    output = PID['Kp'] * error + PID['Ki'] * I + PID['Kd'] * D

    # === Convert velocity command to body frame (for drones or mobile robots) ===
    vel_cmd = to_body_frame(output[:3], current_status[3])

    # === Clip output to avoid excessive commands ===
    vel_cmd = np.clip(vel_cmd, -1.0, 1.0)  # Limit linear velocities
    yaw_rate_cmd = np.clip(output[3], -np.radians(90), np.radians(90))  # Limit yaw rate

    return (*vel_cmd, yaw_rate_cmd)  # Return 4 control values: vx, vy, vz, yaw_rate
