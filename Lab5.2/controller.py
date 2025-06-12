import numpy as np

# PID gains for x, y, z, yaw
PID = {
    'Kp': np.array([1.6562, 0.9747, 2.9853, 2.8345]),
    'Ki': np.array([0.0785, 0.0305, 0.0802, 0.0292]),
    'Kd': np.array([0.5273, 0.2283, 0.0192, 0.0000])
}

initialized = False
previous_target = None
I = np.zeros(4)
P_pre = np.zeros(4)

# New: Storage for logging
log_data = []

# New: Log save filename
log_filename = 'controller_log.txt'

def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def to_body_frame(v, yaw):
    cos_yaw = np.cos(-yaw)
    sin_yaw = np.sin(-yaw)
    x_body = cos_yaw * v[0] - sin_yaw * v[1]
    y_body = sin_yaw * v[0] + cos_yaw * v[1]
    return np.array([x_body, y_body, v[2]])

def controller(state, target_pos, dt):
    global initialized, previous_target, PID, I, P_pre, log_data

    # Initialize the program at start and target changed
    if not initialized or previous_target is None or not np.allclose(previous_target, target_pos, atol=1e-4):
        initialized = True
        if previous_target is None:
            previous_target = np.zeros(4)
        previous_target = target_pos
        I[:] = 0.0
        P_pre[:] = 0.0

    # Compute error vector
    current_status = np.hstack((state[:3], state[5]))
    error = target_pos - current_status
    error[3] = wrap_angle(error[3])

    # PID matrix calculation
    I += error * dt
    D = (error - P_pre) / dt
    P_pre[:] = error
    output = PID['Kp'] * error + PID['Ki'] * I + PID['Kd'] * D

    # Transform velocity to body frame
    vel_cmd = to_body_frame(output[:3], current_status[3])
    yaw_rate_cmd = np.clip(output[3], -np.radians(90), np.radians(90))

    # Log current frame (only state, target, dt, vel_cmd)
    frame_log = {
        'state': list(state),
        'target_pos': list(target_pos),
        'dt': dt,
        'vel_cmd': list(vel_cmd)
    }
    log_data.append(frame_log)

    # Every 100 frames, append to file
    if len(log_data) % 100 == 0:
        append_log(log_filename, log_data)
        log_data = []

    return (*vel_cmd, yaw_rate_cmd)

def append_log(filename, data):
    with open(filename, 'a') as f:
        for entry in data:
            f.write(f"State: {entry['state']}, Target: {entry['target_pos']}, dt: {entry['dt']}, Vel_cmd: {entry['vel_cmd']}\n")

def save_log(filename):
    global log_data
    append_log(filename, log_data)
    log_data = []
