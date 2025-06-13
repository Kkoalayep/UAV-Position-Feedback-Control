import numpy as np

# Global state variables
prev_pos = None
prev_u = np.zeros(3)
active_target = None
target_phase = "POSITION"
processed_target_count = 0
phase_timeout = 0

# Stability counters
position_stable_counter = 0
yaw_stable_counter = 0
required_stable_counts = 10  # [Param 1] Can be increased to 12-15 to let the drone hover longer
position_switching = False

def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def compute_lqr_gain(A, B, Q, R):
    P = Q
    for _ in range(100):
        K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
        P_new = Q + A.T @ P @ A - A.T @ P @ B @ K
        if np.allclose(P, P_new):
            break
        P = P_new
    return K

def controller(state, target_pos, dt):
    global prev_pos, prev_u, active_target, target_phase, processed_target_count, phase_timeout
    global position_stable_counter, yaw_stable_counter, position_switching

    pos = np.array(state[0:3])
    yaw = float(state[5])

    if prev_pos is None:
        vel = np.zeros(3)
    else:
        vel = (pos - prev_pos) / dt
    prev_pos = pos.copy()

    if active_target is None:
        active_target = tuple(target_pos)
        processed_target_count += 1
        target_phase = "POSITION"
        phase_timeout = 0
        prev_u = np.zeros(3)
        position_stable_counter = 0
        yaw_stable_counter = 0
        position_switching = False

    if tuple(target_pos) != active_target and not position_switching:
        active_target = tuple(target_pos)
        processed_target_count += 1
        target_phase = "POSITION"
        phase_timeout = 0
        prev_u = np.zeros(3)
        position_stable_counter = 0
        yaw_stable_counter = 0

    pos_error = np.array(pos) - np.array(target_pos[0:3])
    yaw_error = wrap_to_pi(float(target_pos[3]) - yaw)

    xy_error = np.linalg.norm(pos_error[:2])
    z_error = abs(pos_error[2])
    vel_norm = np.linalg.norm(vel)

    if target_phase == "POSITION":
        if xy_error < 0.05 and z_error < 0.05 and vel_norm < 0.03:  # [Param 2] Can be adjusted to 0.04/0.04/0.02 for higher precision
            position_stable_counter += 1
        else:
            position_stable_counter = 0
    elif target_phase == "YAW":
        if abs(yaw_error) < np.deg2rad(5):  # [Param 2] Can be adjusted to np.deg2rad(4)
            yaw_stable_counter += 1
        else:
            yaw_stable_counter = 0

    phase_timeout += dt

    if target_phase == "POSITION" and position_stable_counter >= required_stable_counts:
        target_phase = "YAW"
        phase_timeout = 0
        position_stable_counter = 0

    if phase_timeout > 15.0:
        if target_phase == "POSITION":
            target_phase = "YAW"
            position_stable_counter = 0
        phase_timeout = 0

    if target_phase == "YAW" and yaw_stable_counter >= required_stable_counts:
        yaw_stable_counter = 0
        position_switching = True

    if target_phase == "POSITION":
        A = np.zeros((6, 6))
        A[0:3, 0:3] = np.eye(3)
        A[0:3, 3:6] = np.eye(3) * dt
        A[3:6, 3:6] = np.eye(3)

        B = np.zeros((6, 3))
        B[3:6, 0:3] = np.eye(3) * dt

        state_vec = np.concatenate([pos, vel])
        target_state = np.concatenate([np.array(target_pos[0:3]), np.zeros(3)])

        # [Param 3] LQR weight matrices - key parameters
        Q = np.diag([12.0, 12.0, 25.0, 4.0, 4.0, 5.0])  # Z-axis weight (3rd value) can be tuned to 18-22
        R = np.diag([0.2, 0.2, 0.15])  # Z-axis control cost (3rd value) can be tuned to 0.12-0.18
        K = compute_lqr_gain(A, B, Q, R)

        error = state_vec - target_state
        dist = np.linalg.norm(pos_error)
        scaling = min(1.0, dist / 0.5)
        
        # [Param 4] Z-axis scaling factor - key parameter
        z_scaling = min(1.0, abs(pos_error[2]) / 0.3) * 1.5  # The second multiplier can be adjusted to 1.3-1.7
        u = -scaling * (K @ error)
        u[2] = -z_scaling * (K @ error)[2]  # Handle Z-axis separately
        
        # [Param 5] Ascending force boost - key parameter
        if pos_error[2] > 0.05:  # Threshold can be adjusted to 0.04-0.06
            u[2] *= 1.3  # Boost coefficient can be tuned to 1.2-1.4
        
        # [Param 6] Max velocity limit - key parameter
        max_v = np.array([0.4, 0.4, 0.15])  # Z-axis max speed can be adjusted to 0.12-0.18
        v_cmd = np.clip(u, -max_v, max_v)

        yaw_rate_cmd = 0.0
    else:  # YAW phase
        v_cmd = np.zeros(3)
        
        # [Param 7] Position control during yaw phase - key parameter
        if abs(pos_error[2]) > 0.04:  # Threshold can be adjusted to 0.06-0.10
            v_cmd[2] = -0.15 * np.sign(pos_error[2])  # Coefficient can be adjusted to 0.10-0.18
            
        # Maintain XY position
        if xy_error > 0.05:
            v_cmd[0:2] = -0.1 * pos_error[0:2] / (xy_error + 1e-6)
            
        yaw_gain = 1.5
        yaw_rate_cmd = np.clip(yaw_gain * yaw_error, -1.0, 1.0)

    # [Param 8] Altitude protection
    if pos[2] < 0.2 and v_cmd[2] < 0:  # Altitude threshold can be adjusted to 0.18-0.25
        v_cmd[2] = 0
    
    # [Param 9] Distance-based smoothing - affects response speed
    if target_phase == "POSITION":
        dist = np.linalg.norm(pos_error)
        # Smaller smoothing factors lead to faster response
        alpha_near = 0.9  # Can be tuned to 0.85-0.95
        alpha_far = 0.3   # Can be tuned to 0.25-0.4
        ratio = min(1.0, dist / 1.0)
        alpha = (1 - ratio) * alpha_near + ratio * alpha_far
    else:
        alpha = 0.7  # Can be tuned to 0.6-0.8

    v_cmd = alpha * v_cmd + (1 - alpha) * prev_u
    prev_u = v_cmd.copy()

    # [Param 10] Z-axis lock control
    if abs(pos_error[2]) < 0.04 and abs(vel[2]) < 0.02:  # Can be adjusted to 0.03/0.01 for higher precision
        v_cmd[2] = 0.0
    
    # [Param 11] Deadzone filtering - affects precise positioning
    v_cmd[0:2][np.abs(v_cmd[0:2]) < 0.04] = 0.0  # Can be adjusted to 0.04-0.06
    if abs(v_cmd[2]) < 0.05:  # Z-axis deadzone can be adjusted to 0.04-0.06
        v_cmd[2] = 0.0
    
    # [Param 12] Special handling: ensure enough lift force
    if pos_error[2] > 0.1 and 0 < v_cmd[2] < 0.08:  # Threshold can be adjusted to 0.08-0.12 and 0.06-0.09
        v_cmd[2] = 0.08  # Force lift speed can be tuned to 0.07-0.10

    # Convert from world frame to body frame
    cy, sy = np.cos(yaw), np.sin(yaw)
    v_body = np.array([
        cy * v_cmd[0] + sy * v_cmd[1],
        -sy * v_cmd[0] + cy * v_cmd[1],
        v_cmd[2]
    ])

    return (v_body[0], v_body[1], v_body[2], yaw_rate_cmd)
