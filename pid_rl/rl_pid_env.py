import numpy as np
import gym
from gym import spaces
import subprocess
import os

# === Custom Gym Environment for PID Controller Tuning ===
# This environment simulates a system where the PID gains for x, y, z, yaw are applied to a real/virtual controller.
# It reads PID parameters, writes them into `controller.py`, runs an external simulation (`run.py`),
# and parses a result file (`result.txt`) to retrieve the reward for each axis.

class PIDEnv(gym.Env):
    def __init__(self):
        super(PIDEnv, self).__init__()

        # === Define real-world lower and upper bounds for PID parameters per axis ===
        # Each axis (x/y/z/yaw) has 3 values: [Kp, Ki, Kd]
        self.low = np.array([
            0.5, 0.0, 0.0,    # x
            0.5, 0.0, 0.0,    # y
            0.5, 0.0, 0.0,    # z
            0.5, 0.0, 0.0     # yaw
        ])
        self.high = np.array([
            4.0, 0.5, 0.6,
            4.0, 0.5, 0.6,
            4.0, 0.5, 0.6,
            4.0, 0.5, 0.6
        ])

        # === Define action and observation spaces for Gym compatibility ===
        # Action: a 12-dimensional vector (normalized values between 0 and 1)
        # Observation: dummy single value (not used, but required by Gym)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(12,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

    def scale_action(self, action):
        """
        Scale a normalized action vector [0~1] to real PID parameter ranges.
        """
        action = np.clip(action, 0.0, 1.0)
        return self.low + action * (self.high - self.low)

    def step(self, action_dict):
        """
        Apply PID parameters, run external simulation, and return reward.

        Args:
            action_dict (dict): A dictionary with keys ['x', 'y', 'z', 'yaw'], each a 3-element PID vector.

        Returns:
            observation (np.ndarray): Placeholder observation
            reward_dict (dict): Reward for each axis
            done (bool): Always True (since it's a single-step environment)
            info (dict): Empty info dictionary
        """
        # === Concatenate all axis PID values into one 12D array ===
        action_all = np.concatenate([
            action_dict['x'],
            action_dict['y'],
            action_dict['z'],
            action_dict['yaw']
        ])

        # === Scale to real-world PID parameter ranges ===
        action_scaled = self.scale_action(action_all)

        # Split into Kp, Ki, Kd per axis
        Kp = action_scaled[0::3]
        Ki = action_scaled[1::3]
        Kd = action_scaled[2::3]

        print(f"[Scaled Kp] {Kp}")
        print(f"[Scaled Ki] {Ki}")
        print(f"[Scaled Kd] {Kd}")

        # === Modify controller.py file by injecting the new PID values ===
        with open("controller.py", "r", encoding="utf-8") as f:
            lines = f.readlines()

        var_name = "PID = {"
        skip = False
        with open("controller.py", "w", encoding="utf-8") as f:
            for line in lines:
                if var_name in line:
                    indent = line[:line.index("P")]
                    f.write(f"{indent}{var_name}\n")
                    f.write(f"{indent}    'Kp': np.array([{', '.join(f'{x:.4f}' for x in Kp)}]),\n")
                    f.write(f"{indent}    'Ki': np.array([{', '.join(f'{x:.4f}' for x in Ki)}]),\n")
                    f.write(f"{indent}    'Kd': np.array([{', '.join(f'{x:.4f}' for x in Kd)}])\n")
                    f.write(f"{indent}}}\n")
                    skip = True
                elif skip and "}" in line:
                    skip = False
                    continue  # Skip original closing brace
                elif not skip:
                    f.write(line)

        # === Run external simulation via run.py ===
        try:
            subprocess.run(["python", "run.py"], timeout=30)
        except subprocess.TimeoutExpired:
            # Simulation failed or took too long → assign large negative rewards
            return np.array([0.0]), {'x': -100.0, 'y': -100.0, 'z': -100.0, 'yaw': -100.0}, True, {}

        # === Read reward from simulation output file ===
        rewards = {'x': -100.0, 'y': -100.0, 'z': -100.0, 'yaw': -100.0}
        if os.path.exists("result.txt"):
            with open("result.txt", "r") as f:
                for line in f:
                    if "reward_x=" in line:
                        rewards['x'] = float(line.strip().split("=")[1])
                    elif "reward_y=" in line:
                        rewards['y'] = float(line.strip().split("=")[1])
                    elif "reward_z=" in line:
                        rewards['z'] = float(line.strip().split("=")[1])
                    elif "reward_yaw=" in line:
                        rewards['yaw'] = float(line.strip().split("=")[1])

        # Always return dummy observation and done=True (single-step environment)
        return np.array([0.0]), rewards, True, {}

    def reset(self):
        """
        Reset the environment (required by Gym, but unused here).
        """
        return np.array([0.0])
