import torch
import numpy as np
import os
from rl_pid_env import PIDEnv

# === Set working directory to the project root ===
project_dir = os.path.abspath(os.path.dirname(__file__))
root_dir = os.path.abspath(os.path.join(project_dir, ".."))
os.chdir(root_dir)

# === Load best PID parameter checkpoints ===
model_dir = "model"
best_xyz_path = os.path.join(model_dir, "best_pid_xyz.pth")
best_yaw_path = os.path.join(model_dir, "best_pid_yaw.pth")

# Ensure the checkpoints exist
assert os.path.exists(best_xyz_path), f"Cannot find {best_xyz_path}"
assert os.path.exists(best_yaw_path), f"Cannot find {best_yaw_path}"

# Load checkpoint data
best_xyz = torch.load(best_xyz_path, weights_only=False)
best_yaw = torch.load(best_yaw_path, weights_only=False)

# === Extract the best PID parameters from checkpoint ===
# These are the best parameters found during training, saved per axis.
# If not found (e.g., due to missing keys), fallback to a default value.
x_pid = best_xyz.get('best_params_x', np.array([0.5, 0.0, 0.0]))
y_pid = best_xyz.get('best_params_y', np.array([0.5, 0.0, 0.0]))
z_pid = best_xyz.get('best_params_z', np.array([0.5, 0.0, 0.0]))
yaw_pid = best_yaw.get('best_params_yaw', np.array([0.5, 0.0, 0.0]))

# === Create the simulation environment ===
env = PIDEnv()

# Create a dictionary containing all best PID settings
actions = {
    'x': x_pid,
    'y': y_pid,
    'z': z_pid,
    'yaw': yaw_pid
}

print("📝 Applying best PID parameters and starting simulation...")

# Run one simulation using the best PID parameters
# The env.step() will internally performs:
# - action scaling
# - writing to controller.py
# - executing the simulation (e.g., run.py)
_, _, _, _ = env.step(actions)

print("✅ Simulation finished.")
