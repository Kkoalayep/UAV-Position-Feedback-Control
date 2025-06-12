import torch
import torch.nn as nn
import torch.optim as optim
from pid_policy import PolicyNet          # Neural network model for outputting PID parameters
from rl_pid_env import PIDEnv             # Custom environment for evaluating PID controllers
import numpy as np
import os
import matplotlib.pyplot as plt
import csv

# Set working directory to project root
project_dir = os.path.abspath(os.path.dirname(__file__))
root_dir = os.path.abspath(os.path.join(project_dir, ".."))
os.chdir(root_dir)

# Write current target setpoint to CSV for the environment to use
def write_target_to_csv(target, filename="targets.csv"):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["x", "y", "z", "yaw"])
        writer.writerow([f"{v:.4f}" for v in target])

# Initialize environment and training setup
env = PIDEnv()
axes_to_train = ['x', 'y', 'z']

# Create separate policy network and optimizer for each axis
policies = {k: PolicyNet() for k in axes_to_train}
optimizers = {k: optim.Adam(p.parameters(), lr=0.01) for k, p in policies.items()}

# Load pretrained yaw parameters (fixed during training)
yaw_params = np.array([0.5, 0.0, 0.0])
yaw_path = "model/best_pid_yaw.pth"
if os.path.exists(yaw_path):
    yaw_ckpt = torch.load(yaw_path, weights_only=False)
    yaw_params = yaw_ckpt.get("best_params_yaw", yaw_params)
    print(f"🔁 Loaded best yaw PID params: {yaw_params}")

# Training hyperparameters and bookkeeping
n_episodes = 500
best_rewards = {k: -float("inf") for k in axes_to_train}    # Track best reward per axis
best_params = {k: None for k in axes_to_train}              # Track best PID params per axis
reward_history = {k: [] for k in axes_to_train}             # Save reward per episode per axis
save_path = "model/best_pid_xyz.pth"

# Load previous model and best parameters if available
if os.path.exists(save_path):
    checkpoint = torch.load(save_path, weights_only=False)
    for axis in axes_to_train:
        policies[axis].load_state_dict(checkpoint[f'model_{axis}'])
        best_rewards[axis] = checkpoint.get(f'best_reward_{axis}', -float("inf"))
        best_params[axis] = checkpoint.get(f'best_params_{axis}', None)
    print(f"🔁 Loaded existing model. Best rewards: {best_rewards}")

try:
    for ep in range(n_episodes):
        # Define multiple targets to test controller generalization
        targets = [
            np.array([2.0, 2.0, 1.0, 2.0]),
            np.array([-5.0, -5.0, 2.0, -2.0]),
            np.array([8.0, 8.0, 3.0, 3.0])
        ]

        # === Train each axis independently ===
        for axis in axes_to_train:
            all_rewards = []

            # Forward pass through the policy network to sample PID parameters
            state_tensor = torch.FloatTensor([[0.0]])  # Dummy input
            out = policies[axis](state_tensor)
            dist = torch.distributions.Normal(out, 0.1)  # Gaussian sampling for exploration
            sample = dist.sample()
            action = sample.squeeze().detach().numpy()  # Sampled PID: [Kp, Ki, Kd]
            log_prob = dist.log_prob(sample).sum()      # Log probability for policy gradient

            # === Evaluate this PID setting on all targets ===
            for target in targets:
                write_target_to_csv(target)
                env.reset()

                # Default actions for all axes
                actions = {
                    'x': np.array([0.0, 0.0, 0.0]),
                    'y': np.array([0.0, 0.0, 0.0]),
                    'z': np.array([0.0, 0.0, 0.0]),
                    'yaw': yaw_params
                }

                # Apply current PID only to the current axis being trained
                actions[axis] = action

                # Perform one environment step and get reward
                _, reward_dict, _, _ = env.step(actions)
                all_rewards.append(reward_dict[axis])

            # === Policy update ===
            avg_reward = np.mean(all_rewards)
            returns = torch.tensor([avg_reward], dtype=torch.float32)
            loss = -log_prob * returns  # REINFORCE loss
            optimizers[axis].zero_grad()
            loss.backward()
            optimizers[axis].step()

            reward_history[axis].append(avg_reward)
            print(f"🎯 Episode {ep + 1} [{axis.upper()}], Avg Reward: {avg_reward:.2f}")

            # Save best parameters if improved
            if avg_reward > best_rewards[axis]:
                best_rewards[axis] = avg_reward
                best_params[axis] = action
                print(f"💾 Best {axis.upper()} updated: reward={avg_reward:.2f}, PID={action}")

        # Save model and best parameters after every episode
        torch.save({
            **{f'model_{axis}': policies[axis].state_dict() for axis in axes_to_train},
            **{f'best_reward_{axis}': best_rewards[axis] for axis in axes_to_train},
            **{f'best_params_{axis}': best_params[axis] for axis in axes_to_train},
            'best_params_yaw': yaw_params
        }, save_path)

except KeyboardInterrupt:
    print("⏹️ Training interrupted by user.")

# === Summary printout ===
print("✅ Training complete.")
print("🏆 Best PID parameters:")
for axis in axes_to_train:
    print(f"  {axis.upper()}: {best_params[axis]}")
print(f"  YAW (fixed): {yaw_params}")

# === Plotting reward curves ===
plt.figure(figsize=(10, 6))
for axis in axes_to_train:
    plt.plot(reward_history[axis], label=f'{axis.upper()} Reward')
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.title('XYZ PID Training Reward Curve (Independent Axes)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('model/pid_xyz_reward_curve.png')
plt.show()
print("📈 Saved reward curve as pid_xyz_reward_curve.png")
