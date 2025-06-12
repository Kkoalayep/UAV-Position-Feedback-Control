import torch
import torch.nn as nn
import torch.optim as optim
from pid_policy import PolicyNet
from rl_pid_env import PIDEnv
import numpy as np
import matplotlib.pyplot as plt
import os

# Set a fixed yaw target (135 degrees)
def set_targets():
    yaw = np.pi * 0.75
    print(f"✅ The yaw target is {yaw:.4f}")
    return yaw

# Setup working directory
project_dir = os.path.abspath(os.path.dirname(__file__))
root_dir = os.path.abspath(os.path.join(project_dir, ".."))
os.chdir(root_dir)

# Ensure the model saving directory exists
model_dir = "model"
os.makedirs(model_dir, exist_ok=True)

# Define file paths for model and reward plot
save_path = os.path.join(model_dir, "best_pid_yaw.pth")
fig_path = os.path.join(model_dir, "pid_yaw_reward_curve.png")

# Initialize the environment and policy network
env = PIDEnv()
policy = PolicyNet()
optimizer = optim.Adam(policy.parameters(), lr=0.01)

# Training parameters
n_episodes = 500
best_reward = -float("inf")
best_params = None
rewards_all = []

# 🔁 Try to load existing model if available
if os.path.exists(save_path):
    checkpoint = torch.load(save_path, weights_only=False)
    policy.load_state_dict(checkpoint["model_yaw"])
    best_reward = checkpoint.get("best_reward_yaw", -float("inf"))
    best_params = checkpoint.get("best_params_yaw", None)
    print(f"🔁 Loaded existing model. Best yaw reward: {best_reward:.2f}")

try:
    for ep in range(n_episodes):
        # Set target for this episode
        yaw = set_targets()

        log_probs = []
        rewards = []

        # Reset the environment
        state = env.reset()
        done = False

        while not done:
            # Convert state to tensor and get network output
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            out = policy(state_tensor)

            # Check for NaN in output
            if torch.isnan(out).any():
                print("⚠️ NaN detected in policy output")
                raise ValueError("NaN in policy output")

            # Clamp output and sample action from Normal distribution
            out = torch.clamp(out, 0.0, 1.0)
            dist = torch.distributions.Normal(out, 0.1)
            sample = dist.rsample()
            action = sample.squeeze().detach().numpy()
            log_probs.append(dist.log_prob(sample).sum())

            # Construct action dictionary (only yaw is active)
            action_dict = {
                'x': np.array([0.0, 0.0, 0.0]),
                'y': np.array([0.0, 0.0, 0.0]),
                'z': np.array([0.0, 0.0, 0.0]),
                'yaw': action
            }

            # Step the environment
            next_state, reward_dict, done, _ = env.step(action_dict)
            rewards.append(reward_dict['yaw'])
            state = next_state

        # Compute discounted returns
        R = 0
        returns = []
        for r in reversed(rewards):
            R = r + 0.99 * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32)

        # Normalize returns if possible
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        else:
            returns = returns * 0  # fallback to zero

        # Compute loss using policy gradient + entropy regularization
        entropy = dist.entropy().sum()
        loss = -torch.stack(log_probs) @ returns - 0.01 * entropy

        # Backprop and optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track reward
        total_reward = sum(rewards)
        rewards_all.append(total_reward)

        print(f"🎯 Episode {ep + 1} [YAW], Reward: {total_reward:.2f}")

        # Save best model if reward improves
        if total_reward > best_reward:
            best_reward = total_reward
            best_params = action
            print(f"💾 Best YAW updated: reward={total_reward:.2f}, PID={action}")

            torch.save({
                "model_yaw": policy.state_dict(),
                "best_reward_yaw": best_reward,
                "best_params_yaw": best_params
            }, save_path)

            # Also print scaled PID values (converted from [0, 1] to real-world range)
            raw = best_params
            low = env.action_space.low[9:]
            high = env.action_space.high[9:]
            scaled = low + raw * (high - low)
            print(f"🔎 Best so far [YAW]: Kp={scaled[0]:.4f}, Ki={scaled[1]:.4f}, Kd={scaled[2]:.4f}")

except KeyboardInterrupt:
    print("⏹️ Training interrupted by user.")

# Save and show reward curve after training
plt.plot(rewards_all)
plt.xlabel("Episode")
plt.ylabel("Yaw Reward")
plt.title("Yaw PID Training Curve")
plt.grid(True)
plt.savefig(fig_path)
plt.show()
print(f"📈 Saved reward curve to {fig_path}")

print("✅ YAW training complete.")
print(f"🏆 Best yaw PID = {best_params}")
