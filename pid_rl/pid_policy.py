import torch
import torch.nn as nn

# === Policy Network for PID parameter generation ===
# This neural network maps a dummy input (constant) to a 3D output representing [Kp, Ki, Kd]
# Each axis (x/y/z) uses its own instance of this network.

class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()

        # Define the neural network architecture
        # Input: 1D dummy value (e.g., [0.0])
        # Output: 3 values representing PID parameters (Kp, Ki, Kd), all in (0, 1)
        self.net = nn.Sequential(
            nn.Linear(1, 64),    # First layer: fully connected, expands to 64 neurons
            nn.ReLU(),           # Nonlinear activation
            nn.Linear(64, 3),    # Output layer: 3 values for [Kp, Ki, Kd]
            nn.Sigmoid()         # Ensure output range is (0, 1) for scaling purposes
        )

        # Initialize the output layer bias to center predictions around desired initial PID values
        self._init_output_bias()

    def _init_output_bias(self):
        """
        Initialize the bias of the last linear layer so that the Sigmoid output is close
        to predefined initial PID values (e.g., [1.0, 0.1, 0.2] scaled by 0.5).
        This helps with faster convergence early in training.
        """
        init_pid = [1.0, 0.1, 0.2]           # Desired starting values for Kp, Ki, Kd
        target = torch.tensor(init_pid) / 2.0  # Assume values will be scaled later (e.g., *2)
        target = torch.clamp(target, 1e-5, 1 - 1e-5)  # Avoid extreme sigmoid boundaries
        bias = torch.log(target / (1 - target))      # Inverse sigmoid to get proper bias

        # Apply the bias to the last Linear layer
        with torch.no_grad():
            self.net[2].bias.copy_(bias)

    def forward(self, x):
        """
        Forward pass through the policy network.

        Args:
            x (torch.Tensor): Dummy input tensor (typically [[0.0]])

        Returns:
            torch.Tensor: A tensor of shape [batch_size, 3] with values in (0, 1)
                          representing scaled Kp, Ki, Kd.
        """
        return self.net(x)
