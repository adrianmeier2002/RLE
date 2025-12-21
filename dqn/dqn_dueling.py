import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from dqn.dqn_initial import DQNAgent, train_dqn
from utils.replay_buffer import ReplayBuffer

class DuelingQNetwork(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DuelingQNetwork, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        c, h, w = input_shape

        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        conv_out_size = self._get_conv_out(input_shape)

        # Value stream
        self.value = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        # Advantage stream
        self.advantage = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def _get_conv_out(self, shape):
        x = self.conv(torch.zeros(1, *shape)).to(self.device)
        return int(np.prod(x.size()[1:]))
    
    def forward(self, x):
        x = x.float() / 255.0
        conv_out = self.conv(x)
        conv_out = conv_out.view(conv_out.size(0), -1)

        value = self.value(conv_out)
        advantage = self.advantage(conv_out)

        # Combine: Q(s, a) = V(s) + (A(s, a) - mean_a A(s, a))
        q = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q
    
class DuelingDQNAgent(DQNAgent):
    def __init__(self, input_shape, num_actions, **kwargs):
        
        # Standard DQNAgent initialization
        super().__init__(input_shape, num_actions, **kwargs)

        # Replace Q-network and target network with dueling architecture
        self.q_net = DuelingQNetwork(input_shape, num_actions).to(self.device)
        self.target_net = DuelingQNetwork(input_shape, num_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        # Optimizer remains the same but has to be redefined for new network parameters
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=kwargs.get('lr', 1e-4))

if __name__ == "__main__":
    agent, history = train_dqn(
        env_id="ALE/SpaceInvaders-v5",
        agent=DuelingDQNAgent,
        buffer_class=ReplayBuffer,
        num_steps=5000000,
        batch_size=32,
        target_update_freq=10000,
        learning_starts=50000,
        train_freq=4,
        video_every=100,

        video_folder="videos/dqn_dueling",
        writer_path="runs/dqn_dueling",
        model_save="dqn/models/dqn_dueling.pt"
    )