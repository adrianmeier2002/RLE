import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random

from utils.replay_buffer import ReplayBuffer
from dqn.dqn_initial import train_dqn

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.017):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init)

    def reset_noise(self):
        epsilon_in = torch.randn(self.in_features)
        epsilon_out = torch.randn(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)
    

class NoisyQNetwork(nn.Module):
    def __init__(self, input_shape, num_actions):
        super().__init__()
        c, h, w = input_shape
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU()
        )
        conv_out_size = self._get_conv_out(input_shape)
        self.fc1 = NoisyLinear(conv_out_size, 512)
        self.fc2 = NoisyLinear(512, num_actions)

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()[1:]))

    def forward(self, x):
        x = x.float() / 255.0
        x = self.conv(x).view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def reset_noise(self):
        self.fc1.reset_noise()
        self.fc2.reset_noise()
    
class NoisyDQNAgent:
    def __init__(self, input_shape, num_actions, lr=1e-4, gamma=0.99):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_actions = num_actions
        self.gamma = gamma

        self.q_net = NoisyQNetwork(input_shape, num_actions).to(self.device)
        self.target_net = NoisyQNetwork(input_shape, num_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
    
    def select_action(self, state):
        state_tensor = torch.tensor(np.array([state]), dtype=torch.float32).to(self.device)
        q_values = self.q_net(state_tensor)
        return int(torch.argmax(q_values, dim=1)[0])

    def update(self, batch):
        states, actions, rewards, next_states, dones = batch
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)

        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target = rewards + self.gamma * next_q_values * (1 - dones)

        loss = F.mse_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.q_net.reset_noise()
        self.target_net.reset_noise()
        return loss.item()
    
    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())
    
if __name__ == "__main__":
    agent, rewards_history = train_dqn(
        env_id="ALE/SpaceInvaders-v5",
        agent=NoisyDQNAgent,
        buffer_class=ReplayBuffer,
        num_steps=5000000,
        batch_size=32,
        target_update_freq=10000,
        learning_starts=50000,
        train_freq=4,
        video_every=100,

        video_folder="videos/dqn_noisy/",
        writer_path="runs/dqn_noisy",
        model_save="dqn/models/dqn_noisy.pt"
    )