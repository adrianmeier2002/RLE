import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
import os
from datetime import datetime


from utils.env_utils import make_env
from utils.replay_buffer import ReplayBuffer
from utils.logging_utils import create_writer, log_scalar

class QNetwork(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(QNetwork, self).__init__()
        
        c, h, w = input_shape
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )
    
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()[1:]))
    
    def forward(self, x):
        x = x.float() / 255.0  # Normalize pixel values
        conv_out = self.conv(x)
        conv_out = conv_out.view(conv_out.size(0), -1)
        return self.fc(conv_out)
    
class DQNAgent:
    def __init__(self, input_shape, num_actions, lr=1e-4, gamma=0.99, epsilon_start=1.0, 
                 epsilon_final=0.01, epsilon_decay=50000):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_actions = num_actions
        self.gamma = gamma

        self.q_net = QNetwork(input_shape, num_actions).to(self.device)
        self.target_net = QNetwork(input_shape, num_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        self.epsilon = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.step_count = 0

    def select_action(self, state):
        self.step_count += 1
        eps = self.epsilon_final + (self.epsilon - self.epsilon_final) * \
              np.exp(-1. * self.step_count / self.epsilon_decay)            # calculate epsilon
        if random.random() < eps:                                           # explore a random action if less than epsilon
            return random.randrange(self.num_actions)
        else:                                                               # exploit the best action
            state_tensor = torch.tensor(np.array([state]), dtype=torch.float32).to(self.device)
            q_values = self.q_net(state_tensor)
            return int(torch.argmax(q_values, dim=1)[0])                    # return the action with highest Q-value
        
    def update(self, batch):
        states, actions, rewards, next_states, dones = batch
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)
        
        # Q(s,a)
        q_values = self.q_net(states).gather(1, actions)                     # Q(s,a)
        
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)  # max_a' Q(s',a')
            target = rewards + self.gamma * next_q_values * (1 - dones)  # target Q-value

        loss = F.mse_loss(q_values, target)                                 # MSE loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

def train_dqn(env_id="ALE/SpaceInvaders-v5", 
              agent=DQNAgent, num_episodes=1000, 
              batch_size=32, target_update_freq=1000, 
              video_every=None, 
              video_folder = "videos/dqn_initial/",
              writer_path = "runs/dqn_initial",
              model_save = "dqn/models/dqn_initial.pt"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    else:
        print("Using CPU")

    def preprocess_obs(obs):
        obs = np.array(obs, copy=False)
        if obs.ndim == 2:
            obs = obs[None, :, :]
        elif obs.ndim == 3:
            obs = np.transpose(obs, (2, 0, 1))
        else:
            raise ValueError(f"Unexpected observation shape: {obs.shape}")
        return obs

    env = make_env(
        env_id=env_id,
        record_video=bool(video_every),
        video_folder=video_folder,
        video_freq=video_every
    )
    obs, info = env.reset()
    obs = preprocess_obs(obs)
    input_shape = obs.shape
    num_actions = env.action_space.n

    agent = agent(input_shape, num_actions)
    agent.q_net.to(device)
    agent.target_net.to(device)
    replay_buffer = ReplayBuffer(capacity=100000, state_shape=input_shape)

    writer = create_writer(writer_path, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    rewards_history = []
    global_step = 0

    for ep in range(num_episodes):
        obs, info = env.reset()
        obs = preprocess_obs(obs)
        done = False
        truncated = False
        ep_reward = 0

        while not (done or truncated):
            action = agent.select_action(obs)
            next_obs, reward, done, truncated, info = env.step(action)
            next_obs = preprocess_obs(next_obs)
            replay_buffer.add(obs, action, reward, next_obs, done)
            ep_reward += reward
            obs = next_obs

            global_step += 1

            if len(replay_buffer) >= batch_size:
                for _ in range(4):  # Perform multiple updates per step
                    batch = replay_buffer.sample(batch_size)
                    loss = agent.update(batch)
                    log_scalar(writer, "loss", loss, global_step)

        rewards_history.append(ep_reward)
        log_scalar(writer, "Reward/Episode", ep_reward, ep)

        if global_step % target_update_freq == 0:
            agent.update_target()

        print(f"Episode {ep + 1}/{num_episodes} - Reward: {ep_reward}")
        
    env.close()
    torch.save(agent.q_net.state_dict(), model_save)
    return agent, rewards_history


if __name__ == "__main__":
    agent, rewards_history = train_dqn(
        env_id="ALE/SpaceInvaders-v5",
        agent=DQNAgent,
        num_episodes=1000,
        batch_size=128,
        target_update_freq=1000,
        video_every=100
    )