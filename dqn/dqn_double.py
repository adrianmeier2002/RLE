import torch
import torch.nn.functional as F

from dqn.dqn_initial import DQNAgent, train_dqn
from utils.replay_buffer import ReplayBuffer

class DoubleDQNAgent(DQNAgent):
    # Inherits everything from DQNAgent except the update method
    def update(self, batch):
        states, actions, rewards, next_states, dones = batch

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        # Q(s,a)
        q_values = self.q_net(states).gather(1, actions)

        with torch.no_grad():
            # Double DQN: action selection from q_net, evaluation from target_net
            next_actions = torch.argmax(self.q_net(next_states), dim=1, keepdim=True)
            next_q_values = self.target_net(next_states).gather(1, next_actions)
            target = rewards + self.gamma * next_q_values * (1 - dones)

        loss = F.mse_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
if __name__ == "__main__":
    agent, rewards_history = train_dqn(
        env_id="ALE/SpaceInvaders-v5",
        agent=DoubleDQNAgent,
        buffer_class=ReplayBuffer,
        num_steps=5000000,
        batch_size=32,
        target_update_freq=10000,
        learning_starts=50000,
        train_freq=4,

        video_every=100,
        video_folder="videos/dqn_double/",
        writer_path="runs/dqn_double",
        model_save="dqn/models/dqn_double.pt"
    )
              