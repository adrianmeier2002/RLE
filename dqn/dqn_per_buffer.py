import torch

from utils.replay_buffer_per import PrioritizedReplayBuffer
from dqn.dqn_initial import DQNAgent, train_dqn

class PerDQNAgent(DQNAgent):
    def update(self, batch):
        states, actions, rewards, next_states, dones, idx, weights = batch

        if not isinstance(states, torch.Tensor):
            states = torch.tensor(states, dtype=torch.float32).to(self.device)
            next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
            actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device)
            rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
            dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)
            weights = torch.tensor(weights, dtype=torch.float32).unsqueeze(1).to(self.device)

        else:
            actions = actions.unsqueeze(1) if actions.dim() == 1 else actions
            rewards = rewards.unsqueeze(1) if rewards.dim() == 1 else rewards
            dones = dones.float()
            dones = dones.unsqueeze(1) if dones.dim() == 1 else dones
            weights = weights.float()
            weights = weights.unsqueeze(1) if weights.dim() == 1 else weights


        # Q(s,a)
        q_values = self.q_net(states).gather(1, actions)

        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(dim=1)[0].unsqueeze(1)  # max_a' Q(s',a')
            target = rewards + self.gamma * next_q_values * (1 - dones)  # target Q-value

        td_errors = target - q_values
        abs_td = td_errors.abs().detach()

        # Weighted MSE loss with PER
        loss = (weights * (td_errors ** 2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Return TD errors for updating priorities
        return loss.item(), abs_td
    
if __name__ == "__main__":
    agent, rewards_history = train_dqn(
        env_id="ALE/SpaceInvaders-v5",
        agent=PerDQNAgent,
        buffer_class=PrioritizedReplayBuffer,
        num_steps=5000000,
        batch_size=32,
        target_update_freq=10000,
        learning_starts=50000,
        train_freq=4,
        video_every=100,

        video_folder="videos/dqn_per/",
        writer_path="runs/dqn_per",
        model_save="dqn/models/dqn_per.pt"
    )
