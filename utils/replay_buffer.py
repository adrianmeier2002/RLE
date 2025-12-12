import numpy as np
import torch

class ReplayBuffer:
    """"
    Simple Experience Replay Buffer for DQN.
    Stores transitions of shape:
    (state, action, reward, next_state, done)
    """

    def __init__(self, capacity: int, state_shape: tuple, dtype=torch.uint8):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.capacity = capacity
        self.ptr = 0           # Pointer to the current index
        self.size = 0          # Current size of the buffer

        self.states = torch.zeros((capacity, *state_shape), dtype=dtype).to(self.device)
        self.next_states = torch.zeros((capacity, *state_shape), dtype=dtype).to(self.device)
        self.actions = torch.zeros((capacity,), dtype=torch.int64).to(self.device)
        self.rewards = torch.zeros((capacity,), dtype=torch.float32).to(self.device)
        self.dones = torch.zeros((capacity,), dtype=torch.bool).to(self.device)

    def add(self, state, action, reward, next_state, done):
        """Adds a new transition to the replay buffer."""
        state = torch.tensor(state, dtype=self.states.dtype, device=self.device)
        next_state = torch.tensor(next_state, dtype=self.next_states.dtype, device=self.device)

        self.states[self.ptr] = state
        self.next_states[self.ptr] = next_state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done

        # Move pointer
        self.ptr = (self.ptr + 1) % self.capacity           # Circular buffer (overwrite old data)
        self.size = min(self.size + 1, self.capacity)       # Update current size


    def sample(self, batch_size: int):
        """Randomly sample a batch of transitions."""
        idx = np.random.randint(0, self.size, size=batch_size) # Sample random indices
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx]
        )
    
    def __len__(self):
        """Returns the current size of the buffer."""
        return self.size