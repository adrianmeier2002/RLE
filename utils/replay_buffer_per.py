import numpy as np
import torch

class PrioritizedReplayBuffer:
    """
    Proportional Prioritized Experience Replay (PER) Buffer.

    Reuse a lot of the structure from the simple ReplayBuffer, but add priority sampling.

    Parameters:
    capacity : int
        Maximum size of the buffer.
    state_shape : tuple
        Shape of state observations.
    alpha : float
        How much prioritization to use (0 = uniform, 1 = full prioritization).
    beta : float
        Initial importance sampling weight (annealing to 1 over time).
    beta_increment : float
        Amount to increment beta per sample.
    """

    def __init__(self, capacity: int, 
                 state_shape: tuple, 
                 alpha: float = 0.6, 
                 beta: float = 0.4, 
                 beta_increment: float = 1e-6):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.capacity = capacity
        self.ptr = 0           # Pointer to the current index
        self.size = 0          # Current size of the buffer

        # Data storage
        self.states = torch.zeros((capacity, *state_shape), dtype=torch.uint8, device=self.device)
        self.next_states = torch.zeros((capacity, *state_shape), dtype=torch.uint8, device=self.device)
        self.actions = torch.zeros((capacity,), dtype=torch.int64, device=self.device)
        self.rewards = torch.zeros((capacity,), dtype=torch.float32, device=self.device)
        self.dones = torch.zeros((capacity,), dtype=torch.bool, device=self.device)

        # PER parameters
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = 1e-6

        # Priorities
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        """Adds a new transition to the replay buffer."""
        state = torch.tensor(state, dtype=self.states.dtype, device=self.device)
        next_state = torch.tensor(next_state, dtype=self.next_states.dtype, device=self.device)

        self.states[self.ptr] = state
        self.next_states[self.ptr] = next_state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done

        # Highest priority -> new samples will be sampled first
        max_prio = self.priorities[:self.size].max() if self.size > 0 else 1.0
        self.priorities[self.ptr] = max_prio

        # Move pointer
        self.ptr = (self.ptr + 1) % self.capacity           # Circular buffer (overwrite old data)
        self.size = min(self.size + 1, self.capacity)       # Update current size

    def sample(self, batch_size: int):
        """Proportional sampling based on priorities."""
        prios = self.priorities[:self.size]

        # Convert priorities to probabilities
        probs = prios ** self.alpha
        probs /= probs.sum()

        # Sample indices
        idx = np.random.choice(self.size, batch_size, p=probs, replace=False)

        # Importance sampling weights
        self.beta = min(1.0, self.beta + self.beta_increment)
        weights = (self.size * probs[idx]) ** (-self.beta)
        weights /= weights.max()  # Normalize for stability

        # Convert to tensors
        weights = torch.tensor(weights, dtype=torch.float32, device=self.device)

        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
            idx,
            weights
        )
    
    def update_priorities(self, idx, td_errors):
        """Update priorities using TD-error."""
        if isinstance(td_errors, torch.Tensor):
            td_errors = td_errors.detach().cpu().numpy()
        td_errors = td_errors.flatten()
        
        self.priorities[idx] = np.abs(td_errors) + self.epsilon

    def __len__(self):
        """Returns the current size of the buffer."""
        return self.size