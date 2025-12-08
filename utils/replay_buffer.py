import numpy as np

class ReplayBuffer:
    """"
    Simple Experience Replay Buffer for DQN.
    Stores transitions of shape:
    (state, action, reward, next_state, done)
    """

    def __init__(self, capacity: int, state_shape: tuple, dtype=np.uint8):
        self.capacity = capacity
        self.ptr = 0           # Pointer to the current index
        self.size = 0          # Current size of the buffer

        self.states = np.zeros((capacity, *state_shape), dtype=dtype)
        self.next_states = np.zeros((capacity, *state_shape), dtype=dtype)
        self.actions = np.zeros((capacity,), dtype=np.int32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.bool_)

    def add(self, state, action, reward, next_state, done):
        """Adds a new transition to the replay buffer."""
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