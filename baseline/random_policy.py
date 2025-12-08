class RandomAgent:
    """A simple random agent for evaluation purposes."""
    def __init__(self, action_space):
        self.action_space = action_space

    def select_action(self, observation):
        return self.action_space.sample()