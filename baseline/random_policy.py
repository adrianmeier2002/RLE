class RandomAgent:
    """
    A simple random agent that selects actions uniformly at random.
    Used as a baseline to measure improvement of trained agents.
    """
    def __init__(self, action_space):
        self.action_space = action_space
        self.num_actions = action_space.n
    
    def select_action(self, observation, eval_mode=False):
        """
        Select a random action.
        
        Parameters
        ----------
        observation : np.ndarray
            Current observation (unused for random agent)
        eval_mode : bool
            Evaluation mode flag (unused for random agent)
            
        Returns
        -------
        action : int
            Random action from action space
        """
        return self.action_space.sample()
    
    def __repr__(self):
        return f"RandomAgent(num_actions={self.num_actions})"