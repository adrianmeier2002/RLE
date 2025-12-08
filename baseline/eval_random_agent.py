from utils.eval_utils import evaluate_agent
from utils.env_utils import make_env
from baseline.random_policy import RandomAgent

def eval_random():
    env = make_env(eval_mode=True)
    agent = RandomAgent(env.action_space)

    evaluate_agent(
        agent,
        episodes=50
    )

if __name__ == "__main__":
    eval_random()