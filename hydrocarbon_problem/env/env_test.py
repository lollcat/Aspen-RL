import numpy as np

from hydrocarbon_problem.env.env import AspenDistillation
from hydrocarbon_problem.api.fake_api import FakeDistillationAPI

def make_fake_agent(env: AspenDistillation):
    def fake_agent(obs):
        del(obs)
        discrete_spec, continuous_spec = env.action_spec()
        discrete_action = np.random.randint(0, discrete_spec.num_values, size=())
        continuous_action = np.random.uniform(low=continuous_spec.minimum,
                                              high=continuous_spec.maximum,
                                              size=continuous_spec.shape)
        return discrete_action, continuous_action
    return fake_agent

def test():
    api = FakeDistillationAPI()
    env = AspenDistillation(flowsheet_api=api)
    agent = make_fake_agent(env)

    n_episodes = 3
    for i in range(n_episodes):
        timestep = env.reset()
        episode_return = 0
        while not timestep.last():
            observation = timestep.observation.upcoming_state
            action = agent(observation)
            timestep = env.step(action)
            print(timestep)
            episode_return += timestep.reward
            discrete_action = action[0]
            if discrete_action == 0: # choose not to seperate
                assert timestep.reward == 0.0
                assert timestep.discount.created_states == (0, 0)
                assert (timestep.observation.created_states[1] == env._blank_state).all()
                assert (timestep.observation.created_states[0] == env._blank_state).all()
            if timestep.discount.overall == 0:
                assert not (timestep.observation.upcoming_state == env._blank_state).all()
            else:
                assert (timestep.observation.upcoming_state == env._blank_state).all()
        print(f"episode complete with return of {episode_return}")



if __name__ == '__main__':
    test()
