import numpy as np

from hydrocarbon_problem.env.env import AspenDistillation
from hydrocarbon_problem.api.fake_api import FakeDistillationAPI

def make_fake_agent(env: AspenDistillation):
    def fake_agent(obs):
        del(obs)
        discrete_spec, continuous_spec = env.action_spec()
        discrete_action = np.random.randint(0, 1, size=discrete_spec.num_values)
        continuous_action = np.random.uniform(low=continuous_spec.minimum,
                                              high=continuous_spec.maximum,
                                              size=continuous_spec.shape)
        return discrete_action, continuous_action
    return fake_agent

def test():
    api = FakeDistillationAPI()
    env = AspenDistillation(flowsheet_api=api)
    agent = make_fake_agent(env)

    timestep = env.reset()
    while not timestep.last():
        observation = timestep.observation.upcoming_state
        action = agent(observation)
        timestep = env.step(action)


if __name__ == '__main__':
    test()
