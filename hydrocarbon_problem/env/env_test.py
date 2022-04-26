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

def test(n_episodes: int = 5):
    """This test runs multiple environment episodes, running some simple sanity
    checks along the way.
    """
    api = FakeDistillationAPI()  # this can be changed to AspenAPI to test with Aspen
    env = AspenDistillation(flowsheet_api=api)
    agent = make_fake_agent(env)

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
            if discrete_action == 0:  # choose not to seperate
                # if we don't seperate then the created states are black, 0 reward is given, and
                # the discount for the created states is zero
                assert timestep.reward == 0.0
                assert timestep.discount.created_states == (0, 0)
                assert (timestep.observation.created_states[1] == env._blank_state).all()
                assert (timestep.observation.created_states[0] == env._blank_state).all()
            else:
                # if we choose to seperate a stream, then the reward should be non-zero, the created state
                # discount's should both be 1, the created_states should have non-zero values.
                assert not timestep.reward == 0.0
                assert timestep.discount.created_states == (1, 1)
                assert not (timestep.observation.created_states[1] == env._blank_state).all()
                assert not (timestep.observation.created_states[0] == env._blank_state).all()
            if timestep.discount.overall == 0:
                assert (timestep.observation.upcoming_state == env._blank_state).all()
            else:
                assert not (timestep.observation.upcoming_state == env._blank_state).all()
        print(f"episode complete with return of {episode_return}")



if __name__ == '__main__':
    test()

