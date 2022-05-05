import numpy as np
import matplotlib.pyplot as plt

from hydrocarbon_problem.env.env import AspenDistillation
from hydrocarbon_problem.api.fake_api import FakeDistillationAPI
from hydrocarbon_problem.api.aspen_api import AspenAPI

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
    # api = FakeDistillationAPI()  # this can be changed to AspenAPI to test with Aspen
    api = AspenAPI()
    env = AspenDistillation(flowsheet_api=api)
    agent = make_fake_agent(env)
    simulation_time = []
    converged = []


    for i in range(n_episodes):
        timestep = env.reset()
        episode_return = 0
        n_streams = 1
        while not timestep.last():
            observation = timestep.observation.upcoming_state
            action = agent(observation)
            timestep, duration, run_converged = env.step(action)
            # if duration > 0 and run_converged == True:
            simulation_time.append(duration)
            converged.append(run_converged)
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
                n_streams += 2  # 2 new streams created
                # if we choose to seperate a stream, then the reward should be non-zero, the created state
                # discount's should both be 1, the created_states should have non-zero values.
                assert not timestep.reward == 0.0
                if env._stream_table[-2].is_product:
                    # if tops is product, check discount is 0 else, check discount is 1
                    assert timestep.discount.created_states[0] == 0
                else:
                    assert timestep.discount.created_states[0] == 1
                if env._stream_table[-1].is_product:
                    # if bots is product, check discount is 0 else, check discount is 1
                    assert timestep.discount.created_states[1] == 0
                else:
                    assert timestep.discount.created_states[1] == 1
                assert not (timestep.observation.created_states[1] == env._blank_state).all()
                assert not (timestep.observation.created_states[0] == env._blank_state).all()
                if not timestep.last():
                    # if the episode is not done, then check that the upcoming observation has
                    # non-zero values
                    assert not (timestep.observation.upcoming_state == env._blank_state).all()

            # check the stream table has the correct number of streams
            assert len(env._stream_table) == n_streams
        print(f"episode complete with return of {episode_return}")
    return simulation_time, converged


if __name__ == '__main__':
    simulation_time, converged = test()

    # Separate the convergence data
    unconverged_separations = [index for (index, item) in enumerate(converged) if item == False]
    iterations_without_separation = [index for (index, item) in enumerate(converged) if item == "no separation"]
    converged_separation = [index for (index, item) in enumerate(converged) if item == True]

    number_of_iterations_without_separation = len(iterations_without_separation)

    number_of_unconverged_separations = len(unconverged_separations)
    number_of_converged_separations = len(converged_separation)
    number_of_non_separations = len(iterations_without_separation)
    total_separations = number_of_unconverged_separations + number_of_converged_separations

    percent_unconverged_separations = 100 * number_of_unconverged_separations/total_separations
    percent_converged_separations = 100 * number_of_converged_separations/total_separations

    if number_of_converged_separations == 0 and number_of_unconverged_separations == 0:
        print("no separations were performed")
        print(f"Number of iterations = {len(converged)}")

    else:
        print(f"Number of iterations: {len(converged)}")
        print(f"Number of unconverged separations: {number_of_unconverged_separations}, "
              f"{percent_unconverged_separations} %")
        print(f"Number of converged separations: {number_of_converged_separations}, "
              f"{percent_converged_separations} %")
        print(f"Number of non separations: {number_of_non_separations}")

        # print(f"Total sim array {simulation_time}")
        # x = list(range(0, len(simulation_time)))
        # plt.plot(x, simulation_time)
        # plt.xlabel("Iteration [-]")
        # plt.ylabel("Simulation time [s]")
        # plt.show()


