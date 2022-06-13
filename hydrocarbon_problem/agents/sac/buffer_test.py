import chex

from hydrocarbon_problem.api.fake_api import FakeDistillationAPI
from hydrocarbon_problem.agents.sac.buffer import ReplayBuffer
import jax
from hydrocarbon_problem.env.env import AspenDistillation
from hydrocarbon_problem.agents.sac.agent_test import create_fake_batch

if __name__ == '__main__':
    # to check that the replay buffer runs
    batch_size = 3
    n_batches_total_length = 2
    length = n_batches_total_length * batch_size
    min_sample_length = int(length * 0.5)
    rng_key = jax.random.PRNGKey(0)
    buffer = ReplayBuffer(length, min_sample_length)

    env = AspenDistillation(flowsheet_api=FakeDistillationAPI())
    buffer_state = buffer.init(rng_key, env)
    print("initialisation worked")

    batch_size = 2
    batch = buffer.sample(buffer_state, rng_key, batch_size=2)
    fake_batch = create_fake_batch(env, batch_size)
    chex.assert_tree_all_equal_structs(batch, fake_batch)
    print("sampling worked")

    dummy_data_point = jax.tree_map(lambda x: x[0], batch)
    buffer_state = buffer.add(dummy_data_point, buffer_state)
    print("add worked")