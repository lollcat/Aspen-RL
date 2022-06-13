import chex
import jax
import numpy as np
import optax
import jax.numpy as jnp
import time
import matplotlib.pyplot as plt
from tqdm import tqdm


from hydrocarbon_problem.api.fake_api import FakeDistillationAPI
from hydrocarbon_problem.agents.sac.agent import create_agent, Agent
from hydrocarbon_problem.agents.sac.buffer import ReplayBuffer
from hydrocarbon_problem.agents.sac.create_networks import create_sac_networks
from hydrocarbon_problem.env.env import AspenDistillation
from hydrocarbon_problem.agents.base import NextObservation, Transition
from hydrocarbon_problem.agents.logger import ListLogger, plot_history



def train(n_iterations: int,
          env: AspenDistillation,
          agent: Agent,
          buffer: ReplayBuffer,
          key = jax.random.PRNGKey(0),
          n_sac_updates_per_episode: int = 3,
          batch_size: int = 32
          ):
    # initialise the buffer state (filling it with random experience)
    key, subkey = jax.random.split(key)
    buffer_state = buffer.init(subkey, env)

    logger = ListLogger()

    pbar = tqdm(range(n_iterations))
    # now run the training loop
    for i in pbar:
        # run an episode, adding the new experience to the buffer
        episode_return = 0.0
        timestep = env.reset()
        previous_timestep = timestep
        episode_start_time = time.time()
        while not timestep.last():
            # get the action
            key, subkey = jax.random.split(key)
            action = agent.select_action(agent.state, timestep.observation.upcoming_state,
                                         subkey)
            action = np.asarray(action[0]), np.asarray(action[1])

            # step the environment
            timestep = env.step(action)
            episode_return += timestep.reward

            # add to the buffer
            next_obs = NextObservation(observation=(timestep.observation.created_states),
                                       discounts=timestep.discount.created_states)
            transition = Transition(
                observation=previous_timestep.observation.upcoming_state,
                action=action,
                reward=timestep.reward,
                discount=timestep.discount.overall,
                next_observation=next_obs,
                extras=()
            )
            transition = jax.tree_map(jnp.asarray, transition)
            buffer_state = buffer.add(transition, buffer_state)
            previous_timestep = timestep

        # save useful metrics
        metrics = {"episode_return": episode_return,
                   "episode_time": time.time() - episode_start_time}
        logger.write(metrics)

        pbar.set_description(f"episode return of {episode_return:.2f}")

        # now update the SAC agent
        sac_start_time = time.time()
        for j in range(n_sac_updates_per_episode):
            # sample a batch from the replay buffer
            key, subkey = jax.random.split(key)
            batch = buffer.sample(buffer_state, subkey, batch_size)
            # chex.assert_tree_all_finite(batch)

            # update the agent using the sampled batch
            agent_state, info = agent.update(agent.state, batch)
            # chex.assert_tree_all_finite(agent_state)
            agent._replace(state=agent_state)
            logger.write(info)

        logger.write({"agent_step_time": time.time() - sac_start_time})

    plot_history(logger.history)
    plt.show()


if __name__ == '__main__':
    DISABLE_JIT = False  # useful for debugging
    from jax.config import config
    config.update('jax_disable_jit', DISABLE_JIT)
    SUPRESS_WARNINGS = True
    if SUPRESS_WARNINGS:
        # If we don't want to print warnings.
        # Should be used with care.
        import logging
        logger = logging.getLogger("root")
        class CheckTypesFilter(logging.Filter):
            def filter(self, record):
                return "check_types" not in record.getMessage()
        logger.addFilter(CheckTypesFilter())

    n_iterations = 20
    batch_size = 12
    n_sac_updates_per_episode = 1

    env = AspenDistillation(flowsheet_api=FakeDistillationAPI())
    sac_net = create_sac_networks(env=env,
                        policy_hidden_units = (3,), q_value_hidden_units = (10, 10))

    agent = create_agent(networks=sac_net,
                         rng_key=jax.random.PRNGKey(0),
                         policy_optimizer=optax.adam(1e-3),
                         q_optimizer=optax.adam(1e-3)
                         )

    min_sample_length = batch_size
    max_buffer_length = batch_size*100
    rng_key = jax.random.PRNGKey(0)
    buffer = ReplayBuffer(min_sample_length=min_sample_length, max_length=max_buffer_length)

    train(
        n_iterations=n_iterations, agent=agent, buffer=buffer, env=env,
        batch_size=batch_size, n_sac_updates_per_episode=n_sac_updates_per_episode,
          )