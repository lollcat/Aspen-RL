import os
from typing import Optional

import jax
import numpy as np
import optax
import jax.numpy as jnp
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from functools import partial


from acme.jax.savers import restore_from_path, save_to_path

# from hydrocarbon_problem.api.aspen_api import AspenAPI
from hydrocarbon_problem.api.fake_api import FakeDistillationAPI
from hydrocarbon_problem.agents.sac_pr.prioritised_buffer import PrioritisedReplayBuffer, \
    make_prioritised_replay
from hydrocarbon_problem.agents.sac_pr.agent import create_agent, Agent
from hydrocarbon_problem.agents.sac_pr.create_networks import create_sac_networks
from hydrocarbon_problem.env.env import AspenDistillation, ProductSpecification
from hydrocarbon_problem.agents.base import NextObservation, Transition
from hydrocarbon_problem.agents.logger import ListLogger, plot_history



def train(n_iterations: int,
          env: AspenDistillation,
          agent: Agent,
          buffer: PrioritisedReplayBuffer,
          key = jax.random.PRNGKey(0),
          n_sac_updates_per_episode: int = 3,
          agent_checkpoint_load_dir: Optional[str] = None,
          buffer_state_load_dir: Optional[str] = None,
          do_checkpointing: bool = False,  # if we save checkpoints
          iter_per_checkpoint: int = 100  # how often we save checkpoints
          ):

    if agent_checkpoint_load_dir:
        agent = agent._replace(state=restore_from_path(agent_checkpoint_load_dir))

    if buffer_state_load_dir:
        buffer_state = restore_from_path(buffer_state_load_dir)

    # DEBUG = True
    # if DEBUG:
    #     batch = buffer.sample(buffer_state, subkey, batch_size)
    #     # chex.assert_tree_all_finite(batch)
    #
    #     # update the agent using the sampled batch
    #     # Now we can step through the update function for debugging.
    #     agent_state, info = agent.learner._unjitted_update_step(agent.state, batch)

    logger = ListLogger(save_period=1, save=True, save_path="./results/logging_hist.pkl")

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
            next_obs = NextObservation(observation=timestep.observation.created_states,
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
            if i == 0:
                buffer_state = buffer.init(transition)
            else:
                transition_to_batch_size_of_1 = jax.tree_map(lambda x: x[None, ...],
                                                             transition)
                buffer_state = buffer.add(transition_to_batch_size_of_1, buffer_state)
            previous_timestep = timestep

            step_metrics = env.info
            logger.write(step_metrics)

        # save useful metrics
        metrics = {"episode_return": episode_return,
                   "episode_time": time.time() - episode_start_time}
        logger.write(metrics)
        print(f"Episode return: {episode_return}")
        pbar.set_description(f"episode return of {episode_return:.2f}")

        # Once buffer is fill enough we can start training the agent.
        training_started = False
        if buffer.can_sample(buffer_state):
            if training_started is False:
                print("buffer full, training has begun!")
            training_started = True
            # now update the SAC agent
            sac_start_time = time.time()
            for j in range(n_sac_updates_per_episode):
                # sample a batch from the replay buffer
                key, subkey = jax.random.split(key)
                batch = buffer.sample(buffer_state, subkey)
                batch = batch.transition
                # chex.assert_tree_all_finite(batch)

                # update the agent using the sampled batch
                agent_state, info = agent.update(agent.state, batch)
                # chex.assert_tree_all_finite(agent_state)
                agent = agent._replace(state=agent_state)
                logger.write(info)

            logger.write({"agent_step_time": time.time() - sac_start_time})

            if do_checkpointing:
                if i % iter_per_checkpoint == 0:
                    print(f"saving checkpoint at iteration {i}")
                    save_to_path(f"agent_state_iter{i}", agent.state)
                    save_to_path(f"buffer_state_iter_{i}", buffer_state)


    plot_history(logger.history)
    plt.show()


if __name__ == '__main__':

    DISABLE_JIT = False  # useful for debugging
    if DISABLE_JIT:
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

    n_iterations = 2
    batch_size = 3
    n_sac_updates_per_episode = 1
    agent_name = "SAC"
    do_checkpointing = True
    iter_per_checkpoint = 1 # how often to save checkpoints
    cwd = os.getcwd()
    # agent_checkpoint_load_dir = f"{cwd}/hydrocarbon_problem/agents/sac/agent_state_iter0
    agent_checkpoint_load_dir = None

    # You can replay the fake flowsheet here with the actual aspen flowsheet.
    env = AspenDistillation(flowsheet_api=FakeDistillationAPI(),  # FakeDistillationAPI(), AspenAPI()
                            product_spec=ProductSpecification(purity=0.95),
                            )

    if agent_name == "SAC":
        sac_net = create_sac_networks(env=env,
                                      policy_hidden_units=(10, 10),
                                      q_value_hidden_units=(10, 10))

        agent = create_agent(networks=sac_net,
                             rng_key=jax.random.PRNGKey(0),
                             policy_optimizer=optax.adam(1e-4),
                             q_optimizer=optax.adam(1e-3)
                             )
    else:
        from hydrocarbon_problem.agents.random_agent.random_agent import create_random_agent
        assert agent_name == "random"
        agent = create_random_agent(env)

    min_sample_length = batch_size
    max_buffer_length = batch_size*100
    rng_key = jax.random.PRNGKey(0)
    buffer = make_prioritised_replay(min_length=min_sample_length, max_length=max_buffer_length,
                                     priority_exponent=0.0, sampling_batch_size=batch_size)

    train(
        n_iterations=n_iterations, agent=agent, buffer=buffer, env=env,
        n_sac_updates_per_episode=n_sac_updates_per_episode,
        do_checkpointing=do_checkpointing,
        iter_per_checkpoint=iter_per_checkpoint,
        agent_checkpoint_load_dir=agent_checkpoint_load_dir
          )
