import jax
import numpy as np
import optax
import jax.numpy as jnp
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from functools import partial
import os
from datetime import datetime, date

from hydrocarbon_problem.api.aspen_api import AspenAPI
from hydrocarbon_problem.api.fake_api import FakeDistillationAPI
from hydrocarbon_problem.agents.sac.agent import create_agent, Agent
from hydrocarbon_problem.agents.sac.buffer import ReplayBuffer
from hydrocarbon_problem.agents.sac.create_networks import create_sac_networks
from hydrocarbon_problem.env.env import AspenDistillation, ProductSpecification
from hydrocarbon_problem.agents.base import NextObservation, Transition
from hydrocarbon_problem.agents.logger import ListLogger, plot_history
from hydrocarbon_problem.api.Simulation import Simulation



def train(n_iterations: int,
          env: AspenDistillation,
          agent: Agent,
          buffer: ReplayBuffer,
          key = jax.random.PRNGKey(0),
          n_sac_updates_per_episode: int = 3,
          batch_size: int = 32,
          set_agent: str = "random",
          ):

    # initialise the buffer state (filling it with random experience)
    key, subkey = jax.random.split(key)
    buffer_select_action = partial(agent.select_action, agent.state)
    buffer_state = buffer.init(subkey, env, select_action=buffer_select_action)


    today = date.today()
    current_time = datetime.now()
    current_time = current_time.strftime("%H-%M-%S")
    today = today.strftime("%Y-%m-%d")

    if set_agent == "sac":
        logger = ListLogger(save_period=1, save=True, save_path=f"./results/{today}-{current_time}_logging_hist_DDPG_PID"
                                                                f"_{n_iterations}_batch_and_NN_64_LR_1e-4.pkl")
    elif set_agent == "random":
        logger = ListLogger(save_period=1, save=True, save_path=f"./results/{today}_{current_time}_logging_hist_random_agent_{n_iterations}_scaled_reward.pkl")

    pbar = tqdm(range(n_iterations))
    # now run the training loop
    for i in pbar:
        # run an episode, adding the new experience to the buffer
        episode_return = 0.0
        timestep = env.reset()
        previous_timestep = timestep
        episode_start_time = time.time()
        counter = 1
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
            buffer_state = buffer.add(transition, buffer_state)
            previous_timestep = timestep

            step_metrics = env.info
            step_metrics["Contact"] = env.contact

            if env.contact:
                step_metrics["TopStream"] = step_metrics["TopStream"]._replace(episode=i)
                step_metrics["BottomStream"] = step_metrics["BottomStream"]._replace(episode=i)
                step_metrics["Column"] = step_metrics["Column"]._replace(episode=i)
                step_metrics["Column"] = step_metrics["Column"]._replace(diameter=step_metrics["Diameter"])
                step_metrics["Column"] = step_metrics["Column"]._replace(height=step_metrics["Height"])
                step_metrics["Column"] = step_metrics["Column"]._replace(n_stages=step_metrics["n_stages"])
                step_metrics["Column"] = step_metrics["Column"]._replace(column_number=counter)
                logger.write(step_metrics)
            else:
                step_metrics["Unconverged"] = step_metrics
                logger.write(step_metrics)
            counter += 1

        # save useful metrics
        if env.contact:
            metrics = {"episode_return": episode_return,
                       "episode_time": time.time() - episode_start_time}
            logger.write(metrics)
            print(f"Episode return: {episode_return}")
        else:
            pass
        pbar.set_description(f"episode return of {episode_return:.2f}")

        # now update the SAC agent
        if agent_type == "sac":
            sac_start_time = time.time()
            for j in range(n_sac_updates_per_episode):
                # sample a batch from the replay buffer
                key, subkey = jax.random.split(key)
                batch = buffer.sample(buffer_state, subkey, batch_size)
                # chex.assert_tree_all_finite(batch)

                # update the agent using the sampled batch
                agent_state, info = agent.update(agent.state, batch)
                # chex.assert_tree_all_finite(agent_state)
                agent = agent._replace(state=agent_state)
                logger.write(info)

            logger.write({"agent_step_time": time.time() - sac_start_time})
        # if (i % 1000) == 0:
        #     print("100 episodes passed, restart Aspen")
        #     env.flowsheet_api.restart_aspen()

    plot_history(logger.history)
    plt.show()


if __name__ == '__main__':

    agent_type = "random"

    DISABLE_JIT = False  # useful for debugging
    if DISABLE_JIT:
        from jax.config import config
        config.update('jax_disable_jit', DISABLE_JIT)
    SUPRESS_WARNINGS = True
    if SUPRESS_WARNINGS:
        # If we don't want to print warnings.
        # Should be used with care.
        import logging
        os.chdir("results")
        logger = logging.getLogger("root")
        logger_unconverged = logging.getLogger("root")

        class CheckTypesFilter(logging.Filter):
            def filter(self, record):
                return "check_types" not in record.getMessage()
        logger.addFilter(CheckTypesFilter())
        logger_unconverged.addFilter(CheckTypesFilter())

    n_iterations = 3000
    batch_size = 32
    n_sac_updates_per_episode = 1

    # You can replay the fake flowsheet here with the actual aspen flowsheet.
    env = AspenDistillation(flowsheet_api=AspenAPI(),  # FakeDistillationAPI(),  # FakeDistillationAPI(), AspenAPI()
                            product_spec=ProductSpecification(purity=0.95),
                            max_steps=8)

    if agent_type == "sac":
        sac_net = create_sac_networks(env=env,
                                      policy_hidden_units=(32, 32),
                                      q_value_hidden_units=(64, 64))

        agent = create_agent(networks=sac_net,
                             rng_key=jax.random.PRNGKey(0),
                             policy_optimizer=optax.adam(1e-4),
                             q_optimizer=optax.adam(1e-4)
                             )
    else:
        from hydrocarbon_problem.agents.random_agent.random_agent import create_random_agent
        assert agent_type == "random"
        agent = create_random_agent(env)

    min_sample_length = batch_size * 1#0
    max_buffer_length = batch_size*10#0000
    rng_key = jax.random.PRNGKey(0)
    buffer = ReplayBuffer(min_sample_length=min_sample_length, max_length=max_buffer_length)

    train(
        n_iterations=n_iterations, agent=agent, buffer=buffer, env=env,
        batch_size=batch_size, n_sac_updates_per_episode=n_sac_updates_per_episode, set_agent=agent_type)
