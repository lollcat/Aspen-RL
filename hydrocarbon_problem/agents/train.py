from typing import Optional

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

from acme.jax.savers import restore_from_path, save_to_path

# from hydrocarbon_problem.api.aspen_api import AspenAPI
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
          agent_checkpoint_load_dir: Optional[str] = None,
          buffer_state_load_dir: Optional[str] = None,
          do_checkpointing: bool = False,  # if we save checkpoints
          iter_per_checkpoint: int = 100,  # how often we save checkpoints
          set_agent: str = "random"
          ):
    today = date.today()
    current_time = datetime.now()
    current_time = current_time.strftime("%H-%M-%S")
    today = today.strftime("%Y-%m-%d")
    # pyRAPL.setup()
    # meter = pyRAPL.Measurement('bar')
    checkpoint_path = f"C:/Users/s2399016/Documents/Aspen-RL_v2/Aspen-RL/hydrocarbon_problem/agents/results/{today}-{current_time}/checkpoint"
    logger_path = f"C:/Users/s2399016/Documents/Aspen-RL_v2/Aspen-RL/hydrocarbon_problem/agents/results/{today}-{current_time}/SAC_ActionSpace_Full_logger_{n_iterations}_LRalpha7_5e-5_SAC_updates_{n_sac_updates_per_episode}_steps_{max_steps}"

    isExist = os.path.exists(path=checkpoint_path)
    if not isExist:
        os.makedirs(checkpoint_path)

    # initialise the buffer state (filling it with random experience)

    if agent_checkpoint_load_dir:
        print(os.getcwd())
        # os.chdir("C:/Users/s2399016/Documents/Aspen-RL_v2/Aspen-RL/hydrocarbon_problem/agents/results/updates2")
        agent = agent._replace(state=restore_from_path(agent_checkpoint_load_dir))

    key, subkey = jax.random.split(key)

    if buffer_state_load_dir:
        print(os.getcwd())
        buffer_state = restore_from_path(buffer_state_load_dir)
    else:

        buffer_select_action = partial(agent.select_action, agent.state)
        buffer_state = buffer.init(subkey, env, select_action=buffer_select_action)

    DEBUG = False
    if DEBUG:
        batch = buffer.sample(buffer_state, subkey, batch_size)
        # chex.assert_tree_all_finite(batch)

        # update the agent using the sampled batch
        # Now we can step through the update function for debugging.
        agent_state, info = agent.learner.blob(agent.state,batch)
        agent_state, info = agent.learner._unjitted_update_step(agent.state, batch)

    logger = ListLogger(save_period=1, save=True, save_path=f"{logger_path}.pkl")

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
            step_metrics["Converged"] = env.converged
            step_metrics["Separate"] = env.choose_separate
            step_metrics["FeedStream"] = step_metrics["FeedStream"]._replace(episode=i)
            step_metrics["FeedStream"] = step_metrics["FeedStream"]._replace(separate=env.choose_separate)

            if env.choose_separate and env.contact and (env.converged == 0 or env.converged == 2):
                step_metrics["TopStream"] = step_metrics["TopStream"]._replace(episode=i)
                step_metrics["BottomStream"] = step_metrics["BottomStream"]._replace(episode=i)
                step_metrics["Column"] = step_metrics["Column"]._replace(episode=i)
                step_metrics["Column"] = step_metrics["Column"]._replace(diameter=step_metrics["Diameter"])
                step_metrics["Column"] = step_metrics["Column"]._replace(height=step_metrics["Height"])
                step_metrics["Column"] = step_metrics["Column"]._replace(n_stages=step_metrics["n_stages"])
                step_metrics["Column"] = step_metrics["Column"]._replace(column_number=counter)

            elif not env.choose_separate:
                step_metrics["Feedstream no-separate"] = step_metrics["FeedStream"]
            elif env.choose_separate and (not env.contact or env.converged == 1):
                step_metrics["Unconverged"] = step_metrics
            logger.write(step_metrics)

            counter += 1

        # save useful metrics
        once_per_episode_info = env.once_per_episode_info
        metrics = {"episode_return": episode_return,
                   "episode_time": time.time() - episode_start_time,
                   "Streams yet to be acted on": once_per_episode_info["Streams yet to be acted on"],
                   "n_stagesFirst stream": once_per_episode_info["n_stagesFirst stream"],
                   "feed_stage_locationFirst stream": once_per_episode_info["feed_stage_locationFirst stream"],
                   "reflux_ratioFirst stream": once_per_episode_info["reflux_ratioFirst stream"],
                   "reboil_ratioFirst stream": once_per_episode_info["reboil_ratioFirst stream"],
                   "condensor_pressureFirst stream": once_per_episode_info["condensor_pressureFirst stream"]}


        # metrics = metrics.update(once_per_episode_info)
        print(metrics)
        logger.write(metrics)
        print(f"Episode return: {episode_return}")

        pbar.set_description(f"episode return of {episode_return:.2f}")

        # now update the SAC agent
        if set_agent == "sac":
            sac_start_time = time.time()
            for j in range(n_sac_updates_per_episode):
                # sample a batch from the replay buffer
                key, subkey = jax.random.split(key)
                start = time.time()
                batch = buffer.sample(buffer_state, subkey, batch_size)
                time_to_sample_from_buffer = time.time() - start
                # chex.assert_tree_all_finite(batch)

                # update the agent using the sampled batch
                start = time.time()
                agent_state, info = agent.update(agent.state, batch)
                time_to_update_agent = time.time() - start
                # chex.assert_tree_all_finite(agent_state)
                agent = agent._replace(state=agent_state)
                logger.write(info)

            logger.write({"agent_step_time": time.time() - sac_start_time})
            logger.write({"time_to_sample_from_buffer": time_to_sample_from_buffer})
            logger.write({"time_to_update_agent": time_to_update_agent})

        if (i % 500) == 0:
            print("500 episodes passed, restart Aspen")
            env.flowsheet_api.restart_aspen()

        if do_checkpointing:
            if i % iter_per_checkpoint == 0:
                print(os.getcwd())
                os.chdir(checkpoint_path)
                print(f"saving checkpoint at iteration {i}")
                save_to_path(f"agent_state_iter{i}test", agent.state)
                save_to_path(f"buffer_state_iter_{i}test", buffer_state)


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
        current_time = datetime.now()
        current_time = current_time.strftime("%H-%M-%S")
        today = date.today()
        today = today.strftime("%Y-%m-%d")
        path = f"C:/Users/s2399016/Documents/Aspen-RL_v2/Aspen-RL/hydrocarbon_problem/agents/results/{today}"
        isExist = os.path.exists(path=path)
        if not isExist:
            os.makedirs(path)
        # os.chdir("results")
        logger = logging.getLogger("root")

        class CheckTypesFilter(logging.Filter):
            def filter(self, record):
                return "check_types" not in record.getMessage()
        logger.addFilter(CheckTypesFilter())

    n_iterations = 20000
    reward_scale = 10
    punishment = -10
    max_steps = 4
    batch_size = 1  # 32
    n_sac_updates_per_episode = 4
    do_checkpointing = True
    iter_per_checkpoint = 500  # how often to save checkpoints
    agent_checkpoint_load_dir = None  # r"C:\Users\s2399016\Documents\Aspen-RL_v2\Aspen-RL\hydrocarbon_problem\agents\results\2022-09-02-12-01-31\checkpoint\agent_state_iter10000test"  # None
    buffer_checkpoint_load_dir = None  # r"C:\Users\s2399016\Documents\Aspen-RL_v2\Aspen-RL\hydrocarbon_problem\agents\results\2022-09-02-12-01-31\checkpoint\buffer_state_iter_10000test"  # None

    # You can replay the fake flowsheet here with the actual aspen flowsheet.
    env = AspenDistillation(flowsheet_api=AspenAPI(visibility=False, suppress=True, max_solve_iterations=100),  # FakeDistillationAPI(),
                            product_spec=ProductSpecification(purity=0.95),
                            max_steps=max_steps, small_action_space=False, reward_scale=reward_scale, punishment=punishment)

    if agent_type == "sac":
        sac_net = create_sac_networks(env=env,
                                      policy_hidden_units=(128, 128),
                                      q_value_hidden_units=(128, 128))

        agent = create_agent(networks=sac_net,
                             rng_key=jax.random.PRNGKey(0),
                             policy_optimizer=optax.adam(3e-4),
                             q_optimizer=optax.adam(3e-4)
                             )
    else:
        from hydrocarbon_problem.agents.random_agent.random_agent import create_random_agent
        assert agent_type == "random"
        agent = create_random_agent(env)

    min_sample_length = batch_size * 1#0
    max_buffer_length = batch_size * 10#000
    rng_key = jax.random.PRNGKey(0)
    buffer = ReplayBuffer(min_sample_length=min_sample_length, max_length=max_buffer_length)

    train(
        n_iterations=n_iterations, agent=agent, buffer=buffer, env=env, key=rng_key,
        batch_size=batch_size, n_sac_updates_per_episode=n_sac_updates_per_episode,
        do_checkpointing=do_checkpointing,
        iter_per_checkpoint=iter_per_checkpoint, set_agent=agent_type,
        agent_checkpoint_load_dir=agent_checkpoint_load_dir,
        buffer_state_load_dir=buffer_checkpoint_load_dir
    )
