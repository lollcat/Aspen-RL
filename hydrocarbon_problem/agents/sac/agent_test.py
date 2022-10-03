import os

import chex
import jax.random
import jax.numpy as jnp
import pickle
import optax
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


from hydrocarbon_problem.agents.sac.agent import create_agent, Agent
from hydrocarbon_problem.agents.sac.create_networks import create_sac_networks
from hydrocarbon_problem.env.env import AspenDistillation
from hydrocarbon_problem.agents.base import NextObservation, Transition
from hydrocarbon_problem.agents.sac.networks import SACNetworks
from hydrocarbon_problem.agents.logger import ListLogger, plot_history


def create_fake_batch(env: AspenDistillation, batch_size: int = 10) -> Transition:
    obs = env.observation_spec().generate_value()
    next_obs = NextObservation(observation=(obs, obs), discounts=(np.array(1.0), np.array(1.0)))
    transition = Transition(
        observation=obs,
        action=(env.action_spec()[0].generate_value(), env.action_spec()[1].generate_value()),
        reward = env.reward_spec().generate_value(),
        discount=np.array(1.0),
        next_observation=next_obs
    )
    batch = jax.tree_map(
        lambda x: jnp.broadcast_to(x, shape=(batch_size, *x.shape)), transition
    )
    batch = batch._replace(discount=batch.discount.at[-1].set(0),
                           observation=jax.random.normal(shape=batch.observation.shape,
                                                  key=jax.random.PRNGKey(0)))
    return batch


def test_agent_select_action(agent: Agent, env: AspenDistillation) -> None:
    # test action selection
    key = jax.random.PRNGKey(0)

    timestep = env.reset()
    while not timestep.last():
        key, subkey = jax.random.split(key)
        action = agent.select_action(agent.state, timestep.observation.upcoming_state, subkey)
        chex.assert_tree_all_finite(action)
        timestep = env.step(action)
        print(f"took action {action}")

    print("passed agent select action test")


def test_agent_update(agent: Agent, env: AspenDistillation) -> None:
    batch = create_fake_batch(env)
    agent_state, info = agent.update(agent.state, batch)
    chex.assert_tree_all_finite(agent_state)
    print("passed agent update test")


def test_agent_overfit(agent: Agent, env: AspenDistillation) -> None:
    batch = create_fake_batch(env)
    logger = ListLogger(
        save_path="C:/Users/s2399016/Documents/Aspen-RL_v2/Aspen-RL/hydrocarbon_problem/agents/sac/agent_test.pkl")
    # logger = ListLogger()
    for i in tqdm(range(10000)):
        agent_state, info = agent.update(agent.state, batch)
        agent = agent._replace(state=agent_state)
        chex.assert_tree_all_finite(agent_state)
        logger.write(info)
    path_to_saved_hist = r"C:/Users/s2399016/Documents/Aspen-RL_v2/Aspen-RL/hydrocarbon_problem/agents/sac/agent_test.pkl"
    hist = pickle.load(open(path_to_saved_hist, "rb"))

    for key in hist:
        a = hist[key]
        a[0] = float(a[0])
    plot_history(logger.history)
    plt.show()
    name = "test"
    plt.savefig(f'Agent_data_{name}.pdf')
    # state, info = agent.learner._unjitted_update_step(agent_state, batch)



if __name__ == '__main__':
    from hydrocarbon_problem.api.fake_api import FakeDistillationAPI
    from hydrocarbon_problem.api.aspen_api import AspenAPI
    env = AspenDistillation(flowsheet_api=FakeDistillationAPI()) #FakeDistillationAPI()AspenAPI()
    sac_net = create_sac_networks(env=env,
                                  policy_hidden_units=(128,128),
                                  q_value_hidden_units=(128, 128))

    agent = create_agent(networks=sac_net,
                         rng_key=jax.random.PRNGKey(0),
                         policy_optimizer=optax.adam(3e-4),
                         q_optimizer=optax.adam(3e-4),
                         auto_tune_alpha=True
                         )

    test_agent_select_action(agent, env)
    test_agent_update(agent, env)
    test_agent_overfit(agent, env)
