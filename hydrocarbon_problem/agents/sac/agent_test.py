import chex
import jax.random
import jax.numpy as jnp
import optax
import numpy as np

from hydrocarbon_problem.agents.sac.agent import create_agent, Agent
from hydrocarbon_problem.agents.sac.create_networks import create_sac_networks
from hydrocarbon_problem.env.env import AspenDistillation
from hydrocarbon_problem.agents.base import NextObservation, Transition
from hydrocarbon_problem.agents.sac.networks import SACNetworks



def test_agent_select_action(agent: Agent, env: AspenDistillation) -> None:
    # test action selection
    key = jax.random.PRNGKey(0)

    timestep = env.reset()
    while not timestep.last():
        key, subkey = jax.random.split(key)
        action = agent.select_action(agent.params, timestep.observation.upcoming_state, subkey)
        chex.assert_tree_all_finite(action)
        timestep = env.step(action)
        print(f"took action {action}")

    print("passed agent select action test")


def test_agent_update(agent: Agent, env: AspenDistillation) -> None:
    obs = env.observation_spec().generate_value()
    next_obs = NextObservation(observation=(obs, obs), discounts=(np.array([1.0]), np.array([1.0])))
    batch_size = 10
    transition = Transition(
        observation=obs,
        action=(env.action_spec()[0].generate_value(), env.action_spec()[1].generate_value()),
        reward = env.reward_spec().generate_value(),
        discount=env.discount_spec().overall.generate_value(),
        next_observation=next_obs
    )
    batch = jax.tree_map(
        lambda x: jnp.broadcast_to(x, shape=(batch_size, *x.shape)), transition
    )

    agent_state, info = agent.update(agent.params, batch)
    print("passed agent update test")


if __name__ == '__main__':
    from hydrocarbon_problem.api.fake_api import FakeDistillationAPI

    env = AspenDistillation(flowsheet_api=FakeDistillationAPI())
    sac_net = create_sac_networks(env=env,
                        policy_hidden_units = (3,), q_value_hidden_units = (10, 10))

    agent = create_agent(networks=sac_net,
                         rng_key=jax.random.PRNGKey(0),
                         policy_optimizer=optax.adam(1e-3),
                         q_optimizer=optax.adam(1e-3)
                         )

    test_agent_select_action(agent, env)
    test_agent_update(agent, env)
