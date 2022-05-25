import chex

from hydrocarbon_problem.agents.random_agent import create_random_agent, RandomAgentState
from hydrocarbon_problem.env.env import AspenDistillation
from hydrocarbon_problem.env.types_ import Discount
from hydrocarbon_problem.agents.base import SelectAction, AgentUpdate, Transition

import jax
import jax.numpy as jnp
import numpy as np


def create_fake_batch(env: AspenDistillation,
                      batch_size: int = 11) -> Transition:
    # generate fake action
    discrete_spec, continuous_spec = env.action_spec()
    fake_action = (discrete_spec.generate_value(), continuous_spec.generate_value())

    # generate fake discount
    discount_spec = env.discount_spec()
    fake_discount = Discount(overall=discount_spec.overall.generate_value(),
                             created_states=(
                                 discount_spec.created_states[0].generate_value(),
                                 discount_spec.created_states[1].generate_value()
                                            )
                             )

    # generate fake next observation
    next_observation = (env.observation_spec().generate_value(),
                       env.observation_spec().generate_value())

    transition = Transition(observation=env.observation_spec().generate_value(),
                            action=fake_action,
                            reward=env.reward_spec().generate_value(),
                            discount=fake_discount,
                            next_observation=next_observation)


    def stack_array_to_batch(array: chex.Array) -> chex.Array:
        return jnp.stack((array, ) * batch_size)

    batch = jax.tree_map(stack_array_to_batch, transition)
    return batch



def test_select_action(select_action: SelectAction,
                       observation: chex.Array,
                       agent_params: chex.ArrayTree,
                       random_key: chex.PRNGKey,
                       env: AspenDistillation) -> None:

    action = select_action(agent_params=agent_params, observation=observation,
                           random_key=random_key)
    discrete_spec, continuous_spec = env.action_spec()

    discrete_action, continuous_action = action_to_numpy(action)
    discrete_spec.validate(discrete_action)
    continuous_spec.validate(continuous_action)

    print("action selection tested")


def test_agent_update(agent_update: AgentUpdate,
                      agent_params: chex.ArrayTree,
                      batch: Transition) -> None:
    new_agent_params, info = agent_update(agent_params, batch)

    equal_leaves = jax.tree_map(lambda x, y: (x == y).all(), new_agent_params, agent_params)
    agent_params_different = jax.tree_flatten(equal_leaves)[0]
    assert not np.all(agent_params_different)

    print("agent update tested")



if __name__ == '__main__':
    # setup
    from hydrocarbon_problem.api.fake_api import FakeDistillationAPI
    from hydrocarbon_problem.agents.base import action_to_numpy
    env = AspenDistillation(flowsheet_api=FakeDistillationAPI())
    select_action, update_agent = create_random_agent(env)
    agent_state = RandomAgentState()

    timestep = env.reset()
    random_key = jax.random.PRNGKey(0)

    # Test action selection.
    test_select_action(select_action, timestep.observation, agent_state.params, random_key, env)


    # Mow test agent update.
    batch = create_fake_batch(env)
    test_agent_update(update_agent, agent_state, batch)





