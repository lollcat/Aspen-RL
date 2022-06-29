from typing import NamedTuple, Tuple, Dict

from hydrocarbon_problem.agents.base import SelectAction, Action, AgentUpdate, Transition
from hydrocarbon_problem.env.env import AspenDistillation
import jax.numpy as jnp
import jax
import chex


class RandomAgentState(NamedTuple):
    counter: chex.Array = jnp.zeros(1)
    params: chex.ArrayTree = ()

class Agent(NamedTuple):
    select_action: SelectAction
    update: AgentUpdate
    state: RandomAgentState

def create_random_agent(env: AspenDistillation) -> Tuple[SelectAction, AgentUpdate]:
    discrete_spec, continuous_spec = env.action_spec()

    def select_action(agent_params: chex.ArrayTree, observation: chex.Array,
                      random_key: chex.PRNGKey) -> Action:
        """Randomly choose an action."""
        del observation, agent_params
        discrete_action_key, continuous_action_key = jax.random.split(random_key)
        # discrete_action = jax.random.choice(key=discrete_action_key,
        #                                            a=jnp.arange(discrete_spec.num_values, dtype=int))
        discrete_action = True
        continuous_action = jax.random.uniform(key=continuous_action_key,
                                               shape=continuous_spec.shape,
                                               minval=continuous_spec.minimum,
                                               maxval=continuous_spec.maximum)
        return discrete_action, continuous_action

    def agent_update(agent_state: RandomAgentState, batch: Transition) -> \
            Tuple[RandomAgentState, Dict]:
        """The random agent update does not do stochastic gradient descent with the batch,
        and rather just increments a counter."""
        del batch
        agent_state = RandomAgentState(agent_state.counter + 1)
        info = {"blank": 0.0}
        return agent_state, info

    init_state = RandomAgentState()

    return Agent(select_action=select_action,
                 update=agent_update,
                 state=init_state)


