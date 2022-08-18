from typing import Tuple, Dict, Iterator, NamedTuple, Callable

from hydrocarbon_problem.agents.base import SelectAction, Action, AgentUpdate, Transition,\
    Observation
import jax
import jax.numpy as jnp
import numpy as np
import optax
import chex
from hydrocarbon_problem.agents.sac.networks import SACNetworks
from hydrocarbon_problem.agents.sac.learning import TrainingState, SACLearner


class Agent(NamedTuple):
    select_action: SelectAction
    update: AgentUpdate
    state: TrainingState
    learner: SACLearner  # useful for debugging


def create_agent(networks: SACNetworks,
                 rng_key: chex.PRNGKey,
                 policy_optimizer: optax.GradientTransformation,
                 q_optimizer: optax.GradientTransformation,
                 auto_tune_alpha: bool = True
                 ) -> Agent:
    """
    To bake in the next_state discounting, we add this to the next_observation field
    so that it can be done internally without effecting the SAC code.

    We have to comment out the processing of next_observation in Acme's SAC agent, thus we
    redefine `learning.py` in this module.
    """

    def select_action(
            agent_params: TrainingState,
            observation: Observation,
            random_key: chex.PRNGKey
    ) -> Action:
        """Select an action in the environment based on an observation"""

        @jax.jit
        def _select_action(agent_params: TrainingState, observation: Observation,
                           random_key: chex.PRNGKey):
            """Select action with jit"""
            dist_params = networks.policy_network.apply(agent_params.policy_params, observation)
            action = networks.sample(dist_params, random_key)  # [1]
            q_value = networks.q_network.apply(agent_params.q_params, observation, action)

            discrete_action = jnp.where(q_value.max() > 0.0, 1, 0)
            continuous_action = action[1]
            action = discrete_action, continuous_action
            return action
        action = _select_action(agent_params, observation, random_key)
        # Convert into numpy array as expected.
        #continuous_action = action[1]
        action = action[0], action[1]
        # action = np.asarray(continuous_action[0]), np.asarray(continuous_action[1])#
        return action


    leaner = SACLearner(networks=networks,
                        rng=rng_key,
                        iterator=None,
                        policy_optimizer=policy_optimizer,
                        q_optimizer=q_optimizer,
                        entropy_coefficient=None if auto_tune_alpha else 1.0
                        )

    @jax.jit
    def update_agent(agent_state: TrainingState, batch: Transition) -> Tuple[TrainingState, Dict]:
        """
        Args:
            agent_state: Current state of the agent (state, optimizer state etc).
            batch: Batch of experience generated through interaction with the environment.

        Returns:
            agent_state: Updated agent state.
            info: Relevant information from the agent update.

        """
        agent_state, info = leaner._update_step(agent_state, batch)
        return agent_state, info


    agent = Agent(select_action=select_action, update=update_agent,
                  state=leaner._state,
                  learner=leaner)

    return agent




