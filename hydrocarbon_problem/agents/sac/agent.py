from typing import Tuple, Dict, Iterator

from hydrocarbon_problem.agents.base import SelectAction, Action, AgentUpdate, Transition,\
    Observation
import jax
import optax
import chex
import reverb
from hydrocarbon_problem.agents.sac.networks import SACNetworks
from acme.agents.jax.sac.learning import TrainingState, SACLearner



def create_agent(networks: SACNetworks,
                 rng_key: chex.PRNGKey,
                 iterator: Iterator[reverb.ReplaySample],
                 policy_optimizer: optax.GradientTransformation,
                 q_optimizer: optax.GradientTransformation,
                 ) -> Tuple[SelectAction, AgentUpdate]:
    """
    To bake in the next_state discounting, we add this to the next_observation field
    so that it can be done internally without effecting the SAC code.
    """


    def select_action(
            agent_params: chex.ArrayTree,
            observation: Observation,
            random_key: chex.PRNGKey
    ) -> Action:
        """Select an action in the environment based on an observation"""
        policy_params, q_network_params = agent_params
        dist_params = networks.policy_network.apply(policy_params, observation)
        action = networks.sample(dist_params, random_key)
        return action


    leaner = SACLearner(networks=networks,
                        rng=rng_key,
                        iterator=iterator,
                        policy_optimizer=policy_optimizer,
                        q_optimizer=q_optimizer,
                        )

    def update_agent(agent_state: TrainingState, batch: Transition) -> Tuple[TrainingState, Dict]:
        """
        Args:
            agent_state: Current state of the agent (params, optimizer params etc).
            batch: Batch of experience generated through interaction with the environment.

        Returns:
            agent_state: Updated agent state.
            info: Relevant information from the agent update.

        """
        agent_state, info = leaner._update_step(agent_state, batch)
        return agent_state, info

    return select_action, update_agent



if __name__ == '__main__':
    rng_key = jax.random.PRNGKey(0)
    policy_optimizer = optax.adam(learning_rate=1e-4)
    q_optimizer = optax.adam(learning_rate=1e-3)




