from typing import NamedTuple, Tuple, Optional, Dict, Iterator

from hydrocarbon_problem.agents.base import SelectAction, Action, AgentUpdate, Transition, Observation
from hydrocarbon_problem.env.env import AspenDistillation
import jax.numpy as jnp
import jax
import optax
import chex
import reverb
from hydrocarbon_problem.agents.D4PG.networks import D4PGNetworks
from acme.agents.jax.d4pg.learning import TrainingState, D4PGLearner


def create_agent(networks: D4PGNetworks,
                 rng_key: chex.PRNGKey,
                 discount: float,
                 target_update_period: int,
                 iterator: Iterator[reverb.ReplaySample],
                 policy_optimizer: Optional[optax.GradientTransformation] = None,
                 critic_optimizer: Optional[optax.GradientTransformation] = None,
                 clipping: bool = True,
                 counter: Optional[counting.Counter] = None,
                 logger: Optional[loggers.Logger] = None,
                 jit: bool = True,
                 num_sgd_steps_per_step: int = 1,):

    def select_action(params: chex.ArrayTree,
                      observation:Observation,
                      ) -> Action:
        """Select an action in the environment based on an observation
        Adapted from https://github.com/deepmind/acme/blob/57493053729b9a3e74a152d7a574fa1ef57121b9/acme/agents/jax/d4pg/networks.py"""

        action = networks.policy_network.apply(policy_params=params, observation=observation)
        return action

    learner = D4PGLearner(policy_network=D4PGNetworks.policy_network,
                          critic_network=D4PGNetworks.critic_network,
                          random_key=rng_key,
                          discount=discount,
                          target_update_period=target_update_period,
                          iterator=iterator,
                          policy_optimizer=policy_optimizer,
                          critic_optimizer=critic_optimizer,
                          clipping=clipping,
                          counter=counter,
                          logger=logger,
                          jit=jit,
                          num_sgd_steps_per_step=num_sgd_steps_per_step
                          )

    def update_agent(agent_state: TrainingState, batch: Transition) -> Tuple[TrainingState, Dict[str, jnp.ndarray]]:
        """
        Args:
            agent_state: Current state of the agent (params, optimizer params etc).
            batch: Batch of experience generated through interaction with the environment.
        Returns:
            agent_state: Updated agent state.
            info: Relevant information from the agent update.
        """
        agent_state, info = learner._sgd_step(state=agent_state, transition=batch)
        return agent_state, info

    return select_action, update_agent


if __name__=='__main__':
    rng_key = jax.random.PRNGKey(0)
    policy_optimizer = optax.adam(learning_rate=1e-4)
    critic_optimizer = optax.adam(learning_rate=1e-3)

