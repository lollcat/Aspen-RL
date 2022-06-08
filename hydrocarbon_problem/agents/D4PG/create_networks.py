from typing import Tuple, Union

import chex
import haiku as hk
import jax.numpy as jnp
import jax.random
import tensorflow_probability.substrates.jax as tfp

from hydrocarbon_problem.agents.D4PG.networks import D4PGNetworks, NextAction
from hydrocarbon_problem.agents.base import Observation, Action, NextObservation
from hydrocarbon_problem.env.env import AspenDistillation


def create_d4pg_networks(env: AspenDistillation,
                         policy_hidden_units: Tuple[int,...] = (10,10),
                         critic_hidden_units: Tuple[int, ...] = (10,10),
                         ):
    """Create D4PG networks"""

    continuous_action_shape = env.action_spec()[1].shape
    # Check that the actions are a vector.
    # Should n_continuous_action be equal to the number of parameters we set in Aspen?
    assert len(continuous_action_shape) == 1
    n_continuous_action = continuous_action_shape[0]

    # Define the critic network when handling 1 action and observation
    def critic_network_forward_single(observation: Observation, action: Action):
        action = jnp.concatenate([action[0][..., None], action[1]])
