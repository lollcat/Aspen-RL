"""Adapted from acme, https://github.com/deepmind/acme/blob/57493053729b9a3e74a152d7a574fa1ef57121b9/acme/agents/jax/d4pg/networks.py"""
from typing import Tuple, Union

import chex
import haiku as hk
import jax.numpy as jnp
import numpy as np
import jax.random
import tensorflow_probability.substrates.jax as tfp

from hydrocarbon_problem.agents.D4PG.networks import D4PGNetworks, NextAction
from hydrocarbon_problem.agents.base import Observation, Action, NextObservation
from hydrocarbon_problem.env.env import AspenDistillation


def create_d4pg_networks(env: AspenDistillation,
                         policy_layer_size: Tuple[int,...] = (300,200),
                         critic_layer_size: Tuple[int, ...] = (400,300),
                         vmin: float = -150.,
                         vmax: float = 150.,
                         num_atoms: int = 51,
                         ) -> D4PGNetworks:
    """Create D4PG networks"""

    action_spec = env.action_spec()[1].shape
    # Check that the actions are a vector.
    # Should n_continuous_action be equal to the number of parameters we set in Aspen?
    assert len(action_spec) == 1
    n_continuous_action = action_spec[0]

    num_dimensions = np.prod(action_spec.shape, dtype=int)
    critic_atoms = jnp.linspace(vmin, vmax, num_atoms)

    # Critic definition
    # At the moment I followed acme, but here I say the output of the critic network is a q_value,
    # while on github it is said that the output is a distribution over state-action values
    # adapted from https://github.com/deepmind/acme/blob/57493053729b9a3e74a152d7a574fa1ef57121b9/acme/agents/jax/d4pg/networks.py
    @hk.without_apply_rng
    @hk.transform
    def critic_network_forward_single(observation: Observation, action: Action) -> chex.Array:
        critic_net = hk.Sequential([
            utils.batch_concat,
            networks_lib.LayerNormMLP(layer_sizes=[*critic_layer_size, num_atoms]),
        ])
        q_value = critic_net([observation, action])
        return q_value, critic_atoms

    def critic_network_forward(critic_params: chex.ArrayTree,
                       observation: Union[Observation, NextObservation],
                       action: Union[Action, NextAction]) -> chex.Array:
        if isinstance(observation, tuple):
            assert len(observation) == 2  # Check for top and bottom observation
            assert len(action) == 2       # Check for top and bottom action
            tops_obs, bottoms_obs = observation
            tops_action, bottoms_action = action
            q_top = critic_network_forward_single.apply(critic_params, tops_obs, tops_action)
            q_bottom = critic_network_forward_single.apply(critic_params, bottoms_obs, bottoms_action)
            q_value = q_top + q_bottom
        else:
            q_value = critic_network_forward_single.apply(critic_params, observation, action)

        return q_value

    critic_network = hk.Transformed(init=critic_network_forward_single.init,
                                    apply=critic_network_forward)

    # Actor definition
    # The policy actually returns the network, not sure if the expected return of type chex.ArrayTree is correct
    @hk.without_apply_rng
    @hk.transform

    def policy_forward_single(observation: Observation) -> chex.ArrayTree:
        """Create policy network:
        LayerNormMLP = definition for the MultiLayer Perceptron
        NearZeroInitializedLinear = linear layer, initialized near zero weights and biases
        TanhToSpec = function to make real-valued inputs match a BoundedArraySpec"""
        policy_net = hk.Sequential([
            utils.batch_concat,
            networks_lib.LayerNormMLP(policy_layer_size, activate_final = False),
            networks_lib.NearZeroInitializedLinear(num_dimensions),
            networks_lib.TanhToSpec(action_spec)
        ])
        return policy_net(observation)

    def policy_forward(observation: Union[Observation, NextObservation]) -> chex.ArrayTree:
        if isinstance(observation, tuple):
            assert len(observation) == 2
            tops_obs, bottoms_obs = observation
            policy_net_top = policy_forward_single.apply(tops_obs)
            policy_net_bot = policy_forward_single.apply(bottoms_obs)
            policy_net = (policy_net_top, policy_net_bot)
        else:
            policy_net = policy_forward_single.apply(observation)
        return policy_net

    policy_network = hk.Transformed(init=policy_forward_single.init,
                                    apply=policy_forward)

    d4pg_networks = D4PGNetworks(policy_network=policy_network,
                                 critic_network=critic_network)

    return d4pg_networks
