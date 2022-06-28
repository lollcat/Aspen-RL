from typing import Tuple, Union

import chex
import haiku as hk
import jax.numpy as jnp
import jax.random
import numpy as np

from hydrocarbon_problem.agents.sac.networks import SACNetworks, DistParams, NextDistParams, NextAction
from hydrocarbon_problem.agents.base import Observation, Action, NextObservation
from hydrocarbon_problem.env.env import AspenDistillation
from acme.jax import networks as networks_lib




def create_sac_networks(env: AspenDistillation,
                        policy_hidden_units: Tuple[int, ...] = (10, 10),
                        q_value_hidden_units: Tuple[int, ...] = (10, 10),
                        ):
    """Create SAC networks, for now the agent always separates the given stream."""

    continuous_action_shape = env.action_spec()[1].shape
    assert len(continuous_action_shape) == 1 # check that actions are vector
    n_continuous_action = continuous_action_shape[0]


    # Define the q-value network
    @hk.without_apply_rng
    @hk.transform
    def q_value_network_forward_single(observation: Observation, action: Action) -> chex.Array:
        network1 = hk.Sequential([
            hk.nets.MLP(
                list(q_value_hidden_units) + [1],
                w_init=hk.initializers.VarianceScaling(1.0, 'fan_in', 'uniform'),
                activation=jax.nn.relu),
        ])
        network2 = hk.Sequential([
            hk.nets.MLP(
                list(q_value_hidden_units) + [1],
                w_init=hk.initializers.VarianceScaling(1.0, 'fan_in', 'uniform'),
                activation=jax.nn.relu),
        ])
        action = jnp.concatenate([action[0][..., None], action[1]], axis=-1)
        input_ = jnp.concatenate([observation, action], axis=-1)
        value1 = network1(input_)
        value2 = network2(input_)
        #TODO for now let's just concat the actions, later we may return 0 value if
        # choose not separate.
        return jnp.concatenate([value1, value2], axis=-1)

    def q_network_forward(q_params: chex.ArrayTree,
                          observation: Union[Observation, NextObservation],
                          action: Union[Action, NextAction]) -> chex.Array:
        if isinstance(observation, NextObservation):
            tops_obs, bottoms_obs = observation.observation
            tops_discount, bottoms_discount = observation.discounts
            tops_action, bottoms_action = action
            q_top = q_value_network_forward_single.apply(q_params, tops_obs, tops_action)
            q_bottom = q_value_network_forward_single.apply(q_params, bottoms_obs, bottoms_action)
            q_value = q_top*tops_discount[..., None] + q_bottom*bottoms_discount[..., None]
        else:
            q_value = q_value_network_forward_single.apply(q_params, observation, action)
        return q_value

    def q_value_net_init(key: chex.PRNGKey) -> chex.ArrayTree:
        """Initialise the q value network using dummy observations"""
        obs = env.observation_spec().generate_value()
        action = env.action_spec()[0].generate_value(), env.action_spec()[1].generate_value()

        critic_params = q_value_network_forward_single.init(key, obs, action)
        return critic_params

    q_value_network = hk.Transformed(init=q_value_net_init,
                                     apply=q_network_forward)



    # Now let's define the policy.
    @hk.without_apply_rng
    @hk.transform
    def policy_forward_single(observation: Observation) -> DistParams:
        network = hk.Sequential([
            hk.nets.MLP(
                policy_hidden_units,
                w_init=hk.initializers.VarianceScaling(1.0, 'fan_in', 'uniform'),
                activation=jax.nn.relu,
                activate_final=True),
            networks_lib.NormalTanhDistribution(n_continuous_action),
        ])
        return network(observation)


    def policy_forward(policy_params: chex.ArrayTree,
                       observation: Union[Observation, NextObservation]) -> \
            Union[DistParams, NextDistParams]:
        if isinstance(observation, NextObservation):
            tops_obs, bottoms_obs = observation.observation
            dist_params_top = policy_forward_single.apply(policy_params, tops_obs)
            dist_params_bot = policy_forward_single.apply(policy_params, bottoms_obs)
            dist_params = NextDistParams(params=(dist_params_top, dist_params_bot),
                                         discounts=observation.discounts)
        else:
            dist_params = policy_forward_single.apply(policy_params, observation)
        return dist_params

    def policy_init(key: chex.PRNGKey) -> chex.ArrayTree:
        obs = env.observation_spec().generate_value()
        policy_params = policy_forward_single.init(key, obs)
        return policy_params

    policy_network = hk.Transformed(init=policy_init,
                                    apply=policy_forward)


    # Now let's create the log prob function jax.nn.softplus(dist_params.log_var)
    def log_prob_single(dist_params: DistParams, action: Action) -> chex.Array:
        continuos_action = action[1]
        return dist_params.log_prob(continuos_action)


    def log_prob(dist_params: Union[DistParams, NextDistParams],
                        action: Union[Action, NextAction]) -> chex.Array:
        if isinstance(dist_params, NextDistParams):
            action_tops = action[0]
            action_bots = action[1]
            log_prob_tops = log_prob_single(dist_params.params[0], action_tops)
            log_prob_bots = log_prob_single(dist_params.params[1], action_bots)
            log_prob = log_prob_bots*dist_params.discounts[1] + \
                       log_prob_tops*dist_params.discounts[0]
        else:
            log_prob = log_prob_single(dist_params, action)
        return log_prob


    def sample_single(dist_params: DistParams, seed: chex.PRNGKey) -> Action:
        continuous_action = dist_params.sample(seed=seed)
        discrete_action = jnp.ones(dist_params.batch_shape)
        action = discrete_action, continuous_action
        return action


    # Lastly let's create
    def sample(dist_params: Union[DistParams, NextDistParams],
                 seed: chex.PRNGKey) -> Union[Action, NextAction]:
        if isinstance(dist_params, NextDistParams):
            key1, key2 = jax.random.split(seed)
            action_tops = sample_single(dist_params.params[0], key1)
            action_bots = sample_single(dist_params.params[1], key2)
            action = action_tops, action_bots
        else:
            action = sample_single(dist_params, seed)
        return action


    sac_networks = SACNetworks(policy_network=policy_network,
                               q_network=q_value_network,
                               log_prob=log_prob,
                               sample=sample
                               )
    return sac_networks
