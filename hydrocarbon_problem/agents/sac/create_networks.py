from typing import Tuple, Union

import chex
import haiku as hk
import jax.numpy as jnp
import jax.random
import tensorflow_probability.substrates.jax as tfp

from hydrocarbon_problem.agents.sac.networks import SACNetworks, DistParams, NextDistParams, NextAction
from hydrocarbon_problem.agents.base import Observation, Action, NextObservation
from hydrocarbon_problem.env.env import AspenDistillation




def create_sac_networks(env: AspenDistillation,
                        policy_hidden_units: Tuple[int, ...] = (10, 10),
                        q_value_hidden_units: Tuple[int, ...] = (10, 10),
                        ):
    """Create SAC networks, for now the agent always separates the given stream."""

    # first let's make the policy network
    continuous_action_shape = env.action_spec()[1].shape
    assert len(continuous_action_shape) == 1 # check that actions are vector
    n_continuous_action = continuous_action_shape[0]

    def policy_forward_single(observation: Observation) -> DistParams:
        policy_net = hk.nets.MLP(policy_hidden_units + (n_continuous_action * 2),
                                 activate_final=False)
        policy_net_out = policy_net(observation)
        mean, log_var = jnp.split(policy_net_out, 2, axis=-1)
        return DistParams(mean=mean, log_var=log_var)


    def policy_forward(observation: Union[Observation, NextObservation]) -> Union[DistParams, NextDistParams]:
        if isinstance(observation, tuple):
            assert len(observation) == 2
            tops_obs, bottoms_obs = observation
            dist_params_top = policy_forward_single(tops_obs)
            dist_params_bot = policy_forward_single(bottoms_obs)
            dist_params = (dist_params_top, dist_params_bot)
        else:
            dist_params = policy_forward_single(observation)
        return dist_params

    policy_network = hk.without_apply_rng(hk.transform(policy_forward))



    # now the q-value network
    def q_value_network_forward_single(observation: Observation, action: Action) -> chex.Array:
        #TODO for now let's just concat the actions, later we may return 0 value if
        # choose not separate.
        action = jnp.concatenate([action[0][..., None], action[1]], axis=-1)
        q_net = hk.nets.MLP(q_value_hidden_units + (1,), activate_final=False)
        q_net_in = jnp.concatenate([observation, action], axis=-1)
        q_value = q_net(q_net_in)
        return q_value

    def q_network_forward(observation: Union[Observation, NextObservation],
                          action: Union[Action, NextAction]) -> chex.Array:
        if isinstance(observation, tuple):
            assert len(observation) == 2
            assert len(action) == 2
            tops_obs, bottoms_obs = observation
            tops_action, bottoms_action = action
            q_top = q_value_network_forward_single(tops_obs, tops_action)
            q_bottom = q_value_network_forward_single(bottoms_obs, bottoms_action)
            q_value = q_top + q_bottom
        else:
            q_value = q_value_network_forward_single(observation, action)
        return q_value

    q_value_network = hk.without_apply_rng(hk.transform(q_network_forward))


    def get_dist(dist_params: DistParams) -> tfp.distributions.Distribution:
        scale_diag = jnp.diag(jnp.exp(dist_params.log_var))
        dist = tfp.distributions.MultivariateNormalDiag(loc=dist_params.mean,
                                                        scale_diag=scale_diag)
        return dist


    # Now let's create the log prob function
    def log_prob_single(dist_params: DistParams, action: Action) -> chex.Array:
        dist = get_dist(dist_params)
        return dist.log_prob(action)

    def log_prob(dist_params: Union[DistParams, NextDistParams],
                        action: Union[Action, NextAction]) -> chex.Array:
        if isinstance(action, Tuple):
            assert len(action) == 2
            assert len(dist_params) == 2
            continuous_action_tops =  action[0][1]
            continuous_action_bots = action[1][1]
            log_prob_tops = log_prob_single(dist_params[0], continuous_action_tops)
            log_prob_bots = log_prob_single(dist_params[1], continuous_action_bots)
            log_prob = log_prob_bots + log_prob_tops
        else:
            log_prob = log_prob_single(dist_params, action)
        return log_prob


    def sample_single(dist_params: DistParams, seed: chex.PRNGKey) -> Action:
        dist = get_dist(dist_params)
        continuous_action = dist.sample(seed=seed)
        discrete_action = jnp.ones(dist.batch_shape)
        action = discrete_action, continuous_action
        return action


    # Lastly let's create
    def sample(dist_params: Union[DistParams, NextDistParams],
                 seed: chex.PRNGKey) -> Union[Action, NextAction]:
        if isinstance(dist_params, Tuple):
            assert len(dist_params) == 2
            key1, key2 = jax.random.split(seed)
            action_tops = sample_single(dist_params[0], key1)
            action_bots = sample_single(dist_params[1], key2)
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

