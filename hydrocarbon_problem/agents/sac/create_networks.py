from typing import Tuple, Union, NamedTuple

import chex

from hydrocarbon_problem.agents.sac.networks import SACNetworks, DistParams, NextDistParams, NextAction
from hydrocarbon_problem.agents.base import Observation, Action, NextObservation
from hydrocarbon_problem.env.env import AspenDistillation
import haiku as hk
import jax.numpy as jnp



def create_sac_networks(env: AspenDistillation,
                        policy_hidden_units: Tuple[int, ...] = (10, 10),
                        q_value_hidden_units: Tuple[int, ...] = (10, 10),
                        ):

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






