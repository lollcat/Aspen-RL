import chex
import jax.random
import jax.numpy as jnp

from hydrocarbon_problem.agents.sac.create_networks import create_sac_networks
from hydrocarbon_problem.env.env import AspenDistillation
from hydrocarbon_problem.agents.base import NextObservation
from hydrocarbon_problem.agents.sac.networks import SACNetworks


def test_critic_net(env: AspenDistillation, network: SACNetworks):
    seed = jax.random.PRNGKey(0)
    obs = env.observation_spec().generate_value()
    action = env.action_spec()[0].generate_value(), env.action_spec()[1].generate_value()

    critic_params = network.q_network.init(seed)

    q_value = network.q_network.apply(critic_params, obs, action)
    chex.assert_shape(q_value, (2,))

    # test with non-0 discount
    next_obs = NextObservation(observation=(obs, obs),
                               discounts=(jnp.array(1.0), jnp.array(1.0)))
    next_action = action, action
    next_q_value = network.q_network.apply(critic_params, next_obs, next_action)
    chex.assert_shape(q_value, (2,))
    assert (next_q_value == q_value*2).all()  # should be double the q value for 2 streams

    # test with 0 discount
    next_obs = NextObservation(observation=(obs, obs),
                               discounts=(jnp.array(0.0), jnp.array(0.0)))
    next_action = action, action
    next_q_value = network.q_network.apply(critic_params, next_obs, next_action)
    chex.assert_shape(q_value, (2,))
    assert (next_q_value == 0.0).all()

    print("passed critic tests")


def test_policy_net(env: AspenDistillation, network: SACNetworks):
    seed = jax.random.PRNGKey(0)
    obs = env.observation_spec().generate_value()

    policy_params = network.policy_network.init(seed)

    # Now check for single obs.
    dist_params = network.policy_network.apply(policy_params, obs)
    action = network.sample(dist_params, seed)
    log_prob = network.log_prob(dist_params, action)
    chex.assert_shape(log_prob, ())


    # Now for next obs
    # test with non-0 discount
    next_obs = NextObservation(observation=(obs, obs),
                               discounts=(jnp.array(1.0), jnp.array(1.0)))
    next_dist_params = network.policy_network.apply(policy_params, next_obs)
    assert isinstance(next_dist_params, tuple)
    next_action_ = network.sample(next_dist_params, seed)
    next_action = (action, action)
    chex.assert_tree_all_equal_structs(next_action, next_action_)
    next_log_prob = network.log_prob(next_dist_params, next_action)
    chex.assert_shape(next_log_prob, ())
    assert next_log_prob == log_prob*2

    # test with 0 discount
    next_obs = NextObservation(observation=(obs, obs),
                               discounts=(jnp.array(0.0), jnp.array(0.0)))
    next_dist_params = network.policy_network.apply(policy_params, next_obs)
    next_log_prob = network.log_prob(next_dist_params, next_action)
    assert next_log_prob == 0.0  # 0 discounting

    print("passed policy tests")

if __name__ == '__main__':
    from hydrocarbon_problem.api.fake_api import FakeDistillationAPI

    env = AspenDistillation(flowsheet_api=FakeDistillationAPI())
    sac_net = create_sac_networks(env=env,
                        policy_hidden_units = (3,), q_value_hidden_units = (10, 10))
    test_critic_net(env, sac_net)
    test_policy_net(env, sac_net)
