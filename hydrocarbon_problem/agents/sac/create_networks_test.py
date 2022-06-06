import chex
import jax.random

from hydrocarbon_problem.agents.sac.create_networks import create_sac_networks
from hydrocarbon_problem.env.env import AspenDistillation
from hydrocarbon_problem.agents.sac.networks import SACNetworks


def test_policy_net(env: AspenDistillation, network: SACNetworks):
    seed = jax.random.PRNGKey(0)
    obs = env.observation_spec().generate_value()
    action = env.action_spec()[0].generate_value(), env.action_spec()[1].generate_value()

    critic_params = network.q_network.init(seed, obs, action)

    q_value = network.q_network.apply(critic_params, obs, action)
    chex.assert_shape(q_value, (1,))



if __name__ == '__main__':
    from hydrocarbon_problem.api.fake_api import FakeDistillationAPI

    env = AspenDistillation(flowsheet_api=FakeDistillationAPI())
    sac_net = create_sac_networks(env=env,
                        policy_hidden_units = (3,), q_value_hidden_units = (10, 10))
    test_policy_net(env, sac_net)