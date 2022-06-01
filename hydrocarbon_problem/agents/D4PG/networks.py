""""Taken from acme: https://github.com/deepmind/acme/blob/master/acme/agents/jax/d4pg/networks.py"""
"""D4PG networks definition."""

import dataclasses
from typing import Tuple, Protocol, Union, NamedTuple

import chex

from hydrocarbon_problem.agents.base import Observation, Action, NextObservation


class PolicyParams(NamedTuple):
    mean: chex.Array
    log_var: chex.Array

NextPolicyParams = Tuple[PolicyParams, PolicyParams]  # 2 values, 1 for top and 1 for bottom
NextAction = Tuple[Action, Action]  # 2 values, 1 for top and 1 for bottom

class PolicyNetwork(Protocol):
    def init(self, seed: chex.PRNGKey, observation: Observation) -> chex.ArrayTree:
        """"Initialises the policy network"""
        raise NotImplementedError

# I guess it returns an action based on: line 89 jax/D4PG/learning.py, does dpg_a_t stand for DPG action @ time t?
    def apply(self, policy_params: chex.ArrayTree,
              observation: Union[Observation, NextObservation]) -> Action:
        """
        Args:
          policy_params: Parameters of the policy network.
          observation: Observation from the environment.
        Returns:
          Action based on the current observation.
        """
        raise NotImplementedError

class CriticNetwork(Protocol):
    def init(self,):

#  Online Observation and Action is actually transition.observation/action. The types are similar, so I think this should be okay
#  The code of ACME returns q_tm1, atoms_tm1. These can both be returned in a single ArrayTree, right?
    def apply(self, critic_param: chex.ArrayTree,
              observation: Union[Observation, NextObservation],
              action: Union[Action, NextAction]) -> chex.ArrayTree:
        """
        Args:
            critic_param: Parameters of the critic network
            observation: The observation of the current stream, or of the 2 created streams
            action: Just performed action

        Returns:
            Q values and Number of atoms in output layer of distributional critic https://github.com/msinto93/D4PG/blob/master/params.py
        """


class D4PGNetworks(NamedTuple):
    """Network and pure functions for the D4PG agent"""
    policy_network: PolicyNetwork
    critic_network: CriticNetwork