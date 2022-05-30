""""Taken from acme: https://github.com/deepmind/acme/blob/master/acme/agents/jax/d4pg/networks.py"""
"""D4PG networks definition."""

import dataclasses
from typing import Optional, Callable, Protocol

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from hydrocarbon_problem.agents.base import Params, Observation, Action

class PolicyNetwork(Protocol):
  def init(self, ):

  def apply(self, policy_params: chex.ArrayTree, observation: Observation) -> Action:
      """
      
      Args:
        policy_params: Parameters of the policy network.
        observation: Observation from the environment.

      Returns:
        Action based on the current observation.
      """"
      raise NotImplementedError

PolicyParams = chex.ArrayTree  # Policy parameters   line 89 jax/D4PG/learning.py, does dpg_a_t stand for DPG action @ time t?
PolicyNetwork = Callable[[PolicyParams, Observation], Action]
CriticNetwork =

@dataclasses.dataclass
class D4PGNetworks:
  """Network and pure functions for the D4PG agent.."""
  policy_network: networks_lib.FeedForwardNetwork
  critic_network: networks_lib.FeedForwardNetwork

