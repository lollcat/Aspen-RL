"""Taken from acme: https://github.com/deepmind/acme/blob/master/acme/agents/jax/sac/networks.py"""
from typing import Tuple, Protocol, Union, NamedTuple

import chex
import tensorflow_probability.substrates.jax as tfp
from hydrocarbon_problem.agents.base import Observation, Action, NextObservation

DistParams = tfp.distributions.Distribution

class NextDistParams(NamedTuple):
    # parameters describing distribution over action in both tops and bottoms
    params: Tuple[DistParams, DistParams]
    discounts: Tuple[chex.Array, chex.Array]

NextAction = Tuple[Action, Action]


class PolicyNetwork(Protocol):
    def init(self, seed) -> chex.ArrayTree:
        """Initialises the policy network."""
        raise NotImplementedError

    def apply(self, policy_params: chex.ArrayTree,
              observation: Union[Observation, NextObservation]) -> \
            Union[DistParams, NextDistParams]:
        """
        Args:
          policy_params: Parameters of the policy network.
          observation: Observation/next observation from the environment.

        Returns:
          distribution_params: Specification of the distribution over actions.
        """
        raise NotImplementedError


class QNetwork(Protocol):
    def init(self, seed) -> chex.ArrayTree:
        """Initialises the value network."""
        raise NotImplementedError


    def apply(self, q_params: chex.ArrayTree, observation: Union[Observation, NextObservation],
              action: Union[Action, NextAction]) -> chex.Array:
        """
        Args:
            q_params: Parameters of the q-network
            observation: Either the observation of the current stream, or of the next stream.
                We have different types for each of these.
            action: Action over current stream, or over both tops and bottoms (as a tuple).

        Returns:
            q_value: Q-Value for the observation/next_observation & action/next_action.

        """

class LogProbFn(Protocol):
    def __call__(self, dist_params: Union[DistParams, NextDistParams],
                        action: Union[Action, NextAction]) -> chex.Array:
        """

        Args:
            dist_params: Parameter of action or next actions.
            action: Action (if current stream) or actions over tops and bottoms.

        Returns:
            log_prob: Log probability of action(s).
        """

class SampleFn(Protocol):
    def __call__(self, dist_params: Union[DistParams, NextDistParams],
                 seed: chex.PRNGKey) -> Union[Action, NextAction]:
        """
        Sample an action or actions.
        Args:
            dist_params: Parameters of the distribution over the action(s).
            seed: Source of randomness.

        Returns:
            action: Action (feed_stream) or tuple pair of actions (tops & bottoms).
        """


class SACNetworks(NamedTuple):
    """Network and pure functions for the SAC agent.."""
    policy_network: PolicyNetwork
    q_network: QNetwork
    log_prob: LogProbFn
    sample: SampleFn
    # sample_eval: Optional[networks_lib.SampleFn] = None