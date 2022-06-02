"""Adapted from acme: https://github.com/deepmind/acme/blob/master/acme/agents/jax/d4pg/learning.py"""

import time
from typing import Dict, Iterator, List, NamedTuple, Optional, Tuple

import acme
from acme import types
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.utils import counting
from acme.utils import loggers
import jax
import jax.numpy as jnp
import optax
import reverb
import rlax

class TrainingState(NamedTuple):
    """Contains training state for the learner."""
    policy_params: chex.ArrayTree
    target_policy_params: chex.ArrayTree
    critic_params: chex.ArrayTree
    target_critic_params: chex.ArrayTree
    policy_opt_state: optax.OptState
    critic_opt_state: optax.OptState
    steps: int

class D4PGLearner(acme.Learner):

