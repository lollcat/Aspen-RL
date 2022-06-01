from typing import NamedTuple, Tuple

from hydrocarbon_problem.agents.base import SelectAction, Action, AgentUpdate, Transition
from hydrocarbon_problem.env.env import AspenDistillation
import jax.numpy as jnp
import jax
import chex
from hydrocarbon_problem.agents.D4PG.networks import D4PGNetworks

def create_agent(networks: D4PGNetworks) -> Tuple[SelectAction, AgentUpdate]:
    def select_action(agent_params:,
                      observation:Observation):

