from typing import Protocol, Tuple, NamedTuple, Dict
import chex
import numpy as np

Action = Tuple[chex.Array, chex.Array]

# next observation contains the tops and bottoms streams
NextObservation = Tuple[chex.Array, chex.Array]
Params = chex.ArrayTree
Observation = chex.Array

class Discount(NamedTuple):
    """Copied from env.types but converted to jax."""
    created_states: Tuple[chex.Array, chex.Array]
    overall: chex.Array


class Transition(NamedTuple):
    """Container for a transition."""
    observation: chex.Array
    action: Action
    reward: chex.Array
    discount: Discount
    next_observation: NextObservation
    extras: chex.ArrayTree = ()


class SelectAction(Protocol):
    def __call__(self, agent_params: chex.ArrayTree, observation: Observation,
                 random_key: chex.PRNGKey) -> Action:
        """
        Function specifying selection of an action by an agent.
        Args:
            agent_params: Parameters defining an agent (i.e. neural net parameters).
            observation: Observation from the environment.
            random_key: Source of randomness.

        Returns:
            to_separate: Whether or not to separate a stream (1 if separate).
            column_spec: Array of floats, used to specify the column.
        """
        raise NotImplementedError


class AgentUpdate(Protocol):
    def __call__(self, agent_state: chex.ArrayTree, batch: Transition) -> \
            Tuple[chex.ArrayTree, Dict]:
        """
        Update the agent's learnt parameters (i.e. weights and biases of its network),
        based on a batch of experience.
        Args:
            agent_state: Parameters of the agent (such as the neural network parameters,
                and the optimizer state)
            batch: Batch of Experience.

        Returns:
            agent_state: Updated agent parameters from stochastic gradient descent using the batch.
            info: Information on the agent update step.
        """


def action_to_numpy(action: Action) -> Tuple[np.array, np.array]:
    """Convert action into what our environment expects."""
    discrete_action = np.asarray(action[0], dtype="int32")
    continuous_action = np.asarray(action[1], dtype=float)
    return discrete_action, continuous_action


