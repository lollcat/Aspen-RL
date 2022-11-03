from typing import NamedTuple, Callable, Optional

import functools
import chex
import jax.numpy as jnp
import jax

from hydrocarbon_problem.agents.sac import sum_tree

Transition = chex.ArrayTree
Batch = Transition
Priorities = chex.Array
Indices = chex.Array


def get_tree_shape_prefix(tree: chex.ArrayTree, n_axes: int = 1) -> chex.Shape:
    """Get the shape of the leading axes (up to n_axes) of a pytree. This assumes all
    leaves have a common leading axes size (e.g. a common batch size)."""
    flat_tree, tree_def = jax.tree_util.tree_flatten(tree)
    leaf = flat_tree[0]
    leading_axis_shape = leaf.shape[0:n_axes]
    chex.assert_tree_shape_prefix(tree, leading_axis_shape)
    return leading_axis_shape


class PrioritisedReplayBufferState(NamedTuple):
    sum_tree_state: sum_tree.SumTreeState
    transitions: chex.ArrayTree
    current_index: jnp.int32  # keep track of where we overwrote the transition data.
    is_full: jnp.bool_


class PrioritisedReplaySample(NamedTuple):
    transition: Transition
    indices: chex.Array
    priorities: Priorities


class PrioritisedReplayBuffer(NamedTuple):
    # Container for replay buffer functions.
    init: Callable[[Transition], PrioritisedReplayBufferState]  # Init buffer state with a (possible dummy) transition.
    sample: Callable[[PrioritisedReplayBufferState, chex.PRNGKey], PrioritisedReplaySample]
    add: Callable[[Batch, PrioritisedReplayBufferState], PrioritisedReplayBufferState]
    set_priorities: Callable[[Priorities, Indices, PrioritisedReplayBufferState], PrioritisedReplayBufferState]
    can_sample: Callable[[PrioritisedReplayBufferState], jnp.bool_]



def init(transition: Transition, max_length: int) -> PrioritisedReplayBufferState:
    # Set transition value to be empty.
    transition = jax.tree_util.tree_map(jnp.empty_like, transition)
    # Broadcast to fill max_len
    transitions = jax.tree_map(
        lambda x: jnp.stack([x] * max_length), transition)

    # Init the sum_tree
    sum_tree_state = sum_tree.init(max_length)

    # Now can create the state
    state = PrioritisedReplayBufferState(
        transitions=transitions,
        sum_tree_state=sum_tree_state,
        is_full=jnp.bool_(False),
        current_index=jnp.int32(0)
    )
    return state


def add(batch: Batch, state: PrioritisedReplayBufferState,
        priorities: Optional[Priorities] = None,
        priority_exponent: Optional[float] = None) -> \
        PrioritisedReplayBufferState:
    batch_size = get_tree_shape_prefix(batch)[0]
    max_length = get_tree_shape_prefix(state.transitions)[0]
    indices = (jnp.arange(batch_size) + state.current_index) % max_length

    if priorities is None:
        # If no priorities then set to the maximum priority.
        priorities = jnp.zeros((batch_size,)) + state.sum_tree_state.max_recorded_priority
    else:
        priorities = jnp.where(priorities == 0, jnp.zeros_like(priorities),
                               priorities**priority_exponent)
        if priority_exponent is None:
            raise Exception("Must be specified if priority specififed.")
    sum_tree_state = sum_tree.set_batch(state.sum_tree_state, indices, priorities)
    transitions = jax.tree_map(lambda x, y: x.at[indices].set(y),
                               state.transitions, batch)
    new_index = state.current_index + batch_size
    is_full = state.is_full | (new_index >= (max_length-1))
    new_index = new_index % max_length
    state = PrioritisedReplayBufferState(
        sum_tree_state=sum_tree_state,
        transitions=transitions,
        current_index=new_index,
        is_full=is_full
    )
    return state



def sample(state: PrioritisedReplayBufferState, key: chex.PRNGKey, batch_size: int) -> \
        PrioritisedReplaySample:
    indices = sum_tree.stratified_sample(
        state.sum_tree_state,
        batch_size=batch_size,
        rng_key=key,
    )
    transition = jax.tree_util.tree_map(lambda x: x[indices], state.transitions)
    priorities = sum_tree.get(state.sum_tree_state, indices)
    sample = PrioritisedReplaySample(transition, indices, priorities)
    return sample



def set_priorities(
        priorities: Priorities,
        indices: Indices,
        state: PrioritisedReplayBufferState,
        priority_exponent: float
) -> PrioritisedReplayBufferState:
    priorities = jnp.where(priorities == 0, jnp.zeros_like(priorities),
                           priorities ** priority_exponent)
    sum_tree_state = sum_tree.set_batch(state.sum_tree_state, indices, priorities)
    return state._replace(sum_tree_state=sum_tree_state)


def can_sample(state: PrioritisedReplayBufferState, min_length: int) -> jnp.bool_:
    can_sample = state.is_full | (state.current_index >= min_length)
    return can_sample


def make_prioritised_replay(max_length: int, min_length: int, sampling_batch_size: int,
                            priority_exponent: float = 0.6) -> \
        PrioritisedReplayBuffer:
    # Useful Acme links:
    # Use sample default priority exponent as Acme.
    #    - https://github.com/deepmind/acme/blob/1cc8378e4d3418f15c30731e4a8953804a45a519/acme/agents/jax/dqn/builder.py#L121
    init_fn = functools.partial(init, max_length=max_length)
    add_fn = functools.partial(add)
    sample_fn = functools.partial(sample, batch_size=sampling_batch_size)
    set_priorities_fn = functools.partial(set_priorities, priority_exponent=priority_exponent)
    can_sample_fn = functools.partial(can_sample, min_length=min_length)
    return PrioritisedReplayBuffer(
        init=init_fn,
        add=add_fn,
        sample=sample_fn,
        set_priorities=set_priorities_fn,
        can_sample=can_sample_fn
    )