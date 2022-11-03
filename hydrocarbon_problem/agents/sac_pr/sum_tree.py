from typing import List, NamedTuple, Optional

import chex
import jax
import jax.numpy as jnp
import numpy as np

"""
 Pure functions defining a sum-tree data structure. The desired use is within a prioritised replay
 buffer, see Prioritized Experience Replay by Schaul et al. (2015) and prioritised_replay.py.
 This is an adaption of the sum-tree implementation from
 dopamine: https://github.com/google/dopamine/blob/master/dopamine/replay_memory/sum_tree.py.
 Lots of the code is verbatim copied.
 The key differences between this implementation and the dopamine implementation are (1) This
 implementation is in jax with a functional style, and (2) this implementation focuses on
 vectorised adding (rather sequential adding).
"""


class SumTreeState(NamedTuple):
    nodes: List[chex.Array]
    max_recorded_priority: jnp.float32


def init(capacity: int) -> SumTreeState:
    """Creates the sum tree data structure for the given replay capacity.
    Args:
        capacity: The maximum number of elements that can be stored in this
            data structure.
    """
    nodes = []
    tree_depth = int(np.ceil(np.log2(capacity)))
    level_size = 1
    for _ in range(tree_depth + 1):
        nodes_at_this_depth = jnp.zeros(level_size)
        nodes.append(nodes_at_this_depth)
        level_size *= 2

    max_recorded_priority = jnp.array(1.0)
    return SumTreeState(nodes=nodes, max_recorded_priority=max_recorded_priority)


def _total_priority(state: SumTreeState) -> jnp.float32:
    """Returns the sum of all priorities stored in this sum tree.
    Returns:
        Sum of priorities stored in this sum tree.
    """
    return state.nodes[0][0]


def sample(
    state: SumTreeState,
    rng_key: Optional[chex.PRNGKey] = None,
    query_value: Optional[jnp.float32] = None,
) -> jnp.int32:
    """Samples an element from the sum tree.
    Each element has probability p_i / sum_j p_j of being picked, where p_i is
    the (positive) value associated with node i (possibly unnormalized).
    Args:
        query_value: float in [0, 1], used as the random value to select a
            sample. If None, will select one randomly in [0, 1).
    Returns:
        A random element from the sum tree.
    """

    # Sample a value in range [0, R), where R is the value stored at the root.
    if query_value is None:
        assert rng_key is not None
        query_value = jax.random.uniform(rng_key)
    else:
        assert query_value is not None
    query_value = query_value * _total_priority(state)

    # Now traverse the sum tree.
    node_index = 0
    for nodes_at_this_depth in state.nodes[1:]:
        # Compute children of previous depth's node.
        left_child = node_index * 2

        left_sum = nodes_at_this_depth[left_child]
        # Each subtree describes a range [0, a), where a is its value.
        query_value, node_index = jax.lax.cond(
            query_value < left_sum,
            lambda qv, lc, ls: (qv, lc),
            lambda qv, lc, ls: (qv - ls, lc + 1),
            query_value,
            left_child,
            left_sum,
        )
    return node_index


def stratified_sample(
    state: SumTreeState, batch_size: int, rng_key: chex.PRNGKey
) -> chex.Array:
    """Performs stratified sampling using the sum tree.
    Let R be the value at the root (total value of sum tree). This method will
    divide [0, R) into batch_size segments, pick a random number from each of
    those segments, and use that random number to sample from the sum_tree. This
    is as specified in Schaul et al. (2015).
    Args:
        state: Current state of the sum-tree.
        batch_size: The number of strata to use.
    Returns:
        Batch of indices sampled from the sum tree.
    """
    bounds = jnp.linspace(0.0, 1.0, batch_size + 1)
    assert len(bounds) == batch_size + 1
    segments = [(bounds[i], bounds[i + 1]) for i in range(batch_size)]
    query_values = [
        jax.random.uniform(key=rng_key, minval=x[0], maxval=x[1]) for x in segments
    ]
    return jnp.asarray([sample(state, query_value=x) for x in query_values])


def get(state: SumTreeState, node_indices: jnp.float32) -> jnp.float32:
    """Returns the value of the leaf node corresponding to the index.
    Args:
        node_index: The index of the leaf node.
    Returns:
        The value of the leaf node.
    """
    return state.nodes[-1][node_indices]


def set_non_batched(
    state: SumTreeState, node_index: jnp.int32, value: jnp.float32
) -> SumTreeState:
    """Sets the value of a leaf node and updates internal nodes accordingly.
    This operation takes O(log(capacity)).
    Args:
        state: Current state of the sum-tree.
        node_index: The index of the leaf node to be updated.
        value: The value which we assign to the node. This value must be
            nonnegative. Setting value = 0 will cause the element to never be sampled.
    This is not used in practice within the prioritised replay, as it is not vmap-able. However,
    it is useful for comparisons and testing. See `set_single` and `set_batch` for the functions
    that we use within the prioritised replay in practice.
    """
    state = state._replace(
        max_recorded_priority=jnp.maximum(value, state.max_recorded_priority)
    )

    delta_value = value - state.nodes[-1][node_index]

    # Now traverse back the tree, adjusting all sums along the way.
    for i, nodes_at_this_depth in enumerate(reversed(state.nodes)):
        # Note: Adding a delta leads to some tolerable numerical inaccuracies.
        nodes_at_this_depth = nodes_at_this_depth.at[node_index].set(
            nodes_at_this_depth[node_index] + delta_value
        )
        node_index //= 2
        state.nodes[-1 - i] = nodes_at_this_depth
    return state


def set_single(
    state: SumTreeState, node_index: jnp.int32, value: jnp.float32, vmap_axis_name="i"
) -> SumTreeState:
    """Sets the value of a leaf node and updates internal nodes accordingly.
    This operation takes O(log(capacity)). This is for use within the `set_batch` function.
    Args:
        state: Current state of the buffer.
        node_index: The index of the leaf node to be updated.
        value: The value which we assign to the node. This value must be
            nonnegative. Setting value = 0 will cause the element to never be sampled.
        vmap_axis_name: Vmap axis used inside the `set_batch` function.
    Returns:
        A state with the new value set at the given node_index.
    Note: When this is vmapped, if there are multiple `node_indexes` with the same `value`, then
    we pick the maximum of these values for the leaf of the tree. To understand the changes we
    introduce for this, it is useful to compare this function with `set_non_batched` which closely
    matches the dopamine sum-tree `set` function and does not deal with the clashes.
    """
    max_recorded_priority = jax.lax.pmax(
        jnp.maximum(value, state.max_recorded_priority), axis_name=vmap_axis_name
    )
    state = state._replace(max_recorded_priority=max_recorded_priority)

    # Need to be careful to deal with clashing node indices. For clashing node indices we set the
    # leaf to the maximum of the values corresponding to the clashing indices.
    values_all_nodes = jnp.zeros_like(state.nodes[-1]).at[node_index].set(value)
    ones_where_vals = jnp.where(values_all_nodes == 0, 0, 1)
    counts = jax.lax.psum(ones_where_vals, axis_name=vmap_axis_name)
    values_pmaxed = jax.lax.pmax(values_all_nodes, axis_name=vmap_axis_name)
    delta_value_total = values_pmaxed[node_index] - state.nodes[-1][node_index]

    # We divide the delta_value by the number of clashes, so that in the `psum` in the loop below,
    # the values will sum to delta_value_total.
    delta_value = delta_value_total / counts[node_index]

    # Now traverse back the tree, adjusting all sums along the way.
    for i, nodes_at_this_depth in enumerate(reversed(state.nodes)):
        # Note: Adding a delta leads to some tolerable numerical inaccuracies.
        change_to_nodes_at_this_depth = (
            jnp.zeros_like(nodes_at_this_depth).at[node_index].set(delta_value)
        )
        # Sum changes across batch.
        change_to_nodes_at_this_depth = jax.lax.psum(
            change_to_nodes_at_this_depth, axis_name=vmap_axis_name
        )
        # Apply changes.
        state.nodes[-1 - i] = nodes_at_this_depth + change_to_nodes_at_this_depth
        node_index //= 2
    return state


def set_batch(
    state: SumTreeState,
    node_indexes: chex.Array,
    values: chex.Array,
    vmap_axis_name="i",
) -> SumTreeState:
    """Sets the values of a batch of leaf nodes and updates internal nodes accordingly.
    This operation takes O(log(capacity)). As sampling sometimes returns repeats of the same
    index, we also deal with repeats but setting the leaf to the maximum value.
    Args:
        state: Current state of the buffer.
        node_index: The index of the leaf node to be updated.
        value: The value which we assign to the node. This value must be
            nonnegative. Setting value = 0 will cause the element to never be sampled.
        vmap_axis_name: Vmap axis used when we vmap the `set_single` function.
    Returns:
        A buffer state with updates nodes.
    """
    state = jax.vmap(set_single, axis_name=vmap_axis_name, in_axes=(None, 0, 0, None))(
        state, node_indexes, values, vmap_axis_name
    )
    # Grab first "element" within vmapped state. The `psum` in the `set_single` function ensures
    # that all elements across the state returned by the above vmap are identical.
    state = jax.tree_util.tree_map(lambda x: x[0], state)
    return state