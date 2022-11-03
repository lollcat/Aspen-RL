import chex
import jax.tree_util
import jax.numpy as jnp
import numpy as np

from hydrocarbon_problem.agents.sac import prioritised_buffer as pr


if __name__ == '__main__':
    max_length = 32
    sampling_batch_size = 10
    min_length = int(sampling_batch_size * 3 + 1)
    fake_transition = {"artichoke": jnp.array(4.20),
            "silly_dogfish": jnp.array([42, 420])}
    prioritised_buffer = pr.make_prioritised_replay(
        max_length=max_length,
        sampling_batch_size=sampling_batch_size,
        min_length=min_length
    )


    adding_batch_size = 17
    key = jax.random.PRNGKey(0)

    fake_batch = jax.tree_util.tree_map(lambda x: jnp.stack([x]*adding_batch_size),
                                        fake_transition)

    # Init the state.
    state = prioritised_buffer.init(fake_transition)

    n_batches_to_fill = int(np.ceil(min_length / adding_batch_size))

    # Add to the buffer until it is filled to the point that we can sample from it.
    for i in range(n_batches_to_fill):
        # priorities = jnp.ones(adding_batch_size) + i
        batch = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x) + i, fake_batch)
        # state = prioritised_buffer.add(batch, state)
        state = jax.jit(prioritised_buffer.add)(batch, state)

        if i != n_batches_to_fill - 1:
            assert jax.jit(prioritised_buffer.can_sample)(state) == False
        else:
            # after last step we should be able to sample
            assert prioritised_buffer.can_sample(state) == True

        print(state.sum_tree_state.max_recorded_priority)

    # Check sampling from the buffer.
    key, subkey = jax.random.split(key)
    sample = jax.jit(prioritised_buffer.sample)(state, subkey)
    chex.assert_tree_shape_prefix(sample, (sampling_batch_size,))

    # Check adjusting the priorities
    new_priorities = jnp.zeros((sampling_batch_size,)) + 0.1
    state = jax.jit(prioritised_buffer.set_priorities)(
        new_priorities, sample.indices, state)