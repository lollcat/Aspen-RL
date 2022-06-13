from typing import NamedTuple, Tuple, Iterable, Callable
from hydrocarbon_problem.agents.base import NextObservation, Transition, Observation, Action
from hydrocarbon_problem.env.env import AspenDistillation
from hydrocarbon_problem.agents.sac.agent_test import create_fake_batch
import numpy as np
from functools import partial

import jax.lax
import jax.numpy as jnp
import chex


class BufferState(NamedTuple):
    data: Transition
    is_full: jnp.bool_
    can_sample: jnp.bool_
    current_index: jnp.int32


class ReplayBuffer:
    def __init__(self,
                 max_length: int,
                 min_sample_length: int,
                 ):
        """
        Create replay buffer for batched sampling and adding of data.
        Args:
            max_length: maximum length of the buffer
            min_sample_length: minimum length of buffer required for sampling

        The `max_length` and `min_sample_length` should be sufficiently long to prevent overfitting
        to the replay data. For example, if `min_sample_length` is equal to the
        sampling batch size, then we may overfit to the first batch of data, as we would update
        on it many times during the start of training.
        """
        assert min_sample_length < max_length
        self.max_length = max_length
        self.min_sample_length = min_sample_length

    def init(self, key: chex.PRNGKey, env: AspenDistillation) -> BufferState:
        """
        key: source of randomness
        initial_sampler: sampler producing x and log_w, used to fill the buffer up to
            the min sample length. The initialised flow + AIS may be used here,
            or we may desire to use AIS with more distributions to give the flow a "good start".
        """
        current_index = 0
        is_full = False  # whether the buffer is full
        can_sample = False  # whether the buffer is full enough to begin sampling
        fake_batch = jax.tree_map(jnp.zeros_like, create_fake_batch(env, self.max_length))

        buffer_state = BufferState(data=fake_batch, is_full=is_full, can_sample=can_sample,
                                   current_index=current_index)
        buffer_state = jax.tree_map(jnp.asarray, buffer_state)

        discrete_spec, continuous_spec = env.action_spec()

        
        while not buffer_state.can_sample:
            timestep = env.reset()
            previous_timestep = timestep
            while not timestep.last():
                # fill buffer up minimum length
                key, subkey = jax.random.split(key)
                discrete_action = jnp.array(1) # always choose to separater
                continuous_action = jax.random.uniform(key=subkey,
                                                       shape=continuous_spec.shape,
                                                       minval=continuous_spec.minimum,
                                                       maxval=continuous_spec.maximum)
                action = np.asarray(discrete_action), np.asarray(continuous_action)
                timestep = env.step(action)
                next_obs = NextObservation(observation=(timestep.observation.created_states),
                                           discounts=timestep.discount.created_states)
                transition = Transition(
                    observation=previous_timestep.observation.upcoming_state,
                    action=action,
                    reward=timestep.reward,
                    discount=timestep.discount.overall,
                    next_observation=next_obs,
                    extras=()
                )
                transition = jax.tree_map(jnp.asarray, transition)
                buffer_state = self.add(transition, buffer_state)
                previous_timestep = timestep
        return buffer_state

    @partial(jax.jit, static_argnums=0)
    def add(self, transition: Transition, buffer_state: BufferState) -> BufferState:
        """Add a batch of generated data to the replay buffer"""
        index = buffer_state.current_index % self.max_length
        new_data = jax.tree_map(lambda x, y: x.at[index].set(y),
                                buffer_state.data, transition)
        new_index = buffer_state.current_index + 1
        is_full = jax.lax.select(buffer_state.is_full, buffer_state.is_full,
                                 new_index >= self.max_length)
        can_sample = jax.lax.select(buffer_state.is_full, buffer_state.can_sample,
                                    new_index >= self.min_sample_length)
        current_index = new_index % self.max_length
        state = BufferState(data=new_data,
                            current_index=current_index,
                            is_full=is_full,
                            can_sample=can_sample)
        return state

    @partial(jax.jit, static_argnums=(0, 3))
    def sample(self, buffer_state: BufferState, key: chex.PRNGKey,
               batch_size: int) -> Transition:
        """Return a batch of sampled data, if the batch size is specified then the batch will have a
        leading axis of length batch_size, otherwise the default self.batch_size will be used."""
        # if not buffer_state.can_sample:
        #     raise Exception("Buffer must be at minimum length before calling sample")
        max_index = jax.lax.select(buffer_state.is_full,
                                   self.max_length, buffer_state.current_index)
        # mask for which data to sample
        probs = jnp.where(jnp.arange(self.max_length) < max_index, jnp.ones(self.max_length,),
                          jnp.zeros((self.max_length,)))
        indices = jax.random.choice(key, jnp.arange(self.max_length), shape=(batch_size,),
                                    replace=False, p=probs)
        return jax.tree_map(lambda x: x[indices], buffer_state.data)