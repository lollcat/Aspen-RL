from typing import List
import dm_env
from collections import deque
import numpy as np
from scipy.special import expit

from hydrocarbon_problem.api.api_base import BaseAspenDistillationAPI
from hydrocarbon_problem.api.types_ import StreamSpecification, PerCompoundProperty, \
    ColumnInputSpecification, ColumnOutputSpecification
from hydrocarbon_problem.env.types_ import Stream, Column, Observation, Done

_DEFAULT_INITIAL_FEED_FLOWS = PerCompoundProperty(ethane=17.0,
                                                  propane=1110.0,
                                                  isobutane=1198.0,
                                                  n_butane=516.0,
                                                  isopentane=334.0,
                                                  n_pentane=173.0)
DEFAULT_INITIAL_FEED_SPEC = StreamSpecification(temperature=105.0,
                                                pressure=17.4,
                                                molar_flows=_DEFAULT_INITIAL_FEED_FLOWS)

class AspenDistillation(dm_env.Environment):
    """
    Action space assumes unbounded continuous values. This environment has a very not-standard
    step function (2 streams created from 1 stream - # TODO describe).
    """
    flowsheet_api: BaseAspenDistillationAPI  # TODO: add actual API here

    def __init__(self, initial_feed_spec: StreamSpecification = DEFAULT_INITIAL_FEED_SPEC,
                 min_n_stages: int = 2, max_n_stages: int = 100, min_pressure: float = 0.1,
                 max_pressure: float = 10.0,
                 max_steps: int = 30):
        # hyper-parameters of the distillation environment
        self._min_n_stages = min_n_stages
        self._max_n_stages = max_n_stages
        # Note: Stream number matches steam index in self._stream_table.
        self._initial_feed = Stream(specification=initial_feed_spec, is_product=False, number=0)
        self._min_pressure = min_pressure  # atm
        self._max_pressure = max_pressure  # atm
        self._max_n_steps = max_steps  # Maximum number of environment steps.

        # Initialise stream and column tables.
        self._stream_table: List[Stream, ...] = [self._initial_feed]
        self._column_table: List[Column, ...] = []
        self._stream_numbers_yet_to_be_acted_on = deque()
        self._current_stream_number = self._initial_feed.number
        self._steps = 0

    
    def observation_spec(self):
        pass

    def action_spec(self):
        pass

    def reward_spec(self):
        pass

    def discount_spec(self):
        pass


    def reset(self) -> dm_env.TimeStep:
        self._stream_table: List[Stream, ...] = [self._initial_feed]
        self._column_table: List[Column, ...] = []
        self._stream_numbers_yet_to_be_acted_on = deque()
        self._current_stream_number = self._initial_feed.number
        observation = Observation([self._initial_feed.specification])
        timestep = dm_env.TimeStep(step_type=dm_env.StepType.FIRST, observation=observation,
                                   reward=None, discount=None)
        self._steps = 0
        return timestep

    def step(self, action: np.ndarray) -> dm_env.TimeStep:
        self._steps += 1
        column_input_spec = self._action_to_column_spec(action)
        self.flowsheet_api.set_column_specification(column_input_spec)
        self.flowsheet_api.solve_flowsheet()
        tops_stream, bottoms_stream, column_output_spec = self._get_simulated_flowsheet_info()
        self._manage_environment_internals(tops_stream, bottoms_stream, column_output_spec)
        reward = self.calculate_reward(tops_stream, bottoms_stream, column_output_spec)
        upcoming_stream = self._get_upcoming_stream()
        observation = Observation(
            created_states=(self._stream_to_observation(tops_stream),
                            self._stream_to_observation(bottoms_stream)),
            upcoming_state=self._stream_to_observation(upcoming_stream))
        done_overall = self._steps >= self._max_n_steps
        done = Done((tops_stream.is_product, bottoms_stream.is_product), done_overall)
        timestep_type = dm_env.StepType.MID if not done_overall else dm_env.StepType.LAST
        timestep = dm_env.TimeStep(step_type=timestep_type, observation=observation,
                                   reward=reward, discount=done)
        return timestep



    def _action_to_column_spec(self, action: np.ndarray) -> ColumnInputSpecification:
        """All actions are assumed to be unbounded and we then translate these into
        the relevant values for the ColumnInputSpecification."""
        n_stages = round(np.interp(expit(action[0]), [0, 1], [self._min_n_stages,
                                                             self._max_n_stages]) + 0.5)
        feed_stage_location = expit(action[1]) * n_stages
        reflux_ratio = np.exp(action[2])
        reboil_ratio = np.exp(action[3])
        condensor_pressure = np.interp(expit(action[1]), [0, 1], [self._min_pressure,
                                                                  self._max_pressure])
        return ColumnInputSpecification(n_stages=n_stages, feed_stage_location=feed_stage_location,
                                        reflux_ratio=reflux_ratio, reboil_ratio=reboil_ratio,
                                        condensor_pressure=condensor_pressure)

    def _get_simulated_flowsheet_info(self):
        """Grabs relevant info from simualted flowsheet."""
        tops_stream_spec, bottoms_stream_spec = \
            self.flowsheet_api.get_output_stream_specifications()
        tops_stream = self._stream_specification_to_stream(tops_stream_spec)
        bottoms_stream = self._stream_specification_to_stream(bottoms_stream_spec)
        column_output_spec = self.flowsheet_api.get_simulated_column_properties()
        return tops_stream, bottoms_stream, column_output_spec

    def _manage_environment_internals(self, tops_stream_spec: Stream, bottoms_stream_spec: Stream,
                         column_output_spec: ColumnOutputSpecification) -> None:
        """Add info to stream and column table, and add streams that don't meet product spec to
        self._stream_numbers_yet_to_be_acted_on"""
        raise NotImplementedError# TODO

    def _get_upcoming_stream(self) -> Stream:
        stream_number = self._stream_numbers_yet_to_be_acted_on.pop()
        stream = self._stream_table[stream_number]
        return stream

    def _stream_specification_to_stream(self, stream_spec: StreamSpecification) -> Stream:
        raise NotImplementedError# TODO

    def calculate_reward(self, tops_stream_spec: Stream, bottoms_stream_spec: Stream,
                         column_output_spec: ColumnOutputSpecification) -> float:
        """Calculate potential revenue from selling top streams and bottoms streams, and compare
        relative to selling the input stream, subtract TAC and normalise."""
        raise NotImplementedError # TODO


    def _stream_to_observation(self, stream: Stream) -> np.array:
        raise NotImplementedError # TODO

