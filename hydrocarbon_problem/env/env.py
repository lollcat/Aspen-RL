from typing import List, Tuple, Optional
import dm_env
from dm_env import specs
from collections import deque
import numpy as np


from hydrocarbon_problem.api.api_base import BaseAspenDistillationAPI
from hydrocarbon_problem.api.fake_api import FakeDistillationAPI
from hydrocarbon_problem.api.types_ import StreamSpecification, PerCompoundProperty, \
    ColumnInputSpecification, ColumnOutputSpecification, ProductSpecification
from hydrocarbon_problem.env.types_ import Stream, Column, Observation, Done, Discount

Action = Tuple[np.ndarray, np.ndarray]

_DEFAULT_INITIAL_FEED_FLOWS = PerCompoundProperty(ethane=0.0170,
                                                  propane=1.11,
                                                  isobutane=1.1980,
                                                  n_butane=0.516,
                                                  isopentane=0.334,
                                                  n_pentane=0.173)
DEFAULT_INITIAL_FEED_SPEC = StreamSpecification(temperature=105.0,
                                                pressure=17.4,
                                                molar_flows=_DEFAULT_INITIAL_FEED_FLOWS)

class AspenDistillation(dm_env.Environment):
    """
    This environment has a very not-standard step function (2 streams created from 1 stream)
    To manage this we have an observation object that contains "created states" which are the states
    that have been recently produced by an action, and "upcoming state" which is the observation of
    the next stream that will be used to create a distillation column.
    We also split "done" and "discount" into values for the recently created streams
    from a specific action - for indicating whether these streams are product streams that leave the process,
    as well as an "overall" value which indicates whether the overall episode is complete
    (this occurs when the maximum number of columns have been used, or if there are no non-product
    streams yet to be acted on in the `._stream_numbers_yet_to_be_acted_on` object.
    """

    def __init__(self,
                 initial_feed_spec: StreamSpecification = DEFAULT_INITIAL_FEED_SPEC,
                 product_spec: ProductSpecification = ProductSpecification(purity=0.95),
                 n_stages_bounds: Tuple[int, int] = (2, 150),
                 pressure_bounds: Tuple[float, float] = (0.01, 50),
                 reflux_ratio_bounds: Tuple[float, float] = (0.01, 20.0),
                 max_steps: int = 30,
                 flowsheet_api: Optional[BaseAspenDistillationAPI] = None):
        self.flowsheet_api = flowsheet_api if flowsheet_api else FakeDistillationAPI()
        # hyper-parameters of the distillation environment
        self.product_spec = product_spec
        self._pressure_bounds = pressure_bounds
        self._n_stages_bounds = n_stages_bounds
        self._reflux_ratio_bounds = reflux_ratio_bounds
        # Note: Stream number matches steam index in self._stream_table.
        self._initial_feed = Stream(specification=initial_feed_spec,
                                    is_outlet=False,
                                    is_product=False, number=0,
                                    value=0.0)
        self._max_n_steps = max_steps  # Maximum number of environment steps.

        # Initialise stream and column tables.
        self._stream_table: List[Stream, ...] = [self._initial_feed]
        self._column_table: List[Column, ...] = []
        self._stream_numbers_yet_to_be_acted_on = deque()
        self._current_stream_number = self._initial_feed.number
        self._steps = 0

        self._blank_state = np.zeros(self.observation_spec().shape)

    def observation_spec(self) -> specs.Array:
        input_obs = self._stream_to_observation(self._initial_feed)
        return specs.Array(shape=input_obs.shape, dtype=float)

    def discount_spec(self):
        discount_tops = specs.BoundedArray(
            shape=(), dtype=float, minimum=0., maximum=1., name='discount_tops')
        discount_bots = specs.BoundedArray(
            shape=(), dtype=float, minimum=0., maximum=1., name='discount_bots')
        overall = specs.BoundedArray(
            shape=(), dtype=float, minimum=0., maximum=1., name='discount_overall')
        return Discount(
            overall=overall,
            created_states=(discount_tops, discount_bots))

    def action_spec(self) -> Tuple[specs.DiscreteArray, specs.BoundedArray]:
        continuous_spec = specs.BoundedArray(shape=(5,), dtype=float, minimum=-1, maximum=1,
                                             name="action_continuous")
        discrete_spec = specs.DiscreteArray(num_values=2, name="action_discrete")
        return discrete_spec, continuous_spec

    def reset(self) -> dm_env.TimeStep:
        self._stream_table: List[Stream, ...] = [self._initial_feed]
        self._column_table: List[Column, ...] = []
        self._stream_numbers_yet_to_be_acted_on = deque()
        self._current_stream_number = self._initial_feed.number
        obs = self._stream_to_observation(self._initial_feed)
        observation = Observation(
            created_states=(np.zeros_like(obs), np.zeros_like(obs)),
            upcoming_state=obs)
        timestep = dm_env.TimeStep(step_type=dm_env.StepType.FIRST, observation=observation,
                                   reward=None, discount=None)
        self._steps = 0
        return timestep

    def step(self, action: Action) -> Tuple[dm_env.TimeStep, float, bool]:
        """The step function of the environment, which takes in an action and returns a
        dm_env.TimeStep object which contains the step_type, reward, discount and observation."""
        self._steps += 1
        duration = "no separation"
        run_converged = "no separation"
        feed_stream = self._stream_table[self._current_stream_number]
        choose_separate, column_input_spec = self._action_to_column_spec(action)

        if choose_separate:
            self.flowsheet_api.set_input_stream_specification(feed_stream.specification)
            self.flowsheet_api.set_column_specification(column_input_spec)
            self.flowsheet_api.solve_flowsheet()
            tops_stream, bottoms_stream, column_output_spec = \
            self._get_simulated_flowsheet_info(column_input_spec)
            self._manage_environment_internals(tops_stream, bottoms_stream, column_input_spec,
                                               column_output_spec)
            reward = self.calculate_reward(feed_stream, tops_stream, bottoms_stream, column_input_spec,
                                           column_output_spec)
            done_overall = self.get_done_overall()
            if not done_overall:
                upcoming_stream = self._get_upcoming_stream()
                upcoming_stream_obs = self._stream_to_observation(upcoming_stream)
            else:
                upcoming_stream_obs = self._blank_state
            observation = Observation(
                created_states=(self._stream_to_observation(tops_stream),
                                self._stream_to_observation(bottoms_stream)),
                upcoming_state=upcoming_stream_obs)
            done = Done((tops_stream.is_outlet, bottoms_stream.is_outlet), done_overall)
        else:
            # choose not to separate the stream
            self._manage_environment_internals_no_act()
            done_overall = self.get_done_overall()
            reward = 0.0
            if not done_overall:
                upcoming_stream = self._get_upcoming_stream()
                upcoming_stream_obs = self._stream_to_observation(upcoming_stream)
            else:
                upcoming_stream_obs = self._blank_state
            observation = Observation(
                created_states=(self._blank_state, self._blank_state),
                upcoming_state=upcoming_stream_obs)
            done = Done((True, True), done_overall)
        discount = Discount((np.array(1 - done.created_states[0]),
                             np.array(1 - done.created_states[1])),
                        np.array(1-done.overall))
        timestep_type = dm_env.StepType.MID if not done_overall else dm_env.StepType.LAST
        timestep = dm_env.TimeStep(step_type=timestep_type, observation=observation,
                                   reward=reward, discount=discount)
        return timestep


    def get_done_overall(self):
        """Indicates whether the overall episode is complete, because either the
        maximum number of environment steps have been taken, or the
        self._stream_numbers_yet_to_be_acted_on is empty. """
        return self._steps >= self._max_n_steps or \
                           len(self._stream_numbers_yet_to_be_acted_on) == 0



    def _action_to_column_spec(self, action: Action) -> Tuple[bool,
                                                              Optional[ColumnInputSpecification]]:
        """All actions are assumed to be bounded between -1 and 1, and we then translate these into
        the relevant values for the ColumnInputSpecification."""
        discrete_action, continuous_action = action
        choose_seperate = True if discrete_action == 1 else False
        if choose_seperate:
            n_stages = round(np.interp(continuous_action[0], [-1, 1], self._n_stages_bounds) + 0.5)
            # feed as fraction between stage 0 and n_stages
            feed_stage_location = round(np.interp(continuous_action[1], [-1, 1], [0, n_stages]) + 0.5)
            reflux_ratio = np.interp(continuous_action[2], [-1, 1], self._reflux_ratio_bounds)
            reboil_ratio = np.interp(continuous_action[3], [-1, 1], self._reflux_ratio_bounds)
            condensor_pressure = np.interp(continuous_action[4], [-1, 1], self._pressure_bounds)
            column_spec = ColumnInputSpecification(
                n_stages=n_stages,
                feed_stage_location=feed_stage_location,
                reflux_ratio=reflux_ratio,
                reboil_ratio=reboil_ratio,
                condensor_pressure=condensor_pressure)
        else:
            column_spec = None
        return choose_seperate, column_spec

    def _get_simulated_flowsheet_info(self, column_input_specification: ColumnInputSpecification):
        """Grabs relevant info from simulated flowsheet."""
        tops_stream_spec, bottoms_stream_spec = \
            self.flowsheet_api.get_output_stream_specifications()
        tops_stream = self._stream_specification_to_stream(tops_stream_spec)
        bottoms_stream = self._stream_specification_to_stream(bottoms_stream_spec)
        column_output_spec = self.flowsheet_api.get_simulated_column_properties(column_input_specification)
        # column_output_spec = self.flowsheet_api.get_simulated_column_properties(column_input_spec=column_input_specification)
        return tops_stream, bottoms_stream, column_output_spec

    def _manage_environment_internals(self, tops_stream: Stream,
                                      bottoms_stream: Stream,
                                      column_input_spec: ColumnInputSpecification,
                                      column_output_spec: ColumnOutputSpecification) -> None:
        """Update the stream table, column table, and _stream_numbers_yet_to_be_acted_on objects."""
        for stream in [tops_stream, bottoms_stream]:
            self._stream_table.append(stream)
            if not stream.is_outlet:
                self._stream_numbers_yet_to_be_acted_on.append(stream.number)
                if self._stream_numbers_yet_to_be_acted_on == deque([]):
                    print(self._stream_table)
                    print(tops_stream)
                    print(bottoms_stream)
                    breakpoint()
        column = Column(input_spec=column_input_spec,
                        output_spec=column_output_spec,
                        input_stream_number=self._current_stream_number,
                        tops_stream_number=tops_stream.number,
                        bottoms_stream_number=bottoms_stream.number
                        )
        self._column_table.append(column)


    def _manage_environment_internals_no_act(self):
        # If no action is taken then no new streams are added to the stream table, so currently
        # this function is just an empty placeholder.
        pass

    def _get_upcoming_stream(self) -> Stream:
        """For the next action, get a stream from the
        self._stream_numbers_yet_to_be_acted_on list."""
        try:
            self._current_stream_number = self._stream_numbers_yet_to_be_acted_on.pop()
        except IndexError:
            print(f"{self._stream_numbers_yet_to_be_acted_on}")
            breakpoint()

        stream = self._stream_table[self._current_stream_number]
        return stream

    def _stream_specification_to_stream(self, stream_spec: StreamSpecification) -> Stream:
        """Convert StreamSpecification object into a StreamObject for a recently added stream. This
        function also gives the stream a unique number."""
        is_product, is_outlet = self.flowsheet_api.stream_is_product_or_outlet(stream_spec, self.product_spec)
        stream_value = self.flowsheet_api.get_stream_value(stream_spec, self.product_spec)
        stream = Stream(specification=stream_spec, is_product=is_product, is_outlet=is_outlet,
                        value=stream_value, number=len(self._stream_table) + 1)
        return stream

    def calculate_reward(self, feed_stream: Stream,
                         tops_stream: Stream,
                         bottoms_stream: Stream,
                         column_input_spec: ColumnInputSpecification,
                         column_output_spec: ColumnOutputSpecification) -> float:
        """Calculate potential revenue from selling top streams and bottoms streams, and compare
        relative to selling the input stream, subtract TAC and normalise."""
        total_annual_cost = self.flowsheet_api.get_column_cost(feed_stream.specification,
                                                  column_input_specification=column_input_spec,
                                                  column_output_specification=column_output_spec)
        total_annual_revenue = tops_stream.value + bottoms_stream.value
        return total_annual_revenue - total_annual_cost


    def _stream_to_observation(self, stream: Stream) -> np.array:
        """Convert the Stream object into an observation that can be passed to the agent,
        in the form of an array."""
        stream_spec = stream.specification
        obs = np.array([stream_spec.temperature, stream_spec.pressure] +
                       list(stream_spec.molar_flows))
        return obs

