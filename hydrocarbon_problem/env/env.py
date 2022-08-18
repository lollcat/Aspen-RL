from typing import List, Tuple, Optional
import dm_env
from dm_env import specs
from collections import deque
import numpy as np
import time

from hydrocarbon_problem.api.api_base import BaseAspenDistillationAPI
from hydrocarbon_problem.api.fake_api import FakeDistillationAPI
from hydrocarbon_problem.api.types_ import StreamSpecification, PerCompoundProperty, \
    ColumnInputSpecification, ColumnOutputSpecification, ProductSpecification
from hydrocarbon_problem.env.types_ import Stream, Column, TimestepObservation, Done, Discount


Action = Tuple[np.ndarray, np.ndarray]

_DEFAULT_INITIAL_FEED_FLOWS = PerCompoundProperty(ethane=0.0017,
                                                  propane=1.1098,
                                                  isobutane=1.1977,
                                                  n_butane=0.5158,
                                                  isopentane=0.3443,
                                                  n_pentane=0.1732)
FLOW_OBS_SCALING = 1

DEFAULT_INITIAL_FEED_SPEC = StreamSpecification(temperature=105.0,
                                                pressure=25,
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
                 n_stages_bounds: Tuple[int, int] = (30, 100),
                 pressure_bounds: Tuple[float, float] = (0.5, 30),
                 reflux_ratio_bounds: Tuple[float, float] = (1, 20.0),
                 max_steps: int = 8,
                 flowsheet_api: Optional[BaseAspenDistillationAPI] = None,
                 small_action_space=False,
                 reward_scale=100,
                 punishment=-10):
        self.reward_scale = reward_scale
        self.punishment = punishment
        self.small_action_space = small_action_space
        self.tops_stream = None
        self.bots_stream = None
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
                                    value=0.0,
                                    episode=0,
                                    separate=0)
        self._max_n_steps = max_steps  # Maximum number of environment steps.

        # Initialise stream and column tables.
        self._stream_table: List[Stream, ...] = [self._initial_feed]
        self._column_table: List[Column, ...] = []
        self._stream_numbers_yet_to_be_acted_on = deque()
        self._flow_yet_to_be_acted_on = deque()
        self._current_stream_number = self._initial_feed.number
        self._steps = 0
        self.contact = -1
        self.converged = 0
        self._blank_state = np.zeros(self.observation_spec().shape)
        self.terminate_on_flowsheet_fail = False
        self.info = {}
        self.once_per_episode_info = {}
        self.choose_separate = True

    def observation_spec(self) -> specs.Array:
        input_obs = self._stream_to_observation(self._initial_feed)
        single_stream_obs = specs.Array(shape=input_obs.shape, dtype=float)
        return single_stream_obs

    def next_observation(self) -> TimestepObservation:
        """This is the observation stored in the timestep returned by the
        environment step function."""
        single_stream_obs = self.observation_spec()
        return TimestepObservation(single_stream_obs, single_stream_obs)

    def discount_spec(self):
        discount_tops = specs.BoundedArray(
            shape=(1,), dtype=float, minimum=0., maximum=1., name='discount_tops')
        discount_bots = specs.BoundedArray(
            shape=(1,), dtype=float, minimum=0., maximum=1., name='discount_bots')
        overall = specs.BoundedArray(
            shape=(1,), dtype=float, minimum=0., maximum=1., name='discount_overall')
        return Discount(
            overall=overall,
            created_states=(discount_tops, discount_bots))

    def action_spec(self) -> Tuple[specs.DiscreteArray, specs.BoundedArray]:
        if self.small_action_space:
            continuous_spec = specs.BoundedArray(shape=(2,), dtype=float, minimum=-1, maximum=1,
                                             name="action_continuous")
        else:
            continuous_spec = specs.BoundedArray(shape=(5,), dtype=float, minimum=-1, maximum=1,
                                                 name="action_continuous")
        discrete_spec = specs.DiscreteArray(num_values=2, name="action_discrete")
        return discrete_spec, continuous_spec

    def reset(self) -> dm_env.TimeStep:
        self._stream_table: List[Stream, ...] = [self._initial_feed]
        self._column_table: List[Column, ...] = []
        self._stream_numbers_yet_to_be_acted_on = deque()
        self._flow_yet_to_be_acted_on = deque()
        self._current_stream_number = self._initial_feed.number
        obs = self._stream_to_observation(self._initial_feed)
        observation = TimestepObservation(
            created_states=(np.zeros_like(obs), np.zeros_like(obs)),
            upcoming_state=obs)
        timestep = dm_env.TimeStep(step_type=dm_env.StepType.FIRST, observation=observation,
                                   reward=None, discount=None)
        self._steps = 0
        self.info = {}
        return timestep

    def step(self, action: Action) -> dm_env.TimeStep:
        """The step function of the environment, which takes in an action and returns a
        dm_env.TimeStep object which contains the step_type, reward, discount and observation."""
        self._steps += 1
        self.feed_stream = self._stream_table[self._current_stream_number-1]
        self.choose_separate, column_input_spec = self._action_to_column_spec(action)

        self.info["FeedStream"] = self.feed_stream
        if self.choose_separate:
            start = time.time()
            self.flowsheet_api.set_input_stream_specification(self.feed_stream.specification)
            self.flowsheet_api.set_column_specification(column_input_spec)
            time_to_set_aspen = time.time() - start
            start = time.time()
            self.contact, self.converged = self.flowsheet_api.solve_flowsheet(stream_input=self.feed_stream.specification,
                                                                          column_input=column_input_spec)
            time_to_run_aspen = time.time() - start
            if self.contact == 1 and (self.converged == 0 or self.converged == 2):
                start = time.time()
                self.tops_stream, self.bottoms_stream, column_output_spec = \
                self._get_simulated_flowsheet_info(column_input_spec)
                time_to_retrieve_aspen_data = time.time() - start
                self._manage_environment_internals(self.tops_stream, self.bottoms_stream, column_input_spec,
                                                   column_output_spec)
                start = time.time()
                reward = self.calculate_reward(self.feed_stream, self.tops_stream, self.bottoms_stream, column_input_spec,
                                               column_output_spec)
                time_to_calculate_reward = time.time() - start
                done_overall = self.get_done_overall()
                if not done_overall:
                    upcoming_stream = self._get_upcoming_stream()
                    upcoming_stream_obs = self._stream_to_observation(upcoming_stream)
                else:
                    upcoming_stream_obs = self._blank_state
                observation = TimestepObservation(
                    created_states=(self._stream_to_observation(self.tops_stream),
                                    self._stream_to_observation(self.bottoms_stream)),
                    upcoming_state=upcoming_stream_obs)
                done = Done((self.tops_stream.is_outlet, self.bottoms_stream.is_outlet), done_overall)

            elif self.contact == 0 or self.converged == 1:
                # assert not (self.contact == 0 and self.converged == 1)
                if self.converged:
                    self.info["Column_error"] = column_input_spec
                    time_to_retrieve_aspen_data = -0.1
                    time_to_calculate_reward = -0.1
                elif not self.contact:
                    time_to_retrieve_aspen_data = -0.1
                    time_to_calculate_reward = -0.1
                self._manage_environment_internals_no_act()
                done_overall = self.get_done_overall()
                reward = self.punishment if self.converged == 1 else 0
                if not done_overall:
                    upcoming_stream = self._get_upcoming_stream()
                    upcoming_stream_obs = self._stream_to_observation(upcoming_stream)
                else:
                    upcoming_stream_obs = self._blank_state
                observation = TimestepObservation(
                    created_states=(self._blank_state, self._blank_state),
                    upcoming_state=upcoming_stream_obs)
                done = Done((True, True), done_overall)

        if not self.choose_separate:
            self.info["Feedstream no-separate"] = self.feed_stream
            time_to_retrieve_aspen_data = -0.2
            time_to_calculate_reward = -0.2
            time_to_set_aspen = -0.2
            time_to_run_aspen = -0.2
            self._manage_environment_internals_no_act()
            done_overall = self.get_done_overall()
            reward = 0.0
            if not done_overall:
                upcoming_stream = self._get_upcoming_stream()
                upcoming_stream_obs = self._stream_to_observation(upcoming_stream)
            else:
                upcoming_stream_obs = self._blank_state
            observation = TimestepObservation(
                created_states=(self._blank_state, self._blank_state),
                upcoming_state=upcoming_stream_obs)
            done = Done((True, True), done_overall)

        discount = Discount((np.array(1 - done.created_states[0]),
                             np.array(1 - done.created_states[1])),
                        np.array(1-done.overall))
        timestep_type = dm_env.StepType.MID if not done_overall else dm_env.StepType.LAST
        timestep = dm_env.TimeStep(step_type=timestep_type, observation=observation,
                                   reward=reward, discount=discount)
        if timestep.last():
            self.once_per_episode_info["Streams yet to be acted on"] = len(self._stream_numbers_yet_to_be_acted_on)

        if self.choose_separate:
            self.info.update(column_input_spec._asdict())
            if self._steps == 1:
                self.once_per_episode_info.update({key + "First stream":value for key, value in column_input_spec._asdict().items()})

        self.info['Time to set aspen'] = time_to_set_aspen
        self.info['Time to run aspen'] = time_to_run_aspen
        self.info['Time to retrieve aspen data'] = time_to_retrieve_aspen_data
        self.info['Time to calculate reward'] = time_to_calculate_reward
        self.info['Choose to separate'] = self.choose_separate

        return timestep

    def get_done_overall(self):
        """Indicates whether the overall episode is complete, because either the
        maximum number of environment steps have been taken, or the
        self._stream_numbers_yet_to_be_acted_on is empty. """
        done_from_flowsheet_crash = True if (self.terminate_on_flowsheet_fail and (self.converged == 1 or self.contact == 0)) else False

        if self._steps >= self._max_n_steps or done_from_flowsheet_crash or (self._steps > 1 and len(self._stream_numbers_yet_to_be_acted_on) == 0):
            done_overall = True

        elif self._steps > 1 and self._steps < self._max_n_steps and len(self._stream_numbers_yet_to_be_acted_on) != 0:
            done_overall = False

        elif self._steps == 1:
            done_overall = False

        return done_overall  # self._steps >= self._max_n_steps or len(self._stream_numbers_yet_to_be_acted_on) == 0 or done_from_flowsheet_crash


    def _action_to_column_spec(self, action: Action) -> Tuple[bool,
                                                              Optional[ColumnInputSpecification]]:
        """All actions are assumed to be bounded between -1 and 1, and we then translate these into
        the relevant values for the ColumnInputSpecification."""
        discrete_action, continuous_action = action
        choose_seperate = True if discrete_action == 1 else False
        if choose_seperate:
            if not self.small_action_space:
                n_stages = round(np.interp(continuous_action[0], [-1, 1], self._n_stages_bounds) + 0.5)
                # feed as fraction between stage 0 and n_stages
                feed_stage_location = round(np.interp(continuous_action[1], [-1, 1], [max(0.2 * n_stages, 2),
                                                                                      min(n_stages-1, n_stages * 0.8)]))  # + 0.5
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
                n_stages = 77
                # feed as fraction between stage 0 and n_stages
                feed_stage_location = 44
                reflux_ratio = np.interp(continuous_action[0], [-1, 1], self._reflux_ratio_bounds)
                reboil_ratio = np.interp(continuous_action[1], [-1, 1], self._reflux_ratio_bounds)
                condensor_pressure = 1
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
        self.info["TopStream"] = tops_stream
        self.info["BottomStream"] = bottoms_stream
        return tops_stream, bottoms_stream, column_output_spec

    def _manage_environment_internals(self, tops_stream: Stream,
                                      bottoms_stream: Stream,
                                      column_input_spec: ColumnInputSpecification,
                                      column_output_spec: ColumnOutputSpecification) -> None:
        """Update the stream table, column table, _flow_yet_to_be_acted_on and _stream_numbers_yet_to_be_acted_on objects."""
        for stream in [tops_stream, bottoms_stream]:
            if not stream.is_outlet:
                flow = sum(stream.specification.molar_flows)
                self._flow_yet_to_be_acted_on.append(flow)
                self._stream_numbers_yet_to_be_acted_on.append(stream.number)
        column = Column(input_spec=column_input_spec,
                        output_spec=column_output_spec,
                        input_stream_number=self._current_stream_number,
                        tops_stream_number=tops_stream.number,
                        bottoms_stream_number=bottoms_stream.number,
                        diameter=0,
                        height=0,
                        n_stages=0,
                        a_cnd=0,
                        a_rbl=0,
                        cost_col=0,
                        cost_int=0,
                        cost_cnd=0,
                        cost_rbl=0,
                        cost_util_cnd=0,
                        cost_util_rbl=0,
                        column_number=0,
                        episode=0
                        )
        self._column_table.append(column)
        self.info["Column"] = column


    def _manage_environment_internals_no_act(self):
        # If no action is taken then no new streams are added to the stream table, so currently
        # this function is just an empty placeholder.
        pass

    def _get_upcoming_stream(self) -> Stream:
        """For the next action, get a stream from the
        self._stream_numbers_yet_to_be_acted_on list."""
        if self.contact == 1 and ((self.choose_separate and (self.converged == 0 or self.converged == 2)) or \
           (self._steps > 1 and not self.choose_separate)):
            max_flow_index = self._flow_yet_to_be_acted_on.index(max(self._flow_yet_to_be_acted_on))
            self._current_stream_number = self._stream_numbers_yet_to_be_acted_on[max_flow_index]
            del self._flow_yet_to_be_acted_on[max_flow_index]
            del self._stream_numbers_yet_to_be_acted_on[max_flow_index]

            stream = self._stream_table[self._current_stream_number-1]
            while stream.is_outlet == True:
                max_flow_index = self._flow_yet_to_be_acted_on.index(sum(self._flow_yet_to_be_acted_on))
                self._current_stream_number = self._stream_numbers_yet_to_be_acted_on[max_flow_index]
                # self._current_stream_number = self._stream_numbers_yet_to_be_acted_on.pop()
                stream = self._stream_table[self._current_stream_number]

        elif (self._steps == 1 and not self.choose_separate) or (self.choose_separate and self.converged == 1) or self.contact == 0:
            stream = self.feed_stream
        else:
            pass
        return stream

    def _stream_specification_to_stream(self, stream_spec: StreamSpecification) -> Stream:
        """Convert StreamSpecification object into a StreamObject for a recently added stream. This
        function also gives the stream a unique number."""
        is_outlet_forced = len(self._stream_numbers_yet_to_be_acted_on) > (self._max_n_steps - self._steps)
        is_product, is_outlet = self.flowsheet_api.stream_is_product_or_outlet(stream_spec, self.product_spec)
        is_outlet = is_outlet or is_outlet_forced
        if is_product == True:
            stream_value = self.flowsheet_api.get_stream_value(stream_spec, self.product_spec)
        if is_product == False:
            stream_value = 0.0
        stream = Stream(specification=stream_spec, is_product=is_product, is_outlet=is_outlet,
                        value=stream_value, number=len(self._stream_table) + 1, episode=0, separate=self.choose_separate)
        self._stream_table.append(stream)
        return stream

    def calculate_reward(self, feed_stream: Stream,
                         tops_stream: Stream,
                         bottoms_stream: Stream,
                         column_input_spec: ColumnInputSpecification,
                         column_output_spec: ColumnOutputSpecification) -> float:
        """Calculate potential revenue from selling top streams and bottoms streams, and compare
        relative to selling the input stream, subtract TAC and normalise."""
        # reward_scaler = 10  # reward 100M€
        total_annual_cost, col_info = self.flowsheet_api.get_column_cost(feed_stream.specification,
                                                  column_input_specification=column_input_spec,
                                                  column_output_specification=column_output_spec)
        total_annual_revenue = tops_stream.value + bottoms_stream.value

        self.info["Revenue"] = total_annual_revenue
        self.info["Diameter"] = col_info[1]
        self.info["Height"] = col_info[0]
        self.info["Column"] = self.info["Column"]._replace(a_cnd=col_info[2])
        self.info["Column"] = self.info["Column"]._replace(a_rbl=col_info[3])
        self.info["Column"] = self.info["Column"]._replace(cost_col=col_info[4])
        self.info["Column"] = self.info["Column"]._replace(cost_int=col_info[5])
        self.info["Column"] = self.info["Column"]._replace(cost_cnd=col_info[6])
        self.info["Column"] = self.info["Column"]._replace(cost_rbl=col_info[7])
        self.info["Column"] = self.info["Column"]._replace(cost_util_cnd=col_info[8])
        self.info["Column"] = self.info["Column"]._replace(cost_util_rbl=col_info[9])

        # self.info["Costs internals"] = col_info[5]
        # self.info["Costs condenser"] = col_info[6]
        # self.info["Costs reboiler"] = col_info[7]
        # self.info["Utility costs condenser"] = col_info[8]
        # self.info["Utility costs reboiler"] = col_info[9]

        return (total_annual_revenue - total_annual_cost)/self.reward_scale


    def _stream_to_observation(self, stream: Stream) -> np.array:
        """Convert the Stream object into an observation that can be passed to the agent,
        in the form of an array."""
        stream_spec = stream.specification
        temp_scaling = 100
        pressure_scaling = 25
        molar_flows = [flow / FLOW_OBS_SCALING for flow in stream_spec.molar_flows]
        obs = np.array([stream_spec.temperature/temp_scaling,
                        stream_spec.pressure/pressure_scaling, self._steps] + molar_flows
                       )
        return obs

