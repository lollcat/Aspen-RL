from typing import List
import dm_env
import numpy as np

from hydrocarbon_problem.api.api_base import AspenDistillationAPI
from hydrocarbon_problem.api.types import StreamSpecification, PerCompoundProperty
from hydrocarbon_problem.env.types import Stream, Column, Observation

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
    flowsheet_api: AspenDistillationAPI  # TODO: add actual API here

    def __init__(self, initial_feed_spec: StreamSpecification = DEFAULT_INITIAL_FEED_SPEC):
        initial_feed = Stream(specification=initial_feed_spec, is_product=False, number=0)
        self.initial_feed = initial_feed
        self.stream_table: List[Stream, ...] = [initial_feed]
        self.column_table: List[Column, ...] = []


    def reset(self) -> dm_env.TimeStep:
        self.stream_table = [self.initial_feed]
        self.column_table = []
        observation = Observation(self.initial_feed.specification)
        # TODO: current place of work
        # timestep = dm_env.TimeStep(step_type=dm_env.StepType.FIRST, observation=)


    def stream_specification_to_observation(self, stream_specification: StreamSpecification) -> \
            np.array:
        pass

