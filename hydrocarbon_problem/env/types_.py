from typing import NamedTuple, Tuple

import numpy as np

from hydrocarbon_problem.api.types_ import StreamSpecification, ColumnInputSpecification, ColumnOutputSpecification


SingleStreamObservation = np.array

"""Types for the environment."""
class TimestepObservation(NamedTuple):
    """The observation used within the Timestep returned by the environment step."""
    # the states created by the current action.
    created_states: Tuple[SingleStreamObservation, SingleStreamObservation]
    # the next state that will be acted upon
    upcoming_state: SingleStreamObservation

class Done(NamedTuple):
    created_states: Tuple[bool, bool]  # if the created streams are product or still need to be
    # separated.
    overall: bool  # the whole env loop is done


class Discount(NamedTuple):
    created_states: Tuple[np.ndarray, np.ndarray]
    overall: np.ndarray


class Stream(NamedTuple):
    """Defines stream type, which are managed within the stream table."""
    specification: StreamSpecification
    is_product: bool
    is_outlet: bool
    number: int
    value: float
    episode: int


class Column(NamedTuple):
    input_spec: ColumnInputSpecification
    output_spec: ColumnOutputSpecification
    input_stream_number: int
    tops_stream_number: int
    bottoms_stream_number: int
    diameter: float
    height: float
    n_stages: int
    column_number: int
    episode: int
