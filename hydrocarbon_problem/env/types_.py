from typing import NamedTuple, Tuple

import numpy as np

from hydrocarbon_problem.api.types_ import StreamSpecification, ColumnInputSpecification, ColumnOutputSpecification


"""Types for the environment."""
class Observation(NamedTuple):
    created_states: Tuple[np.ndarray, np.ndarray]  # the states created by the current action.
    upcoming_state: np.ndarray  # the next state that will be acted upon

class Done(NamedTuple):
    created_states: Tuple[bool, bool]  # if the created streams are product or still need to be
    # separated.
    overall: bool # the whole env loop is done

class Stream(NamedTuple):
    """Defines stream type, which are managed within the stream table."""
    specification: StreamSpecification
    is_product: bool
    number: int


class Column(NamedTuple):
    input_spec: ColumnInputSpecification
    output_spec: ColumnOutputSpecification
    input_stream_number: int
    tops_stream_number: int
    bottoms_stream_number: int