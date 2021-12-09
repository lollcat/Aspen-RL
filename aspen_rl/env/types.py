from typing import NamedTuple, Sequence
from aspen_rl.api.types import StreamSpecification, ColumnInputSpecification, ColumnOutputSpecification


"""Types for the environment."""
Observation = Sequence[StreamSpecification]  # 1 stream if initial state, otherwise 2 streams

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