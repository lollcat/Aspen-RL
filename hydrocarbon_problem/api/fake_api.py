from hydrocarbon_problem.api.api_base import BaseAspenDistillationAPI
from hydrocarbon_problem.api.types import StreamSpecification

class FakeDistillationAPI(BaseAspenDistillationAPI):
    """
    A fake API that returns random numbers, and only checks that the inputs are of the expected
    types.
    """
    def set_input_stream_specification(self, stream_specification: StreamSpecification) -> None:
        assert isinstance(stream_specification, StreamSpecification)
        # TODO: could be improved by writing a validate_stream specification function


