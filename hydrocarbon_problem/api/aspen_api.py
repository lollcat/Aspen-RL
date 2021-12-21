from typing import Tuple

from hydrocarbon_problem.api.api_base import BaseAspenDistillationAPI
from hydrocarbon_problem.api.types import StreamSpecification, ColumnInputSpecification, \
    ColumnOutputSpecification, ProductSpecification


class AspenAPI(BaseAspenDistillationAPI):
    def __init__(self):
        pass

    def set_input_stream_specification(self, stream_specification: StreamSpecification) -> None:
        """Sets the input stream to a column to fit the stream specification"""
        # TODO: Christos - fill in each of these functions
        raise NotImplementedError

    def get_output_stream_specifications(self) -> Tuple[StreamSpecification, StreamSpecification]:
        """Returns the stream specification for the top and bottom output."""
        raise NotImplementedError

    def get_simulated_column_properties(self) -> ColumnOutputSpecification:
        """Returns the specification of the simulated column."""
        raise NotImplementedError


    def set_column_specification(self, column_specification: ColumnInputSpecification) -> None:
        """Sets the column specification"""
        raise NotImplementedError


    def solve_flowsheet(self) -> bool:
        """Solves the flowsheet. Returns True if the solve was successful."""
        raise NotImplementedError


    def get_column_cost(self, column_specification: ColumnOutputSpecification) -> float:
        """Calculates the TAC of the column."""
        raise NotImplementedError


    def get_stream_value(self, stream_specification: StreamSpecification) -> float:
        """Calculates the value (per year) of a stream."""
        raise NotImplementedError


    def stream_is_product(self, stream_specification: StreamSpecification, product_specification:
                                ProductSpecification) -> bool:
        """Checks whether a stream meets the product specification."""
        raise NotImplementedError


if __name__ == '__main__':
    aspen_api = AspenAPI()

