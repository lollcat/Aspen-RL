from typing import Tuple
import abc

from hydrocarbon_problem.api.types_ import StreamSpecification, ColumnInputSpecification, \
    ColumnOutputSpecification, ProductSpecification


class BaseAspenDistillationAPI(abc.ABC):
    """Define Base Class for the Aspen-python interface. This defines the key methods that are
    called by the reinforcement learning environment."""

    @abc.abstractmethod
    def set_input_stream_specification(self, stream_specification: StreamSpecification) -> None:
        """Sets the input stream to a column to fit the stream specification"""
        raise NotImplementedError

    @abc.abstractmethod
    def get_output_stream_specifications(self) -> Tuple[StreamSpecification, StreamSpecification]:
        """Returns the stream specification for the top and bottom output."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_simulated_column_properties(self, column_input_specification: ColumnInputSpecification) -> \
            ColumnOutputSpecification:
        """Returns the specification of the simulated column."""
        raise NotImplementedError

    @abc.abstractmethod
    def set_column_specification(self, column_input_specification: ColumnInputSpecification) -> None:
        """Sets the column specification"""
        raise NotImplementedError

    @abc.abstractmethod
    def solve_flowsheet(self) -> Tuple[float, bool]:
        """Solves the flowsheet. Returns True if the solve was successful."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_column_cost(self, stream_specification: StreamSpecification,
                        column_input_specification: ColumnInputSpecification,
                        column_output_specification: ColumnOutputSpecification) -> float:
        """Calculates the TAC of the column."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_stream_value(self, stream: StreamSpecification, product_specification: ProductSpecification) -> float:
        """Calculates the value (per year) of a stream."""
        raise NotImplementedError

    @abc.abstractmethod
    def stream_is_product(self, stream: StreamSpecification, product_specification: ProductSpecification) -> bool:
        """Checks whether a stream meets the product specification."""
        raise NotImplementedError


