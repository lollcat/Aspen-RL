from typing import Tuple
import abc

from hydrocarbon_problem.api.types import StreamSpecification, ColumnInputSpecification


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
    def get_simulated_column_properties(self) -> Tuple[StreamSpecification, StreamSpecification]:
        """Returns the stream specification for the top and bottom output."""
        raise NotImplementedError

    @abc.abstractmethod
    def set_column_specification(self, column_specification: ColumnInputSpecification) -> None:
        """Sets the column specification"""
        raise NotImplementedError

    @abc.abstractmethod
    def solve_flowsheet(self):
        """Solves the flowsheet."""
        raise NotImplementedError