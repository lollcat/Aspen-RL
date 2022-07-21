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
    def solve_flowsheet(self, stream_input:StreamSpecification, column_input:ColumnInputSpecification) -> None:
        """Solves the flowsheet. Returns True if the solve was successful."""
        raise NotImplementedError

    # @abc.abstractmethod
    def restart_aspen(self) -> None:
        """Restarts Aspen Plus"""
        raise NotImplementedError

    def pid_com(self, process_name, instance):
        """Setting PID communicator in PID_check.txt"""
        raise NotImplementedError

    def retrieve_pids(self, process_name):
        """Retrieving PID values of AspenPlus"""
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
    def stream_is_product_or_outlet(self, stream: StreamSpecification,
                                    product_specification: ProductSpecification) -> \
            Tuple[bool, bool]:
        """
        Checks whether a stream meets the product specification
        and if the stream exits the process. Some streams may not meet product spec
        but still exit the process.

        Args:
            stream: The stream to be classified.
            product_specification: The definition of what makes a stream a product.

        Returns:
            is_product: Whether the stream is a product or not.
            is_outlet: Whether the steam exits the process.
        """
        raise NotImplementedError


