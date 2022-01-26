from typing import Tuple

from hydrocarbon_problem.api.api_base import BaseAspenDistillationAPI
from hydrocarbon_problem.api.types import StreamSpecification, PerCompoundProperty, \
    ColumnOutputSpecification, ColumnInputSpecification, ProductSpecification

class FakeDistillationAPI(BaseAspenDistillationAPI):
    """
    A fake API that returns random numbers, and only checks that the inputs are of the expected
    types.
    """
    # TODO: could add randomness to this class and greater testing to make it more useful for
    #  plugging into the environment.

    def set_input_stream_specification(self, stream_specification: StreamSpecification) -> None:
        assert isinstance(stream_specification, StreamSpecification)

    def get_output_stream_specifications(self) -> Tuple[StreamSpecification,
                                                        StreamSpecification]:
        fake_stream = StreamSpecification(temperature=50.8,
                                          pressure=80.2,
                                          molar_flows=PerCompoundProperty(
                                              ethane=2.1,
                                              propane=1.1,
                                              isobutane=0.1,
                                              n_butane=0.5,
                                              isopentane=0.25,
                                              n_pentane=5.3
                                          ))
        return fake_stream, fake_stream

    def get_simulated_column_properties(self, column_input_spec: ColumnInputSpecification) -> \
            ColumnOutputSpecification:
        return ColumnOutputSpecification(
            condensor_duty=10.0, reboiler_duty=1.1,
            molar_weight_per_stage=(1.1,) * column_input_spec.n_stages,
            vapor_flow_per_stage=(1.2,) * column_input_spec.n_stages,
            temperature_per_stage=(100.1, ) * column_input_spec.n_stages)

    def set_column_specification(self, column_specification: ColumnInputSpecification) -> None:
        assert isinstance(column_specification, ColumnInputSpecification)

    def solve_flowsheet(self) -> bool:
        return True

    def get_column_cost(self, column_specification: ColumnOutputSpecification) -> float:
        return 100.0

    def get_stream_value(self, stream_specification: StreamSpecification,
                         product_specification: ProductSpecification) -> float:
        if self.stream_is_product(stream_specification, product_specification):
            return 10.0
        else:
            return 0.0

    def stream_is_product(self, stream_specification: StreamSpecification, product_specification:
                                ProductSpecification) -> bool:
        # for simplicity define the product definition only based off ethane
        total_flow = stream_specification.molar_flows.ethane + \
                     stream_specification.molar_flows.propane + \
                     stream_specification.molar_flows.isobutane + \
                     stream_specification.molar_flows.n_butane + \
                     stream_specification.molar_flows.isobutane + \
                     stream_specification.molar_flows.n_pentane
        if stream_specification.molar_flows.ethane / total_flow > product_specification.purity:
            return True
        else:
            return False

