import os
print(os.getcwd())

from hydrocarbon_problem.api.api_base import BaseAspenDistillationAPI
from hydrocarbon_problem.api.types import StreamSpecification, PerCompoundProperty, \
    ColumnInputSpecification, ColumnOutputSpecification, ProductSpecification

def test_api(api: BaseAspenDistillationAPI):
    """Run tests on the api that checks that each of the methods works as desired.
    Currently, to run these tests one has to manually inspect the flowsheet GUI to confirm that
    the values that are set, and retrieved make sense."""

    # set input stream
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
    # now manually check that the stream in the flowsheet has changed
    api.set_input_stream_specification(fake_stream)

    # set column input spec
    fake_column_input_spec = ColumnInputSpecification(n_stages=10, feed_stage_location=5,
                                                      reflux_ratio=1.0,
                                                      reboil_ratio=1.0,
                                                      condensor_pressure=101.0)
    # now manually check that the column specification has changed
    api.set_column_specification(fake_column_input_spec)

    # simulate the column
    solved = api.solve_flowsheet()
    assert solved

    # retrieve values from the flowsheet
    column_output_spec = api.get_simulated_column_properties(fake_column_input_spec)
    assert isinstance(column_output_spec, ColumnOutputSpecification)
    # check that the per stage properties of the column are of the correct length
    assert len(column_output_spec.molar_weight_per_stage) == fake_column_input_spec.n_stages
    assert len(column_output_spec.vapor_flow_per_stage) == fake_column_input_spec.n_stages
    assert len(column_output_spec.temperature_per_stage) == fake_column_input_spec.n_stages

    tops, bottoms = api.get_output_stream_specifications()
    assert isinstance(tops, StreamSpecification)
    assert isinstance(bottoms, StreamSpecification)

    fake_product_specification = ProductSpecification(purity=0.9)
    for stream in [tops, bottoms]:
        stream_value = api.get_stream_value(stream, fake_product_specification)
        if api.stream_is_product(stream, fake_product_specification):
            assert stream_value > 0.0
        else:
            assert stream_value == 0.0

    column_cost = api.get_column_cost(column_output_spec)
    assert isinstance(column_cost, float)
    assert column_cost >= 0.0

    print("Test passed, please inspect flowsheet GUI to confirm the values match the following "
          "values.")
    print(f"Input stream: {fake_stream}")
    print(f"Column input spec: {fake_column_input_spec}")
    print(f"Tops: {tops}")
    print(f"Bottoms: {bottoms}")
    print(f"Column output spec: {column_output_spec}")
    print(f"Column cost: {column_cost}")


if __name__ == '__main__':
    # example run with the fake distillation api
    from hydrocarbon_problem.api.fake_api import FakeDistillationAPI
    api = FakeDistillationAPI()
    test_api(api)



