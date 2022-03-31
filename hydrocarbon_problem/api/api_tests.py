from hydrocarbon_problem.api.api_base import BaseAspenDistillationAPI
from hydrocarbon_problem.api.types_ import StreamSpecification, PerCompoundProperty, \
    ColumnInputSpecification, ColumnOutputSpecification, ProductSpecification
import os
import time
print(os.getcwd())


def test_api(api: BaseAspenDistillationAPI):
    """Run tests on the api that checks that each of the methods works as desired.
    Currently, to run these tests one has to manually inspect the flowsheet GUI to confirm that
    the values that are set, and retrieved make sense."""

    # set input stream
    fake_stream = StreamSpecification(temperature=105,
                                      pressure=17.4,
                                      molar_flows=PerCompoundProperty(
                                          ethane=0.017,
                                          propane=1.110,
                                          isobutane=1.198,
                                          n_butane=0.516,
                                          isopentane=0.334,
                                          n_pentane=0.173
                                      ))
    # now manually check that the stream in the flowsheet has changed
    api.set_input_stream_specification(fake_stream)

    # set column input spec
    fake_column_input_spec = ColumnInputSpecification(n_stages=50, feed_stage_location=25,
                                                      reflux_ratio=1,
                                                      reboil_ratio=1,
                                                      condensor_pressure=17.4)
    # now manually check that the column specification has changed
    api.set_column_specification(fake_column_input_spec)

    # simulate the column
    start = time.time()
    solved = api.solve_flowsheet()
    print(time.time() - start)
    if solved == False:
        print("Aspen did not converge")
    elif solved == True:
        print("Aspen did converge")
    assert solved

    # retrieve values from the flowsheet
    column_output_spec = api.get_simulated_column_properties(fake_column_input_spec)
    assert isinstance(column_output_spec, ColumnOutputSpecification)
    # check that the per stage properties of the column are of the correct length
    # assert len(column_output_spec.molar_weight_per_stage) == fake_column_input_spec.n_stages
    # assert len(column_output_spec.vapor_flow_per_stage) == fake_column_input_spec.n_stages
    # assert len(column_output_spec.temperature_per_stage) == fake_column_input_spec.n_stages

    tops, bottoms = api.get_output_stream_specifications()
    assert isinstance(tops, StreamSpecification)
    assert isinstance(bottoms, StreamSpecification)

    fake_product_specification = ProductSpecification(purity=0.9)
    top_stream_value, top_stream_purity = api.get_stream_value(tops, fake_product_specification)
    bottom_stream_value, bottom_stream_purity = api.get_stream_value(bottoms, fake_product_specification)

    print(top_stream_purity)
    print(bottom_stream_purity)


    for stream, stream_value in zip([tops, bottoms], [top_stream_value, bottom_stream_value]):
        if api.stream_is_product(stream, fake_product_specification):
            assert stream_value > 0.0
        else:
            assert stream_value == 0.0

    column_cost = api.get_column_cost(fake_stream, fake_column_input_spec, column_output_spec)
    assert isinstance(column_cost, float)
    assert column_cost >= 0.0

    revenue = top_stream_value + bottom_stream_value - column_cost

    print("Test passed, please inspect flowsheet GUI to confirm the values match the following "
          "values.")
    print(f"Input stream: {fake_stream}")
    print(f"Column input spec: {fake_column_input_spec}")
    print(f"Tops: {tops}")
    print(f"Bottoms: {bottoms}")
    print(f"Column output spec: {column_output_spec}")
    print(f"TAC [k€]: {column_cost}")
    print(f"Top stream value [k€]: {top_stream_value}")
    print(f"Bottom stream value [k€]: {bottom_stream_value}")
    print(f"Revenue [k€]: {revenue}")


if __name__ == '__main__':
    # example run with the fake distillation api
    # from hydrocarbon_problem.api.fake_api import FakeDistillationAPI
    # api = FakeDistillationAPI()
    # test_api(api)

    from hydrocarbon_problem.api.aspen_api import BaseAspenDistillationAPI, AspenAPI
    api = AspenAPI()
    test_api(api)
