from typing import Tuple
from Simulation import Simulation

from hydrocarbon_problem.api.api_base import BaseAspenDistillationAPI
from hydrocarbon_problem.api.types import StreamSpecification, ColumnInputSpecification, \
    ColumnOutputSpecification, ProductSpecification, PerCompoundProperty

PATH = 'C:/Users/s2199718/Desktop/RL_PD/AspenSimulation/HydrocarbonMixture.bkp'


class AspenAPI(BaseAspenDistillationAPI):
    def __init__(self):
        self._flowsheet: Simulation = Simulation(PATH=PATH, VISIBILITY=False)
        self._feed_name: str = "S1"
        self._tops_name: str = "S2"
        self._bottoms_name: str = "S3"
        self._name_to_aspen_name = PerCompoundProperty(ethane="ETHANE",
                                                       propane="PROPANE", isobutane="I-BUTANE",
                                                       n_butane="N-BUTANE", isopentane="I-PENTAN", n_pentane="N-PENTAN")

    def set_input_stream_specification(self, stream_specification: StreamSpecification) -> None:
        """Sets the input stream to a column to fit the stream specification"""
        # Defining the Thermodynamic Properties
        self._flowsheet.STRM_Temperature(self._feed_name, stream_specification.temperature)
        self._flowsheet.STRM_Pressure(self._feed_name, stream_specification.pressure)
        # Defining the Stream Composition for the Feed
        self._flowsheet.STRM_Flowrate(self._feed_name, self._name_to_aspen_name.ethane,
                                      stream_specification.molar_flows.ethane)
        self._flowsheet.STRM_Flowrate(self._feed_name, self._name_to_aspen_name.propane,
                                      stream_specification.molar_flows.propane)
        self._flowsheet.STRM_Flowrate(self._feed_name, self._name_to_aspen_name.isobutane,
                                      stream_specification.molar_flows.isobutane)
        self._flowsheet.STRM_Flowrate(self._feed_name, self._name_to_aspen_name.n_butane,
                                      stream_specification.molar_flows.n_butane)
        self._flowsheet.STRM_Flowrate(self._feed_name, self._name_to_aspen_name.isopentane,
                                      stream_specification.molar_flows.isopentane)
        self._flowsheet.STRM_Flowrate(self._feed_name, self._name_to_aspen_name.n_pentane,
                                      stream_specification.molar_flows.n_pentane)

    def get_output_stream_specifications(self) -> Tuple[StreamSpecification, StreamSpecification]:
        # Getting the physical values of Top streams
        tops_temperature = self._flowsheet.STRM_Get_Temperature(self._tops_name)
        tops_pressure = self._flowsheet.STRM_Get_Pressure(self._tops_name)
        # Acquiring the outputs out of the Destillate (Top Stream)
        tops_ethane = self._flowsheet.STRM_Get_Outputs(self._tops_name, self._name_to_aspen_name.ethane)
        tops_propane = self._flowsheet.STRM_Get_Outputs(self._tops_name, self._name_to_aspen_name.propane)
        tops_isobutane = self._flowsheet.STRM_Get_Outputs(self._tops_name, self._name_to_aspen_name.isobutane)
        tops_n_butane = self._flowsheet.STRM_Get_Outputs(self._tops_name, self._name_to_aspen_name.n_butane)
        tops_isopentane = self._flowsheet.STRM_Get_Outputs(self._tops_name, self._name_to_aspen_name.isopentane)
        tops_n_pentane = self._flowsheet.STRM_Get_Outputs(self._tops_name, self._name_to_aspen_name.n_pentane)
        # Passing all the variables to their respective "slot"
        tops_specifications = StreamSpecification(temperature=tops_temperature, pressure=tops_pressure,
                                                  molar_flows=PerCompoundProperty(ethane=tops_ethane,
                                                                                  propane=tops_propane,
                                                                                  isobutane=tops_isobutane,
                                                                                  n_butane=tops_n_butane,
                                                                                  isopentane=tops_isopentane,
                                                                                  n_pentane=tops_n_pentane))

        # Getting the physical values of Top streams
        bots_temperature = self._flowsheet.STRM_Get_Temperature(self._bottoms_name)
        bots_pressure = self._flowsheet.STRM_Get_Pressure(self._bottoms_name)
        # Acquiring the outputs out of the Bottom (Bottom Stream)
        bots_ethane = self._flowsheet.STRM_Get_Outputs(self._bottoms_name, self._name_to_aspen_name.ethane)
        bots_propane = self._flowsheet.STRM_Get_Outputs(self._bottoms_name, self._name_to_aspen_name.propane)
        bots_isobutane = self._flowsheet.STRM_Get_Outputs(self._bottoms_name, self._name_to_aspen_name.isobutane)
        bots_n_butane = self._flowsheet.STRM_Get_Outputs(self._bottoms_name, self._name_to_aspen_name.n_butane)
        bots_isopentane = self._flowsheet.STRM_Get_Outputs(self._bottoms_name, self._name_to_aspen_name.isopentane)
        bots_n_pentane = self._flowsheet.STRM_Get_Outputs(self._bottoms_name, self._name_to_aspen_name.n_pentane)
        # Tubulating the Results of the Bottom Stream
        bots_specifications = StreamSpecification(temperature=bots_temperature, pressure=bots_pressure,
                                                  molar_flows=PerCompoundProperty(ethane=bots_ethane,
                                                                                  propane=bots_propane,
                                                                                  isobutane=bots_isobutane,
                                                                                  n_butane=bots_n_butane,
                                                                                  isopentane=bots_isopentane,
                                                                                  n_pentane=bots_n_pentane))

        return tops_specifications, bots_specifications

    def get_simulated_column_properties(self) -> ColumnOutputSpecification:
        # D_F_Location = self._flowsheet.BLK_Get_FeedLocation()
        # D_Pressure = self._flowsheet.BLK_Get_Pressure()
        # D_Reflux_Ratio = self._flowsheet.BLK_Get_RefluxRatio()
        # D_Reboiler_Ratio = self._flowsheet.BLK_Get_ReboilerRatio()
        D_Cond_Duty = self._flowsheet.BLK_Get_Condenser_Duty()
        D_Reb_Duty = self._flowsheet.BLK_Get_Reboiler_Duty()
        D_Col_Diameter = self._flowsheet.BLK_Get_Column_Diameter()

        D_Specifications = ColumnOutputSpecification(condensor_duty=D_Cond_Duty, reboiler_duty=D_Reb_Duty,
                                                     diameter=D_Col_Diameter)

        return D_Specifications

    def set_column_specification(self, column_specification: ColumnInputSpecification) -> None:
        self._flowsheet.BLK_NumberOfStages(column_specification.n_stages)
        self._flowsheet.BLK_FeedLocation(column_specification.feed_stage_location)
        self._flowsheet.BLK_Pressure(column_specification.condensor_pressure)
        self._flowsheet.BLK_RefluxRatio(column_specification.reflux_ratio)
        self._flowsheet.BLK_ReboilerRatio(column_specification.reboil_ratio)

    def solve_flowsheet(self) -> bool:
        self._flowsheet.Run()

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
    test_stream = StreamSpecification
    aspen_api.set_input_stream_specification(test_stream)