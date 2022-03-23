from typing import Tuple
import numpy as np
# from sqlalchemy import column
from hydrocarbon_problem.api.Simulation import Simulation

from hydrocarbon_problem.api.api_base import BaseAspenDistillationAPI
from hydrocarbon_problem.api.types_ import StreamSpecification, ColumnInputSpecification, \
    ColumnOutputSpecification, ProductSpecification, PerCompoundProperty

PATH = 'C:/Users/s2399016/Documents/Aspen-RL_v2/Aspen-RL/hydrocarbon_problem/AspenSimulation/HydrocarbonMixture.bkp'


class AspenAPI(BaseAspenDistillationAPI):
    def __init__(self):
        self._flowsheet: Simulation = Simulation(PATH=PATH, VISIBILITY=False)
        self._feed_name: str = "S1"
        self._tops_name: str = "S2"
        self._bottoms_name: str = "S3"
        self._name_to_aspen_name = PerCompoundProperty(ethane="ETHANE",
                                                       propane="PROPANE",
                                                       isobutane="I-BUTANE",
                                                       n_butane="N-BUTANE",
                                                       isopentane="I-PENTAN",
                                                       n_pentane="N-PENTAN")
        
    def set_input_stream_specification(self, stream_specification: StreamSpecification) -> None:
        """Sets the input stream to a column to fit the stream specification"""
        # Defining the Thermodynamic Properties
        self._flowsheet.STRM_Temperature(self._feed_name,  stream_specification.temperature)
        self._flowsheet.STRM_Pressure(self._feed_name, stream_specification.pressure)
        # Defining the Stream Composition for the Feed
        self._flowsheet.STRM_Flowrate(self._feed_name, self._name_to_aspen_name.ethane,     stream_specification.molar_flows.ethane)
        self._flowsheet.STRM_Flowrate(self._feed_name, self._name_to_aspen_name.propane,    stream_specification.molar_flows.propane)
        self._flowsheet.STRM_Flowrate(self._feed_name, self._name_to_aspen_name.isobutane,  stream_specification.molar_flows.isobutane)
        self._flowsheet.STRM_Flowrate(self._feed_name, self._name_to_aspen_name.n_butane,   stream_specification.molar_flows.n_butane)
        self._flowsheet.STRM_Flowrate(self._feed_name, self._name_to_aspen_name.isopentane, stream_specification.molar_flows.isopentane)
        self._flowsheet.STRM_Flowrate(self._feed_name, self._name_to_aspen_name.n_pentane,  stream_specification.molar_flows.n_pentane)

    def get_output_stream_specifications(self) -> Tuple[StreamSpecification, StreamSpecification]:
        # Getting the physical values of Top streams
        tops_temperature = self._flowsheet.STRM_Get_Temperature(self._tops_name)
        tops_pressure = self._flowsheet.STRM_Get_Pressure(self._tops_name)
        # Acquiring the outputs out of the Distillate (Top Stream)
        tops_ethane = self._flowsheet.STRM_Get_Outputs(self._tops_name, self._name_to_aspen_name.ethane)
        tops_propane = self._flowsheet.STRM_Get_Outputs(self._tops_name, self._name_to_aspen_name.propane)
        tops_isobutane = self._flowsheet.STRM_Get_Outputs(self._tops_name, self._name_to_aspen_name.isobutane)
        tops_n_butane = self._flowsheet.STRM_Get_Outputs(self._tops_name, self._name_to_aspen_name.n_butane)
        tops_isopentane = self._flowsheet.STRM_Get_Outputs(self._tops_name, self._name_to_aspen_name.isopentane)
        tops_n_pentane = self._flowsheet.STRM_Get_Outputs(self._tops_name, self._name_to_aspen_name.n_pentane)
        # Passing all the variables to their respective "slot"
        tops_specifications = StreamSpecification(temperature=tops_temperature, pressure= tops_pressure,
                                                  molar_flows =PerCompoundProperty(ethane=tops_ethane,
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
                                                  molar_flows =PerCompoundProperty(ethane=bots_ethane,
                                                                                   propane=bots_propane,
                                                                                   isobutane=bots_isobutane,
                                                                                   n_butane=bots_n_butane,
                                                                                   isopentane=bots_isopentane,
                                                                                   n_pentane=bots_n_pentane))

        return tops_specifications, bots_specifications

    def get_simulated_column_properties(self, column_input_specification: ColumnInputSpecification) -> ColumnOutputSpecification:

        D_Cond_Duty = self._flowsheet.BLK_Get_Condenser_Duty()
        D_Reb_Duty = self._flowsheet.BLK_Get_Reboiler_Duty()
        vap_flows = self._flowsheet.BLK_Get_Column_Stage_Vapor_Flows(column_input_specification.n_stages)
        stage_temp = self._flowsheet.BLK_Get_Column_Stage_Temperatures(column_input_specification.n_stages)
        stage_mw = self._flowsheet.BLK_Get_Column_Stage_Molar_Weights(column_input_specification.n_stages)

        D_Specifications = ColumnOutputSpecification(condenser_duty=D_Cond_Duty,
                                                     reboiler_duty=D_Reb_Duty,
                                                     vapor_flow_per_stage=vap_flows,
                                                     temperature_per_stage=stage_temp,
                                                     molar_weight_per_stage=stage_mw)

        return D_Specifications

    def set_column_specification(self, column_input_specification: ColumnInputSpecification) -> None:
        self._flowsheet.BLK_NumberOfStages(column_input_specification.n_stages)
        self._flowsheet.BLK_FeedLocation(column_input_specification.feed_stage_location, "S1")
        self._flowsheet.BLK_Pressure(column_input_specification.condensor_pressure)
        self._flowsheet.BLK_RefluxRatio(column_input_specification.reflux_ratio)
        self._flowsheet.BLK_ReboilerRatio(column_input_specification.reboil_ratio)

    def solve_flowsheet(self) -> bool:
        print(self._flowsheet.Run())
        return True

    def get_column_cost(self, column_input_specification: ColumnInputSpecification,
                        column_output_specification: ColumnOutputSpecification) -> float:
        t_reboiler = column_output_specification.temperature_per_stage[-1]
        t_condenser = column_output_specification.temperature_per_stage[0]
        total_cost = self._flowsheet.CAL_InvestmentCost(column_input_specification.n_stages,
                                                        column_output_specification.condenser_duty,
                                                        t_reboiler,
                                                        column_output_specification.reboiler_duty,
                                                        t_condenser,
                                                        column_output_specification.vapor_flow_per_stage,
                                                        column_output_specification.molar_weight_per_stage,
                                                        column_output_specification.temperature_per_stage) + \
                     self._flowsheet.CAL_Annual_OperatingCost(column_output_specification.reboiler_duty,
                                                              column_output_specification.condenser_duty)
        return total_cost

    def get_stream_value(self, stream, ProductSpecification) -> float:
        """Calculates the value (per year) of a stream."""
        stream_value = self._flowsheet.CAL_stream_value(stream, ProductSpecification.purity)
        # top_stream_value = self._flowsheet.CAL_stream_value(tops_specifications.molar_flows, product_specification.purity)
        # bottom_stream_value = self._flowsheet.CAL_stream_value(bots_specifications.molar_flows, product_specification.purity)

        # total_stream_value = top_stream_value + bottom_stream_value

        """"
        tops, bottoms = aspen_api.get_output_stream_specifications()

        for stream in [tops, bottoms]:
            stream_value = self._flowsheet.CAL_stream_value(stream, ProductSpecification.purity)

        top_stream_value = self._flowsheet.CAL_stream_value(component_specifications, stream_specification.molar_flows,
                                                            top_component_specifications)
        # bottom_stream_value = self._flowsheet.CAL_Stream_Value(component_specifications, bots_specifications.molar_flows, bottom_component_specifications)

        return top_stream_value  # , bottom_stream_value
        """

        return stream_value  # total_stream_value

    def stream_is_product(self, stream, ProductSpecification) -> int:  # Tuple[StreamSpecification, StreamSpecification]:
        """Checks whether a stream meets the product specification."""
        is_purity, component_purities = self._flowsheet.CAL_purity_check(stream, ProductSpecification.purity)
        # top_purity_check = self._flowsheet.CAL_purity_check(tops_specifications, product_specification.purity)
        # bottom_purity_check = self._flowsheet.CAL_purity_check(bots_specifications, product_specification.purity)
        if np.any(is_purity):
            purity = 1
        else:
            purity = 0

        return purity  # top_purity_check, bottom_purity_check

    def error_check(self):
        error = self._flowsheet.get_error()
        return error


if __name__ == '__main__':
    aspen_api = AspenAPI()
    test_stream = StreamSpecification
    aspen_api.set_input_stream_specification(test_stream)