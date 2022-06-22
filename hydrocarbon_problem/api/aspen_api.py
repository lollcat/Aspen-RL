from typing import Tuple
import numpy as np
# from sqlalchemy import column
from hydrocarbon_problem.api.Simulation import Simulation

from hydrocarbon_problem.api.api_base import BaseAspenDistillationAPI
from hydrocarbon_problem.api.types_ import StreamSpecification, ColumnInputSpecification, \
    ColumnOutputSpecification, ProductSpecification, PerCompoundProperty


class AspenAPI(BaseAspenDistillationAPI):
    def __init__(self, max_solve_iterations: int = 100):
        self._flowsheet: Simulation = Simulation(VISIBILITY=True,
                                                 max_iterations=max_solve_iterations)
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
        self._flowsheet.STRM_Temperature(Name=self._feed_name, Temp=stream_specification.temperature)
        self._flowsheet.STRM_Pressure(Name=self._feed_name, Pressure=stream_specification.pressure)
        # Defining the Stream Composition for the Feed
        self._flowsheet.STRM_Flowrate(Name=self._feed_name, Chemical=self._name_to_aspen_name.ethane,
                                      Flowrate=stream_specification.molar_flows.ethane)
        self._flowsheet.STRM_Flowrate(Name=self._feed_name, Chemical=self._name_to_aspen_name.propane,
                                      Flowrate=stream_specification.molar_flows.propane)
        self._flowsheet.STRM_Flowrate(Name=self._feed_name, Chemical=self._name_to_aspen_name.isobutane,
                                      Flowrate=stream_specification.molar_flows.isobutane)
        self._flowsheet.STRM_Flowrate(Name=self._feed_name, Chemical=self._name_to_aspen_name.n_butane,
                                      Flowrate=stream_specification.molar_flows.n_butane)
        self._flowsheet.STRM_Flowrate(Name=self._feed_name, Chemical=self._name_to_aspen_name.isopentane,
                                      Flowrate=stream_specification.molar_flows.isopentane)
        self._flowsheet.STRM_Flowrate(Name=self._feed_name, Chemical=self._name_to_aspen_name.n_pentane,
                                      Flowrate=stream_specification.molar_flows.n_pentane)

    def get_output_stream_specifications(self) -> Tuple[StreamSpecification, StreamSpecification]:
        # Getting the physical values of Top streams
        tops_temperature = self._flowsheet.STRM_Get_Temperature(Name=self._tops_name)
        tops_pressure = self._flowsheet.STRM_Get_Pressure(Name=self._tops_name)
        # Acquiring the outputs out of the Distillate (Top Stream)
        tops_ethane = self._flowsheet.STRM_Get_Outputs(Name=self._tops_name, Chemical=self._name_to_aspen_name.ethane)
        tops_propane = self._flowsheet.STRM_Get_Outputs(Name=self._tops_name, Chemical=self._name_to_aspen_name.propane)
        tops_isobutane = self._flowsheet.STRM_Get_Outputs(Name=self._tops_name, Chemical=self._name_to_aspen_name.isobutane)
        tops_n_butane = self._flowsheet.STRM_Get_Outputs(Name=self._tops_name, Chemical=self._name_to_aspen_name.n_butane)
        tops_isopentane = self._flowsheet.STRM_Get_Outputs(Name=self._tops_name, Chemical=self._name_to_aspen_name.isopentane)
        tops_n_pentane = self._flowsheet.STRM_Get_Outputs(Name=self._tops_name, Chemical=self._name_to_aspen_name.n_pentane)
        # Passing all the variables to their respective "slot"
        tops_specifications = StreamSpecification(temperature=tops_temperature, pressure=tops_pressure,
                                                  molar_flows=PerCompoundProperty(ethane=tops_ethane,
                                                                                  propane=tops_propane,
                                                                                  isobutane=tops_isobutane,
                                                                                  n_butane=tops_n_butane,
                                                                                  isopentane=tops_isopentane,
                                                                                  n_pentane=tops_n_pentane))

        # Getting the physical values of Top streams
        bots_temperature = self._flowsheet.STRM_Get_Temperature(Name=self._bottoms_name)

        bots_pressure = self._flowsheet.STRM_Get_Pressure(Name=self._bottoms_name)
        # Acquiring the outputs out of the Bottom (Bottom Stream)
        bots_ethane = self._flowsheet.STRM_Get_Outputs(Name=self._bottoms_name,Chemical=self._name_to_aspen_name.ethane)
        bots_propane = self._flowsheet.STRM_Get_Outputs(Name=self._bottoms_name, Chemical=self._name_to_aspen_name.propane)
        bots_isobutane = self._flowsheet.STRM_Get_Outputs(Name=self._bottoms_name, Chemical=self._name_to_aspen_name.isobutane)
        bots_n_butane = self._flowsheet.STRM_Get_Outputs(Name=self._bottoms_name, Chemical=self._name_to_aspen_name.n_butane)
        bots_isopentane = self._flowsheet.STRM_Get_Outputs(Name=self._bottoms_name, Chemical=self._name_to_aspen_name.isopentane)
        bots_n_pentane = self._flowsheet.STRM_Get_Outputs(Name=self._bottoms_name, Chemical=self._name_to_aspen_name.n_pentane)
        # Tubulating the Results of the Bottom Stream
        bots_specifications = StreamSpecification(temperature=bots_temperature, pressure=bots_pressure,
                                                  molar_flows=PerCompoundProperty(ethane=bots_ethane,
                                                                                  propane=bots_propane,
                                                                                  isobutane=bots_isobutane,
                                                                                  n_butane=bots_n_butane,
                                                                                  isopentane=bots_isopentane,
                                                                                  n_pentane=bots_n_pentane))

        return tops_specifications, bots_specifications

    def get_simulated_column_properties(self,
                                        column_input_specification: ColumnInputSpecification) -> ColumnOutputSpecification:

        D_Cond_Duty = self._flowsheet.BLK_Get_Condenser_Duty()
        D_Reb_Duty = self._flowsheet.BLK_Get_Reboiler_Duty()
        vap_flows = self._flowsheet.BLK_Get_Column_Stage_Vapor_Flows(N_stages=column_input_specification.n_stages)
        stage_temp = self._flowsheet.BLK_Get_Column_Stage_Temperatures(N_stages=column_input_specification.n_stages)
        stage_mw = self._flowsheet.BLK_Get_Column_Stage_Molar_Weights(N_stages=column_input_specification.n_stages)

        D_Specifications = ColumnOutputSpecification(condenser_duty=D_Cond_Duty,
                                                     reboiler_duty=D_Reb_Duty,
                                                     vapor_flow_per_stage=vap_flows,
                                                     temperature_per_stage=stage_temp,
                                                     molar_weight_per_stage=stage_mw)

        return D_Specifications

    def set_column_specification(self, column_input_specification: ColumnInputSpecification) -> None:
        self._flowsheet.BLK_NumberOfStages(nstages=column_input_specification.n_stages)
        self._flowsheet.BLK_FeedLocation(Feed_Location=column_input_specification.feed_stage_location, Feed_Name="S1")
        self._flowsheet.BLK_Pressure(Pressure=column_input_specification.condensor_pressure)
        self._flowsheet.BLK_RefluxRatio(RfxR=column_input_specification.reflux_ratio)
        self._flowsheet.BLK_ReboilerRatio(RblR=column_input_specification.reboil_ratio)

    def solve_flowsheet(self) -> None:
        self._flowsheet.Run()

    def get_column_cost(self, stream_specification: StreamSpecification,
                        column_input_specification: ColumnInputSpecification,
                        column_output_specification: ColumnOutputSpecification) -> float:

        t_reboiler = column_output_specification.temperature_per_stage[-1]
        t_condenser = column_output_specification.temperature_per_stage[0]

        invest = self._flowsheet.CAL_InvestmentCost(pressure=stream_specification.pressure,
                                                    n_stages=column_input_specification.n_stages,
                                                    condenser_duty=column_output_specification.condenser_duty,
                                                    reboiler_temperature=t_reboiler,
                                                    reboiler_duty=column_output_specification.reboiler_duty,
                                                    tops_temperature=t_condenser,
                                                    vapor_flows=column_output_specification.vapor_flow_per_stage,
                                                    stage_mw=column_output_specification.molar_weight_per_stage,
                                                    stage_temp=column_output_specification.temperature_per_stage)
        operating = self._flowsheet.CAL_Annual_OperatingCost(reboiler_duty=column_output_specification.reboiler_duty,
                                                             condenser_duty=column_output_specification.condenser_duty)

        total_cost = invest + operating
        return total_cost

    def get_stream_value(self, stream, product_specification) -> float:
        """Calculates the value (per year) of a stream."""
        stream_value = self._flowsheet.CAL_stream_value(stream_specification=stream,
                                                        product_specification=product_specification)

        return stream_value

    def stream_is_product_or_outlet(self, stream: StreamSpecification,
                                    product_specification: ProductSpecification) -> \
            Tuple[bool, bool]:
        """Checks whether a stream meets the product specification."""
        # is_product = False
        # total_flow = sum(stream.molar_flows)
        # if total_flow < 0.001:
        #     is_outlet = True
        #     is_product = False
        # else:
        #     is_outlet = False
        #     is_purity, component_purities = self._flowsheet.CAL_purity_check(stream_specification=stream,
        #                                                                      product_specification=product_specification.purity)
        #     if np.any(is_purity):
        #         is_product = True
        total_flow = sum(stream.molar_flows)
        if total_flow > 0.001:
            is_purity, component_purities = self._flowsheet.CAL_purity_check(
                stream_specification=stream, product_specification=product_specification.purity)
        else:
            is_purity = [0, 0, 0, 0, 0, 0]
        if np.any(is_purity):
            is_product = True
            is_outlet = True
        else:
            is_product = False
            if total_flow < 0.001:
                is_outlet = True
            else:
                is_outlet = False
        return is_product, is_outlet


if __name__ == '__main__':
    aspen_api = AspenAPI()
    test_stream = StreamSpecification
    aspen_api.set_input_stream_specification(test_stream)