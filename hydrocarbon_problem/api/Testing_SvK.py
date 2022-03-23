# The purpose of the following file is to make a testing of already made properties
# Import all the needed libraries
import os
from Simulation import Simulation
import numpy as np

#Defining Testing Values
PATH = 'C:/Users/s2399016/Documents/ASPEN-RL_V2/Aspen-RL/hydrocarbon_problem/AspenSimulation/HydrocarbonMixture.bkp'
VISIBILITY = True
NSTAGES = 20
FEED_STAGE = 10
REFLUX = 2
REBOILER = 2
PRESSURE = 1
FEED_NAME = "S1"
CHEMICALS = ["ETHANE","PROPANE","I-BUTANE","N-BUTANE","I-PENTAN","N-PENTAN"]
FLOWRATE = [1,1,1,1,1,1]
TEMPERATURE = 60

#Initiate the Flowsheet
flowsheet = Simulation(PATH,VISIBILITY)

#Defining the Distillation Column Properties
FEED_LOC_CHOOSE = flowsheet.BLK_FeedLocation(Feed_Location=FEED_STAGE,Feed_Name="S1")
N_STAGES_CHOOSE = flowsheet.BLK_NumberOfStages(NSTAGES)
PRES_DIST_CHOOSE = flowsheet.BLK_Pressure(PRESSURE)
RFX_DIST_CHOOSE = flowsheet.BLK_RefluxRatio(REFLUX)
RBL_DIST_CHOOSE = flowsheet.BLK_ReboilerRatio(REBOILER)
#Defining the FEED Stream Properties
FEED_TEMP = flowsheet.STRM_Temperature(FEED_NAME,TEMPERATURE)
FEED_PRES = flowsheet.STRM_Pressure(FEED_NAME,PRESSURE)
FEED_COMP_1_FLOW = flowsheet.STRM_Flowrate(FEED_NAME,CHEMICALS[0],FLOWRATE[0])
FEED_COMP_2_FLOW = flowsheet.STRM_Flowrate(FEED_NAME,CHEMICALS[1],FLOWRATE[1])
FEED_COMP_3_FLOW = flowsheet.STRM_Flowrate(FEED_NAME,CHEMICALS[2],FLOWRATE[2])
FEED_COMP_4_FLOW = flowsheet.STRM_Flowrate(FEED_NAME,CHEMICALS[3],FLOWRATE[3])
FEED_COMP_5_FLOW = flowsheet.STRM_Flowrate(FEED_NAME,CHEMICALS[4],FLOWRATE[4])
FEED_COMP_6_FLOW = flowsheet.STRM_Flowrate(FEED_NAME,CHEMICALS[5],FLOWRATE[5])

# We can run the Simulation
flowsheet.Run()

#printing all the changed values
print("feed location: ", flowsheet.BLK_Get_FeedLocation("S1"))
NSTAGES = flowsheet.BLK_Get_NStages()
CHEMICALS = ["ETHANE","PROPANE","I-BUTANE","N-BUTANE","I-PENTAN","N-PENTAN"]
print("column pressure: ", flowsheet.BLK_Get_Pressure())
print("column reflux ratio: ", flowsheet.BLK_Get_RefluxRatio())
print("column reboiler ratio: ", flowsheet.BLK_Get_ReboilerRatio())
print("feed temperature: ",flowsheet.STRM_Get_Temperature("S1"))
print("destillation reboiler duty: ", flowsheet.BLK_Get_Reboiler_Duty())
print("distillation condenser duty: ", flowsheet.BLK_Get_Condenser_Duty())

# How to indirectly calculate the column diameter
print("column diameter: ", flowsheet.CAL_Column_Diameter(column_input_specification.n_stages,
                                                       column_output_specification.vapor_flow_per_stage,
                                                       column_output_specification.molar_weight_per_stage,
                                                       column_output_specification.temperature_per_stage))
print("column height: ", flowsheet.CAL_Column_Height(column_input_specifications.n_stages))
print("heat transfer area condenser: ", flowsheet.CAL_HT_Condenser_Area(column_output_specification.condenser_duty,
                                                                       column_output_specification.temperature_per_stage[0]))
print("heat transfer area reboiler: ", flowsheet.CAL_HT_Reboiler_Area(column_output_specification.reboiler_duty,
                                                                     column_output_specification.temperature_per_stage[-1]))
print("column LMTD: ", flowsheet.CAL_LMTD(column_output_specification.top_temperature))
print("column Investment Cost: ", flowsheet.CAL_InvestmentCost(column_input_specification.n_stages,
                                                             column_output_specification.condenser_duty,
                                                              t_reboiler,
                                                              column_output_specification.reboiler_duty,
                                                              t_condenser,
                                                              column_output_specification.vapor_flow_per_stage,
                                                              column_output_specification.molar_weight_per_stage,
                                                              column_output_specification.temperature_per_stage))
print("column Operating Cost: ", flowsheet.CAL_OperatingCost(column_output_specification.reboiler_duty,
                                                            column_output_specification.condenser_duty))
print("column Annually Operating Cost: ", flowsheet.CAL_Annual_OperatingCost(column_output_specification.reboiler_duty,
                                                                            column_output_specification.condenser_duty))