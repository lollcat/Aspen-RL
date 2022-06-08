import pprint

import pandas as pd
import math
from hydrocarbon_problem.api.aspen_api import AspenAPI
from hydrocarbon_problem.api import Simulation


column_number = [1, 2, 3, 4]
path = r"C:\Users\s2399016\Documents\Python_RL\Hydrocarbon problem base case\Linear\Linear BaseCase - LowFlow.xlsx"
Cost = []
StreamSpec = 8
ColumnInput = 5
distil = AspenAPI()
stream = Simulation.Simulation(VISIBILITY=False)
TotalCosts = {
    "Column 1": {'Number': 1,
                 'Costs M€': 0,
                 'Topvalue M€': 0,
                 'Bottomvalue M€': 0},
    "Column 2": {'Number': 2,
                 'Costs M€': 0,
                 'Topvalue M€': 0,
                 'Bottomvalue M€': 0},
    "Column 3": {'Number': 3,
                 'Costs M€': 0,
                 'Topvalue M€': 0,
                 'Bottomvalue M€':0},
    "Column 4": {'Number': 4,
                 'Costs M€': 0,
                 'Topvalue M€': 0,
                 'Bottomvalue M€': 0}
}

for i in column_number:
    data = pd.read_excel(path, sheet_name="Linear RADFRAC Column "+str(i)+" DifP")

    stream_spec = round(data.StreamSpecification, 2)
    stream_spec = stream_spec.values.tolist()
    stream_spec = [x for x in stream_spec if math.isnan(x) == False]

    column_input = round(data.ColumnInputSpecification, 2)
    column_input = column_input.values.tolist()
    column_input = [x for x in column_input if math.isnan(x) == False]

    column_output = round(data.ColumnOutputSpecification, 2)
    column_output = column_output.values.tolist()
    column_output = [x for x in column_output if math.isnan(x) == False]

    temp_profile = round(data.TemperatureDegree, 2)
    temp_profile = temp_profile.values.tolist()
    temp_profile = [x for x in temp_profile if math.isnan(x) == False]

    vap_flow_profile = round(data.Vapourflow, 2)
    vap_flow_profile = vap_flow_profile.values.tolist()
    vap_flow_profile = [x for x in vap_flow_profile if math.isnan(x) == False]


    vap_MW_profile = round(data.VapourMW, 2)
    vap_MW_profile = vap_MW_profile.values.tolist()
    vap_MW_profile = [x for x in vap_MW_profile if math.isnan(x) == False]

    top_stream = round(data.TopStream, 2)
    top_stream = top_stream.values.tolist()
    top_stream = [x for x in top_stream if math.isnan(x) == False]

    bot_stream = round(data.BotStream, 2)
    bot_stream = bot_stream.values.tolist()
    bot_stream = [x for x in bot_stream if math.isnan(x) == False]

    top_product_specification = round(data.TopPurity, 2)
    top_product_specification = top_product_specification.values.tolist()
    top_product_specification = [x for x in top_product_specification if math.isnan(x) == False]

    bot_product_specification = round(data.BotPurity, 2)
    bot_product_specification = bot_product_specification.values.tolist()
    bot_product_specification = [x for x in bot_product_specification if math.isnan(x) == False]

    TAC, D = distil.EXCEL_get_column_cost(stream_spec=stream_spec[-1],
                                       column_output=column_output,
                                       temp_profile=temp_profile,
                                       vap_flow_profile=vap_flow_profile,
                                       vap_MW_profile=vap_MW_profile)

    top_stream_revenue = stream.Excel_CAL_stream_value(stream_specification=top_stream,
                                                   product_specification=top_product_specification)
    bottom_stream_revenue = stream.Excel_CAL_stream_value(stream_specification=bot_stream,
                                                              product_specification=bot_product_specification)

    TotalCosts[str('Column '+str(i))]['Costs M€']=round(-TAC,2)
    TotalCosts[str('Column '+str(i))]['Topvalue M€']=round(top_stream_revenue, 2)
    TotalCosts[str('Column '+str(i))]['Bottomvalue M€'] = round(bottom_stream_revenue, 2)
    TotalCosts[str('Column '+str(i))]['Diameter m'] = round(D, 2)

pprint.pprint(TotalCosts, sort_dicts=False)
total_TAC = sum(d['Costs M€'] for d in TotalCosts.values() if d)
total_top = sum(d['Topvalue M€'] for d in TotalCosts.values() if d)
total_bottom = sum(d['Bottomvalue M€'] for d in TotalCosts.values() if d)

print(f"Total TAC:  M€ {total_TAC}")
print(f"Total top:  M€ {round(total_top,2)}")
print(f"Total bottom:  M€ {total_bottom}")

yearly_revenue = total_TAC + total_top + total_bottom
print(f"Yearly profit:  M€ {yearly_revenue}")





