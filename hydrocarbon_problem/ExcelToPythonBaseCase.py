import pprint

import pandas as pd
import math
from hydrocarbon_problem.api.aspen_api import AspenAPI
from hydrocarbon_problem.api import Simulation


column_number = [1, 2, 3, 4]
sep = [0, 1]  # 0 = linear, 1 = tree
path = [r"C:\Users\s2399016\Documents\Python_RL\Hydrocarbon problem base case\Linear\Linear BaseCase - LowFlow.xlsx",
        r"C:\Users\s2399016\Documents\Python_RL\Hydrocarbon problem base case\Tree\Tree BaseCase.xlsx"]
Cost = []
StreamSpec = 8
ColumnInput = 5
distil = AspenAPI()
stream = Simulation.Simulation(VISIBILITY=False)
TotalCosts = {
    "Column 1": {'Number': 1,
                 'Costs M€': {'linear': 0,
                              'tree': 0},
                 'Topvalue M€': {'linear': 0,
                                 'tree': 0},
                 'Bottomvalue M€': {'linear': 0,
                                    'tree': 0},
                 'Diameter m': {'linear': 0,
                                'tree': 0}},
    "Column 2": {'Number': 2,
                 'Costs M€': {'linear': 0,
                              'tree': 0},
                 'Topvalue M€': {'linear': 0,
                                 'tree': 0},
                 'Bottomvalue M€': {'linear': 0,
                                    'tree': 0},
                 'Diameter m': {'linear': 0,
                                'tree': 0}},
    "Column 3": {'Number': 3,
                 'Costs M€': {'linear': 0,
                              'tree': 0},
                 'Topvalue M€': {'linear': 0,
                                 'tree': 0},
                 'Bottomvalue M€': {'linear': 0,
                                    'tree': 0},
                 'Diameter m': {'linear': 0,
                                'tree': 0}},
    "Column 4": {'Number': 4,
                 'Costs M€': {'linear': 0,
                              'tree': 0},
                 'Topvalue M€': {'linear': 0,
                                 'tree': 0},
                 'Bottomvalue M€': {'linear': 0,
                                    'tree': 0},
                 'Diameter m': {'linear': 0,
                                'tree': 0}}
}

for j in sep:
    for i in column_number:
        print(j,i)
        data = pd.read_excel(path[j], sheet_name="RADFRAC Column "+str(i)+" DifP")

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
        print(TAC)
        top_stream_revenue = stream.Excel_CAL_stream_value(stream_specification=top_stream,
                                                       product_specification=top_product_specification)
        print(top_stream_revenue)
        bottom_stream_revenue = stream.Excel_CAL_stream_value(stream_specification=bot_stream,
                                                                  product_specification=bot_product_specification)
        print(bottom_stream_revenue)

        if j == 0:
            method = "linear"
        else:
            method = "tree"

        TotalCosts[str('Column '+str(i))]['Costs M€'][method]=round(-TAC,2)
        TotalCosts[str('Column '+str(i))]['Topvalue M€'][method]=round(top_stream_revenue, 2)
        TotalCosts[str('Column '+str(i))]['Bottomvalue M€'][method] = round(bottom_stream_revenue, 2)
        TotalCosts[str('Column '+str(i))]['Diameter m'][method] = round(D, 2)

pprint.pprint(TotalCosts, sort_dicts=False)

total_TAC_linear = sum(d['Costs M€']['linear'] for d in TotalCosts.values() if d)
total_top_linear = sum(d['Topvalue M€']['linear'] for d in TotalCosts.values() if d)
total_bottom_linear = sum(d['Bottomvalue M€']['linear'] for d in TotalCosts.values() if d)

total_TAC_tree = sum(d['Costs M€']['tree'] for d in TotalCosts.values() if d)
total_top_tree = sum(d['Topvalue M€']['tree'] for d in TotalCosts.values() if d)
total_bottom_tree = sum(d['Bottomvalue M€']['tree'] for d in TotalCosts.values() if d)

print(f"Total TAC:  M€ linear: {total_TAC_linear}, tree: {total_TAC_tree}")
print(f"Total top:  M€ linear: {round(total_top_linear,2)}, tree: {round(total_top_tree,2)}")
print(f"Total bottom:  M€ linear: {round(total_bottom_linear,2)}, tree: {round(total_bottom_tree,2)}")

yearly_revenue_linear = total_TAC_linear + total_top_linear + total_bottom_linear
yearly_revenue_tree = total_TAC_tree + total_top_tree + total_bottom_tree

print(f"Yearly profit:  M€ linear: {yearly_revenue_linear}, tree: {yearly_revenue_tree}")





