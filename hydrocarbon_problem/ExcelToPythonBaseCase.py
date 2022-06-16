
import os
import pprint

import pandas as pd
import math
from hydrocarbon_problem.api.aspen_api import AspenAPI
from hydrocarbon_problem.api import Simulation

Total_costs = []
topvalue = []
botsvalue = []
Diameter = []
A_cnd = []
A_rbl = []
cost_column = []
cost_internals = []
cost_condenser = []
cost_reboiler = []
Operating_cost = []

def filter(enter_data):
    enter_data = enter_data.values.tolist()
    enter_data = [x for x in enter_data if math.isnan(x) == False]
    return enter_data


dataset = 1  # 0=excel, 1 = luyben

if dataset == 0:
    column_number = [1, 2, 3, 4]
    sep = [0, 1]  # 0 = linear, 1 = tree
    # ok = ['Linear', 'Tree']

    path_high = [r"C:\Users\s2399016\Documents\Aspen-RL_v2\Hydrocarbon problem base case\Linear\Linear BaseCase - HighFlow.xlsx",
            r"C:\Users\s2399016\Documents\Aspen-RL_v2\Hydrocarbon problem base case\Tree\Tree BaseCase - HighFlow.xlsx"]
    path_low = [
    [r"C:\Users\s2399016\Documents\Aspen-RL_v2\Hydrocarbon problem base case\Linear\Linear BaseCase - LowFlow.xlsx",
     r"C:\Users\s2399016\Documents\Aspen-RL_v2\Hydrocarbon problem base case\Tree\Tree BaseCase - LowFlow.xlsx"]
    ]
    Cost = []
    StreamSpec = 8
    ColumnInput = 5
    distil = AspenAPI()
    stream = Simulation.Simulation(VISIBILITY=False)

    for j in sep:

        Number = [1, 2, 3, 4]
        Total_costs.clear()
        topvalue.clear()
        botsvalue.clear()
        Diameter.clear()
        A_cnd.clear()
        A_rbl.clear()
        cost_column.clear()
        cost_internals.clear()
        cost_condenser.clear()
        cost_reboiler.clear()
        Operating_cost.clear()

        for i in Number:
            print(j,i)
            data = pd.read_excel(path_high[j], sheet_name="RADFRAC Column "+str(i)+" DifP")
            stream_spec = filter(round(data.StreamSpecification, 2))
            column_input = filter(round(data.ColumnInputSpecification, 2))
            column_output = filter(round(data.ColumnOutputSpecification, 2))
            temp_profile = filter(round(data.TemperatureDegree, 2))
            vap_flow_profile = filter(round(data.Vapourflow, 2))
            vap_MW_profile = filter(round(data.VapourMW, 2))
            top_stream = filter(round(data.TopStream, 2))
            bot_stream = filter(round(data.BotStream, 2))
            top_product_specification = filter(round(data.TopPurity, 2))
            bot_product_specification = filter(round(data.BotPurity, 2))

            TAC, D, a_cnd, a_rbl, c_col, c_int, c_cnd, c_rbl, operating = distil.EXCEL_get_column_cost(
                stream_spec=stream_spec[-1],
                column_output=column_output,
                temp_profile=temp_profile,
                vap_flow_profile=vap_flow_profile,
                vap_MW_profile=vap_MW_profile
            )

            top_stream_revenue = stream.Excel_CAL_stream_value(
                stream_specification=top_stream,
                product_specification=top_product_specification
            )

            bottom_stream_revenue = stream.Excel_CAL_stream_value(
                stream_specification=bot_stream,
                product_specification=bot_product_specification
            )

            Total_costs.append(TAC)
            topvalue.append(top_stream_revenue)
            botsvalue.append(bottom_stream_revenue)
            Diameter.append(D)
            A_cnd.append(a_cnd)
            A_rbl.append(a_rbl)
            cost_column.append(c_col)
            cost_internals.append(c_int)
            cost_condenser.append(c_cnd)
            cost_reboiler.append(c_rbl)
            Operating_cost.append(operating)

            if j==0 and i==4:
                df_linear = pd.DataFrame({
                    "Number": Number,
                    "Costs M€": Total_costs,
                    'Topvalue M€': topvalue,
                    'Bottomvalue M€': botsvalue,
                    'Diameter m': Diameter,
                    'A cnd m2': A_cnd,
                    'A rbl m2': A_rbl,
                    'Cost column M€': cost_column,
                    'Cost internals M€': cost_internals,
                    'Cost condenser M€': cost_condenser,
                    'Cost reboiler M€': cost_reboiler,
                    'Operating cost M€': operating,
                })


            elif j == 1 and i == 4:
                df_tree = pd.DataFrame({
                    "Number": Number,
                    "Costs M€": Total_costs,
                    'Topvalue M€': topvalue,
                    'Bottomvalue M€': botsvalue,
                    'Diameter m': Diameter,
                    'A cnd m2': A_cnd,
                    'A rbl m2': A_rbl,
                    'Cost column M€': cost_column,
                    'Cost internals M€': cost_internals,
                    'Cost condenser M€': cost_condenser,
                    'Cost reboiler M€': cost_reboiler,
                    'Operating cost M€': operating,
                })




#             TotalCosts[str('Column '+str(i))]['Costs M€'][method] = round(-TAC, 2)
#             TotalCosts[str('Column '+str(i))]['Topvalue M€'][method] = round(top_stream_revenue, 2)
#             TotalCosts[str('Column '+str(i))]['Bottomvalue M€'][method] = round(bottom_stream_revenue, 2)
#             TotalCosts[str('Column '+str(i))]['Diameter m'][method] = round(D, 2)
#             TotalCosts[str('Column '+str(i))]['A cnd m2'][method] = round(a_cnd, 2)
#             TotalCosts[str('Column '+str(i))]['A rbl m2'][method] = round(a_rbl, 2)
#             TotalCosts[str('Column ' + str(i))]['Cost column M€'][method] = round(c_col / 1000000, 2)
#             TotalCosts[str('Column ' + str(i))]['Cost internals M€'][method] = round(c_int/1000000, 2)
#             TotalCosts[str('Column ' + str(i))]['Cost condenser M€'][method] = round(c_cnd/1000000, 2)
#             TotalCosts[str('Column ' + str(i))]['Cost reboiler M€'][method] = round(c_rbl/1000000, 2)
#             TotalCosts[str('Column ' + str(i))]['Operating cost M€'][method] = round(operating, 2)
#
#     df = pd.DataFrame(TotalCosts)
#     print(df)
#     pprint.pprint(TotalCosts, sort_dicts=False)
#
#     total_TAC_linear = sum(d['Costs M€']['linear'] for d in TotalCosts.values() if d)
#     total_top_linear = sum(d['Topvalue M€']['linear'] for d in TotalCosts.values() if d)
#     total_bottom_linear = sum(d['Bottomvalue M€']['linear'] for d in TotalCosts.values() if d)
#
#     total_TAC_tree = sum(d['Costs M€']['tree'] for d in TotalCosts.values() if d)
#     total_top_tree = sum(d['Topvalue M€']['tree'] for d in TotalCosts.values() if d)
#     total_bottom_tree = sum(d['Bottomvalue M€']['tree'] for d in TotalCosts.values() if d)
#
#     print(f"Total TAC:  M€ linear: {total_TAC_linear}, tree: {total_TAC_tree}")
#     print(f"Total top:  M€ linear: {round(total_top_linear,2)}, tree: {round(total_top_tree,2)}")
#     print(f"Total bottom:  M€ linear: {round(total_bottom_linear,2)}, tree: {round(total_bottom_tree,2)}")
#
#     yearly_revenue_linear = total_TAC_linear + total_top_linear + total_bottom_linear
#     yearly_revenue_tree = total_TAC_tree + total_top_tree + total_bottom_tree
#
#     print(f"Yearly profit:  M€ linear: {yearly_revenue_linear}, tree: {yearly_revenue_tree}")
#
else:
    Number = [1, 2, 3]
    path = r"C:\Users\s2399016\Documents\Aspen-RL_v2\Hydrocarbon problem base case\Tree\Tree BaseCase - HighFlow.xlsx"
    Cost = []
    StreamSpec = 8
    ColumnInput = 5
    distil = AspenAPI()
    stream = Simulation.Simulation(VISIBILITY=False)

    Total_costs.clear()
    topvalue.clear()
    botsvalue.clear()
    Diameter.clear()
    A_cnd.clear()
    A_rbl.clear()
    cost_column.clear()
    cost_internals.clear()
    cost_condenser.clear()
    cost_reboiler.clear()
    Operating_cost.clear()

    for i in Number:
        data = pd.read_excel(path, sheet_name="Luyben Col "+str(i))

        stream_spec = filter(round(data.StreamSpecification, 2))
        column_input = filter(round(data.ColumnInputSpecification, 2))
        column_output = filter(round(data.ColumnOutputSpecification, 2))
        temperature = filter(round(data.Temperature, 2))
        diameter = filter(round(data.diameter, 2))
        top_stream = filter(round(data.TopStream, 2))
        bot_stream = filter(round(data.BotStream, 2))
        top_product_specification = filter(round(data.TopPurity, 2))
        bot_product_specification = filter(round(data.BotPurity, 2))

        TAC, a_cnd, a_rbl, c_col, c_int, c_cnd, c_rbl, operating = distil.Luyben_get_column_cost(stream_spec=stream_spec[-1],
                                            n_stages=column_input[0],
                                           column_output=column_output,
                                           temperature=temperature,
                                            diameter=diameter[0])

        top_stream_revenue = stream.Excel_CAL_stream_value(stream_specification=top_stream,
                                                       product_specification=top_product_specification)
        bottom_stream_revenue = stream.Excel_CAL_stream_value(stream_specification=bot_stream,
                                                                  product_specification=bot_product_specification)

        Total_costs.append(TAC)
        topvalue.append(top_stream_revenue)
        botsvalue.append(bottom_stream_revenue)
        Diameter.append(diameter)
        A_cnd.append(a_cnd)
        A_rbl.append(a_rbl)
        cost_column.append(c_col/1000000)
        cost_internals.append(c_int/1000000)
        cost_condenser.append(c_cnd/1000000)
        cost_reboiler.append(c_rbl/1000000)
        Operating_cost.append(operating)

        if i == 3:
            df_luyben = pd.DataFrame({
                "Number": Number,
                "Costs M€": Total_costs,
                'Topvalue M€': topvalue,
                'Bottomvalue M€': botsvalue,
                'Diameter m': Diameter,
                'A cnd m2': A_cnd,
                'A rbl m2': A_rbl,
                'Cost column M€': cost_column,
                'Cost internals M€': cost_internals,
                'Cost condenser M€': cost_condenser,
                'Cost reboiler M€': cost_reboiler,
                'Operating cost M€': operating,
            })

#         TotalCosts[str('Column '+str(i))]['Costs']=-TAC
#         TotalCosts[str('Column '+str(i))]['Topvalue']=top_stream_revenue
#         TotalCosts[str('Column '+str(i))]['Bottomvalue'] = bottom_stream_revenue
#
#     pprint.pprint(TotalCosts,sort_dicts=False)
#     total_TAC = sum(d['Costs'] for d in TotalCosts.values() if d)
#     total_top = sum(d['Topvalue'] for d in TotalCosts.values() if d)
#     total_bottom = sum(d['Bottomvalue'] for d in TotalCosts.values() if d)
#
#     print(f"Total TAC: {total_TAC}")
#     print(f"Total top: {total_top}")
#     print(f"Total bottom: {total_bottom}")
#
#     yearly_revenue = total_TAC + total_top + total_bottom
#     print(f"Yearly revenue: {yearly_revenue}")
#
#
#
#

