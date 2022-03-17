import os 
import win32com.client as win32
import numpy as np


class Simulation():
    AspenSimulation = win32.gencache.EnsureDispatch("Apwn.Document")

    def __init__(self, PATH, VISIBILITY):
        self.AspenSimulation.InitFromArchive2(os.path.abspath(PATH))
        self.AspenSimulation.Visible = VISIBILITY


    @property
    def BLK(self):
        return self.AspenSimulation.Tree.Elements("Data").Elements("Blocks")


    def BLK_NumberOfStages(self, NStages):
        self.BLK.Elements("B1").Elements("Input").Elements("NSTAGE").Value = NStages


    def BLK_FeedLocation(self, Feed_Location, Feed_Name):
        self.BLK.Elements("B1").Elements("Input").Elements("FEED_STAGE").Elements(Feed_Name).Value = Feed_Location


    def BLK_Pressure(self, Pressure):
        self.BLK.Elements("B1").Elements("Input").Elements("PRES1").Value = Pressure


    def BLK_RefluxRatio(self, RfxR):
        self.BLK.Elements("B1").Elements("Input").Elements("BASIS_RR").Value = RfxR


    def BLK_ReboilerRatio(self, RblR):
        self.BLK.Elements("B1").Elements("Input").Elements("BASIS_BR").Value = RblR


    @property
    def STRM(self):
        return self.AspenSimulation.Tree.Elements("Data").Elements("Streams")


    def STRM_Temperature(self, Name, Temp):
        self.STRM.Elements(Name).Elements("Input").Elements("TEMP").Elements("MIXED").Value = Temp


    def STRM_Pressure(self, Name, Pressure):
        self.STRM.Elements(Name).Elements("Input").Elements("PRES").Elements("MIXED").Value = Pressure


    def STRM_Flowrate(self, Name, Chemical, Flowrate):
        self.STRM.Elements(Name).Elements("Input").Elements("FLOW").Elements("MIXED").Elements(Chemical).Value = Flowrate


    def STRM_Get_Outputs(self, Name, Chemical):
        STRM_COMP = self.STRM.Elements(Name).Elements("Output").Elements("MOLEFLOW").Elements("MIXED")
        COMP_1 = STRM_COMP.Elements(Chemical).Value

        return COMP_1


    def STRM_Get_Temperature(self, Name):
        return self.STRM.Elements(Name).Elements("Input").Elements("TEMP").Elements("MIXED").Value


    def STRM_Get_Pressure(self, Name):
        return self.STRM.Elements(Name).Elements("Input").Elements("PRES").Elements("MIXED").Value


    def BLK_Get_NStages(self):
        return self.BLK.Elements("B1").Elements("Input").Elements("NSTAGE").Value


    def BLK_Get_FeedLocation(self, Name):
        return self.BLK.Elements("B1").Elements("Input").Elements("FEED_STAGE").Elements(Name).Value


    def BLK_Get_Pressure(self):
        return self.BLK.Elements("B1").Elements("Input").Elements("PRES1").Value


    def BLK_Get_RefluxRatio(self):
        return self.BLK.Elements("B1").Elements("Input").Elements("BASIS_RR").Value


    def BLK_Get_ReboilerRatio(self):
        return self.BLK.Elements("B1").Elements("Input").Elements("BASIS_BR").Value


    def BLK_Get_Condenser_Duty(self):
        return self.BLK.Elements("B1").Elements("Output").Elements("COND_DUTY").Value


    def BLK_Get_Reboiler_Duty(self):
        return self.BLK.Elements("B1").Elements("Output").Elements("REB_DUTY").Value

    def BLK_Get_Column_Diameter(self):
        return self.BLK.Elements("B1").Elements("Input").Elements("CA_DIAM").Value

    def BLK_Get_Column_Stage_Molar_Weights(self, N_stages):
        M = []
        for i in range(1,N_stages+1):
            M += [self.BLK.Elements("B1").Elements("Output").Elements("MW_GAS").Elements(str(i)).Value]
        return M

    def BLK_Get_Column_Stage_Temperatures(self, N_stages):
        T = []
        for i in range(1, N_stages+1):
            T += [self.BLK.Elements("B1").Elements("Output").Elements("B_TEMP").Elements(str(i)).Value]
        return T

    def BLK_Get_Column_Stage_Vapor_Flows(self, N_stages):
        V = []
        for i in range(1, N_stages+1):
            V += [self.BLK.Elements("B1").Elements("Output").Elements("VAP_FLOW").Elements(str(i)).Value]
        return V



    def Run(self):
        self.AspenSimulation.Engine.Run2()

    def CAL_Column_Diameter(self, n_stages, vapor_flows, stage_mw, stage_temp):
        P = int(1)
        f = float(1.6)
        R = float(8.314)
        Effective_Diameter = []

        for i in range(0,n_stages-1):
            Effective_Diameter += [np.sqrt((4*vapor_flows[i])/(3.1416*f)*np.sqrt(R*(stage_temp[i]+273.15)*stage_mw[i]*1000/(P*1e5)))]

        Diameter = 1.1*max(Effective_Diameter)
        return Diameter

    def CAL_Column_Height(self, n_stages):
        HEPT = 0.5                  # HEPT constant [m]
        H_0 = 0.4                   # Clearance [m]
        return n_stages*HEPT + H_0

    def CAL_LMTD(self, tops_temperature):
        T_cool_in = 30              # Supply temperaturer of cooling water [oC]
        T_cool_out = 40             # Return temperature of cooling water [oC]
        delta_Tm_cnd = (((tops_temperature - T_cool_in) * (tops_temperature - T_cool_out) * (
                (tops_temperature - T_cool_in) + (tops_temperature - T_cool_out)) / 2) ** (1 / 3))
        return delta_Tm_cnd.real

    def CAL_HT_Condenser_Area(self, condenser_duty, tops_temperature):
        K_cnd = 500                 # Heat transfer coefficient [W/m2 K] 
        delta_Tm_cnd = self.CAL_LMTD(tops_temperature)
        A_cnd = -condenser_duty / (K_cnd * delta_Tm_cnd)
        return A_cnd

    def CAL_HT_Reboiler_Area(self, reboiler_temperature, reboiler_duty):
        K_rbl = 800                  #Heat transfer coefficient [W/m2*K] (800, fixed) 
        T_steam = 201                #Temperature of 16 bar steam [°C] (201, fixed) 
        delta_tm_rbl = T_steam - reboiler_temperature
        A_rbl = reboiler_duty / (K_rbl * delta_tm_rbl)
        return A_rbl

    def CAL_InvestmentCost(self, n_stages, condenser_duty, reboiler_temperature, reboiler_duty, tops_temperature, vapor_flows, stage_mw, stage_temp):
        # Define in Column Specifications
        L = self.CAL_Column_Height(n_stages)                           # Column length [m]
        D = self.CAL_Column_Diameter(n_stages, vapor_flows, stage_mw, stage_temp)                                                   # Column diameter [m]
        A_cnd =  self.CAL_HT_Condenser_Area(condenser_duty, tops_temperature)            # Heat transfer area of condenser [m2]
        A_rbl = self.CAL_HT_Reboiler_Area(reboiler_temperature, reboiler_duty)           # Heat transfer area of reboiler [m2]
        # Predefined values.
        F_m = 1                                                        # Correction factor for column shell material (1.0, fixed)
        F_p = 1                                                        # Correction factor for column pressure (1.0, fixed) 
        F_int_m = 0                                                    # Correction factor for internals material [-] (0.0, fixed) 
        F_int_t = 0                                                    # Correction factor for tray type [-] (0.0, fixed) 
        F_int_s = 1.4                                                  # Correction factor for tray spacing [-] (1.4, fixed) 
        F_htx_d = 0.8                                                  # Correction factor for design type: fixed-tube sheet [-] (0.8, fixed) 
        F_htx_p = 0                                                    # Correction factor for pressure [-] (0.0, fixed) 
        F_htx_m = 1                                                    # Correction factor for material [-] (1.0, fixed) 
        M_S = 1638.2                                                   # Marshall & Swift equipment index 2018 (1638.2, fixed) 
        F_c = F_m + F_p
        F_int_c = F_int_s + F_int_t + F_int_m
        F_cnd_c = (F_htx_d + F_htx_p) * F_htx_m
        F_rbl_c = (F_htx_d + F_htx_p) * F_htx_m
        C_col = 0.9 * (M_S / 280) * 937.64 * D ** 1.066 * L ** 0.802 * F_c
        C_int = 0.9 * (M_S / 280) * 97.24 * D ** 1.55 * L * F_int_c
        C_cnd = 0.9 * (M_S / 280) * 474.67 * A_cnd ** 0.65 * F_cnd_c
        C_rbl = 0.9 * (M_S / 280) * 474.67 * A_rbl ** 0.65 * F_rbl_c
        C_eqp = (C_col + C_int + C_cnd + C_rbl) / 1000
        F_cap = 0.2                                                     #Capital charge factor (0.2, fixed) 
        F_L = 5                                                         #Lang factor (5, fixed) 
        C_inv = F_L * C_eqp
        InvestmentCost = F_cap * C_inv
        return InvestmentCost

    def CAL_OperatingCost(self, reboiler_duty, condenser_duty):
        M = 18                                                          # Molar weight of water [g/mol] (18, fixed) 
        c_steam = 18                                                    # Steam price [€/t] (18, fixed) 
        c_cw = 0.006                                                    # Cooling water price [€/t] (0.006, fixed) 
        delta_hv = 34794                                                # Molar heat of condensation of 16 bar steam [J/mol] (34794, fixed) 
        c_p = 4.2                                                       # Heat capacity of water [kJ/(kg*K)] (4.2, fixed) 
        T_cool_in = 30                                                  # Supply cooling water temperature [°C] (30, fixed) 
        T_cool_out = 40                                                 # Return cooling water temperature [°C] (40, fixed) 
        C_op_rbl = reboiler_duty / 1000000 * M * c_steam * 3600 / delta_hv
        C_op_cnd = condenser_duty / 1000000 * c_cw * 3600 / (c_p * (T_cool_out - T_cool_in))
        C_op = C_op_rbl + C_op_cnd
        return C_op

    def CAL_Annual_OperatingCost(self, reboiler_duty, condenser_duty):
        t_a = 8000
        OperatingCost = self.CAL_OperatingCost(reboiler_duty, condenser_duty)*t_a/1000
        return OperatingCost



    def CAL_stream_value(self, component_specifications, molar_flows, stream_component_specifications):
        """Calculates the value (per year) of a stream."""
        up_time = 8400 * 3600                                       # seconds per year, assuming 8400 hours of uptime
        total_mol_flows = sum(molar_flows)                    # total molar flow in top stream
        # bot_mol_flows = bots_specifications.molar_flows             # molar flows in bottom stream per component
        # total_bot_mol_flows = sum(bot_mol_flows)                    # total molar flow in bottom stream

        """component_specifications = {
            'ethane':     {'index': 0, 'molar weight': 30.07, 'price': 125.0, 'required purity': 0.95},
            'propane':    {'index': 1, 'molar weight': 44.1,  'price': 204.0, 'required purity': 0.95},
            'isobutane':  {'index': 2, 'molar weight': 58.12, 'price': 272.0, 'required purity': 0.95},
            'n_butane':   {'index': 3, 'molar weight': 58.12, 'price': 249.0, 'required purity': 0.95},
            'isopentane': {'index': 4, 'molar weight': 72.15, 'price': 545.0, 'required purity': 0.95},
            'n_pentane':  {'index': 5, 'molar weight': 72.15, 'price': 545.0, 'required purity': 0.95}
        }"""

        top_component_specifications = {
            'ethane':     {'index': 0, 'mass flow': [], 'purity': [], 'stream value': []},
            'propane':    {'index': 1, 'mass flow': [], 'purity': [], 'stream value': []},
            'isobutane':  {'index': 2, 'mass flow': [], 'purity': [], 'stream value': []},
            'n_butane':   {'index': 3, 'mass flow': [], 'purity': [], 'stream value': []},
            'isopentane': {'index': 4, 'mass flow': [], 'purity': [], 'stream value': []},
            'n_pentane':  {'index': 5, 'mass flow': [], 'purity': [], 'stream value': []}
        }                            # a dictionary which stores mass flows, molar purities and stream values

        bottom_component_specifications = {
            'ethane':     {'index': 0, 'mass flow': [], 'purity': [], 'stream value': []},
            'propane':    {'index': 1, 'mass flow': [], 'purity': [], 'stream value': []},
            'isobutane':  {'index': 2, 'mass flow': [], 'purity': [], 'stream value': []},
            'n_butane':   {'index': 3, 'mass flow': [], 'purity': [], 'stream value': []},
            'isopentane': {'index': 4, 'mass flow': [], 'purity': [], 'stream value': []},
            'n_pentane':  {'index': 5, 'mass flow': [], 'purity': [], 'stream value': []},
        }

        for entry in stream_component_specifications:
            # calculate mass flows from molar flows and store in component_specifications
            stream_component_specifications[entry]['mass flow'] = molar_flows[stream_component_specifications[entry]['index']] * \
                                                               component_specifications[entry]['molar weight'] / 1000 * up_time
            stream_component_specifications[entry]['purity'] = molar_flows[stream_component_specifications[entry]['index']] / \
                                                            total_mol_flows         # calculate molar purities in top flow

            if stream_component_specifications[entry]['purity'] >= component_specifications[entry]['required purity']: # check whether requirement is met in the top stream
                stream_component_specifications[entry]['stream value'] = stream_component_specifications[entry]['mass flow'] * \
                                                                      component_specifications[entry]['price'] # if requirement is met, stream value is calculated
            else:
                stream_component_specifications[entry]['stream value'] = 0         # if requirement isn't met, stream value is 0

        total_stream_value = sum(d['top stream value'] for d in top_component_specifications.values() if d)

        """for entry in component_specifications:
            component_specifications[entry]['bot mass flow'] = bot_mol_flows[component_specifications[entry]['index']] * component_specifications[entry]['molar weight'] / 1000
            component_specifications[entry]['bot purity'] = bot_mol_flows[component_specifications[entry]['index']] / total_bot_mol_flows

            if component_specifications[entry]['bot purity'] >= component_specifications[entry]['required purity']:
                component_specifications[entry]['bot stream value'] = component_specifications[entry]['bot mass flow'] * component_specifications[entry]['price']
            else:
                component_specifications[entry]['bot stream value'] = 0

        total_bot_stream_value = sum(d['bot stream value'] for d in component_specifications.values() if d)
        """
        return total_stream_value

    def CAL_purity_comparison(self, component_specifications, molar_flows, stream_component_specifications):
        purity_check = []
        for entry in stream_component_specifications:
            stream_component_specifications[entry]['purity'] = molar_flows[stream_component_specifications[entry]['index']] / \
                                                               sum(molar_flows)
            if stream_component_specifications[entry]['purity'] >= component_specifications[entry]['required purity']:
                purity_check[entry] = 1
            else:
                purity_check[entry] = 0

        if sum(purity_check) > 0:
            meet_purity = True
        else:
            meet_purity = False

        return meet_purity


