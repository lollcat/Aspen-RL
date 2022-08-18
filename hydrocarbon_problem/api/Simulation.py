import os
import win32com.client as win32
import psutil
import pywintypes
import signal
import subprocess
import numpy as np
import time
from datetime import datetime
from collections import Counter


class Simulation():

    def __init__(self, VISIBILITY, SUPPRESS, max_iterations: int = 100,
                 flowsheet_path: str = "HydrocarbonMixture.bkp"):
        print("Start Aspen")
        # start = time.time()
        # self.running_pid = self.pid_com_v2(process_name="blob",instance="old")  # , visibility=VISIBILITY, suppress=SUPPRESS, path=flowsheet_path, max_iterations=max_iterations)
        # self.running_pid, self.pid_info = self.pid_com(process_name="AspenPlus", instance='old')
        self.AspenSimulation = win32.gencache.EnsureDispatch("Apwn.Document")
        # self.pid, self.pid_info = self.pid_com(process_name="AspenPlus", instance="new")
        os.chdir(r'C:\Users\s2399016\Documents\Aspen-RL_v2\Aspen-RL\hydrocarbon_problem\AspenSimulation')
        self.AspenSimulation.InitFromArchive2(os.path.abspath(flowsheet_path))
        self.AspenSimulation.Visible = VISIBILITY
        self.AspenSimulation.SuppressDialogs = SUPPRESS
        self.max_iterations = max_iterations
        self.BLK.Elements("B1").Elements("Input").Elements("MAXOL").Value = self.max_iterations
        # self.pid = self.pid_com_v2(process_name="blob", instance="new")  #, visibility=VISIBILITY, suppress=SUPPRESS, path=flowsheet_path, max_iterations=max_iterations)
        # print(f"PID duration = {time.time() - start}")
        self.duration = 0
        self.converged = False
        self.tries = 0
        self.pywin_error = False
        self.info = {}

    def pid_com(self, process_name, instance):
        notepad_stat = False                    # 2168540
        # while not notepad_stat:
        #     os.chdir('C:/Users/s2399016/Documents/Aspen-RL_v2/Aspen-RL/hydrocarbon_problem')
        #     with open('PID_check.txt', 'r') as file:
        #         pid_check = file.read().rstrip()
        #     if pid_check == "Y" and instance == 'old':  # Y to proceed with PID retrieval, N is wait for other process
        #         notepad_stat = True
        #         with open('PID_check.txt', 'w') as file:
        #             file.write("N")
        #             file.close()
        #         print("Retrieving PIDs")
        #         PIDs, procObjList = self.retrieve_pids(process_name=process_name)
        #     elif pid_check == "N" and instance == 'new':
        #         PIDs, procObjList = self.retrieve_pids(process_name=process_name)
        #         pid_info = self.set_aspen_pid(running_pid=self.running_pid, new_pids=PIDs, pid_info=procObjList)
        #         with open('PID_check.txt', 'w') as file:
        #             file.write("Y")
        #             file.close()
        #         notepad_stat = True
        #         print(f"PID is done, {pid_info}")
        #         blob = 0
        #     else:
        #         print('PID waiting')
        #         time.sleep(2)
        # if instance == 'old':
        #     return PIDs, procObjList
        # elif instance == 'new':
        #     return pid_info, blob

    def pid_com_v2(self, process_name, instance):  # , path, visibility, suppress, max_iterations):
        notepad_stat = False
        while not notepad_stat:
            os.chdir('C:/Users/s2399016/Documents/Aspen-RL_v2/Aspen-RL/hydrocarbon_problem')
            with open('PID_check.txt', 'r') as file:
                pid_check = file.read().rstrip()
            if pid_check == "Y" and instance == 'old':  # Y to proceed with PID retrieval, N is wait for other process
                notepad_stat = True
                with open('PID_check.txt', 'w') as file:
                    file.write("N")
                    file.close()
                print("Retrieving PIDs")
                PIDs = self.retrieve_pids(process_name=process_name)
            elif pid_check == "N" and instance == 'new':
                PIDs = self.retrieve_pids(process_name=process_name)
                aspen_id = self.set_aspen_pid(running_pid=self.running_pid, new_pids=PIDs)
                # self.AspenSimulation.InitFromArchive2(os.path.abspath(path=path))
                # self.AspenSimulation.Visible = visibility
                # self.AspenSimulation.SuppressDialogs = suppress
                # self.max_iterations = max_iterations
                # self.BLK.Elements("B1").Elements("Input").Elements("MAXOL").Value = self.max_iterations
                with open('PID_check.txt', 'w') as file:
                    file.write("Y")
                    file.close()
                notepad_stat = True
            else:
                print('PID waiting')
                time.sleep(2)
        if instance == 'old':
            return PIDs
        elif instance == 'new':
            return int(aspen_id)

    def retrieve_pids(self, process_name):
        a = [line.split() for line in subprocess.check_output("tasklist").splitlines()]
        PIDs = []
        a.pop(0)
        for i in a:
            if i[0] == b'AspenPlus.exe':
                k = i[1].decode('UTF-8')
                # print(k)
                PIDs.append(k)

        # procObjList = [procObj for procObj in psutil.process_iter() if process_name in procObj.name()]
        # PIDs = []
        # for i in range(len(procObjList)):
        #     prog_ = procObjList[i]
        #     prog_PID = prog_.pid
        #     PIDs.append(prog_PID)
        # print(f"Aspen PIDS: {PIDs}")
        return PIDs  #, procObjList

    def set_aspen_pid(self, running_pid, new_pids):
        running_and_new = running_pid + new_pids  # Combine arrays and use counter to extract unique value
        mp = Counter(running_and_new)
        for it in mp:
            if mp[it] == 1:
                aspen_pid = it
                print(f"Aspen PID: {aspen_pid}")
        # for i in range(len(pid_info)):
        #     p = pid_info[i]
        #     if aspen_pid == p.pid:
        #         current_process = p
        return aspen_pid #current_process

    @property
    def BLK(self):
        return self.AspenSimulation.Tree.Elements("Data").Elements("Blocks")

    def BLK_NumberOfStages(self, nstages):
        self.BLK.Elements("B1").Elements("Input").Elements("NSTAGE").Value = nstages

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
        self.STRM.Elements(Name).Elements("Input").Elements("FLOW").Elements("MIXED").Elements(
            Chemical).Value = Flowrate

    def STRM_Get_Outputs(self, Name, Chemical):
        STRM_COMP = self.STRM.Elements(Name).Elements("Output").Elements("MOLEFLOW").Elements("MIXED")
        COMP_1 = STRM_COMP.Elements(Chemical).Value
        return COMP_1  # kmol/s

    def STRM_Get_Temperature(self, Name):
        return self.STRM.Elements(Name).Elements("Output").Elements("TEMP_OUT").Elements("MIXED").Value

    def STRM_Get_Pressure(self, Name):
        return self.STRM.Elements(Name).Elements("Output").Elements("PRES_OUT").Elements("MIXED").Value

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

    def BLK_Get_ReboilerTemperature(self):
        return self.BLK.Elements("B1").Elements("Output").Elements("BOTTOM_TEMP").Value

    def BLK_Get_CondenserTemperature(self):
        return self.BLK.Elements("B1").Elements("Output").Elements("TOP_TEMP").Value

    def BLK_Get_Condenser_Duty(self):
        condenser_duty = self.BLK.Elements("B1").Elements("Output").Elements("COND_DUTY").Value
        # We overwrite negative values for the reboiler_duty as they don't make sense
        # (to be looked into further in the future).
        if condenser_duty >= 0:
            condenser_duty = 0
        return condenser_duty  # Watt

    def BLK_Get_Reboiler_Duty(self):
        reboiler_duty = self.BLK.Elements("B1").Elements("Output").Elements("REB_DUTY").Value
        # We overwrite negative values for the reboiler_duty as they don't make sense
        # (to be looked into further in the future).
        if reboiler_duty <= 0:
            reboiler_duty = 0
        return reboiler_duty  # Watt

    def BLK_Get_Column_Stage_Vapor_Flows_Short(self, N_stages, feed_stage):
        V = []
        if feed_stage == N_stages:
            V = [self.BLK.Elements("B1").Elements("Output").Elements("VAP_FLOW").Elements(2).Value * 1000,
                 self.BLK.Elements("B1").Elements("Output").Elements("VAP_FLOW").Elements(str(N_stages-1)).Value * 1000,
                 self.BLK.Elements("B1").Elements("Output").Elements("VAP_FLOW").Elements(str(N_stages - 1)).Value * 1000]
        elif feed_stage != N_stages:
            V = [self.BLK.Elements("B1").Elements("Output").Elements("VAP_FLOW").Elements(2).Value * 1000,
                 self.BLK.Elements("B1").Elements("Output").Elements("VAP_FLOW").Elements(str(feed_stage)).Value * 1000,
                 self.BLK.Elements("B1").Elements("Output").Elements("VAP_FLOW").Elements(str(N_stages - 1)).Value * 1000]
        return V  # mol/s

    def BLK_Get_Column_Stage_Molar_Weights_Short(self, N_stages, feed_stage):
        M = []
        if feed_stage == N_stages:
            M = [self.BLK.Elements("B1").Elements("Output").Elements("MW_GAS").Elements(2).Value,
                 self.BLK.Elements("B1").Elements("Output").Elements("MW_GAS").Elements(str(N_stages-1)).Value,
                 self.BLK.Elements("B1").Elements("Output").Elements("MW_GAS").Elements(str(N_stages-1)).Value]

        elif feed_stage != N_stages:
            M = [self.BLK.Elements("B1").Elements("Output").Elements("MW_GAS").Elements(2).Value,
                 self.BLK.Elements("B1").Elements("Output").Elements("MW_GAS").Elements(str(feed_stage)).Value,
                 self.BLK.Elements("B1").Elements("Output").Elements("MW_GAS").Elements(str(N_stages - 1)).Value]
        return M  # g/mol

    def BLK_Get_Column_Stage_Temperatures_Short(self, Nstages, feed_stage):
        T = []
        if feed_stage == Nstages:
            raise Exception("Agent should never pick nstages as feed location")
        elif feed_stage != Nstages:
            T = [self.BLK.Elements("B1").Elements("Output").Elements("B_TEMP").Elements(str(2)).Value,
                 self.BLK.Elements("B1").Elements("Output").Elements("B_TEMP").Elements(str(feed_stage)).Value,
                 self.BLK.Elements("B1").Elements("Output").Elements("B_TEMP").Elements(str(Nstages - 1)).Value]
        return T  # degree C

    def Run(self):
        self.tries = 0
        while self.tries != 2:
            self.AspenSimulation.Reinit()
            start = time.time()
            try:
                self.AspenSimulation.Engine.Run2()
            # """"Implement com_error exception, when encountered try self.AspenSimulation.Reinit()"""
            except pywintypes.com_error:  # pywintypes.com_error as error:
                print("No contact, pywintypes.com_error")
                self.pywin_error = True
                self.converged = -1
                break
            self.duration = time.time() - start
            self.converged = self.AspenSimulation.Tree.Elements("Data").Elements("Blocks").Elements(
                "B1").Elements("Output").Elements("BLKSTAT").Value
            if self.converged == 0 or self.converged == 2:
                break
            else:
                time.sleep(1)
                self.tries += 1
            if self.tries == 1 and self.converged == 1:
                print("Aspen convergence error")
            self.info['Convergence'] = self.converged


    def restart(self, visibility, suppress, max_iterations):
        x = win32.Dispatch("Apwn.Document")
        os.chdir(r'C:\Users\s2399016\Documents\Aspen-RL_v2\Aspen-RL\hydrocarbon_problem\AspenSimulation')
        # print(os.getcwd())
        x.InitFromArchive2(os.path.abspath("HydrocarbonMixture.bkp"))
        x.Visible = visibility
        x.SuppressDialogs = suppress
        self.AspenSimulation.Close()
        # terminate = False
        # start = time.time()
        # while not terminate:
        #     os.chdir('C:/Users/s2399016/Documents/Aspen-RL_v2/Aspen-RL/hydrocarbon_problem')
        #     with open('PID_check.txt', 'r') as file:
        #         pid_check = file.read().rstrip()
        #     if pid_check == "Y":
        #         with open('PID_check.txt', 'w') as file:
        #             file.write("N")
        #             file.close()
        #         print(f"Terminate Aspen: {self.pid}")
        #         os.kill(self.pid, signal.SIGINT)
        #         # psutil.Process.terminate(self.pid)
        #         with open('PID_check.txt', 'w') as file:
        #             file.write("Y")
        #             file.close()
        #         terminate = True
        #     else:
        #         print("Wait to terminate Aspen")
        #         time.sleep(2)
        # self.running_pid = self.pid_com_v2(process_name="AspenPlus", instance='old')
        del self.AspenSimulation
        self.AspenSimulation = win32.gencache.EnsureDispatch("Apwn.Document")
        # self.pid = self.pid_com_v2(process_name="AspenPlus", instance="new")
        # print(f"PID duration = {time.time() - start}")
        self.pywin_error = False
        # os.system("taskkill /f /im AspenPlus.exe")
        # self.AspenSimulation = win32.gencache.EnsureDispatch("Apwn.Document")
        os.chdir(r'C:\Users\s2399016\Documents\Aspen-RL_v2\Aspen-RL\hydrocarbon_problem\AspenSimulation')
        self.AspenSimulation.InitFromArchive2(os.path.abspath("HydrocarbonMixture.bkp"))
        self.AspenSimulation.Visible = visibility
        self.AspenSimulation.SuppressDialogs = suppress
        self.max_iterations = max_iterations
        self.BLK.Elements("B1").Elements("Input").Elements("MAXOL").Value = self.max_iterations
        print("Aspen has been restarted")

    def CAL_Column_Diameter_short(self, pressure, n_stages, vapor_flows, stage_mw, stage_temp, feed_stage):
        P = pressure
        f = float(1.6)
        R = float(8.314)
        Effective_Diameter = []

        for i in range(0, len(vapor_flows)):
            Effective_Diameter += [np.sqrt((4 * vapor_flows[i]) / (3.1416 * f) * np.sqrt(
                R * (stage_temp[i] + 273.15) * stage_mw[i] / 1000 / (P * 1e5)))]

        Diameter = 1.1 * max(Effective_Diameter)
        return Diameter

    def CAL_Column_Height(self, n_stages):
        HETP = 0.5  # HETP constant [m]
        H_0 = 0.4  # Clearance [m]
        return n_stages * HETP + H_0

    def CAL_LMTD(self, tops_temperature):

        if tops_temperature > 30:
            T_cool_in = 20
        elif tops_temperature > -20 and tops_temperature <=30:
            T_cool_in = -20
        elif tops_temperature > -45 and tops_temperature <=-20:
            T_cool_in = -45

        T_cool_out = T_cool_in + 10
        # T_cool_in = 30  # Supply temperature of cooling water [oC]
        # T_cool_out = 40  # Return temperature of cooling water [oC]
        if tops_temperature > T_cool_out:
            delta_Tm_cnd = (((tops_temperature - T_cool_in) * (tops_temperature - T_cool_out) * (
                (tops_temperature - T_cool_in) + (tops_temperature - T_cool_out)) / 2) ** (1 / 3))
        else:
            delta_Tm_cnd = 0
        return delta_Tm_cnd

    def CAL_HT_Condenser_Area(self, condenser_duty, tops_temperature):
        K_cnd = 500  # Heat transfer coefficient [W/m2 K]
        delta_Tm_cnd = tops_temperature - 10
        A_cnd = -condenser_duty / (K_cnd * abs(delta_Tm_cnd))
        # self.CAL_LMTD(tops_temperature)
        # Tm_type = isinstance(delta_Tm_cnd, float)
        # if delta_Tm_cnd > 0:
        #     A_cnd = -condenser_duty / (K_cnd * delta_Tm_cnd)
        # else:
        #     A_cnd = 0
        return A_cnd

    def CAL_HT_Reboiler_Area(self, reboiler_temperature, reboiler_duty):
        K_rbl = 800  # Heat transfer coefficient [W/m2*K] (800, fixed)
        T_steam = 201  # Temperature of 16 bar steam [°C] (201, fixed)
        delta_tm_rbl = T_steam - reboiler_temperature
        A_rbl = reboiler_duty / (K_rbl * delta_tm_rbl)
        # if T_steam < reboiler_temperature:
        #     A_rbl = 0
        # else:
        #     delta_tm_rbl = T_steam - reboiler_temperature
        #     A_rbl = reboiler_duty / (K_rbl * delta_tm_rbl)
        return A_rbl

    def CAL_InvestmentCost(self,  n_stages, condenser_duty, reboiler_temperature, reboiler_duty,
                           tops_temperature, pressure, vapor_flows, stage_mw, stage_temp,
                           feed_stage):  # diameter
        # Define in Column Specifications
        L = self.CAL_Column_Height(n_stages)  # Column length [m]
        D = self.CAL_Column_Diameter_short(pressure, n_stages, vapor_flows, stage_mw, stage_temp, feed_stage)
        A_cnd = self.CAL_HT_Condenser_Area(condenser_duty, tops_temperature)  # Heat transfer area of condenser [m2]
        A_rbl = self.CAL_HT_Reboiler_Area(reboiler_temperature, reboiler_duty)  # Heat transfer area of reboiler [m2]
        # Predefined values.
        F_m = 1  # Correction factor for column shell material (1.0, fixed)
        F_p = 1  # Correction factor for column pressure (1.0, fixed)
        F_int_m = 0  # Correction factor for internals material [-] (0.0, fixed)
        F_int_t = 0  # Correction factor for tray type [-] (0.0, fixed)
        F_int_s = 1.4  # Correction factor for tray spacing [-] (1.4, fixed)
        F_htx_d = 0.8  # Correction factor for design type: fixed-tube sheet [-] (0.8, fixed)
        F_htx_p = 0  # Correction factor for pressure [-] (0.0, fixed)
        F_htx_m = 1  # Correction factor for material [-] (1.0, fixed)
        M_S = 1638.2  # Marshall & Swift equipment index 2018 (1638.2, fixed)
        F_c = F_m + F_p
        F_int_c = F_int_s + F_int_t + F_int_m
        F_cnd_c = (F_htx_d + F_htx_p) * F_htx_m
        F_rbl_c = (F_htx_d + F_htx_p) * F_htx_m
        if self.converged == 1:
            C_col = 5  # M€
            C_int = 5  # M€
        else:
            C_col = 0.9 * (M_S / 280) * 937.64 * D ** 1.066 * L ** 0.802 * F_c / 1000000  # M€
            C_int = 0.9 * (M_S / 280) * 97.24 * D ** 1.55 * L * F_int_c / 1000000  # M€
        if A_cnd == 0:
            C_cnd = 5  # €
        else:
            C_cnd = 0.9 * (M_S / 280) * 474.67 * A_cnd ** 0.65 * F_cnd_c / 1000000  # M€
        if A_rbl == 0:
            C_rbl = 5  # M€
        else:
            C_rbl = 0.9 * (M_S / 280) * 474.67 * A_rbl ** 0.65 * F_rbl_c / 1000000  # M€
        C_eqp = (C_col + C_int + C_cnd + C_rbl)  # M€ (x1000000=€)
        F_cap = 0.2  # Capital charge factor (0.2, fixed)
        F_L = 5  # Lang factor (5, fixed)
        C_inv = F_L * C_eqp
        InvestmentCost = F_cap * C_inv  # M€

        invest = isinstance(InvestmentCost, float)
        if invest == False:
            print(f"Length column: {L}")
            print(f"Diameter column: {D}")
            print(f"Condenser area: {A_cnd}")
            print(f"Reboiler area: {A_rbl}")
            print(f"Cost column: {C_col}")
            print(f"Cost interals: {C_int}")
            print(f"Cost condenser: {C_cnd}")
            print(f"Cost reboiler: {C_rbl}")
            print(f"Cost equipment: {C_eqp}")
            print(f"C_inv: {C_inv}")
            print(f"InvestmentCost: {InvestmentCost}")

        column_info = [L, D, A_cnd, A_rbl, C_col, C_int, C_cnd, C_rbl]

        return InvestmentCost, column_info

    def CAL_OperatingCost(self, reboiler_duty, condenser_duty, tops_temperature):
        operational_time = 8400
        M = 18  # Molar weight of water [g/mol] (18, fixed)
        c_steam = 18  # Steam price [€/t] (18, fixed)
        c_cw = 0.006  # Cooling water price [€/t] (0.006, fixed)
        delta_hv = 34794  # Molar heat of condensation of 16 bar steam [J/mol] (34794, fixed)
        c_p = 4.2  # Heat capacity of water [kJ/(kg*K)] (4.2, fixed)
        T_cool = tops_temperature - 10

        # T_cool_out = T_cool_in + 10
        # T_cool_in = 30  # Supply cooling water temperature [°C] (30, fixed)
        # T_cool_out = 40  # Return cooling water temperature [°C] (40, fixed)
        C_op_rbl = (reboiler_duty / 1000000 * M * c_steam * 3600 / delta_hv) * operational_time / 1000000  # M€/year
        energy_cnd = operational_time * -condenser_duty / 1000  # kWh/year
        energy_cost = (6*10**-6) * T_cool - 0.0006 * T_cool + 0.0163  # €/kWh
        C_op_cnd = energy_cnd * energy_cost / 1000000  # M€/year
        # C_op_cnd = -condenser_duty / 1000000 * c_cw * 3600 / (c_p * (T_cool_out - T_cool_in))  # €/h
        self.info["Reboil util costs"] = C_op_rbl
        self.info["Condenser util costs"] = C_op_cnd
        # C_op = C_op_rbl + C_op_cnd  # M€/year

        return C_op_cnd, C_op_rbl  # M€/year

    def CAL_Annual_OperatingCost(self, reboiler_duty, condenser_duty, tops_temperature):
        # t_a = 8400
        OperatingCost_cnd, OperatingCost_rbl = self.CAL_OperatingCost(reboiler_duty, condenser_duty, tops_temperature)  # M€/y
        # self.info['OperatingCost'] = OperatingCost
        return OperatingCost_cnd, OperatingCost_rbl

    def CAL_stream_value(self, stream_specification,
                         product_specification):  # , component_specifications, molar_flows, stream_component_specifications):
        """Calculates the value (per year) of a stream."""

        up_time = 8400 * 3600  # seconds per year, assuming 8400 hours of uptime
        is_purity, component_purities = self.CAL_purity_check(stream_specification, product_specification)

        component_specifications = {
            'ethane': {'index': 0, 'molar weight': 30.07, 'price': 125.0 * 0.91, 'mass flow': 0, 'stream value': 0},
            'propane': {'index': 1, 'molar weight': 44.1, 'price': 204.0 * 0.91, 'mass flow': 0, 'stream value': 0},
            'isobutane': {'index': 2, 'molar weight': 58.12, 'price': 272.0 * 0.91, 'mass flow': 0, 'stream value': 0},
            'n_butane': {'index': 3, 'molar weight': 58.12, 'price': 249.0 * 0.91, 'mass flow': 0, 'stream value': 0},
            'isopentane': {'index': 4, 'molar weight': 72.15, 'price': 545.0 * 0.91, 'mass flow': 0, 'stream value': 0},
            'n_pentane': {'index': 5, 'molar weight': 72.15, 'price': 545.0 * 0.91, 'mass flow': 0, 'stream value': 0}
        }  # molar weight = g/mol, price = $/ton *0.91 (exchange rate @ 24-03-2022), mass flow = ton/year, stream value = T€/year

        for entry in component_specifications:
            component_specifications[entry]['mass flow'] = stream_specification.molar_flows[
                                                               component_specifications[entry]['index']] * \
                                                           component_specifications[entry][
                                                               'molar weight'] / 1000 * up_time  # ton/year
        total_flow = sum(d['mass flow'] for d in component_specifications.values() if d)

        for entry in component_specifications:
            if sum(is_purity) > 0:
                component_specifications[entry]['stream value'] = is_purity[component_specifications[entry]['index']] * \
                                                                  component_specifications[entry]['price'] * \
                                                                  total_flow/1000000  # M€/year
            elif sum(is_purity) == 0:
                component_specifications[entry]['stream value'] = 0.0

        total_stream_value = sum(d['stream value'] for d in component_specifications.values() if d)

        return total_stream_value, component_purities

    def CAL_purity_check(self, stream_specification, product_specification):

        molar_flows = stream_specification.molar_flows
        is_purity = np.zeros(len(molar_flows), dtype=int)
        component_purities = np.zeros(len(molar_flows))
        total_flow = sum(molar_flows)

        # if total_flow > 0.001:
        for entry in range(0, len(molar_flows)):
            component_purities[entry] = molar_flows[entry] / total_flow
            if component_purities[entry] >= product_specification:
                is_purity[entry] = 1
            elif component_purities[entry] < product_specification:
                is_purity[entry] = 0

        return is_purity, component_purities