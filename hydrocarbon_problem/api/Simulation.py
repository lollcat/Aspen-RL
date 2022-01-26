import os
import win32com.client as win32


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
        self.STRM.Elements(Name).Elements("Input").Elements("FLOW").Elements("MIXED").Elements(
            Chemical).Value = Flowrate

    def STRM_Get_Outputs(self, Name, Chemicals):
        STRM_COMP = self.STRM.Elements(Name).Elements("Output").Elements("MOLEFLOW").Elements("MIXED")
        COMP_1 = STRM_COMP.Elements(Chemicals[0]).Value
        COMP_2 = STRM_COMP.Elements(Chemicals[1]).Value
        COMP_3 = STRM_COMP.Elements(Chemicals[2]).Value
        COMP_4 = STRM_COMP.Elements(Chemicals[3]).Value
        COMP_5 = STRM_COMP.Elements(Chemicals[4]).Value
        COMP_6 = STRM_COMP.Elements(Chemicals[5]).Value
        return [COMP_1, COMP_2, COMP_3, COMP_4, COMP_5, COMP_6]

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
        for i in range(1, N_stages + 1):
            M += [self.BLK.Elements("B1").Elements("Output").Elements("MW_GAS").Elements(str(i)).Value]
        return M

    def BLK_Get_Column_Stage_Temperatures(self, N_stages):
        T = []
        for i in range(1, N_stages + 1):
            T += [self.BLK.Elements("B1").Elements("Output").Elements("B_TEMP").Elements(str(i)).Value]
        return T

    def BLK_Get_Column_Stage_Vapor_Flows(self, N_stages):
        V = []
        for i in range(1, N_stages + 1):
            V += [self.BLK.Elements("B1").Elements("Output").Elements("VAP_FLOW").Elements(str(i)).Value]
        return V

    def BLK_Get_Column_Diameter(self, N_stages):
        P = int(1)
        f = float(1.6)
        R = float(8.314)
        Effective_Diameter = []
        V = self.BLK_Get_Column_Stage_Vapor_Flows(N_stages)
        M = self.BLK_Get_Column_Stage_Molar_Weights(N_stages)
        T = self.BLK_Get_Column_Stage_Temperatures(N_stages)
        for i in range(0, N_stages - 1):
            Effective_Diameter += [
                np.sqrt((4 * V[i]) / (3.1416 * f) * np.sqrt(R * (T[i] + 273.15) * M[i] * 1000 / (P * 1e5)))]

        Diameter = 1.1 * max(Effective_Diameter)
        return Diameter

    def Run(self):
        self.AspenSimulation.Engine.Run2()