#####################################################################
#ESNプロジェクト，Chialvoニューロンマップのグラフ描写用
#制作者：田中勝規
#作成日：2024/09/24
#####################################################################

#====================================================================
#＜このプログラムのTodoメモ＞
#・
#・
#・
#====================================================================

#====================================================================
#外部ライブラリ
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#内部プログラム
import Plot_Graph


#====================================================================
class Output:
    """
    評価クラス
    全ての評価はこれを継承
    親オブジェクトは今のところ無し．
    実験パラメータをparamで受け，複製したparamに結果を入れて返す．
    """
    def __init__(self, param: dict, parent: any):
        self.Param = param

        self.Parent = parent

    def __call__(self) -> dict: pass

class Output_TimeLine(Output):
    """
    時系列描写
    """

    #コンストラクタ
    def __init__(self, param: dict, parent: any = None):
        super().__init__(param, parent)

        #どれをプロットするか
        self.Plot_X = self.Param["Project_Plot_TimeLine_X"]
        self.Plot_Y = self.Param["Project_Plot_TimeLine_Y"]
        self.Plot_Phi = self.Param["Project_Plot_TimeLine_Phi"]

        #空走時間
        self.Length_Burnin = self.Param["Length_Burnin"]
        #評価時間
        self.Length_Eva = self.Param["Length_Eva"]
        #プロット時間
        self.Length_Plot = self.Param["Length_Plot"]
        #総合時間
        self.Length_Total = self.Length_Burnin + self.Length_Eva + self.Length_Plot

        #モデル
        self.T_Model = self.Param["Model"]
        param = self.Param.copy()
        self.Model = self.T_Model(param, self)
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def __call__(self):

        print("--- Chialvo Neuron Map RunStart---\n")
        self.X, self.Y, self.Phi, self.Signal = self.Model()

        def TimeDate():
            Time_Delta = datetime.timedelta(hours=9)
            JST = datetime.timezone(Time_Delta ,'JST')
            Now = datetime.datetime.now(JST)
            Date = Now.strftime('%Y_%m_%d_%H_%M_%S')
            return Date
        
        Plot_Date = TimeDate()

        if self.Plot_X:
            print("\n--- Display of TimeLine Chialvo Neuron Map of X---")
            
            #プロット要素
            param = self.Param.copy()
            param.update({
                "PlotData" : self.X,
                "PlotData_Label" : "Chialvo : X",
                "PlotSignal" : self.Signal,
                "PlotSignal_Label" : "Input Signal",
                "PlotTitle" : "Chialvo Map : Timeline of x",
                "PlotXLabel" : "Time Step",
                "PlotYLabel" : "x",
                "PlotPath_Project" : "./Graphical/Results/Project_Chialvo_TimeLine",
                "PlotPath_Date" : f"/TimeLine_{Plot_Date}",
                "PlotName" : f"/{TimeDate()}_TimeLine_X.png"
            })
            Plot_Graph.Plot_TimeLine(param)()

        if self.Plot_Y:
            print("\n--- Display of TimeLine Chialvo Neuron Map of Y---")
            #プロット要素
            param = self.Param.copy()
            param.update({
                "PlotData" : self.Y,
                "PlotData_Label" : "Chialvo : Y",
                "PlotSignal" : self.Signal,
                "PlotSignal_Label" : "Input Signal",
                "PlotTitle" : "Chialvo Map : Timeline of y",
                "PlotXLabel" : "Time Step",
                "PlotYLabel" : "y",
                "PlotPath_Project" : "./Graphical/Results/Project_Chialvo_TimeLine",
                "PlotPath_Date" : f"/TimeLine_{Plot_Date}",
                "PlotName" : f"/{TimeDate()}_TimeLine_Y.png"
            })
            Plot_Graph.Plot_TimeLine(param)()

        if self.Plot_Phi:
            print("\n--- Display of TimeLine Chialvo Neuron Map of phi---")
            
            #プロット要素
            param = self.Param.copy()
            param.update({
                "PlotData" : self.Phi,
                "PlotData_Label" : "Chialvo : Phi",
                "PlotSignal" : self.Signal,
                "PlotSignal_Label" : "Input Signal",
                "PlotTitle" : "Chialvo Map : Timeline of phi",
                "PlotXLabel" : "Time Step",
                "PlotYLabel" : "phi",
                "PlotPath_Project" : "./Graphical/Results/Project_Chialvo_TimeLine",
                "PlotPath_Date" : f"/TimeLine_{Plot_Date}",
                "PlotName" : f"/{TimeDate()}_TimeLine_Phi.png"
                        })
            Plot_Graph.Plot_TimeLine(param)()


class Output_PhaseSpace(Output):
    """
    相平面の描写
    """

    #コンストラクタ
    def __init__(self, param: dict, parent: any = None):
        super().__init__(param, parent)

        #どれをプロットするか
        self.Plot_XY = self.Param["Project_Plot_PhaseSpace_XandY"]
        self.Plot_XP = self.Param["Project_Plot_PhaseSpace_XandPhi"]
        self.Plot_YP = self.Param["Project_Plot_PhaseSpace_YandPhi"]
        self.Plot_3D = self.Param["Project_Plot_PhaseSpace_3D"]

        #空走時間
        self.Length_Burnin = self.Param["Length_Burnin"]
        #評価時間
        self.Length_Eva = self.Param["Length_Eva"]
        #プロット時間
        self.Length_Plot = self.Param["Length_Plot"]
        #総合時間
        self.Length_Total = self.Length_Burnin + self.Length_Eva + self.Length_Plot


        #モデル
        self.T_Model = self.Param["Model"]
        param = self.Param.copy()
        self.Model = self.T_Model(param, self)
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def __call__(self):

        print("--- Chialvo Neuron Map RunStart---\n")
        self.X, self.Y, self.Phi, _ = self.Model()

        def TimeDate():
            Time_Delta = datetime.timedelta(hours=9)
            JST = datetime.timezone(Time_Delta ,'JST')
            Now = datetime.datetime.now(JST)
            Date = Now.strftime('%Y_%m_%d_%H_%M_%S')
            return Date
        
        Plot_Date = TimeDate()

        if self.Plot_XY:
            print("\n--- Display of PhaseSpace Chialvo Neuron Map of X and Y ---")
            
            #プロット要素
            param = self.Param.copy()
            param.update({
                "PlotData_x-axis" : self.X,
                "PlotData_y-axis" : self.Y,

                "PlotTitle" : "Chialvo Map : PhaseSpace of X and Y",
                "PlotXLabel" : "x",
                "PlotYLabel" : "y",
                "PlotPath_Project" : "./Graphical/Results/Project_Chialvo_PhaseSpace",
                "PlotPath_Date" : f"/PhaseSpace_{Plot_Date}",
                "PlotName" : f"/{TimeDate()}_PhaseSpace_XandY.png"
                })
            Plot_Graph.Plot_PhaseSpace(param)()

        if self.Plot_XP:
            print("\n--- Display of PhaseSpace Chialvo Neuron Map of X and Phi ---")
            #プロット要素
            param = self.Param.copy()
            param.update({
                "PlotData_x-axis" : self.X,
                "PlotData_y-axis" : self.Phi,

                "PlotTitle" : "Chialvo Map : PhaseSpace of X and Phi",
                "PlotXLabel" : "x",
                "PlotYLabel" : "phi",
                "PlotPath_Project" : "./Graphical/Results/Project_Chialvo_PhaseSpace",
                "PlotPath_Date" : f"/PhaseSpace_{Plot_Date}",
                "PlotName" : f"/{TimeDate()}_PhaseSpace_XandPhi.png"
            })
            Plot_Graph.Plot_PhaseSpace(param)()

        if self.Plot_YP:
            print("\n--- Display of PhaseSpace Chialvo Neuron Map of Y and Phi ---")
            
            #プロット要素
            param = self.Param.copy()
            param.update({
                "PlotData_x-axis" : self.Y,
                "PlotData_y-axis" : self.Phi,

                "PlotTitle" : "Chialvo Map : PhaseSpace of Y and Phi",
                "PlotXLabel" : "y",
                "PlotYLabel" : "phi",
                "PlotPath_Project" : "./Graphical/Results/Project_Chialvo_PhaseSpace",
                "PlotPath_Date" : f"/PhaseSpace_{Plot_Date}",
                "PlotName" : f"/{TimeDate()}_PhaseSpace_YandPhi.png"    
                })
            Plot_Graph.Plot_PhaseSpace(param)()

        if self.Plot_3D:
            print("\n--- Display of PhaseSpace Chialvo Neuron Map of 3D ---")
            
            #プロット要素
            param = self.Param.copy()
            param.update({
                "PlotData_x-axis" : self.X,
                "PlotData_y-axis" : self.Y,
                "PlotData_z-axis" : self.Phi,

                "PlotTitle" : "Chialvo Map : PhaseSpace of 3D",
                "PlotXLabel" : "x",
                "PlotYLabel" : "y",
                "PlotZLabel" : "phi",
                "PlotPath_Project" : "./Graphical/Results/Project_Chialvo_PhaseSpace",
                "PlotPath_Date" : f"/PhaseSpace_{Plot_Date}",
                "PlotName" : f"/{TimeDate()}_PhaseSpace_3D.png"    
                })
            Plot_Graph.Plot_PhaseSpace_3D(param)()


class Output_OldNullcline(Output):
    """
    ベクトル場とNullclineの描写
    """

    #コンストラクタ
    def __init__(self, param: dict, parent: any = None):
        super().__init__(param, parent)

        #どれをプロットするか
        self.Plot_OldNullcline = self.Param["Project_Plot_OldNullcline"]
        self.Plot_PhaseSpace = self.Param["Project_Plot_PhaseSpace"]

        #実行時間
        self.RunTime = self.Param["RunTime"]
        self.Plot_Start = self.Param["Plot_Start"]
        self.Plot_End = self.Param["Plot_End"]

        #モデル
        self.T_Model = self.Param["Model"]
        param = self.Param.copy()
        self.Model = self.T_Model(param, self)
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def __call__(self):

        print("--- Chialvo Neuron Map RunStart---\n")
        
        self.X, self.Y, self.Vx, self.Vy, \
            self.dx, self.dy, self.fx, self.fy, \
                self.x, self.y = self.Model()

        def TimeDate():
            Time_Delta = datetime.timedelta(hours=9)
            JST = datetime.timezone(Time_Delta ,'JST')
            Now = datetime.datetime.now(JST)
            Date = Now.strftime('%Y_%m_%d_%H_%M_%S')
            return Date
        
        Plot_Date = TimeDate()

        if self.Plot_OldNullcline:
            print("\n--- Display of Nullcline Chialvo Neuron Map---")
            
            #プロット要素
            param = self.Param.copy()
            param.update({
                #--------------------------------------------------------------------
                #ベクトル場の点と変数
                "PlotData_X"        : self.X,
                "PlotData_Y"        : self.Y,

                "PlotData_Vx"       : self.Vx,
                "PlotData_Vy"       : self.Vy,
                #--------------------------------------------------------------------
                #Nullclineの点と変数
                "PlotData_dx"       : self.dx,
                "PlotData_dy"       : self.dy,

                "PlotData_fx"       : self.fx,
                "PlotData_fy"       : self.fy,
                #--------------------------------------------------------------------
                #相平面の変数
                "PlotData_x"        : self.x,
                "PlotData_y"        : self.y,
                #--------------------------------------------------------------------
                "PlotTitle"         : "Chialvo Map : Nullcine",
                "PlotXLabel"        : "x",
                "PlotYLabel"        : "y",
                "PlotPath_Project" : "./Results/Project_Chialvo_OldNullcline",
                "PlotPath_Date" : f"/OldNullcline_{Plot_Date}",
                "PlotName" : f"/{TimeDate()}_OldNullcline.png"
            })
            Plot_Graph.Plot_Nullcline(param)()


class Output_NewNullcline(Output):
    """
    ベクトル場とNullclineの描写
    """

    #コンストラクタ
    def __init__(self, param: dict, parent: any = None):
        super().__init__(param, parent)

        #どれをプロットするか
        self.Plot_NewNullcline = self.Param["Project_Plot_NewNullcline"]
        self.Plot_PhaseSpace = self.Param["Project_Plot_PhaseSpace"]

        #実行時間
        self.RunTime = self.Param["RunTime"]
        self.Plot_Start = self.Param["Plot_Start"]
        self.Plot_End = self.Param["Plot_End"]

        #モデル
        self.T_Model = self.Param["Model"]
        param = self.Param.copy()
        self.Model = self.T_Model(param, self)
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def __call__(self):

        print("--- Chialvo Neuron Map RunStart---\n")
        
        self.X, self.Y, self.Vx, self.Vy, \
            self.dx, self.dy, self.fx, self.fy, \
                self.x, self.y = self.Model()

        def TimeDate():
            Time_Delta = datetime.timedelta(hours=9)
            JST = datetime.timezone(Time_Delta ,'JST')
            Now = datetime.datetime.now(JST)
            Date = Now.strftime('%Y_%m_%d_%H_%M_%S')
            return Date
        
        Plot_Date = TimeDate()

        if self.Plot_NewNullcline:
            print("\n--- Display of Nullcline Chialvo Neuron Map---")
            
            #プロット要素
            param = self.Param.copy()
            param.update({
                #--------------------------------------------------------------------
                #ベクトル場の点と変数
                "PlotData_X"        : self.X,
                "PlotData_Y"        : self.Y,

                "PlotData_Vx"       : self.Vx,
                "PlotData_Vy"       : self.Vy,
                #--------------------------------------------------------------------
                #Nullclineの点と変数
                "PlotData_dx"       : self.dx,
                "PlotData_dy"       : self.dy,

                "PlotData_fx"       : self.fx,
                "PlotData_fy"       : self.fy,
                #--------------------------------------------------------------------
                #相平面の変数
                "PlotData_x"        : self.x,
                "PlotData_y"        : self.y,
                #--------------------------------------------------------------------
                "PlotTitle"         : "Chialvo Map : Nullcine",
                "PlotXLabel"        : "x",
                "PlotYLabel"        : "y",
                "PlotPath_Project" : "./Results/Project_Chialvo_NewNullcline",
                "PlotPath_Date" : f"/NewNullcline_{Plot_Date}",
                "PlotName" : f"/{TimeDate()}_NewNullcline.png"
            })
            Plot_Graph.Plot_Nullcline(param)()


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class Output_TimeLine_NeuronMap(Output):
    """
    時系列描写
    """

    #コンストラクタ
    def __init__(self, param: dict, parent: any = None):
        super().__init__(param, parent)

        #どれをプロットするか
        self.Plot_X = self.Param["Project_Plot_TimeLine_X"]
        self.Plot_Y = self.Param["Project_Plot_TimeLine_Y"]
        self.Plot_Phi = self.Param["Project_Plot_TimeLine_Phi"]

        #空走時間
        self.Length_Burnin = self.Param["Length_Burnin"]
        #評価時間
        self.Length_Eva = self.Param["Length_Eva"]
        #プロット時間
        self.Length_Plot = self.Param["Length_Plot"]
        #総合時間
        self.Length_Total = self.Length_Burnin + self.Length_Eva +self.Length_Plot

        #モデル
        self.T_Model = self.Param["Model"]
        param = self.Param.copy()
        self.Model = self.T_Model(param, self)
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def __call__(self):

        print("--- Chialvo Neuron Map RunStart---\n")
        self.X, self.Y, self.Phi, self.Signal = self.Model()

        def TimeDate():
            Time_Delta = datetime.timedelta(hours=9)
            JST = datetime.timezone(Time_Delta ,'JST')
            Now = datetime.datetime.now(JST)
            Date = Now.strftime('%Y_%m_%d_%H_%M_%S')
            return Date
        
        Plot_Date = TimeDate()

        if self.Plot_X:
            print("\n--- Display of TimeLine Chialvo Neuron Map of X---")
            
            #プロット要素
            param = self.Param.copy()
            param.update({
                "PlotData" : self.X,
                "PlotData_Label" : "Chialvo : X",
                "PlotSignal" : self.Signal,
                "PlotSignal_Label" : "Input Signal",
                "PlotTitle" : "Chialvo Map : Timeline of x",
                "PlotXLabel" : "Time Step",
                "PlotYLabel" : "x",
                "PlotPath_Project" : "./Graphical/Results/Project_ChialvoNeuronMap_TimeLine",
                "PlotPath_Date" : f"/TimeLine_{Plot_Date}",
                "PlotName" : f"/{TimeDate()}_TimeLine_X.png"
            })
            Plot_Graph.Plot_TimeLine_NeuronMap(param)()

        if self.Plot_Y:
            print("\n--- Display of TimeLine Chialvo Neuron Map of Y---")
            #プロット要素
            param = self.Param.copy()
            param.update({
                "PlotData" : self.Y,
                "PlotData_Label" : "Chialvo : Y",
                "PlotSignal" : self.Signal,
                "PlotSignal_Label" : "Input Signal",
                "PlotTitle" : "Chialvo Map : Timeline of y",
                "PlotXLabel" : "Time Step",
                "PlotYLabel" : "y",
                "PlotPath_Project" : "./Graphical/Results/Project_ChialvoNeuronMap_TimeLine",
                "PlotPath_Date" : f"/TimeLine_{Plot_Date}",
                "PlotName" : f"/{TimeDate()}_TimeLine_Y.png"
            })
            Plot_Graph.Plot_TimeLine_NeuronMap(param)()

        if self.Plot_Phi:
            print("\n--- Display of TimeLine Chialvo Neuron Map of phi---")
            
            #プロット要素
            param = self.Param.copy()
            param.update({
                "PlotData" : self.Phi,
                "PlotData_Label" : "Chialvo : Phi",
                "PlotSignal" : self.Signal,
                "PlotSignal_Label" : "Input Signal",
                "PlotTitle" : "Chialvo Map : Timeline of phi",
                "PlotXLabel" : "Time Step",
                "PlotYLabel" : "phi",
                "PlotPath_Project" : "./Graphical/Results/Project_ChialvoNeuronMap_TimeLine",
                "PlotPath_Date" : f"/TimeLine_{Plot_Date}",
                "PlotName" : f"/{TimeDate()}_TimeLine_Phi.png"
                        })
            Plot_Graph.Plot_TimeLine_NeuronMap(param)()

class Output_PhaseSpace_NeuronMap(Output):
    """
    相平面の描写
    """

    #コンストラクタ
    def __init__(self, param: dict, parent: any = None):
        super().__init__(param, parent)

        #どれをプロットするか
        self.Plot_XY = self.Param["Project_Plot_PhaseSpace_XandY"]
        self.Plot_XP = self.Param["Project_Plot_PhaseSpace_XandPhi"]
        self.Plot_YP = self.Param["Project_Plot_PhaseSpace_YandPhi"]
        self.Plot_3D = self.Param["Project_Plot_PhaseSpace_3D"]

        #空走時間
        self.Length_Burnin = self.Param["Length_Burnin"]
        #評価時間
        self.Length_Eva = self.Param["Length_Eva"]
        #プロット時間
        self.Length_Plot = self.Param["Length_Plot"]
        #総合時間
        self.Length_Total = self.Length_Burnin + self.Length_Eva + self.Length_Plot


        #モデル
        self.T_Model = self.Param["Model"]
        param = self.Param.copy()
        self.Model = self.T_Model(param, self)
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def __call__(self):

        print("--- Chialvo Neuron Map RunStart---\n")
        self.X, self.Y, self.Phi, _ = self.Model()

        def TimeDate():
            Time_Delta = datetime.timedelta(hours=9)
            JST = datetime.timezone(Time_Delta ,'JST')
            Now = datetime.datetime.now(JST)
            Date = Now.strftime('%Y_%m_%d_%H_%M_%S')
            return Date
        
        Plot_Date = TimeDate()

        if self.Plot_XY:
            print("\n--- Display of PhaseSpace Chialvo Neuron Map of X and Y ---")
            
            #プロット要素
            param = self.Param.copy()
            param.update({
                "PlotData_x-axis" : self.X,
                "PlotData_y-axis" : self.Y,

                "PlotTitle" : "Chialvo Map : PhaseSpace of X and Y",
                "PlotXLabel" : "x",
                "PlotYLabel" : "y",
                "PlotPath_Project" : "./Graphical/Results/Project_ChialvoNeuronMap_PhaseSpace",
                "PlotPath_Date" : f"/PhaseSpace_{Plot_Date}",
                "PlotName" : f"/{TimeDate()}_PhaseSpace_XandY.png"
                })
            Plot_Graph.Plot_PhaseSpace_NeuronMap(param)()

        if self.Plot_XP:
            print("\n--- Display of PhaseSpace Chialvo Neuron Map of X and Phi ---")
            #プロット要素
            param = self.Param.copy()
            param.update({
                "PlotData_x-axis" : self.X,
                "PlotData_y-axis" : self.Phi,

                "PlotTitle" : "Chialvo Map : PhaseSpace of X and Phi",
                "PlotXLabel" : "x",
                "PlotYLabel" : "phi",
                "PlotPath_Project" : "./Graphical/Results/Project_ChialvoNeuronMap_PhaseSpace",
                "PlotPath_Date" : f"/PhaseSpace_{Plot_Date}",
                "PlotName" : f"/{TimeDate()}_PhaseSpace_XandPhi.png"
            })
            Plot_Graph.Plot_PhaseSpace_NeuronMap(param)()

        if self.Plot_YP:
            print("\n--- Display of PhaseSpace Chialvo Neuron Map of Y and Phi ---")
            
            #プロット要素
            param = self.Param.copy()
            param.update({
                "PlotData_x-axis" : self.Y,
                "PlotData_y-axis" : self.Phi,

                "PlotTitle" : "Chialvo Map : PhaseSpace of Y and Phi",
                "PlotXLabel" : "y",
                "PlotYLabel" : "phi",
                "PlotPath_Project" : "./Graphical/Results/Project_ChialvoNeuronMap_PhaseSpace",
                "PlotPath_Date" : f"/PhaseSpace_{Plot_Date}",
                "PlotName" : f"/{TimeDate()}_PhaseSpace_YandPhi.png"    
                })
            Plot_Graph.Plot_PhaseSpace_NeuronMap(param)()

        if self.Plot_3D:
            print("\n--- Display of PhaseSpace Chialvo Neuron Map of 3D ---")
            
            #プロット要素
            param = self.Param.copy()
            param.update({
                "PlotData_x-axis" : self.X,
                "PlotData_y-axis" : self.Y,
                "PlotData_z-axis" : self.Phi,

                "PlotTitle" : "Chialvo Map : PhaseSpace of 3D",
                "PlotXLabel" : "x",
                "PlotYLabel" : "y",
                "PlotZLabel" : "phi",
                "PlotPath_Project" : "./Graphical/Results/Project_ChialvoNeuronMap_PhaseSpace",
                "PlotPath_Date" : f"/PhaseSpace_{Plot_Date}",
                "PlotName" : f"/{TimeDate()}_PhaseSpace_3D.png"    
                })
            Plot_Graph.Plot_PhaseSpace_3D_NeuronMap(param)()

