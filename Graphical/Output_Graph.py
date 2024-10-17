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

#====================================================================
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

        #実行時間
        self.RunTime = self.Param["RunTime"]
        self.Plot_Start = self.Param["Plot_Start"]
        self.Plot_End = self.Param["Plot_End"]

        #モデル
        self.T_Model = self.Param["Model"]
        param = self.Param.copy()
        self.Model = self.T_Model(param, self)

    def __call__(self):

        print("--- Chialvo Neuron Map RunStart---\n")
        self.X, self.Y , self.Phi = self.Model()

        if self.Plot_X:
            print("\n--- Display of TimeLine Chialvo Neuron Map of X---")
            
            #プロット要素
            param = self.Param.copy()
            param.update({
                "PlotData" : self.X,
                "PlotTitle" : "Chialvo Map : Timeline of x",
                "PlotXLabel" : "Time Step",
                "PlotYLabel" : "x",
                "PlotName" : "./Results/Project_Chialvo_TimeLine/TimeLine_X.png"
            })
            Plot_Graph.Plot_TimeLine(param)()

        if self.Plot_Y:
            print("\n--- Display of TimeLine Chialvo Neuron Map of Y---")
            #プロット要素
            param = self.Param.copy()
            param.update({
                "PlotData" : self.Y,
                "PlotTitle" : "Chialvo Map : Timeline of y",
                "PlotXLabel" : "Time Step",
                "PlotYLabel" : "y",
                "PlotName" : "./Results/Project_Chialvo_TimeLine/TimeLine_Y.png"
            })
            Plot_Graph.Plot_TimeLine(param)()

        if self.Plot_Phi:
            print("\n--- Display of TimeLine Chialvo Neuron Map of phi---")
            
            #プロット要素
            param = self.Param.copy()
            param.update({
                "PlotData" : self.Phi,
                "PlotTitle" : "Chialvo Map : Timeline of phi",
                "PlotXLabel" : "Time Step",
                "PlotYLabel" : "phi",
                "PlotName" : "./Results/Project_Chialvo_TimeLine/TimeLine_Phi.png"
                        })
            Plot_Graph.Plot_TimeLine(param)()

class Output_PhaseSpace(Output):
    """
    時系列描写
    """
    #コンストラクタ
    def __init__(self, param: dict, parent: any = None):
        super().__init__(param, parent)

        #どれをプロットするか
        self.Plot_XY = self.Param["Project_Plot_PhaseSpace_XandY"]
        self.Plot_XP = self.Param["Project_Plot_PhaseSpace_XandPhi"]
        self.Plot_YP = self.Param["Project_Plot_PhaseSpace_YandPhi"]

        #実行時間
        self.RunTime = self.Param["RunTime"]
        self.Plot_Start = self.Param["Plot_Start"]
        self.Plot_End = self.Param["Plot_End"]

        #モデル
        self.T_Model = self.Param["Model"]
        param = self.Param.copy()
        self.Model = self.T_Model(param, self)

    def __call__(self):

        print("--- Chialvo Neuron Map RunStart---\n")
        self.X, self.Y , self.Phi = self.Model()

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
                "PlotName" : "./Results/Project_Chialvo_PhaseSpace/PhaseSpace_XandY.png"
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
                "PlotName" : "./Results/Project_Chialvo_PhaseSpace/PhaseSpace_XandPhi.png"
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
                "PlotName" : "./Results/Project_Chialvo_PhaseSpace/PhaseSpace_YandPhi.png"        
                })
            Plot_Graph.Plot_PhaseSpace(param)()

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

    def __call__(self):

        print("--- Chialvo Neuron Map RunStart---\n")
        self.dx, self.dy, self.X, self.Y, self.Ex, self.Ey, self.fx, self.fy, self.x, self.y = self.Model()

        if self.Plot_OldNullcline:
            print("\n--- Display of Nullcline Chialvo Neuron Map---")
            
            #プロット要素
            param = self.Param.copy()
            param.update({
                "PlotData_fx"       : self.fx,
                "PlotData_fy"       : self.fy,

                "PlotData_Ex"       : self.Ex,
                "PlotData_Ey"       : self.Ey,

                "PlotData_x"        : self.x,
                "PlotData_y"        : self.y,

                "PlotData_dx"       : self.dx,
                "PlotData_dy"       : self.dy,

                "PlotData_X"        : self.X,
                "PlotData_Y"        : self.Y,

                "PlotTitle"         : "Chialvo Map : Nullcine",
                "PlotXLabel"        : "x",
                "PlotYLabel"        : "y",
                "PlotName"          : "./Results/Project_Chialvo_OldNullcline/Nullcline_XandY.png"
            })
            Plot_Graph.Plot_Nullcline(param)()