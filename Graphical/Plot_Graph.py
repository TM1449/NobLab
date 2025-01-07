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
from mpl_toolkits.mplot3d import Axes3D
import os
#====================================================================
#内部プログラム


#====================================================================
class Plot:
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

class Plot_TimeLine(Plot):
    """
    時系列描写
    """
    #コンストラクタ
    def __init__(self, param: dict, parent: any = None):
        super().__init__(param, parent)

        #プロット要素
        self.PlotData = self.Param["PlotData"]
        self.PlotData_Label = self.Param["PlotData_Label"]
        self.PlotSignal = self.Param["PlotSignal"]
        self.PlotSignal_Label = self.Param["PlotSignal_Label"]
        self.PlotTitle = self.Param["PlotTitle"]
        self.PlotXLabel = self.Param["PlotXLabel"]
        self.PlotYLabel = self.Param["PlotYLabel"]
        self.PlotPath_Project = self.Param["PlotPath_Project"]
        self.PlotPath_Date = self.Param["PlotPath_Date"]
        self.PlotName = self.Param["PlotName"]

        #モデル
        self.T_Model = self.Param["Model"]
        param = self.Param.copy()
        self.Model = self.T_Model(param, self)

    def __call__(self):

        def New_DirPath():
            Dir_Path = f"{self.PlotPath_Project}{self.PlotPath_Date}"
            os.makedirs(Dir_Path, exist_ok=True)

        #プロット図全体のフォントサイズ
        FigSize = (16,9)
        FontSize_Axis = 32                #「各軸ラベル（Time Stepなど）」のフォントサイズ
        FontSize_Title = 28               #「タイトル（一番上）」のフォントサイズ
        FontSize_TickLabel = 14           #「各ラベルの単位（-2.0,-1.9など）」のフォントサイズ
        FontSize_legend = 10              #「各凡例」のフォントサイズ
        LineWidth = 2                     #線の太さ
        FileFormat = ".png"          #ファイルフォーマット

        fig = plt.figure(figsize = FigSize)
        ax = fig.add_subplot(1,1,1)

        Title = self.PlotTitle
        ax.tick_params(labelsize=FontSize_TickLabel)
        ax.set_title(Title, fontsize = FontSize_Title)
        ax.set_xlabel(self.PlotXLabel, fontsize = FontSize_Axis)
        ax.set_ylabel(self.PlotYLabel, fontsize = FontSize_Axis)

        ax.plot(self.PlotData, label = f"{self.PlotData_Label}", linestyle = '-',lw = LineWidth * 0.9)
        ax.plot(self.PlotSignal, label = f"{self.PlotSignal_Label}", linestyle = '-',lw = LineWidth * 0.85)
        ax.grid()
        ax.legend(fontsize = FontSize_legend)
        fig.tight_layout()
        New_DirPath()
        plt.savefig(self.PlotPath_Project + self.PlotPath_Date + self.PlotName)
        plt.show()

class Plot_PhaseSpace(Plot):
    """
    相平面描写
    """
    #コンストラクタ
    def __init__(self, param: dict, parent: any = None):
        super().__init__(param, parent)

        #プロット要素
        self.PlotData_Xaxis = self.Param["PlotData_x-axis"]
        self.PlotData_Yaxis = self.Param["PlotData_y-axis"]
        
        self.PlotTitle = self.Param["PlotTitle"]
        self.PlotXLabel = self.Param["PlotXLabel"]
        self.PlotYLabel = self.Param["PlotYLabel"]
        self.PlotPath_Project = self.Param["PlotPath_Project"]
        self.PlotPath_Date = self.Param["PlotPath_Date"]
        self.PlotName = self.Param["PlotName"]

        #モデル
        self.T_Model = self.Param["Model"]
        param = self.Param.copy()
        self.Model = self.T_Model(param, self)

    def __call__(self):

        def New_DirPath():
            Dir_Path = f"{self.PlotPath_Project}{self.PlotPath_Date}"
            os.makedirs(Dir_Path, exist_ok=True)

        #プロット図全体のフォントサイズ
        FigSize = (16,9)
        FontSize_Axis = 32                #「各軸ラベル（Time Stepなど）」のフォントサイズ
        FontSize_Title = 28               #「タイトル（一番上）」のフォントサイズ
        FontSize_TickLabel = 14           #「各ラベルの単位（-2.0,-1.9など）」のフォントサイズ
        LineWidth = 2                     #線の太さ
        FileFormat = ".png"          #ファイルフォーマット

        fig = plt.figure(figsize = FigSize)
        ax = fig.add_subplot(1,1,1)

        Title = self.PlotTitle
        plt.tick_params(labelsize=FontSize_TickLabel)
        ax.set_title(Title, fontsize = FontSize_Title)
        ax.set_xlabel(self.PlotXLabel, fontsize = FontSize_Axis)
        ax.set_ylabel(self.PlotYLabel, fontsize = FontSize_Axis)

        ax.plot(self.PlotData_Xaxis, self.PlotData_Yaxis,'-',lw = LineWidth)
        ax.grid()
        fig.tight_layout()
        New_DirPath()
        plt.savefig(self.PlotPath_Project + self.PlotPath_Date + self.PlotName)
        plt.show()

class Plot_PhaseSpace_3D(Plot):
    """
    3Dの相平面描写
    """
    #コンストラクタ
    def __init__(self, param: dict, parent: any = None):
        super().__init__(param, parent)

        #プロット要素
        self.PlotData_Xaxis = self.Param["PlotData_x-axis"]
        self.PlotData_Yaxis = self.Param["PlotData_y-axis"]
        self.PlotData_Zaxis = self.Param["PlotData_z-axis"]
        
        self.PlotTitle = self.Param["PlotTitle"]
        self.PlotXLabel = self.Param["PlotXLabel"]
        self.PlotYLabel = self.Param["PlotYLabel"]
        self.PlotZLabel = self.Param["PlotZLabel"]
        self.PlotPath_Project = self.Param["PlotPath_Project"]
        self.PlotPath_Date = self.Param["PlotPath_Date"]
        self.PlotName = self.Param["PlotName"]

        #モデル
        self.T_Model = self.Param["Model"]
        param = self.Param.copy()
        self.Model = self.T_Model(param, self)

    def __call__(self):

        def New_DirPath():
            Dir_Path = f"{self.PlotPath_Project}{self.PlotPath_Date}"
            os.makedirs(Dir_Path, exist_ok=True)

        #プロット図全体のフォントサイズ
        FigSize = (16,9)
        FontSize_Axis = 22                #「各軸ラベル（Time Stepなど）」のフォントサイズ
        FontSize_Title = 18               #「タイトル（一番上）」のフォントサイズ
        FontSize_TickLabel = 9           #「各ラベルの単位（-2.0,-1.9など）」のフォントサイズ
        FontSize_legend = 10              #「各凡例」のフォントサイズ
        LineWidth = 2                     #線の太さ
        FileFormat = ".png"          #ファイルフォーマット

        fig = plt.figure(figsize = FigSize)
        ax = fig.add_subplot(1,1,1, projection = '3d')

        Title = self.PlotTitle
        plt.tick_params(labelsize=FontSize_TickLabel)
        ax.set_title(Title, fontsize = FontSize_Title)
        ax.set_xlabel(self.PlotXLabel, fontsize = FontSize_Axis)
        ax.set_ylabel(self.PlotYLabel, fontsize = FontSize_Axis)
        ax.set_zlabel(self.PlotZLabel, fontsize = FontSize_Axis)

        ax.plot(self.PlotData_Xaxis, self.PlotData_Yaxis, self.PlotData_Zaxis,'-',lw = LineWidth)
        ax.grid()
        fig.tight_layout()
        New_DirPath()
        plt.savefig(self.PlotPath_Project + self.PlotPath_Date + self.PlotName)
        plt.show()

class Plot_Nullcline(Plot):
    """
    ベクトル場とNullclineの描写
    """
    #コンストラクタ
    def __init__(self, param: dict, parent: any = None):
        super().__init__(param, parent)

        #ベクトル場の点と変数
        self.PlotData_X = self.Param["PlotData_X"]
        self.PlotData_Y = self.Param["PlotData_Y"]
        
        self.PlotData_Vx = self.Param["PlotData_Vx"]
        self.PlotData_Vy = self.Param["PlotData_Vy"]

        #Nullclineの点と変数
        self.PlotData_dx = self.Param["PlotData_dx"]
        self.PlotData_dy = self.Param["PlotData_dy"]
        
        self.PlotData_fx = self.Param["PlotData_fx"]
        self.PlotData_fy = self.Param["PlotData_fy"]
        
        #相平面の変数
        self.PlotData_x = self.Param["PlotData_x"]
        self.PlotData_y = self.Param["PlotData_y"]

        #ベクトル場の点分布の範囲
        self.Plot_x_Start = self.Param["Plot_x_Start"]
        self.Plot_x_End = self.Param["Plot_x_End"]

        self.Plot_y_Start = self.Param["Plot_y_Start"]
        self.Plot_y_End = self.Param["Plot_y_End"]
        
        #プロット要素
        self.PlotTitle = self.Param["PlotTitle"]
        self.PlotXLabel = self.Param["PlotXLabel"]
        self.PlotYLabel = self.Param["PlotYLabel"]
        self.PlotPath_Project = self.Param["PlotPath_Project"]
        self.PlotPath_Date = self.Param["PlotPath_Date"]
        self.PlotName = self.Param["PlotName"]

        #モデル
        self.T_Model = self.Param["Model"]
        param = self.Param.copy()
        self.Model = self.T_Model(param, self)

        self.Plot_PhaseSpace = self.Param["Project_Plot_PhaseSpace"]

    def __call__(self):

        def New_DirPath():
            Dir_Path = f"{self.PlotPath_Project}{self.PlotPath_Date}"
            os.makedirs(Dir_Path, exist_ok=True)

        #プロット図全体のフォントサイズ
        FigSize = (16,9)
        FontSize_Axis = 32                #「各軸ラベル（Time Stepなど）」のフォントサイズ
        FontSize_Title = 28               #「タイトル（一番上）」のフォントサイズ
        FontSize_TickLabel = 14           #「各ラベルの単位（-2.0,-1.9など）」のフォントサイズ
        FontSize_legend = 10              #「各凡例」のフォントサイズ
        LineWidth = 2                     #線の太さ
        FileFormat = ".png"          #ファイルフォーマット

        fig = plt.figure(figsize = FigSize)
        ax = fig.add_subplot(1,1,1)

        Title = self.PlotTitle
        ax.tick_params(labelsize=FontSize_TickLabel)
        ax.set_title(Title, fontsize = FontSize_Title)
        ax.set_xlabel(self.PlotXLabel, fontsize = FontSize_Axis)
        ax.set_ylabel(self.PlotYLabel, fontsize = FontSize_Axis)

        #Nullclineの描写
        ax.plot(self.PlotData_dx, self.PlotData_fx, '-', lw = LineWidth)
        ax.plot(self.PlotData_fy, self.PlotData_dy, '-', lw = LineWidth)
        
        #相平面の描写
        if self.Plot_PhaseSpace:
            ax.plot(self.PlotData_x, self.PlotData_y, '-', lw = LineWidth)

        ax.quiver(self.PlotData_X, self.PlotData_Y, self.PlotData_Vx / np.sqrt(pow(self.PlotData_Vx, 2) + pow(self.PlotData_Vy, 2)), self.PlotData_Vy / np.sqrt(pow(self.PlotData_Vx, 2) + pow(self.PlotData_Vy, 2)))

        ax.set_xlim([self.Plot_x_Start, self.Plot_x_End])
        ax.set_ylim([self.Plot_y_Start, self.Plot_y_End])
        fig.tight_layout()
        New_DirPath()
        plt.savefig(self.PlotPath_Project + self.PlotPath_Date + self.PlotName)
        plt.show()

