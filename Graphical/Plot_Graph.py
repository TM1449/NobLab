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
        self.PlotTitle = self.Param["PlotTitle"]
        self.PlotXLabel = self.Param["PlotXLabel"]
        self.PlotYLabel = self.Param["PlotYLabel"]
        self.PlotName = self.Param["PlotName"]

        #モデル
        self.T_Model = self.Param["Model"]
        param = self.Param.copy()
        self.Model = self.T_Model(param, self)

    def __call__(self):

        #プロット図全体のフォントサイズ
        FigSize = (16,9)
        FontSize_Label = 34                 #ラベルのフォントサイズ
        FontSize_Title = 28                 #タイトルのフォントサイズ
        Labelsize = 28                      #ラベルサイズ
        LineWidth = 2                       #線の太さ
        FileFormat = ".png"          #ファイルフォーマット

        fig = plt.figure(figsize = FigSize)
        ax = fig.add_subplot(1,1,1)

        Title = self.PlotTitle
        plt.tick_params(labelsize=Labelsize)
        ax.set_title(Title, fontsize = FontSize_Title)
        ax.set_xlabel(self.PlotXLabel, fontsize = FontSize_Label)
        ax.set_ylabel(self.PlotYLabel, fontsize = FontSize_Label)

        ax.plot(self.PlotData,'-',lw = LineWidth)
        fig.tight_layout()
        plt.savefig(self.PlotName)
        plt.close()

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
        self.PlotName = self.Param["PlotName"]

        #モデル
        self.T_Model = self.Param["Model"]
        param = self.Param.copy()
        self.Model = self.T_Model(param, self)

    def __call__(self):

        #プロット図全体のフォントサイズ
        FigSize = (16,9)
        FontSize_Label = 34                 #ラベルのフォントサイズ
        FontSize_Title = 28                 #タイトルのフォントサイズ
        Labelsize = 28                      #ラベルサイズ
        LineWidth = 2                       #線の太さ
        FileFormat = ".png"          #ファイルフォーマット

        fig = plt.figure(figsize = FigSize)
        ax = fig.add_subplot(1,1,1)

        Title = self.PlotTitle
        plt.tick_params(labelsize=Labelsize)
        ax.set_title(Title, fontsize = FontSize_Title)
        ax.set_xlabel(self.PlotXLabel, fontsize = FontSize_Label)
        ax.set_ylabel(self.PlotYLabel, fontsize = FontSize_Label)

        ax.plot(self.PlotData_Xaxis, self.PlotData_Yaxis,'-',lw = LineWidth)
        fig.tight_layout()
        plt.savefig(self.PlotName)
        plt.close()

class Plot_Nullcline(Plot):
    """
    ベクトル場とNullclineの描写
    """
    #コンストラクタ
    def __init__(self, param: dict, parent: any = None):
        super().__init__(param, parent)

        #プロット要素
        self.PlotData_fx = self.Param["PlotData_fx"]
        self.PlotData_fy = self.Param["PlotData_fy"]

        self.PlotData_x = self.Param["PlotData_x"]
        self.PlotData_y = self.Param["PlotData_y"]

        self.PlotData_Ex = self.Param["PlotData_Ex"]
        self.PlotData_Ey = self.Param["PlotData_Ey"]

        self.PlotData_dx = self.Param["PlotData_dx"]
        self.PlotData_dy = self.Param["PlotData_dy"]

        self.PlotData_X = self.Param["PlotData_X"]
        self.PlotData_Y = self.Param["PlotData_Y"]
        
        self.PlotTitle = self.Param["PlotTitle"]
        self.PlotXLabel = self.Param["PlotXLabel"]
        self.PlotYLabel = self.Param["PlotYLabel"]
        self.PlotName = self.Param["PlotName"]

        #モデル
        self.T_Model = self.Param["Model"]
        param = self.Param.copy()
        self.Model = self.T_Model(param, self)

        self.Plot_PhaseSpace = self.Param["Project_Plot_PhaseSpace"]

    def __call__(self):

        #プロット図全体のフォントサイズ
        FigSize = (16,9)
        FontSize_Label = 34                 #ラベルのフォントサイズ
        FontSize_Title = 28                 #タイトルのフォントサイズ
        Labelsize = 28                      #ラベルサイズ
        LineWidth = 2                       #線の太さ

        p = 0.5
        xmax, xmin = self.PlotData_x.max() + p, self.PlotData_x.min() - p
        ymax, ymin = self.PlotData_y.max() + p, self.PlotData_y.min() - p

        fig = plt.figure(figsize = FigSize)
        ax = fig.add_subplot(1,1,1)

        Title = self.PlotTitle
        ax.tick_params(labelsize=Labelsize)
        ax.set_title(Title, fontsize = FontSize_Title)
        ax.set_xlabel(self.PlotXLabel, fontsize = FontSize_Label)
        ax.set_ylabel(self.PlotYLabel, fontsize = FontSize_Label)

        #Nullclineの描写
        ax.plot(self.PlotData_dx, self.PlotData_fx, '-', lw = LineWidth)
        ax.plot(self.PlotData_fy, self.PlotData_dy, '-', lw = LineWidth)
        
        #相平面の描写
        if self.Plot_PhaseSpace:
            ax.plot(self.PlotData_x, self.PlotData_y, '-', lw = LineWidth)

        ax.quiver(self.PlotData_X, self.PlotData_Y, self.PlotData_Ex / np.sqrt(pow(self.PlotData_Ex, 2) + pow(self.PlotData_Ey, 2)), self.PlotData_Ey / np.sqrt(pow(self.PlotData_Ex, 2) + pow(self.PlotData_Ey, 2)))

        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        #ax.set_xlim(0,2)
        #ax.set_ylim(-0.5,2.5)
        fig.tight_layout()
        plt.savefig(self.PlotName)
        plt.show()