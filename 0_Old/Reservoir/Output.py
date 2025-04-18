#####################################################################
#信川研ESNプロジェクト用
#制作者：中村亮介 吉原悠斗 吉田聡太 飯沼貴大
#作成日：2023/05/24
"""
本体

maru
"""

#####################################################################

#====================================================================
#＜このプログラムのTodoメモ＞
#・
#・
#・

#====================================================================
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import FileAndDir

#====================================================================
#作図出力

#********************************************************************
#継承元
class Output:
    """
    作図出力クラス
    全ての作図出力はこれを継承．
    親オブジェクトはプロジェクトもしくは評価．
    実験結果をresult_paramで受け出力．
    """
    #コンストラクタ
    def __init__(self, param: dict, parent: any):
        self.Param = param
        
        self.Parent = parent                            #親オブジェクト
        
    #本体
    def __call__(self, result_param): pass
    
#********************************************************************
#利用可能出力（プロジェクトと対応付けること）
class Output_Single_NRMSE_2023_04_19_15_25(Output):
    """
    出力例
    命名規則は Output_[プロジェクト名/Single]_[評価名]_YYYY_MM_DD_HH_MM_[必要であれば識別タグ]
    昔の有意有用な結果の作図出力は全て残し，新しいものは別のクラスで作ること．
    何の実験の出力なのかここにコメントを書くこと．

    モデルとNRMSEのデバッグ用．
    """
    #コンストラクタ
    def __init__(self, param: dict, parent: any):
        super().__init__(param, parent)

        #パラメータ取得
        self.F_OutputLog = self.Param["NRMSE_F_OutputLog"]          #経過の出力を行うか
        self.Length_Burnin = self.Param["NRMSE_Length_Burnin"]      #空走用データ時間長
        self.Length_Train = self.Param["NRMSE_Length_Train"]        #学習用データ時間長
        self.Length_Test = self.Param["NRMSE_Length_Test"]          #評価用データ時間長
        self.Length_Total = self.Length_Burnin + self.Length_Train + self.Length_Test#全体データ時間長

        #フォルダ構造
        self.DirPath_Project = self.Param["DirPath_Project"]        #プロジェクトのフォルダパス
        self.ConstractFileTree_Root(self.DirPath_Project)

        #グリッドサーチ用図題用
        self.AxisTag = self.Param["AxisTag"] if "AxisTag" in self.Param else ""

        #各種図の出力フラグ
        self.F_OutputCharts = self.Param["NRMSE_F_OutputCharts"]
        self.F_OutputCharts_UYYdEWaves = self.Param["NRMSE_F_OutputCharts_UYYdEWaves"]
        
    #本体
    def __call__(self, result_param): 
        #コンソール結果出力
        if self.F_OutputLog : print("+++ Outputing Results +++")
        if self.F_OutputLog : 
            print("NRMSE : " + str(result_param["NRMSE_R_NRMSE"]))
            print("LogNRMSE : " + str(result_param["NRMSE_R_LogNRMSE"]))
            print("TimeForTraining : " + str(result_param["NRMSE_R_TimeForTraining"]))
            print("TimeForTesting : " + str(result_param["NRMSE_R_TimeForTesting"]))

        #作図
        if self.F_OutputLog : print("+++ Making Charts +++")
        if self.F_OutputCharts:
            #結果のフォルダ
            self.ConstractFileTree_Charts_Branch(True)
            self.Save_Charts_Param(result_param, "")

            #入出力波形
            if self.F_OutputCharts_UYYdEWaves:
                FigSize = (16, 9)                   #アスペクト比
                FontSize_Label = 36                 #ラベルのフォントサイズ
                FontSize_Title = 36                 #タイトルのフォントサイズ
                LineWidth = 2                       #線の太さ
                FileFormat = ".png"#".pdf"          #ファイルフォーマット
                
                #出力部分切り取り
                start = self.Length_Burnin + self.Length_Train
                end = self.Length_Burnin + self.Length_Train + self.Length_Test
                T = np.array(result_param["NRMSE_R_T"][start : end])
                TS = np.array(result_param["NRMSE_R_T"][start : end])
                U = np.array(result_param["NRMSE_R_U"][start : end])
                Y = np.array(result_param["NRMSE_R_Y"][start : end])
                Y_d = np.array(result_param["NRMSE_R_Y_d"][start : end])
                E = np.array(result_param["NRMSE_R_E"][start : end])
                RS = np.array(result_param["Reservoir_Move"][:,start : end])

                #各種波形
                Title = "UYYdWaves" + self.AxisTag  #図題
                fig = plt.figure(figsize = FigSize)
                ax = fig.add_subplot(1, 1, 1)
                ax.set_title(Title, fontsize = FontSize_Title)
                ax.set_xlabel("Time Step", fontsize = FontSize_Label)
                ax.set_ylabel("", fontsize = FontSize_Label)
                ax.grid(True)
                ax.plot(T, U, "skyblue", label = "u", lw = LineWidth)
                ax.plot(T, Y, "lawngreen", label = "y", lw = LineWidth)
                ax.plot(T, Y_d, "orange", label = "y_d", lw = LineWidth)
                ax.legend()
                fig.savefig(self.Plt_Charts_UYYdWaves.Path + FileFormat)

                #誤差波形
                Title = "ErrorWaves" + self.AxisTag     #図題
                fig = plt.figure(figsize = FigSize)
                ax = fig.add_subplot(1, 1, 1)
                ax.set_title(Title, fontsize = FontSize_Title)
                ax.set_xlabel("Time Step", fontsize = FontSize_Label)
                ax.set_ylabel("", fontsize = FontSize_Label)
                ax.grid(True)
                ax.plot(T, E, "skyblue", label = "RMSE", lw = LineWidth)
                ax.legend()
                fig.savefig(self.Plt_Charts_ErrorWaves.Path + FileFormat)

                #リザバー層内の状態
                Title =  None
                fig = plt.figure(figsize = FigSize)
                ax = fig.add_subplot(1, 1, 1)
                plt.tick_params(labelsize=18)
                ax.set_title(Title, fontsize = FontSize_Title)
                ax.set_xlabel("Time Step", fontsize = FontSize_Label)
                ax.set_ylabel(r'$x_{R}$', fontsize = FontSize_Label)
                ax.grid(True)
                cmap = plt.get_cmap("tab10")
                for i in range(10):
                    ax.plot(T[:1000], RS[i,:1000], color = cmap(i), label = r'$x_{R}$', lw = LineWidth)
                ax.legend()
                plt.tight_layout()
                fig.savefig(self.Plt_Charts_MOVEWaves.Path + FileFormat)

                plt.close()
    
    #以下ファイルとフォルダ単位のセーブとロード        
    def Load_Project(self, save_type: str):
        pass
        
    def Save_Project(self, save_type: str):
        pass
        
    def Load_Charts_Param(self, save_type: str):
        pass

    def Save_Charts_Param(self, result_param: dict, save_type: str):
        self.CSV_Charts_Param.Save(result_param)
        
    #以下フォルダ構造
    def ConstractFileTree_Root(self, root: str):
        if type(root) is str:
            self.Dir_Project = FileAndDir.RootNode(root)
        else:
            self.Dir_Project = root
        
        self.ConstractFileTree_Project()
        
    def ConstractFileTree_Project(self):
        self.Dir_Results = self.Dir_Project.AddChild(FileAndDir.LogDirNode("."))
        
    def ConstractFileTree_Charts_Branch(self, f_new: bool = False, tag: str = ""):
        self.Dir_Results_Branch = self.Dir_Results.AddChild(FileAndDir.DirNode("NRMSE_"), FileAndDir.Manager.getDate(), tag)

        self.CSV_Charts_Param = self.Dir_Results_Branch.AddChild(FileAndDir.FileNode_dict("ResultAndParam"))
        self.Plt_Charts_UYYdWaves = self.Dir_Results_Branch.AddChild(FileAndDir.FileNode_plt("UYYdWaves"))
        self.Plt_Charts_ErrorWaves = self.Dir_Results_Branch.AddChild(FileAndDir.FileNode_plt("ErrorWaves"))
        self.Plt_Charts_MOVEWaves = self.Dir_Results_Branch.AddChild(FileAndDir.FileNode_plt("MOVEWaves"))
        
#--------------------------------------------------------------------
class Output_Single_MC_2023_05_25_13_28(Output):
    """
    MCのデバッグ用
    """
    #コンストラクタ
    def __init__(self, param: dict, parent: any):
        super().__init__(param, parent)

        #パラメータ取得
        self.F_OutputLog = self.Param["MemoryCapacity_F_OutputLog"]         #経過の出力を行うか
        self.Length_Burnin = self.Param["MemoryCapacity_Length_Burnin"]     #空走用データ時間長
        self.Length_Train = self.Param["MemoryCapacity_Length_Train"]       #学習用データ時間長
        self.Length_Test = self.Param["MemoryCapacity_Length_Test"]         #評価用データ時間長
        self.Length_Total = self.Length_Burnin + self.Length_Train + self.Length_Test#全体データ時間長
        
        #フォルダ構造
        self.DirPath_Project = self.Param["DirPath_Project"]        #プロジェクトのフォルダパス
        self.ConstractFileTree_Root(self.DirPath_Project)

        #グリッドサーチ用図題用
        self.AxisTag = self.Param["AxisTag"] if "AxisTag" in self.Param else ""

        #各種図の出力フラグ
        self.F_OutputCharts = self.Param["MemoryCapacity_F_OutputCharts"]
        self.F_OutputCharts_MCGraph = self.Param["MemoryCapacity_F_OutputCharts_MCGraph"]

    #本体
    def __call__(self, result_param): 
        #コンソール結果出力
        if self.F_OutputLog : print("+++ Outputing Results +++")
        if self.F_OutputLog : 
            print("MC : " + str(result_param["MemoryCapacity_R_MC"]))
            
        #作図
        if self.F_OutputLog : print("+++ Making Charts +++")
        if self.F_OutputCharts:
            #結果のフォルダ
            self.Date = FileAndDir.Manager.getDate()
            self.ConstractFileTree_Charts_Branch(True)
            self.Save_Charts_Param(result_param, "")

            #入出力波形
            if self.F_OutputCharts_MCGraph:
                FigSize = (16, 9)                   #アスペクト比
                FontSize_Label = 24                 #ラベルのフォントサイズ
                FontSize_Title = 24                 #タイトルのフォントサイズ
                LineWidth = 3                       #線の太さ
                FileFormat = ".png"#".pdf"          #ファイルフォーマット
                
                #出力データ
                Tau = result_param["MemoryCapacity_R_Tau"]
                MC_Tau = result_param["MemoryCapacity_R_MC_Tau"]

                #MCカーブ
                Title = "MC" + self.AxisTag         #図題
                fig = plt.figure(figsize = FigSize)
                ax = fig.add_subplot(1, 1, 1)
                ax.set_title(Title, fontsize = FontSize_Title)
                ax.set_xlabel(r"$\tau$", fontsize = FontSize_Label)
                ax.set_ylabel("Memory Capacity", fontsize = FontSize_Label)
                ax.grid(True)
                ax.plot(Tau, MC_Tau, "skyblue", label = r"$\mathrm{MC}_{\tau}$", lw = LineWidth)
                ax.legend()
                fig.savefig(self.Plt_Charts_MCGraph.Path + FileFormat)
                
                plt.close()
            
    #以下ファイルとフォルダ単位のセーブとロード        
    def Load_Project(self, save_type: str):
        pass
        
    def Save_Project(self, save_type: str):
        pass
        
    def Load_Charts_Param(self, save_type: str):
        pass

    def Save_Charts_Param(self, result_param: dict, save_type: str):
        self.CSV_Charts_Param.Save(result_param)
        
    #以下フォルダ構造
    def ConstractFileTree_Root(self, root: str):
        if type(root) is str:
            self.Dir_Project = FileAndDir.RootNode(root)
        else:
            self.Dir_Project = root
        
        self.ConstractFileTree_Project()
        
    def ConstractFileTree_Project(self):
        self.Dir_Results = self.Dir_Project.AddChild(FileAndDir.LogDirNode("."))
        
    def ConstractFileTree_Charts_Branch(self, f_new: bool = False, tag: str = ""):
        self.Dir_Results_Branch = self.Dir_Results.AddChild(FileAndDir.DirNode("MC_"), self.Date, tag)

        self.CSV_Charts_Param = self.Dir_Results_Branch.AddChild(FileAndDir.FileNode_dict("ResultAndParam"))
        self.Plt_Charts_MCGraph = self.Dir_Results_Branch.AddChild(FileAndDir.FileNode_plt("MCGraph"))
        