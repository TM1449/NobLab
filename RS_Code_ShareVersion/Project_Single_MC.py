#####################################################################
#信川研ESNプロジェクト用
#制作者：中村亮介 吉原悠斗 吉田聡太 飯沼貴大
#作成日：2024/12/06
#####################################################################

#====================================================================
#＜このプログラムのTodoメモ＞
#・
#・
#・

#====================================================================
import numpy as np
import torch
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import Evaluation_Performance
import Task
import Model
import Output

import PreAndPostProcess
import FileAndDir

import Project_Single_MC

#====================================================================
#定数
T = True
F = False

#====================================================================
#プロジェクト

#********************************************************************
"""
メモリキャパシティー評価．
"""
if __name__ == '__main__':
    PreAndPostProcess.PreProcess()

    Evaluation_Performance.Evaluation_MC({
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        "Project_F_UsePytorch" : False,                     #Pytorchを使うか（多層リードアウトでは強制的に使用）
        "Project_DeviceCode" : "cpu",                       #CPU/GPUを使うか（CPU -> cpu, GPU -> cuda:n（nはデバイス番号，無くてもいい））
        "Project_DataType" : torch.float,                   #Pytorchのデータ型
        
        "MemoryCapacity_F_OutputLog" : True,                #経過の出力を行うか
        "MemoryCapacity_D_u" : 1,                           #入力信号次元
        "MemoryCapacity_D_y" : 1,                           #出力信号次元
        "MemoryCapacity_Length_Burnin" : 1000,              #空走用データ時間長
        "MemoryCapacity_Length_Train" : 5000,              #学習用データ時間長
        "MemoryCapacity_Length_Test" : 1000,                #評価用データ時間長
        "MemoryCapacity_MaxTau" : 100,                       #評価する最大遅延
        "MemoryCapacity_T_Task" : Task.Task_MC,                                     #評価用タスク（Type型）
        "MemoryCapacity_T_Model" : Model.Model_NormalESN,                           #モデル（Type型）
        "MemoryCapacity_T_Output" : Project_Single_MC.Output_Single_MC_2023_05_25_13_28,#作図出力（Type型）
        
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        "Task_MC_Tau" : 5,                                  #遅延量，MCのτ

        "Task_Parity_Tau" : 5,                              #遅延量
        "Task_Parity_MinTerm" : 100,                         #同じ状態を維持する最小期間
        "Task_Parity_MaxTerm" : 200,                        #同じ状態を維持する最大期間
        
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        "Model_NormalESN_D_u" : 1,                          #入力信号次元
        "Model_NormalESN_D_x" : 100,                        #ニューロン数
        "Model_NormalESN_D_y" : 1,                          #出力信号次元
        
        "Model_NormalAESN_D_u" : 1,                         #入力信号次元
        "Model_NormalAESN_D_x" : 100,                       #サブリザバーニューロン数
        "Model_NormalAESN_D_y" : 1,                         #出力信号次元

        "Model_HetAESN_D_u" : 5,                         #入力信号次元
        "Model_HetAESN_D_x" : 100,                       #サブリザバーニューロン数（リスト型可能）
        "Model_HetAESN_D_y" : 5,                         #出力信号次元
        "Model_HetAESN_LeakingRate" : [0.01, 0.1, 1.0],  #リーク率配列->サブリザバーの数(必ずリスト型)
        "Model_HetAESN_InputScale" : None,               #入力スケーリング配列（リスト型可能，None可能）
        "Model_HetAESN_Rho" : None,                      #スペクトル半径配列（リスト型可能，None可能）
        "Model_HetAESN_Density" : None,                  #結合密度配列（リスト型可能，None可能）

        "Model_AESNwithHub_D_u" : 5,                        #入力信号次元
        "Model_AESNwithHub_D_h" : 100,                      #サブリザバーニューロン数
        "Model_AESNwithHub_D_x" : 100,                      #ハブリザバーニューロン数
        "Model_AESNwithHub_D_y" : 5,                        #出力信号次元
        "Model_AESNwithHub_F_Use_U2HubConnection" : False,  #Hubへの入力信号使用の有無
        "Model_AESNwithHub_F_Use_x2zConnection" : True,     #出力へのHub状態使用の有無
        "Model_AESNwithHub_F_Use_x2AESNConnection" : False, #AESNへのHub信号使用の有無
        "Model_AESNwithHub_F_Use_AverageHInHub" : False,    #HubでHの平均をするか
        "Model_AESNwithHub_Hub_LeakingRate" : None,         #Hubリーク率配列（None可能）
        "Model_AESNwithHub_Hub_InputScale" : None,          #Hub入力スケーリング配列（None可能）
        "Model_AESNwithHub_Hub_Rho" : None,                 #Hubスペクトル半径配列（None可能）
        "Model_AESNwithHub_Hub_Density" : None,             #Hub結合密度配列（None可能）
        
        "Model_ModifiedDeepESN_D_u" : 1,                    #入力信号次元
        "Model_ModifiedDeepESN_D_x" : [100,100,100,100,100,100,100,100,100,100],    #各層ニューロン数（リスト型）
        "Model_ModifiedDeepESN_D_y" : 1,                    #出力信号次元
        "Model_ModifiedDeepESN_LeakingRate" : None,         #リーク率配列（リスト型可能，None可能）
        "Model_ModifiedDeepESN_InputScale" : None,          #入力スケーリング配列（リスト型可能，None可能）
        "Model_ModifiedDeepESN_Rho" : None,                 #スペクトル半径配列（リスト型可能，None可能）
        "Model_ModifiedDeepESN_Density" : None,             #結合密度配列（リスト型可能，None可能）

        "SubReservoir_LeakingRate" : 1,                     #リーク率
        "SubReservoir_InputScale" : 0.1,                    #入力スケーリング
        "SubReservoir_Rho" : 1,                             #スペクトル半径
        "SubReservoir_Density" : 1,                         #結合密度
        "SubReservoir_ActivationFunc" : np.tanh,            #活性化関数

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        "LinearTransformer_Beta" : 0.2,                      #正規化係数
        
        "DNN_LearningRate" : 0.001,                         #学習率
        "DNN_MaxLearningLoop" : 1000,                       #最大学習ループ数（使わない場合は0にする）
        "DNN_AimingError" : 0.001,                          #目標（最小）誤差（使わない場合は0にする）

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        "DirPath_Project" : "./Results/Project_Single_MC_2023_05_25_13_28",
        
        "MemoryCapacity_F_OutputCharts" : True,             #図の出力フラグ
        "MemoryCapacity_F_OutputCharts_MCGraph" : True,     #MC曲線の出力フラグ

        })()

    PreAndPostProcess.PostProcess()
    
#********************************************************************
#利用可能出力（プロジェクトと対応付けること）
class Output_Single_MC_2023_05_25_13_28(Output.Output):
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
        