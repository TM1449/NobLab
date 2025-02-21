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

import Evaluation_Lyapunov
import Task
import Model
import Output

import PreAndPostProcess
import FileAndDir

import Project_Single_LS

#====================================================================
#定数
T = True
F = False

#====================================================================
#プロジェクト

#********************************************************************
"""
リアプノフスペクトル（リアプノフ次元）評価．
"""
if __name__ == '__main__':
    PreAndPostProcess.PreProcess()

    Evaluation_Lyapunov.Evaluation_LS({
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        "Project_F_UsePytorch" : False,                         #Pytorchを使うか（多層リードアウトでは強制的に使用）
        "Project_DeviceCode" : "cpu",                           #CPU/GPUを使うか（CPU -> cpu, GPU -> cuda:n（nはデバイス番号，無くてもいい））
        "Project_DataType" : torch.float,                       #Pytorchのデータ型
        
        "LS_F_OutputLog" : True,                               #経過の出力を行うか
        "LS_D_u" : 3,                                          #入力信号次元
        "LS_D_y" : 3,                                          #出力信号次元
        "LS_Length_Burnin" : 1000,                             #空走用データ時間長
        "LS_Length_Test" : 1000,                               #評価用データ時間長
        "LS_Epsilon" : 1e-7,                                 #摂動
        "LS_T_Task" : Task.Task_NDLorenz,                      #評価用タスク（Type型）
        "LS_T_Model" : Model.Model_NormalESN,                  #モデル（Type型）
        "LS_T_Output" : Project_Single_LS.Output_Single_LS_2024_12_07_01_40,#作図出力（Type型）
        
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        "Task_SinCurve_RK_h" : 0.01,                        #ルンゲクッタ法刻み幅
        
        "Task_MC_Tau" : 5,                                  #遅延量，MCのτ

        "Task_Parity_Tau" : 5,                              #遅延量
        "Task_Parity_MinTerm" : 100,                         #同じ状態を維持する最小期間
        "Task_Parity_MaxTerm" : 200,                        #同じ状態を維持する最大期間
        
        "Task_Rosslor_Scale" : 1 / 30,                      #信号のスケール
        "Task_Rosslor_Mu" : 5.7,                            #レスラー方程式パラメータ
        "Task_Rosslor_Dt" : 0.01,                           #時間スケール
        "Task_Rosslor_A" : 0.005,                            #ギャップジャンクションパラメータ
        "Task_Rosslor_Tau" : 5,                             #どれくらい先を予測するか
        "Task_Rosslor_InitTerm" : 1000,                     #初期状態排除期間
        
        "Task_SRosslor_SelectedInput" :  [T, T, F, F, T,
                                          F, T, F, F, T],   #入力に使用する成分（Tの数がD_u）
        "Task_SRosslor_SelectedOutput" : [F, F, T, T, F,
                                          T, F, T, T, F],   #出力に使用する成分（Tの数がD_y）

        "Task_Lorenz_Scale" : 1 / 50,                       #信号のスケール
        "Task_Lorenz_Sigma" : 10,                           #ローレンツ方程式パラメータ
        "Task_Lorenz_Gamma" : 28,                           #ローレンツ方程式パラメータ
        "Task_Lorenz_Const_B" : 8 / 3,                      #ローレンツ方程式パラメータ
        "Task_Lorenz_Dt" : 0.01,                            #時間スケール
        "Task_Lorenz_A" : 0.001,                             #ギャップジャンクションパラメータ
        "Task_Lorenz_Tau" : 5,                              #どれくらい先を予測するか
        "Task_Lorenz_InitTerm" : 1000,                      #初期状態排除期間
        
        "Task_SLorenz_SelectedInput" :  [T, T, F, F, T,
                                         F, T, F, F, T],    #入力に使用する成分（Tの数がD_u）
        "Task_SLorenz_SelectedOutput" : [F, F, T, T, F,
                                         T, F, T, T, F],    #出力に使用する成分（Tの数がD_y）

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        "Model_NormalESN_D_u" : 3,                          #入力信号次元
        "Model_NormalESN_D_x" : 100,                        #ニューロン数
        "Model_NormalESN_D_y" : 3,                          #出力信号次元
        
        "Model_NormalAESN_D_u" : 5,                         #入力信号次元
        "Model_NormalAESN_D_x" : 100,                       #サブリザバーニューロン数
        "Model_NormalAESN_D_y" : 5,                         #出力信号次元
        
        "Model_HetAESN_D_u" : 3,                         #入力信号次元
        "Model_HetAESN_D_x" : 10,                       #サブリザバーニューロン数（リスト型可能）
        "Model_HetAESN_D_y" : 3,                         #出力信号次元
        "Model_HetAESN_LeakingRate" : [0.01, 0.1, 1],  #リーク率配列->サブリザバーの数(必ずリスト型)
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
        "Model_AESNwithHub_Hub_LeakingRate" : None,             #Hubリーク率配列（None可能）
        "Model_AESNwithHub_Hub_InputScale" : None,              #Hub入力スケーリング配列（None可能）
        "Model_AESNwithHub_Hub_Rho" : None,                     #Hubスペクトル半径配列（None可能）
        "Model_AESNwithHub_Hub_Density" : None,                 #Hub結合密度配列（None可能）

        "Model_ModifiedDeepESN_D_u" : 5,                    #入力信号次元
        "Model_ModifiedDeepESN_D_x" : [100, 100, 100, 100],    #各層ニューロン数（リスト型）
        "Model_ModifiedDeepESN_D_y" : 5,                    #出力信号次元
        "Model_ModifiedDeepESN_LeakingRate" : [1, 0.8, 0.5, 0.1],         #リーク率配列（リスト型可能，None可能）
        "Model_ModifiedDeepESN_InputScale" : None,          #入力スケーリング配列（リスト型可能，None可能）
        "Model_ModifiedDeepESN_Rho" : None,                 #スペクトル半径配列（リスト型可能，None可能）
        "Model_ModifiedDeepESN_Density" : None,             #結合密度配列（リスト型可能，None可能）

        "SubReservoir_LeakingRate" : 1,                     #リーク率
        "SubReservoir_InputScale" : 0.1,                    #入力スケーリング
        "SubReservoir_Rho" : 1,                             #スペクトル半径
        "SubReservoir_Density" : 1,                         #結合密度
        "SubReservoir_ActivationFunc" : np.tanh,            #活性化関数

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        "LinearTransformer_Beta" : 0.2,                     #正規化係数

        "DNN_LearningRate" : 0.001,                         #学習率
        "DNN_MaxLearningLoop" : 1000,                       #最大学習ループ数（使わない場合は0にする）
        "DNN_AimingError" : 0.001,                          #目標（最小）誤差（使わない場合は0にする）

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        "DirPath_Project" : "./Results/Project_Single_LS_2024_12_07_01_40",
        
        "LS_F_OutputCharts" : True,                        #図の出力フラグ
        "LS_F_OutputCharts_MLSWaves" : True,              #入出力＆誤差波形図の出力フラグ

        })()

    PreAndPostProcess.PostProcess()
    
#********************************************************************
#利用可能出力（プロジェクトと対応付けること）
class Output_Single_LS_2024_12_07_01_40(Output.Output):
    """
    LSのデバッグ用
    """
    #コンストラクタ
    def __init__(self, param: dict, parent: any):
        super().__init__(param, parent)

        #パラメータ取得
        self.F_OutputLog = self.Param["LS_F_OutputLog"]                #経過の出力を行うか
        self.Length_Burnin = self.Param["LS_Length_Burnin"]            #空走用データ時間長
        self.Length_Test = self.Param["LS_Length_Test"]                #評価用データ時間長
        self.Length_Total = self.Length_Burnin + self.Length_Test       #全体データ時間長
        
        self.DirPath_Project = self.Param["DirPath_Project"]            #プロジェクトのフォルダパス
        self.ConstractFileTree_Root(self.DirPath_Project)

        #グリッドサーチ用図題用
        self.AxisTag = self.Param["AxisTag"] if "AxisTag" in self.Param else ""

        #各種図の出力フラグ
        self.F_OutputCharts = self.Param["LS_F_OutputCharts"]
        self.F_OutputCharts_MLSWaves = self.Param["LS_F_OutputCharts_MLSWaves"]

    #本体
    def __call__(self, result_param): 
        #コンソール結果出力
        if self.F_OutputLog : print("+++ Outputing Results +++")
        if self.F_OutputLog : 
            print("< LS >")
            for i in range(len(result_param["LS_R_LS"])):
                print(result_param["LS_R_PathShowName"][i] + 
                      "(" + result_param["LS_R_PathName"][i] + ") : " + 
                      str(result_param["LS_R_LS"][i]))
            
            print("< LD >")
            for i in range(len(result_param["LS_R_LD"])):
                print(result_param["LS_R_PathShowName"][i] + 
                      "(" + result_param["LS_R_PathName"][i] + ") : " + 
                      str(result_param["LS_R_LD"][i]))
                
        #作図
        if self.F_OutputLog : print("+++ Making Charts +++")
        if self.F_OutputCharts:
            #結果のフォルダ
            self.NumOfSubReservoirs = len(result_param["LS_R_MLS"])
            self.ConstractFileTree_Charts_Branch(True)
            self.Save_Charts_Param(result_param, "")
            
            #入力波形と瞬時リアプノフ指数
            if self.F_OutputCharts_MLSWaves:
                FigSize = (16, 9)                   #アスペクト比
                FontSize_Label = 24                 #ラベルのフォントサイズ
                FontSize_Title = 24                 #タイトルのフォントサイズ
                LineWidth = 3                       #線の太さ
                FileFormat = ".png"#".pdf"          #ファイルフォーマット
                
                #出力部分切り取り
                start = self.Length_Burnin
                end = self.Length_Burnin + self.Length_Test
                T = np.array(result_param["LS_R_T"][start : end])
                U = np.array(result_param["LS_R_U"][start : end])

                for i in range(self.NumOfSubReservoirs):
                    MLS = result_param["LS_R_MLS"][i]

                    #LS波形
                    fig = plt.figure(figsize = FigSize)
                    ax = fig.add_subplot(2, 1, 1)
                    Title = "U Wave" + self.AxisTag          #図題
                    ax.set_title(Title, fontsize = FontSize_Title)
                    #ax.set_xlabel("Time Step", fontsize = FontSize_Label)
                    ax.set_ylabel("Input Signal", fontsize = FontSize_Label)
                    ax.grid(True)
                    ax.plot(T, U, "skyblue", label = "u", lw = LineWidth)
                    ax.legend()

                    ax = fig.add_subplot(2, 1, 2)
                    Title = "Moument LS Wave" + self.AxisTag          #図題
                    ax.set_title(Title, fontsize = FontSize_Title)
                    ax.set_xlabel("Time Step", fontsize = FontSize_Label)
                    ax.set_ylabel("Moument LS", fontsize = FontSize_Label)
                    ax.grid(True)
                    for j in range(MLS.shape[1]):
                        ax.plot(T, MLS[:, j], label = result_param["LS_R_PathShowName"][i] + "_" + str(j), lw = LineWidth)
                    ax.legend()
                    fig.savefig(self.Plt_Charts_MLSWaves[i].Path + FileFormat)
                
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
        self.Dir_Results_Branch = self.Dir_Results.AddChild(FileAndDir.DirNode("LS_"), FileAndDir.Manager.getDate(), tag)

        self.CSV_Charts_Param = self.Dir_Results_Branch.AddChild(FileAndDir.FileNode_dict("ResultAndParam"))
        self.Plt_Charts_MLSWaves = [None for _ in range(self.NumOfSubReservoirs)]
        for i in range(self.NumOfSubReservoirs):
            self.Plt_Charts_MLSWaves[i] = self.Dir_Results_Branch.AddChild(FileAndDir.FileNode_plt("MLSWaves_"+str(i)))
        