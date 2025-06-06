#####################################################################
#信川研ESNプロジェクト用
#制作者：中村亮介 吉原悠斗 吉田聡太 飯沼貴大 田中勝規
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
import matplotlib.pyplot as plt
import seaborn as sns

import FileAndDir

import Task
import Model
import Output
import Evaluation

import GridSearch

#====================================================================
#グリッドサーチ

#********************************************************************
#********************************************************************
#GSパラメータ
#====================================================================
def Project_GridSearch_SishuESN_NRMSEAndMC():
    """ 
    Sishu提案モデル（結合形態のみ指定、強度は乱数）のNRMSEやMC測定
    """
    GridSearch.Normal({
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        "Project_F_NRMSE" : True,                                   #NRMSEを調査するか
        "Project_F_MemoryCapacity" : True,                          #MCを調査するか

        "Project_F_OutputResults" : True,                           #各評価地点でパラメータを出力するか

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #軸
        #[0.5, 0.4, 0.3, 0.2, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001,0.0009, 0.0008, 0.0007, 0.0006, 0.0005, 0.0004, 0.0003, 0.0002, 0.0001,0.00009, 0.00008, 0.00007, 0.00006, 0.00005, 0.00004, 0.00003, 0.00002, 0.00001]
        #[0.1, 0.099, 0.098, 0.097, 0.096, 0.095, 0.094, 0.093, 0.092, 0.091, 0.09, 0.089, 0.088, 0.087, 0.086, 0.085, 0.084, 0.083, 0.082, 0.081, 0.08, 0.079, 0.078, 0.077, 0.076, 0.075, 0.074, 0.073, 0.072, 0.071, 0.07, 0.069, 0.068, 0.067, 0.066, 0.065, 0.064, 0.063, 0.062, 0.061, 0.06, 0.059, 0.058, 0.057, 0.056, 0.055, 0.054, 0.053, 0.052, 0.051, 0.05, 0.049, 0.048, 0.047, 0.046, 0.045, 0.044, 0.043, 0.042, 0.041, 0.04, 0.039, 0.038, 0.037, 0.036, 0.035, 0.034, 0.033, 0.032, 0.031, 0.03, 0.029, 0.028, 0.027, 0.026, 0.025, 0.024, 0.023, 0.022, 0.021, 0.02, 0.019, 0.018, 0.017, 0.016, 0.015, 0.014, 0.013, 0.012, 0.011, 0.01, 0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001]
        
        "_Param_Rho" : [0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001, 0.0009, 0.0008, 0.0007, 0.0006, 0.0005, 0.0004, 0.0003, 0.0002, 0.0001],
        "_Param_k" : [-10.0, -9.5, -9.0, -8.5, -8.0, -7.5, -7.0, -6.5, -6.0, -5.5, -5.0, -4.5, -4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0],
        "_Param_Model" : ["SishuESN"],

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        "GridSearch_MachineName" : "SishuESN",                            #計算機名
        "GridSearch_StartPoint" : 0,                                #担当開始ポイントインデックス
        "GridSearch_EndPoint" : 11480,                                #担当終了ポイントインデックス(MAX:4410(21*21*10))
        
        "GridSearch_MultiThread" : 12,                               #スレッド数（0で逐次処理）初期値:2
        "GridSearch_MaxNumberInOneFile" : 100,                        #１ファイルの最大のポイント数 初期値:5
        "GridSearch_MaxNumberInOnePool" : 1000,                       #１プール（並列する）最大のポイント数（この分メモリを消費） 初期値:50
        "GridSearch_NumberOfSamples" : 10,                           #サンプル数
        "GridSearch_ProjectName" : "NRMSEAndMC_SishuESN",                    #プロジェクト名
        "GridSearch_ProjectDate" : "2024_06_01_10_30",              #プロジェクト日時
        "GridSearch_T_Process" : Process_SishuESN_GridSearch,     #GS処理指定
        "GridSearch_T_Output" : OutputLog_SishuESN_2024_06_01_10_30             #GS出力処理指定
        })()

def Project_GridSearch_SishuESN_NRMSEAndMC_Tau():
    """ 
    Sishu提案モデル（結合形態のみ指定、強度は乱数）のNRMSEやMC測定
    """
    GridSearch.Normal({
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        "Project_F_NRMSE" : True,                                   #NRMSEを調査するか
        "Project_F_MemoryCapacity" : True,                          #MCを調査するか

        "Project_F_OutputResults" : True,                           #各評価地点でパラメータを出力するか

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #軸
        "_Param_Tau" : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        "_Param_Model" : ["SishuESN"],

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        "GridSearch_MachineName" : "SishuESN",                            #計算機名
        "GridSearch_StartPoint" : 0,                                #担当開始ポイントインデックス
        "GridSearch_EndPoint" : 150,                                #担当終了ポイントインデックス(MAX:4410(21*21*10))
        
        "GridSearch_MultiThread" : 12,                               #スレッド数（0で逐次処理）初期値:2
        "GridSearch_MaxNumberInOneFile" : 100,                        #１ファイルの最大のポイント数 初期値:5
        "GridSearch_MaxNumberInOnePool" : 1000,                       #１プール（並列する）最大のポイント数（この分メモリを消費） 初期値:50
        "GridSearch_NumberOfSamples" : 10,                           #サンプル数
        "GridSearch_ProjectName" : "NRMSEAndMC_SishuESN",                    #プロジェクト名
        "GridSearch_ProjectDate" : "2024_06_01_10_30",              #プロジェクト日時
        "GridSearch_T_Process" : Process_SishuESN_Tau_GridSearch,     #GS処理指定
        "GridSearch_T_Output" : Output_Tau_SishuESN_2024_06_01_10_30             #GS出力処理指定
        })()

#********************************************************************
#********************************************************************
#GS処理指定
#====================================================================
class Process_SishuESN_GridSearch:
    """ 
    上記のパラメータに対応する処理
    """
    #コンストラクタ
    def __init__(self, param: dict, grid_search: any):
        self.Param = param
        self.GridSearch = grid_search
        
    #メイン処理
    def __call__(self, chank_i, param):
        Results = chank_i.copy()
        Results.update(self.Exp_GridWorld2D(chank_i, param))
        
        #表示
        print(("---Index in Chank : %d / %d, Index : %d, Sample : %d\n"
                + "<%s>\n"
                + "NRMSE : %f, LogNRMSE : %f, TimeForTraining : %f, TimeForTesting : %f, MC : %f\n")
                    %(Results["IndexInChank"], Results["NumPointsInChank"], Results["Index"], Results["Samples"],
                    self.getTag(chank_i),
                    Results["NRMSE_R_NRMSE"] if "NRMSE_R_NRMSE" in Results else 0,
                    Results["NRMSE_R_LogNRMSE"] if "NRMSE_R_LogNRMSE" in Results else 0,
                    Results["NRMSE_R_TimeForTraining"] if "NRMSE_R_TimeForTraining" in Results else 0,
                    Results["NRMSE_R_TimeForTesting"] if "NRMSE_R_TimeForTesting" in Results else 0,
                    Results["MemoryCapacity_R_MC"] if "MemoryCapacity_R_MC" in Results else 0))
    
        return Results

    #ポイントのパラメータ設定
    def Exp_GridWorld2D(self, chank_i, gs_param):
        #各軸パラメータ
        Param_Rho = chank_i["Rho"]
        Param_k = chank_i["k"]
        if chank_i["Model"] == "SishuESN":
            Param_Model = Model.Model_SishuESN
            
        #共通パラメータ
        Param = {
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
        
            "Task_Lorenz_Scale" : 1 / 50,                       #信号のスケール
            "Task_Lorenz_Sigma" : 10,                           #ローレンツ方程式パラメータ
            "Task_Lorenz_Gamma" : 28,                           #ローレンツ方程式パラメータ
            "Task_Lorenz_Const_B" : 8 / 3,                      #ローレンツ方程式パラメータ
            "Task_Lorenz_Dt" : 0.01,                            #時間スケール
            "Task_Lorenz_A" : 0.001,                             #ギャップジャンクションパラメータ
            "Task_Lorenz_Tau" : 5,                              #どれくらい先を予測するか
            "Task_Rosslor_InitTerm" : 1000,                     #初期状態排除期間

            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            "Model_SishuESN_D_u" : 1,                  #入力信号次元
            "Model_SishuESN_D_x" : 100,                #ニューロン数
            "Model_SishuESN_D_y" : 1,                  #出力信号次元
            
            "Model_SishuESN_InputScale" : 0.1,                 #入力スケーリング
            "Model_SishuESN_sigma" : "False",                   #リング・ネットワークの結合強度
            "Model_SishuESN_mu" : "False",                     #スター・ネットワークの結合強度
            "Model_SishuESN_k" : Param_k,

            "Model_SishuESN_Rho" : Param_Rho,                             #スペクトル半径
            "SishuReservoir_Density" : 1,                         #結合密度
            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            "LinerTransformer_Beta" : 0.2,                      #正規化係数
            }
    
        Results = dict()
        #NRMSE評価
        if gs_param["Project_F_NRMSE"]:
            param = Param.copy()
            param.update({
                #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                "NRMSE_F_OutputLog" : False,                        #経過の出力を行うか
                "NRMSE_D_u" : 1,                            #入力信号次元
                "NRMSE_D_y" : 1,                            #出力信号次元
                "NRMSE_Length_Burnin" : 1000,                       #空走用データ時間長
                "NRMSE_Length_Train" : 20000,                       #学習用データ時間長
                "NRMSE_Length_Test" : 5000,                         #評価用データ時間長
                "NRMSE_T_Task" : Task.Task_NDRosslor,                                #評価用タスク（Type型）
                "NRMSE_T_Model" : Param_Model,                                      #モデル（Type型）
                "NRMSE_T_Output" : Output.Output_Single_NRMSE_2023_04_19_15_25,     #作図出力（Type型）
        
                #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                "DirPath_Project" : self.GridSearch.Dir_Points_Branch,              #GSのパスを使用
                "AxisTag" : self.getTag(chank_i),                                   #GSのパラメータを図題に使用

                "NRMSE_F_OutputCharts" : False,                      #図の出力フラグ
                "NRMSE_F_OutputCharts_UYYdEWaves" : False,           #入出力＆誤差波形図の出力フラグ

                })
            Results.update(Evaluation.Evaluation_NRMSE(param)())
    
        #MC評価
        if gs_param["Project_F_MemoryCapacity"]:
            param = Param.copy()
            param.update({
                #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                "MemoryCapacity_F_OutputLog" : False,               #経過の出力を行うか
                "MemoryCapacity_D_u" : 1,                   #入力信号次元
                "MemoryCapacity_D_y" : 1,                   #出力信号次元            
                "MemoryCapacity_Length_Burnin" : 1000,              #空走用データ時間長
                "MemoryCapacity_Length_Train" : 5000,               #学習用データ時間長
                "MemoryCapacity_Length_Test" : 1000,                #評価用データ時間長
                "MemoryCapacity_MaxTau" : 100,                      #評価する最大遅延
                "MemoryCapacity_T_Task" : Task.Task_MC,                                     #評価用タスク（Type型）
                "MemoryCapacity_T_Model" : Param_Model,                                     #モデル（Type型）
                "MemoryCapacity_T_Output" : Output.Output_Single_MC_2023_05_25_13_28,       #作図出力（Type型）
        
                #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                "DirPath_Project" : self.GridSearch.Dir_Points_Branch,                      #GSのパスを使用
                "AxisTag" : self.getTag(chank_i),                                           #GSのパラメータを図題に使用

                "MemoryCapacity_F_OutputCharts" : False,             #図の出力フラグ
                "MemoryCapacity_F_OutputCharts_MCGraph" : False,     #MC曲線の出力フラグ

                })
            Results.update(Evaluation.Evaluation_MC(param)())
        
        #地点のパラメータを改めて設定
        if gs_param["Project_F_OutputResults"]:
            self.GridSearch.CSV_Point_Param.Save(Results)

        return Results

    #軸パラメータを文字列化
    def getTag(self, param: dict) -> str:
        return "Rho : " + str(param["Rho"]) + " k : " + str(param["k"]) + " Model : " + param["Model"]

class Process_SishuESN_Tau_GridSearch:
    """ 
    上記のパラメータに対応する処理
    """
    #コンストラクタ
    def __init__(self, param: dict, grid_search: any):
        self.Param = param
        self.GridSearch = grid_search
        
    #メイン処理
    def __call__(self, chank_i, param):
        Results = chank_i.copy()
        Results.update(self.Exp_GridWorld2D(chank_i, param))
        
        #表示
        print(("---Index in Chank : %d / %d, Index : %d, Sample : %d\n"
                + "<%s>\n"
                + "NRMSE : %f, LogNRMSE : %f, TimeForTraining : %f, TimeForTesting : %f, MC : %f\n")
                    %(Results["IndexInChank"], Results["NumPointsInChank"], Results["Index"], Results["Samples"],
                    self.getTag(chank_i),
                    Results["NRMSE_R_NRMSE"] if "NRMSE_R_NRMSE" in Results else 0,
                    Results["NRMSE_R_LogNRMSE"] if "NRMSE_R_LogNRMSE" in Results else 0,
                    Results["NRMSE_R_TimeForTraining"] if "NRMSE_R_TimeForTraining" in Results else 0,
                    Results["NRMSE_R_TimeForTesting"] if "NRMSE_R_TimeForTesting" in Results else 0,
                    Results["MemoryCapacity_R_MC"] if "MemoryCapacity_R_MC" in Results else 0))
    
        return Results

    #ポイントのパラメータ設定
    def Exp_GridWorld2D(self, chank_i, gs_param):
        #各軸パラメータ
        Param_Tau = chank_i["Tau"]
        if chank_i["Model"] == "SishuESN":
            Param_Model = Model.Model_SishuESN
            
        #共通パラメータ
        Param = {
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
        
            "Task_Lorenz_Scale" : 1 / 50,                       #信号のスケール
            "Task_Lorenz_Sigma" : 10,                           #ローレンツ方程式パラメータ
            "Task_Lorenz_Gamma" : 28,                           #ローレンツ方程式パラメータ
            "Task_Lorenz_Const_B" : 8 / 3,                      #ローレンツ方程式パラメータ
            "Task_Lorenz_Dt" : 0.01,                            #時間スケール
            "Task_Lorenz_A" : 0.001,                             #ギャップジャンクションパラメータ
            "Task_Lorenz_Tau" : Param_Tau,                              #どれくらい先を予測するか
            "Task_Rosslor_InitTerm" : 1000,                     #初期状態排除期間

            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            "Model_SishuESN_D_u" : 1,                  #入力信号次元
            "Model_SishuESN_D_x" : 100,                #ニューロン数
            "Model_SishuESN_D_y" : 1,                  #出力信号次元
            
            "Model_SishuESN_InputScale" : 0.1,                 #入力スケーリング
            "Model_SishuESN_sigma" : False,                   #リング・ネットワークの結合強度
            "Model_SishuESN_mu" : False,                     #スター・ネットワークの結合強度
            "Model_SishuESN_k" : -3.2,

            "Model_SishuESN_Rho" : 0.0032,                             #スペクトル半径
            "SishuReservoir_Density" : 1,                         #結合密度
            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            "LinerTransformer_Beta" : 0.2,                      #正規化係数
            }
    
        Results = dict()
        #NRMSE評価
        if gs_param["Project_F_NRMSE"]:
            param = Param.copy()
            param.update({
                #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                "NRMSE_F_OutputLog" : False,                        #経過の出力を行うか
                "NRMSE_D_u" : 1,                            #入力信号次元
                "NRMSE_D_y" : 1,                            #出力信号次元
                "NRMSE_Length_Burnin" : 1000,                       #空走用データ時間長
                "NRMSE_Length_Train" : 20000,                       #学習用データ時間長
                "NRMSE_Length_Test" : 5000,                         #評価用データ時間長
                "NRMSE_T_Task" : Task.Task_NDLorenz,                                #評価用タスク（Type型）
                "NRMSE_T_Model" : Param_Model,                                      #モデル（Type型）
                "NRMSE_T_Output" : Output.Output_Single_NRMSE_2023_04_19_15_25,     #作図出力（Type型）
        
                #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                "DirPath_Project" : self.GridSearch.Dir_Points_Branch,              #GSのパスを使用
                "AxisTag" : self.getTag(chank_i),                                   #GSのパラメータを図題に使用

                "NRMSE_F_OutputCharts" : False,                      #図の出力フラグ
                "NRMSE_F_OutputCharts_UYYdEWaves" : False,           #入出力＆誤差波形図の出力フラグ

                })
            Results.update(Evaluation.Evaluation_NRMSE(param)())
    
        #MC評価
        if gs_param["Project_F_MemoryCapacity"]:
            param = Param.copy()
            param.update({
                #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                "MemoryCapacity_F_OutputLog" : False,               #経過の出力を行うか
                "MemoryCapacity_D_u" : 1,                   #入力信号次元
                "MemoryCapacity_D_y" : 1,                   #出力信号次元            
                "MemoryCapacity_Length_Burnin" : 1000,              #空走用データ時間長
                "MemoryCapacity_Length_Train" : 5000,               #学習用データ時間長
                "MemoryCapacity_Length_Test" : 1000,                #評価用データ時間長
                "MemoryCapacity_MaxTau" : 100,                      #評価する最大遅延
                "MemoryCapacity_T_Task" : Task.Task_MC,                                     #評価用タスク（Type型）
                "MemoryCapacity_T_Model" : Param_Model,                                     #モデル（Type型）
                "MemoryCapacity_T_Output" : Output.Output_Single_MC_2023_05_25_13_28,       #作図出力（Type型）
        
                #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                "DirPath_Project" : self.GridSearch.Dir_Points_Branch,                      #GSのパスを使用
                "AxisTag" : self.getTag(chank_i),                                           #GSのパラメータを図題に使用

                "MemoryCapacity_F_OutputCharts" : False,             #図の出力フラグ
                "MemoryCapacity_F_OutputCharts_MCGraph" : False,     #MC曲線の出力フラグ

                })
            Results.update(Evaluation.Evaluation_MC(param)())
        
        #地点のパラメータを改めて設定
        if gs_param["Project_F_OutputResults"]:
            self.GridSearch.CSV_Point_Param.Save(Results)

        return Results

    #軸パラメータを文字列化
    def getTag(self, param: dict) -> str:
        return "Tau : " + str(param["Tau"]) + " Model : " + param["Model"]

#********************************************************************
#********************************************************************
#GS出力
#====================================================================
class Output_SishuESN_2024_06_01_10_30(Output.Output):
    """
    Sishu提案モデル（結合形態のみ指定、強度は乱数）のNRMSEやMC測定
    """
    #コンストラクタ
    def __init__(self, param: dict, parent: any):
        super().__init__(param, parent)
        
        #グリッドサーチ用図題用
        #self.AxisTag = self.Param["AxisTag"] if "AxisTag" in self.Param else ""

        #各種図の出力フラグ
        #self.F_OutputCharts = self.Param["NRMSE_F_OutputCharts"]
        #self.F_OutputCharts_UYYdEWaves = self.Param["NRMSE_F_OutputCharts_UYYdEWaves"]
        
    #本体
    def __call__(self, result_param): 
        #結果の取得
        AllData = pd.DataFrame(result_param)
        #図までのパスを作成
        self.Parent.ConstractFileTree_Charts_Branch(True)
        #指標の文字列一覧（タイトル用，辞書用，ファイル名用）
        L_Score = [["NRMSE", "NRMSE_R_NRMSE", "NRMSE"], 
                ["Log NRMSE", "NRMSE_R_LogNRMSE", "LogNRMSE"], 
                ["Time For Training", "NRMSE_R_TimeForTraining", "TimeTrain"], 
                ["Time For Testing" , "NRMSE_R_TimeForTesting", "TimeTest"],
                ["Memory Capacity", "MemoryCapacity_R_MC", "MC"]]
        
        #指標でループ
        for score in L_Score:
            #データ抽出
            ScoreData = self.MakeData(AllData, score[1])
            SishuESN_Score = ScoreData[:, :, :, 0]
            #指標の文字列一覧（タイトル用，データ，ファイル名用）
            L_Chart = [["SishuESN", SishuESN_Score, "SishuESN"]]
            #フォルダパスを作成
            Plt_Score = self.Parent.Dir_Charts_Branch.AddChild(
                FileAndDir.DirNode(score[1]))

            #とる軸毎にループ
            for chart in L_Chart:
                #データ生成
                #平均
                mean = np.nanmean(chart[1], axis = 0)
                #標準偏差
                std = np.nanstd(chart[1], axis = 0)
                #サンプル数２以上の場合
                if self.Parent.NumberOfSamples > 1:
                    #縦方向の影響を無くした平均
                    scaled_mean = (mean - np.nanmean(mean, axis = 0)) / (np.nanstd(mean, axis = 0))
                    #縦方向の影響を無くした標準偏差
                    scaled_std = (std - np.nanmean(std, axis = 0)) / (np.nanstd(std, axis = 0))
                else:
                    scaled_mean = np.zeros(mean.shape)
                    scaled_std = np.zeros(mean.shape)
                #統計処理の文字列一覧（タイトル用，データ，ファイル名用）
                L_Stats = [["Mean", mean, "Mean"],
                        ["Std", std, "Std"],
                        ["Mean Scaled in each Row", scaled_mean, "ScaleMean"],
                        ["Std Scaled in each Row", scaled_std, "ScaleStd"]]
                
                #統計処理でループ
                for stats in L_Stats:
                    #軸パラメータ作成
                    AxisX = np.array(self.Param["_Param_k"])
                    AxisY = [0.1, None, None, None, None, 0.095, None, None, None, None, 0.09, None, None, None, None, 0.085, None, None, None, None, 0.08, None, None, None, None, 0.075, None, None, None, None, 0.07, None, None, None, None, 0.065, None, None, None, None, 0.06, None, None, None, None, 0.055, None, None, None, None, 0.05, None, None, None, None, 0.045, None, None, None, None, 0.04, None, None, None, None, 0.035, None, None, None, None, 0.03, None, None, None, None, 0.025, None, None, None, None, 0.02, None, None, None, None, 0.015, None, None, None, None, 0.01, None, None, None, None, 0.005, None, None, None, 0.001]
                    #ファイル作成
                    Plt_Chart = Plt_Score.AddChild(
                        FileAndDir.FileNode_plt(chart[2] + "_" + stats[2]))
                    Plt_Chart.Save()

                    #図の設定
                    LabelsX = "Strength of electromagnetic flux : k"                     #x軸ラベル
                    LabelsY = "Scaling size : \u03c3"                     #y軸ラベル
                    FigSize = (16, 9)                   #図の大きさとアスペクト比
                    FontSize_Label = 36                 #ラベルのフォントサイズ
                    FontSize_Title = 36                 #図題のフォントサイズ
                    LineWidth = 0.5                       #線の太さ
                    FileFormat = ".png"                 #ファイル拡張子
                    MapCode = "jet"                     #カラーマップ
                    Annot = False#True                  #数値を表示するか

                    #図題
                    Title = None
                    #作図
                    fig = plt.figure(figsize = FigSize)
                    sub = fig.add_subplot(1, 1, 1)
                    plt.tick_params(labelsize=15)
                    body = sns.heatmap(stats[1], xticklabels = AxisX, yticklabels = AxisY, cmap = MapCode, linewidth = LineWidth, annot = Annot, ax = sub)
                    body.collections[0].colorbar.set_label('Average NRMSE',fontsize=24)
                    body.set_xlabel(LabelsX, fontsize = FontSize_Label)
                    body.set_ylabel(LabelsY, fontsize = FontSize_Label)
                    body.set_title(Title, fontsize = FontSize_Title)
                    plt.tight_layout()
                    fig.savefig(Plt_Chart.Path + FileFormat)
            
                    #表示せず図を閉じる
                    plt.close()
    
    #DataFrameをNdarrayに変換し，結果のキーから結果を抽出
    def MakeData(self, data, key):
        return data[key].values.reshape([axis.NumberOfPoints for axis in self.Parent.GridSearch_AxisList])

    #以下フォルダ構造
    def ConstractFileTree_Charts(self):
        pass

class OutputLog_SishuESN_2024_06_01_10_30(Output.Output):
    """
    Sishu提案モデル（結合形態のみ指定、強度は乱数）のNRMSEやMC測定
    """
    #コンストラクタ
    def __init__(self, param: dict, parent: any):
        super().__init__(param, parent)
        
        #グリッドサーチ用図題用
        #self.AxisTag = self.Param["AxisTag"] if "AxisTag" in self.Param else ""

        #各種図の出力フラグ
        #self.F_OutputCharts = self.Param["NRMSE_F_OutputCharts"]
        #self.F_OutputCharts_UYYdEWaves = self.Param["NRMSE_F_OutputCharts_UYYdEWaves"]
        
    #本体
    def __call__(self, result_param): 
        #結果の取得
        AllData = pd.DataFrame(result_param)
        #図までのパスを作成
        self.Parent.ConstractFileTree_Charts_Branch(True)
        #指標の文字列一覧（タイトル用，辞書用，ファイル名用）
        L_Score = [["NRMSE", "NRMSE_R_NRMSE", "NRMSE"], 
                ["Log NRMSE", "NRMSE_R_LogNRMSE", "LogNRMSE"], 
                ["Time For Training", "NRMSE_R_TimeForTraining", "TimeTrain"], 
                ["Time For Testing" , "NRMSE_R_TimeForTesting", "TimeTest"],
                ["Memory Capacity", "MemoryCapacity_R_MC", "MC"]]
        
        #指標でループ
        for score in L_Score:
            #データ抽出
            ScoreData = self.MakeData(AllData, score[1])
            SishuESN_Score = ScoreData[:, :, :, 0]
            #指標の文字列一覧（タイトル用，データ，ファイル名用）
            L_Chart = [["SishuESN", SishuESN_Score, "SishuESN"]]
            #フォルダパスを作成
            Plt_Score = self.Parent.Dir_Charts_Branch.AddChild(
                FileAndDir.DirNode(score[1]))

            #とる軸毎にループ
            for chart in L_Chart:
                #データ生成
                #平均
                mean = np.nanmean(chart[1], axis = 0)
                #標準偏差
                std = np.nanstd(chart[1], axis = 0)
                #サンプル数２以上の場合
                if self.Parent.NumberOfSamples > 1:
                    #縦方向の影響を無くした平均
                    scaled_mean = (mean - np.nanmean(mean, axis = 0)) / (np.nanstd(mean, axis = 0))
                    #縦方向の影響を無くした標準偏差
                    scaled_std = (std - np.nanmean(std, axis = 0)) / (np.nanstd(std, axis = 0))
                else:
                    scaled_mean = np.zeros(mean.shape)
                    scaled_std = np.zeros(mean.shape)
                #統計処理の文字列一覧（タイトル用，データ，ファイル名用）
                L_Stats = [["Mean", mean, "Mean"],
                        ["Std", std, "Std"],
                        ["Mean Scaled in each Row", scaled_mean, "ScaleMean"],
                        ["Std Scaled in each Row", scaled_std, "ScaleStd"]]
                
                #統計処理でループ
                for stats in L_Stats:
                    #軸パラメータ作成
                    AxisX = np.array(self.Param["_Param_k"])
                    AxisY = np.array(self.Param["_Param_Rho"])

                    #ファイル作成
                    Plt_Chart = Plt_Score.AddChild(
                        FileAndDir.FileNode_plt(chart[2] + "_" + stats[2]))
                    Plt_Chart.Save()

                    #図の設定
                    LabelsX = "Strength of electromagnetic flux : k"                     #x軸ラベル
                    LabelsY = "Scaling size : \u03c3"                     #y軸ラベル
                    FigSize = (16, 9)                   #図の大きさとアスペクト比
                    FontSize_Label = 36                 #ラベルのフォントサイズ
                    FontSize_Title = 36                 #図題のフォントサイズ
                    LineWidth = 0.5                     #線の太さ
                    FileFormat = ".png"                 #ファイル拡張子
                    MapCode = "jet"                     #カラーマップ
                    Annot = False#True                  #数値を表示するか

                    #図題
                    Title = None
                    #作図
                    fig = plt.figure(figsize = FigSize)
                    sub = fig.add_subplot(1, 1, 1)
                    plt.tick_params(labelsize=15)
                    plt.yscale('log')

                    body = sns.heatmap(stats[1], xticklabels = AxisX, yticklabels = [0.1, None, None, None, None, 0.05, None, None, None, 0.01, None, None, None, None, 0.005, None, None, None, 0.001, None, None, None, None, 0.0005, None, None, None, 0.0001], cmap = MapCode, linewidth = LineWidth, annot = Annot, ax = sub)
                    body.collections[0].colorbar.set_label('Average NRMSE',fontsize=24)
                    body.set_xlabel(LabelsX, fontsize = FontSize_Label)
                    body.set_ylabel(LabelsY,fontsize = FontSize_Label)
                    body.set_title(Title, fontsize = FontSize_Title)
                    plt.tight_layout()
                    fig.savefig(Plt_Chart.Path + FileFormat)
            
                    #表示せず図を閉じる
                    plt.close()
    
    #DataFrameをNdarrayに変換し，結果のキーから結果を抽出
    def MakeData(self, data, key):
        return data[key].values.reshape([axis.NumberOfPoints for axis in self.Parent.GridSearch_AxisList])

    #以下フォルダ構造
    def ConstractFileTree_Charts(self):
        pass

class Output_Tau_SishuESN_2024_06_01_10_30(Output.Output):
    """
    Sishu提案モデル（結合形態のみ指定、強度は乱数）のNRMSEやMC測定
    """
    #コンストラクタ
    def __init__(self, param: dict, parent: any):
        super().__init__(param, parent)
        
        #グリッドサーチ用図題用
        #self.AxisTag = self.Param["AxisTag"] if "AxisTag" in self.Param else ""

        #各種図の出力フラグ
        #self.F_OutputCharts = self.Param["NRMSE_F_OutputCharts"]
        #self.F_OutputCharts_UYYdEWaves = self.Param["NRMSE_F_OutputCharts_UYYdEWaves"]
        
    #本体
    def __call__(self, result_param): 
        #結果の取得
        AllData = pd.DataFrame(result_param)
        #図までのパスを作成
        self.Parent.ConstractFileTree_Charts_Branch(True)
        #指標の文字列一覧（タイトル用，辞書用，ファイル名用）
        L_Score = [["NRMSE", "NRMSE_R_NRMSE", "NRMSE"], 
                ["Log NRMSE", "NRMSE_R_LogNRMSE", "LogNRMSE"], 
                ["Time For Training", "NRMSE_R_TimeForTraining", "TimeTrain"], 
                ["Time For Testing" , "NRMSE_R_TimeForTesting", "TimeTest"],
                ["Memory Capacity", "MemoryCapacity_R_MC", "MC"]]
        
        test = 0

        #指標でループ
        for score in L_Score:
            #データ抽出
            ScoreData = self.MakeData(AllData, score[1])
            SishuESN_Score = ScoreData[:, :, 0]
            #指標の文字列一覧（タイトル用，データ，ファイル名用）
            L_Chart = [["SishuESN", SishuESN_Score, "SishuESN"]]
            #フォルダパスを作成
            Plt_Score = self.Parent.Dir_Charts_Branch.AddChild(
                FileAndDir.DirNode(score[1]))

            #とる軸毎にループ
            for chart in L_Chart:
                #データ生成
                #平均
                mean = np.nanmean(chart[1], axis = 0)
                #標準偏差
                std = np.nanstd(chart[1], axis = 0)
                #サンプル数２以上の場合
                if self.Parent.NumberOfSamples > 1:
                    #縦方向の影響を無くした平均
                    scaled_mean = (mean - np.nanmean(mean, axis = 0)) / (np.nanstd(mean, axis = 0))
                    #縦方向の影響を無くした標準偏差
                    scaled_std = (std - np.nanmean(std, axis = 0)) / (np.nanstd(std, axis = 0))
                else:
                    scaled_mean = np.zeros(mean.shape)
                    scaled_std = np.zeros(mean.shape)
                #統計処理の文字列一覧（タイトル用，データ，ファイル名用）
                L_Stats = [["Mean", mean, "Mean"],
                        ["Std", std, "Std"],
                        ["Mean Scaled in each Row", scaled_mean, "ScaleMean"],
                        ["Std Scaled in each Row", scaled_std, "ScaleStd"]]
                
                #統計処理でループ
                for stats in L_Stats:
                    #軸パラメータ作成
                    AxisX = np.array(self.Param["_Param_Tau"])
                    AxisY = np.array(stats[0])
                    #ファイル作成
                    Plt_Chart = Plt_Score.AddChild(
                        FileAndDir.FileNode_plt(chart[2] + "_" + stats[2]))
                    Plt_Chart.Save()

                    #図の設定
                    LabelsX = "Tau"                     #x軸ラベル
                    LabelsY = L_Score[test][0]                     #y軸ラベル
                    FigSize = (16, 9)                   #図の大きさとアスペクト比
                    FontSize_Label = 36                 #ラベルのフォントサイズ
                    FontSize_Title = 36                 #図題のフォントサイズ
                    LineWidth = 3                       #線の太さ
                    FileFormat = ".png"                 #ファイル拡張子
                    MapCode = "jet"                     #カラーマップ
                    Annot = False#True                  #数値を表示するか

                    #図題
                    Title = None
                    #作図
                    fig  = plt.figure(figsize = FigSize)
                    sub = fig.add_subplot(1, 1, 1)
                    #body = sns.heatmap(stats[1], xticklabels = AxisX, yticklabels = AxisY, cmap = MapCode, linewidth = LineWidth, annot = Annot, ax = sub)
                    plt.plot(self.Param["_Param_Tau"],stats[1])
                    plt.gca().invert_yaxis()
                    plt.xlabel(LabelsX, fontsize = FontSize_Label)
                    plt.ylabel(LabelsY, fontsize = FontSize_Label)
                    plt.title(Title, fontsize = FontSize_Title)
                    plt.tight_layout()
                    fig.savefig(Plt_Chart.Path + FileFormat)
            
                    #表示せず図を閉じる
                    plt.close()
            test += 1
    #DataFrameをNdarrayに変換し，結果のキーから結果を抽出
    def MakeData(self, data, key):
        return data[key].values.reshape([axis.NumberOfPoints for axis in self.Parent.GridSearch_AxisList])

    #以下フォルダ構造
    def ConstractFileTree_Charts(self):
        pass


#********************************************************************
#********************************************************************