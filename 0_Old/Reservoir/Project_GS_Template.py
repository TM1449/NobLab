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
#GSパラメータ
def Project_GridSearch_NRMSEAndMC():
    """ 
    グリッドサーチのデバッグ用
    """
    GridSearch.Normal({
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        "Project_F_NRMSE" : True,                                   #NRMSEを調査するか
        "Project_F_MemoryCapacity" : True,                          #MCを調査するか

        "Project_F_OutputResults" : True,                           #各評価地点でパラメータを出力するか

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #軸
        "_Param_D_u" : [1, 5, 10],
        "_Param_D_x" : [50, 100],
        "_Param_Model" : ["ESN", "AESN"],

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        "GridSearch_MachineName" : "DB",                            #計算機名
        "GridSearch_StartPoint" : 0,                                #担当開始ポイントインデックス
        "GridSearch_EndPoint" : 24,                                 #担当終了ポイントインデックス
        
        "GridSearch_MultiThread" : 10,                               #スレッド数（0で逐次処理）
        "GridSearch_MaxNumberInOneFile" : 5,                        #１ファイルの最大のポイント数
        "GridSearch_MaxNumberInOnePool" : 50,                       #１プール（並列する）最大のポイント数（この分メモリを消費）
        "GridSearch_NumberOfSamples" : 2,                           #サンプル数
        "GridSearch_ProjectName" : "NRMSEAndMC",                    #プロジェクト名
        "GridSearch_ProjectDate" : "2023_06_17_09_36",              #プロジェクト日時
        "GridSearch_T_Process" : Process_GridSearch,                #GS処理指定
        "GridSearch_T_Output" : Output_2023_06_17_11_16             #GS出力処理指定
        })()


def Project_GridSearch_ChialvoESN_RingNetwork_NRMSEAndMC():
    """ 
    ChialvoESNのグリッドサーチ
    """
    GridSearch.Normal({
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        "Project_F_NRMSE" : True,                                   #NRMSEを調査するか
        "Project_F_MemoryCapacity" : True,                          #MCを調査するか

        "Project_F_OutputResults" : True,                           #各評価地点でパラメータを出力するか

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #軸
        "_Param_sigma" : [-0.005, -0.0045, -0.004, -0.0035, -0.003, -0.0025, -0.002, -0.0015, -0.001, -0.0005, 0.0, 0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.0035, 0.004, 0.0045, 0.005],
        "_Param_k" : [-5.0, -4.5, -4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
        "_Param_Model" : ["ChialvoESN"],

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        "GridSearch_MachineName" : "RingESN",                            #計算機名
        "GridSearch_StartPoint" : 0,                                #担当開始ポイントインデックス
        "GridSearch_EndPoint" : 882,                                #担当終了ポイントインデックス
        
        "GridSearch_MultiThread" : 10,                               #スレッド数（0で逐次処理）初期値:2
        "GridSearch_MaxNumberInOneFile" : 100,                        #１ファイルの最大のポイント数 初期値:5
        "GridSearch_MaxNumberInOnePool" : 1000,                       #１プール（並列する）最大のポイント数（この分メモリを消費） 初期値:50
        "GridSearch_NumberOfSamples" : 2,                           #サンプル数
        "GridSearch_ProjectName" : "NRMSEAndMC_RingNetwork_ChialvoESN",                    #プロジェクト名
        "GridSearch_ProjectDate" : "2024_06_01_10_30",              #プロジェクト日時
        "GridSearch_T_Process" : Process_ChialvoESN_RingNetwork_GridSearch,     #GS処理指定
        "GridSearch_T_Output" : Output_RingNetwork_2024_06_01_10_30             #GS出力処理指定
        })()


def Project_GridSearch_ChialvoESN_StarNetwork_NRMSEAndMC():
    """ 
    ChialvoESNのグリッドサーチ
    """
    GridSearch.Normal({
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        "Project_F_NRMSE" : True,                                   #NRMSEを調査するか
        "Project_F_MemoryCapacity" : True,                          #MCを調査するか

        "Project_F_OutputResults" : True,                           #各評価地点でパラメータを出力するか

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #軸
        "_Param_mu" : [-0.0005, -0.00045, -0.0004, -0.00035, -0.0003, -0.00025, -0.0002, -0.00015, -0.0001, -5e-05, 0.0, 5e-05, 0.0001, 0.00015, 0.0002, 0.00025, 0.0003, 0.00035, 0.0004, 0.00045, 0.0005],
        "_Param_k" : [-5.0, -4.5, -4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
        "_Param_Model" : ["ChialvoESN"],

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        "GridSearch_MachineName" : "StarESN",                            #計算機名
        "GridSearch_StartPoint" : 0,                                #担当開始ポイントインデックス
        "GridSearch_EndPoint" : 882,                                #担当終了ポイントインデックス
        
        "GridSearch_MultiThread" : 10,                               #スレッド数（0で逐次処理）初期値:2
        "GridSearch_MaxNumberInOneFile" : 100,                        #１ファイルの最大のポイント数 初期値:5
        "GridSearch_MaxNumberInOnePool" : 1000,                       #１プール（並列する）最大のポイント数（この分メモリを消費） 初期値:50
        "GridSearch_NumberOfSamples" : 2,                           #サンプル数
        "GridSearch_ProjectName" : "NRMSEAndMC_StarNetwork_ChialvoESN",                    #プロジェクト名
        "GridSearch_ProjectDate" : "2024_06_01_10_30",              #プロジェクト日時
        "GridSearch_T_Process" : Process_ChialvoESN_StarNetwork_GridSearch,     #GS処理指定
        "GridSearch_T_Output" : Output_StarNetwork_2024_06_01_10_30             #GS出力処理指定
        })()


def Project_GridSearch_ChialvoESN_RingStarNetwork_NRMSEAndMC():
    """ 
    ChialvoESNのグリッドサーチ
    """
    GridSearch.Normal({
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        "Project_F_NRMSE" : True,                                   #NRMSEを調査するか
        "Project_F_MemoryCapacity" : True,                          #MCを調査するか

        "Project_F_OutputResults" : True,                           #各評価地点でパラメータを出力するか

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #軸
        "_Param_sigma" : [-0.005, -0.0045, -0.004, -0.0035, -0.003, -0.0025, -0.002, -0.0015, -0.001, -0.0005, 0.0, 0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.0035, 0.004, 0.0045, 0.005],
        "_Param_mu" : [-0.0005, -0.00045, -0.0004, -0.00035, -0.0003, -0.00025, -0.0002, -0.00015, -0.0001, -5e-05, 0.0, 5e-05, 0.0001, 0.00015, 0.0002, 0.00025, 0.0003, 0.00035, 0.0004, 0.00045, 0.0005],
        "_Param_Model" : ["ChialvoESN"],

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        "GridSearch_MachineName" : "RingStarESN",                            #計算機名
        "GridSearch_StartPoint" : 0,                                #担当開始ポイントインデックス
        "GridSearch_EndPoint" : 882,                                #担当終了ポイントインデックス(MAX:4410(21*21*10))
        
        "GridSearch_MultiThread" : 10,                               #スレッド数（0で逐次処理）初期値:2
        "GridSearch_MaxNumberInOneFile" : 100,                        #１ファイルの最大のポイント数 初期値:5
        "GridSearch_MaxNumberInOnePool" : 1000,                       #１プール（並列する）最大のポイント数（この分メモリを消費） 初期値:50
        "GridSearch_NumberOfSamples" : 2,                           #サンプル数
        "GridSearch_ProjectName" : "NRMSEAndMC_RingStarNetwork_ChialvoESN",                    #プロジェクト名
        "GridSearch_ProjectDate" : "2024_06_01_10_30",              #プロジェクト日時
        "GridSearch_T_Process" : Process_ChialvoESN_RingStarNetwork_GridSearch,     #GS処理指定
        "GridSearch_T_Output" : Output_RingStarNetwork_2024_06_01_10_30             #GS出力処理指定
        })()


def Project_GridSearch_NewChialvoESN_NRMSEAndMC():
    """ 
    NeoChialvoESNのグリッドサーチ
    """
    GridSearch.Normal({
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        "Project_F_NRMSE" : True,                                   #NRMSEを調査するか
        "Project_F_MemoryCapacity" : True,                          #MCを調査するか

        "Project_F_OutputResults" : True,                           #各評価地点でパラメータを出力するか

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #軸
        "_Param_Rho" : [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085, 0.09, 0.095, 0.1, 0.105, 0.11, 0.115, 0.12, 0.125, 0.13, 0.135, 0.14, 0.145, 0.15, 0.155, 0.16, 0.165, 0.17, 0.175, 0.18, 0.185, 0.19, 0.195, 0.2],
        "_Param_k" : [-5.0, -4.5, -4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
        "_Param_Model" : ["NewChialvoESN"],

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        "GridSearch_MachineName" : "NewChialvo",                            #計算機名
        "GridSearch_StartPoint" : 0,                                #担当開始ポイントインデックス
        "GridSearch_EndPoint" : 1680,                                #担当終了ポイントインデックス(MAX:4410(21*21*10))
        
        "GridSearch_MultiThread" : 10,                               #スレッド数（0で逐次処理）初期値:2
        "GridSearch_MaxNumberInOneFile" : 100,                        #１ファイルの最大のポイント数 初期値:5
        "GridSearch_MaxNumberInOnePool" : 1000,                       #１プール（並列する）最大のポイント数（この分メモリを消費） 初期値:50
        "GridSearch_NumberOfSamples" : 2,                           #サンプル数
        "GridSearch_ProjectName" : "NRMSEAndMC_NewChialvoESN",                    #プロジェクト名
        "GridSearch_ProjectDate" : "2024_06_01_10_30",              #プロジェクト日時
        "GridSearch_T_Process" : Process_NewChialvoESN_GridSearch,     #GS処理指定
        "GridSearch_T_Output" : Output_NewChialvoESN_2024_06_01_10_30             #GS出力処理指定
        })()


#********************************************************************
#GS処理指定
class Process_GridSearch:
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
        Param_D_u = chank_i["D_u"]
        Param_D_y = chank_i["D_u"]
        Param_D_x = chank_i["D_x"]
        if chank_i["Model"] == "ESN":
            Param_Model = Model.Model_NormalESN
        elif chank_i["Model"] == "AESN":
            Param_Model = Model.Model_NormalAESN
            
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
            "Model_NormalESN_D_u" : Param_D_u,                  #入力信号次元
            "Model_NormalESN_D_x" : Param_D_x * Param_D_u,      #ニューロン数
            "Model_NormalESN_D_y" : Param_D_y,                  #出力信号次元
        
            "Model_NormalAESN_D_u" : Param_D_u,                 #入力信号次元
            "Model_NormalAESN_D_x" : Param_D_x,                 #サブリザバーニューロン数
            "Model_NormalAESN_D_y" : Param_D_y,                 #出力信号次元
        
            "Model_DifferentUpdateESN_D_u" : 5,                         #入力信号次元
            "Model_DifferentUpdateESN_D_x" : 100,                       #サブリザバーニューロン数（リスト型可能）
            "Model_DifferentUpdateESN_D_y" : 5,                         #出力信号次元
            "Model_DifferentUpdateESN_LeakingRate" : [0.01, 0.1, 1.0],  #リーク率配列->サブリザバーの数(必ずリスト型)
            "Model_DifferentUpdateESN_InputScale" : None,               #入力スケーリング配列（リスト型可能，None可能）
            "Model_DifferentUpdateESN_Rho" : None,                      #スペクトル半径配列（リスト型可能，None可能）
            "Model_DifferentUpdateESN_Density" : None,                  #結合密度配列（リスト型可能，None可能）
        
            "Model_AESNwithHub_D_u" : 5,                        #入力信号次元
            "Model_AESNwithHub_D_h" : 100,                      #サブリザバーニューロン数
            "Model_AESNwithHub_D_x" : 100,                      #ハブリザバーニューロン数
            "Model_AESNwithHub_D_y" : 5,                        #出力信号次元
            "Model_AESNwithHub_F_Use_U2HubConnection" : False,  #Hubへの入力信号使用の有無
            "Model_AESNwithHub_F_Use_x2zConnection" : True,     #出力へのHub状態使用の有無
            "Model_AESNwithHub_F_Use_x2AESNConnection" : False, #AESNへのHub信号使用の有無
            "Model_AESNwithHub_F_Use_AverageHInHub" : False,    #HubでHの平均をするか
            "Model_AESNwithHub_LeakingRate" : None,             #Hubリーク率配列（None可能）
            "Model_AESNwithHub_InputScale" : None,              #Hub入力スケーリング配列（None可能）
            "Model_AESNwithHub_Rho" : None,                     #Hubスペクトル半径配列（None可能）
            "Model_AESNwithHub_Density" : None,                 #Hub結合密度配列（None可能）

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
            "LinerTransformer_Beta" : 0.2,                      #正規化係数

            "DNN_LearningRate" : 0.001,                         #学習率
            "DNN_MaxLearningLoop" : 1000,                       #最大学習ループ数（使わない場合は0にする）
            "DNN_AimingError" : 0.001,                          #目標（最小）誤差（使わない場合は0にする）
        
            }
    
        Results = dict()
        #NRMSE評価
        if gs_param["Project_F_NRMSE"]:
            param = Param.copy()
            param.update({
                #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                "NRMSE_F_OutputLog" : False,                        #経過の出力を行うか
                "NRMSE_D_u" : Param_D_u,                            #入力信号次元
                "NRMSE_D_y" : Param_D_y,                            #出力信号次元
                "NRMSE_Length_Burnin" : 1000,                       #空走用データ時間長
                "NRMSE_Length_Train" : 20000,                       #学習用データ時間長
                "NRMSE_Length_Test" : 5000,                         #評価用データ時間長
                "NRMSE_T_Task" : Task.Task_NDLorenz,                                #評価用タスク（Type型）
                "NRMSE_T_Model" : Param_Model,                                      #モデル（Type型）
                "NRMSE_T_Output" : Output.Output_Single_NRMSE_2023_04_19_15_25,     #作図出力（Type型）
        
                #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                "DirPath_Project" : self.GridSearch.Dir_Points_Branch,              #GSのパスを使用
                "AxisTag" : self.getTag(chank_i),                                   #GSのパラメータを図題に使用

                "NRMSE_F_OutputCharts" : True,                      #図の出力フラグ
                "NRMSE_F_OutputCharts_UYYdEWaves" : True,           #入出力＆誤差波形図の出力フラグ

                })
            Results.update(Evaluation.Evaluation_NRMSE(param)())
    
        #MC評価
        if gs_param["Project_F_MemoryCapacity"]:
            param = Param.copy()
            param.update({
                #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                "MemoryCapacity_F_OutputLog" : False,               #経過の出力を行うか
                "MemoryCapacity_D_u" : Param_D_u,                   #入力信号次元
                "MemoryCapacity_D_y" : Param_D_y,                   #出力信号次元
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

                "MemoryCapacity_F_OutputCharts" : True,             #図の出力フラグ
                "MemoryCapacity_F_OutputCharts_MCGraph" : True,     #MC曲線の出力フラグ

                })
            Results.update(Evaluation.Evaluation_MC(param)())
        
        #地点のパラメータを改めて設定
        if gs_param["Project_F_OutputResults"]:
            self.GridSearch.CSV_Point_Param.Save(Results)

        return Results

    #軸パラメータを文字列化
    def getTag(self, param: dict) -> str:
        return "D_u : " + str(param["D_u"]) + " D_x : " + str(param["D_x"]) + " Model : " + param["Model"]


class Process_ChialvoESN_RingNetwork_GridSearch:
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
        Param_sigma = chank_i["sigma"]
        Param_k = chank_i["k"]
        if chank_i["Model"] == "ChialvoESN":
            Param_Model = Model.Model_ChialvoESN
            
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
            "Task_Lorenz_Tau" : 10,                              #どれくらい先を予測するか
            "Task_Rosslor_InitTerm" : 1000,                     #初期状態排除期間

            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            "Model_ChialvoESN_D_u" : 1,                  #入力信号次元
            "Model_ChialvoESN_D_x" : 100,                #ニューロン数
            "Model_ChialvoESN_D_y" : 1,                  #出力信号次元
            
            "Model_ChialvoESN_InputScale" : 0.1,                 #入力スケーリング
            "Model_ChialvoESN_sigma" : Param_sigma,                   #リング・ネットワークの結合強度
            "Model_ChialvoESN_mu" : 0 ,                           #スター・ネットワークの結合強度
            "Model_ChialvoESN_k" : Param_k,

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
        return "sigma : " + str(param["sigma"]) + " k : " + str(param["k"]) + " Model : " + param["Model"]


class Process_ChialvoESN_StarNetwork_GridSearch:
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
        Param_mu = chank_i["mu"]
        Param_k = chank_i["k"]
        if chank_i["Model"] == "ChialvoESN":
            Param_Model = Model.Model_ChialvoESN
            
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
            "Task_Lorenz_Tau" : 10,                              #どれくらい先を予測するか
            "Task_Rosslor_InitTerm" : 1000,                     #初期状態排除期間

            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            "Model_ChialvoESN_D_u" : 1,                  #入力信号次元
            "Model_ChialvoESN_D_x" : 100,                #ニューロン数
            "Model_ChialvoESN_D_y" : 1,                  #出力信号次元
            
            "Model_ChialvoESN_InputScale" : 0.1,                 #入力スケーリング
            "Model_ChialvoESN_sigma" : 0,                   #リング・ネットワークの結合強度
            "Model_ChialvoESN_mu" : Param_mu,                           #スター・ネットワークの結合強度
            "Model_ChialvoESN_k" : Param_k,

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
        return "mu : " + str(param["mu"]) + " k : " + str(param["k"]) + " Model : " + param["Model"]


class Process_ChialvoESN_RingStarNetwork_GridSearch:
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
        Param_sigma = chank_i["sigma"]
        Param_mu = chank_i["mu"]
        if chank_i["Model"] == "ChialvoESN":
            Param_Model = Model.Model_ChialvoESN
            
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
            "Task_Lorenz_Tau" : 10,                              #どれくらい先を予測するか
            "Task_Rosslor_InitTerm" : 1000,                     #初期状態排除期間

            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            "Model_ChialvoESN_D_u" : 1,                  #入力信号次元
            "Model_ChialvoESN_D_x" : 100,                #ニューロン数
            "Model_ChialvoESN_D_y" : 1,                  #出力信号次元
            
            "Model_ChialvoESN_InputScale" : 0.1,                 #入力スケーリング
            "Model_ChialvoESN_sigma" : Param_sigma,                   #リング・ネットワークの結合強度
            "Model_ChialvoESN_mu" : Param_mu,                           #スター・ネットワークの結合強度
            "Model_ChialvoESN_k" : -3,

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
        return "sigma : " + str(param["sigma"]) + " mu : " + str(param["mu"]) + " Model : " + param["Model"]


class Process_NewChialvoESN_GridSearch:
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
        if chank_i["Model"] == "NewChialvoESN":
            Param_Model = Model.Model_NewChialvoESN
            
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
            "Task_Lorenz_Tau" : 10,                              #どれくらい先を予測するか
            "Task_Rosslor_InitTerm" : 1000,                     #初期状態排除期間

            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            "Model_ChialvoESN_D_u" : 1,                  #入力信号次元
            "Model_ChialvoESN_D_x" : 100,                #ニューロン数
            "Model_ChialvoESN_D_y" : 1,                  #出力信号次元
            
            "Model_ChialvoESN_InputScale" : 0.1,                 #入力スケーリング
            "Model_ChialvoESN_sigma" : True,                   #リング・ネットワークの結合強度
            "Model_ChialvoESN_mu" : True,                     #スター・ネットワークの結合強度
            "Model_ChialvoESN_k" : Param_k,

            "Model_ChialvoESN_Rho" : Param_Rho,                             #スペクトル半径
            "ChialvoReservoir_ActivationFunc" : np.tanh,            #活性化関数
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
        return "Rho : " + str(param["Rho"]) + " k : " + str(param["k"]) + " Model : " + param["Model"]


#********************************************************************
#GS出力
class Output_2023_06_17_11_16(Output.Output):
    """
    上記のパラメータに対応する出力処理

    命名規則は Output_YYYY_MM_DD_HH_MM_[必要であれば識別タグ]
    昔の有意有用な結果の作図出力は全て残し，新しいものは別のクラスで作ること．
    何の実験の出力なのかここにコメントを書くこと．
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
            ESN_Score = ScoreData[:, :, :, 0]
            AESN_Score = ScoreData[:, :, :, 1]
            #指標の文字列一覧（タイトル用，データ，ファイル名用）
            L_Chart = [["ESN", ESN_Score, "ESN"],
                    ["AESN", AESN_Score, "AESN"]]
            #フォルダパスを作成
            Plt_Score = self.Parent.Dir_Charts_Branch.AddChild(
                FileAndDir.DirNode(score[2]))

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
                    AxisX = np.array(self.Param["_Param_D_x"])
                    AxisY = np.array(self.Param["_Param_D_u"])
                    #ファイル作成
                    Plt_Chart = Plt_Score.AddChild(
                        FileAndDir.FileNode_plt(chart[2] + "_" + stats[2]))
                    Plt_Chart.Save()
                    
                    #図の設定
                    LabelsX = "D_x"                     #x軸ラベル
                    LabelsY = "D_u"                     #y軸ラベル
                    FigSize = (16, 9)                   #図の大きさとアスペクト比
                    FontSize_Label = 24                 #ラベルのフォントサイズ
                    FontSize_Title = 24                 #図題のフォントサイズ
                    LineWidth = 3                       #線の太さ
                    FileFormat = ".png"                 #ファイル拡張子
                    MapCode = "jet"                     #カラーマップ
                    Annot = False#True                  #数値を表示するか

                    #図題
                    Title = stats[0] + " of " + score[0] + " for " + chart[0]
                    #作図
                    fig = plt.figure(figsize = FigSize)
                    sub = fig.add_subplot(1, 1, 1)
                    body = sns.heatmap(stats[1], xticklabels = AxisX, yticklabels = AxisY, cmap = MapCode, linewidth = LineWidth, annot = Annot, ax = sub)
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


class Output_RingNetwork_2024_06_01_10_30(Output.Output):
    """
    ChialvoESNのグリッドサーチ
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
            ChialvoESN_Score = ScoreData[:, :, :, 0]
            #指標の文字列一覧（タイトル用，データ，ファイル名用）
            L_Chart = [["RingNetwork ChialvoESN", ChialvoESN_Score, "RingNetwork_ChialvoESN"]]
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
                    AxisY = np.array(self.Param["_Param_sigma"])
                    #ファイル作成
                    Plt_Chart = Plt_Score.AddChild(
                        FileAndDir.FileNode_plt(chart[2] + "_" + stats[2]))
                    Plt_Chart.Save()
                    #図の設定
                    LabelsX = "k"                     #x軸ラベル
                    LabelsY = "sigma"                     #y軸ラベル
                    FigSize = (16, 9)                   #図の大きさとアスペクト比
                    FontSize_Label = 24                 #ラベルのフォントサイズ
                    FontSize_Title = 24                 #図題のフォントサイズ
                    LineWidth = 0.5                       #線の太さ
                    FileFormat = ".png"                 #ファイル拡張子
                    MapCode = "jet"                     #カラーマップ
                    Annot = False#True                  #数値を表示するか

                    #図題
                    Title = stats[0] + " of " + score[0] + " for " + chart[0]
                    #作図
                    fig = plt.figure(figsize = FigSize)
                    sub = fig.add_subplot(1, 1, 1)
                    body = sns.heatmap(stats[1], xticklabels = AxisX, yticklabels = AxisY, cmap = MapCode, linewidth = LineWidth, annot = Annot, ax = sub)
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


class Output_StarNetwork_2024_06_01_10_30(Output.Output):
    """
    ChialvoESNのグリッドサーチ
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
            ChialvoESN_Score = ScoreData[:, :, :, 0]
            #指標の文字列一覧（タイトル用，データ，ファイル名用）
            L_Chart = [["StarNetwork ChialvoESN", ChialvoESN_Score, "StarNetwork_ChialvoESN"]]
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
                    AxisY = np.array(self.Param["_Param_mu"])
                    #ファイル作成
                    Plt_Chart = Plt_Score.AddChild(
                        FileAndDir.FileNode_plt(chart[2] + "_" + stats[2]))
                    Plt_Chart.Save()
                    #図の設定
                    LabelsX = "k"                     #x軸ラベル
                    LabelsY = "mu"                     #y軸ラベル
                    FigSize = (16, 9)                   #図の大きさとアスペクト比
                    FontSize_Label = 24                 #ラベルのフォントサイズ
                    FontSize_Title = 24                 #図題のフォントサイズ
                    LineWidth = 0.5                       #線の太さ
                    FileFormat = ".png"                 #ファイル拡張子
                    MapCode = "jet"                     #カラーマップ
                    Annot = False#True                  #数値を表示するか

                    #図題
                    Title = stats[0] + " of " + score[0] + " for " + chart[0]
                    #作図
                    fig = plt.figure(figsize = FigSize)
                    sub = fig.add_subplot(1, 1, 1)
                    body = sns.heatmap(stats[1], xticklabels = AxisX, yticklabels = AxisY, cmap = MapCode, linewidth = LineWidth, annot = Annot, ax = sub)
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


class Output_RingStarNetwork_2024_06_01_10_30(Output.Output):
    """
    ChialvoESNのグリッドサーチ
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
            ChialvoESN_Score = ScoreData[:, :, :, 0]
            #指標の文字列一覧（タイトル用，データ，ファイル名用）
            L_Chart = [["RingStarNetwork ChialvoESN", ChialvoESN_Score, "RingStarNetwork_ChialvoESN"]]
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
                    AxisX = np.array(self.Param["_Param_mu"])
                    AxisY = np.array(self.Param["_Param_sigma"])
                    #ファイル作成
                    Plt_Chart = Plt_Score.AddChild(
                        FileAndDir.FileNode_plt(chart[2] + "_" + stats[2]))
                    Plt_Chart.Save()
                    #図の設定
                    LabelsX = "mu"                     #x軸ラベル
                    LabelsY = "sigma"                     #y軸ラベル
                    FigSize = (16, 9)                   #図の大きさとアスペクト比
                    FontSize_Label = 24                 #ラベルのフォントサイズ
                    FontSize_Title = 24                 #図題のフォントサイズ
                    LineWidth = 0.5                       #線の太さ
                    FileFormat = ".png"                 #ファイル拡張子
                    MapCode = "jet"                     #カラーマップ
                    Annot = False#True                  #数値を表示するか

                    #図題
                    Title = stats[0] + " of " + score[0] + " for " + chart[0]
                    #作図
                    fig = plt.figure(figsize = FigSize)
                    sub = fig.add_subplot(1, 1, 1)
                    body = sns.heatmap(stats[1], xticklabels = AxisX, yticklabels = AxisY, cmap = MapCode, linewidth = LineWidth, annot = Annot, ax = sub)
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


class Output_NewChialvoESN_2024_06_01_10_30(Output.Output):
    """
    NewChialvoESNのグリッドサーチ
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
            ChialvoESN_Score = ScoreData[:, :, :, 0]
            #指標の文字列一覧（タイトル用，データ，ファイル名用）
            L_Chart = [["NewChialvoESN", ChialvoESN_Score, "NewChialvoESN"]]
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
                    LabelsX = "k"                     #x軸ラベル
                    LabelsY = "Rho"                     #y軸ラベル
                    FigSize = (16, 9)                   #図の大きさとアスペクト比
                    FontSize_Label = 24                 #ラベルのフォントサイズ
                    FontSize_Title = 24                 #図題のフォントサイズ
                    LineWidth = 0.5                       #線の太さ
                    FileFormat = ".png"                 #ファイル拡張子
                    MapCode = "jet"                     #カラーマップ
                    Annot = False#True                  #数値を表示するか

                    #図題
                    Title = stats[0] + " of " + score[0] + " for " + chart[0]
                    #作図
                    fig = plt.figure(figsize = FigSize)
                    sub = fig.add_subplot(1, 1, 1)
                    body = sns.heatmap(stats[1], xticklabels = AxisX, yticklabels = AxisY, cmap = MapCode, linewidth = LineWidth, annot = Annot, ax = sub)
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

