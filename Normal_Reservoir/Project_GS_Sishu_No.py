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

import FileAndDir_No

import Task_No
import Model_No
import Output_No
import Evaluation_No

import GridSearch_No

#====================================================================
#グリッドサーチ

#********************************************************************
#********************************************************************
#GSパラメータ
#====================================================================
def Project_GridSearch_NoChialvo():
    """ 
    Sishu提案モデル（結合形態のみ指定、強度は乱数）のNRMSEやMC測定
    """
    GridSearch_No.Normal({
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        "Project_F_NRMSE" : True,                                   #NRMSEを調査するか
        "Project_F_MemoryCapacity" : True,                          #MCを調査するか
        "Project_F_MLE" : True,                   #MLEを調査するか
        "Project_F_CovMatrixRank" : True,                  #Covariance Matrix Rankを調査するか
        "Project_F_DelayCapacity" : True,                   #Delay Capacityを調査するか

        "Project_F_OutputResults" : True,                           #各評価地点でパラメータを出力するか

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #軸パラメータ（28 * 41 = 1148）
        #"_Param_Rho" : [0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001,0.0009, 0.0008, 0.0007, 0.0006, 0.0005, 0.0004, 0.0003, 0.0002, 0.0001]
        #"_Param_k" : [-5.0, -4.75, -4.5, -4.25, -4.0, -3.75, -3.5, -3.25, -3.0, -2.75, -2.5, -2.25, -2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0]
        
        #狭い範囲の軸パラメータ（28 * 41 = 1148）
        #"_Param_Rho" : [0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001, 0.0009, 0.0008, 0.0007, 0.0006, 0.0005, 0.0004, 0.0003, 0.0002, 0.0001],
        #"_Param_k" : [-5.0, -4.9, -4.8, -4.7, -4.6, -4.5, -4.4, -4.3, -4.2, -4.1, -4.0, -3.9, -3.8, -3.7, -3.6, -3.5, -3.4, -3.3, -3.2, -3.1, -3.0, -2.9, -2.8, -2.7, -2.6, -2.5, -2.4, -2.3, -2.2, -2.1, -2.0, -1.9, -1.8, -1.7, -1.6, -1.5, -1.4, -1.3, -1.2, -1.1, -1.0],

        "_Param_Rho" : [0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001, 0.0009, 0.0008, 0.0007, 0.0006, 0.0005, 0.0004, 0.0003, 0.0002, 0.0001],
        "_Param_k" : [-5.0, -4.9, -4.8, -4.7, -4.6, -4.5, -4.4, -4.3, -4.2, -4.1, -4.0, -3.9, -3.8, -3.7, -3.6, -3.5, -3.4, -3.3, -3.2, -3.1, -3.0, -2.9, -2.8, -2.7, -2.6, -2.5, -2.4, -2.3, -2.2, -2.1, -2.0, -1.9, -1.8, -1.7, -1.6, -1.5, -1.4, -1.3, -1.2, -1.1, -1.0],
        "_Param_Model" : ["EMChialvo"],

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        "GridSearch_MachineName" : "EMChialvo",                            #計算機名
        "GridSearch_StartPoint" : 0,                                #担当開始ポイントインデックス
        "GridSearch_EndPoint" : 11480,                                #担当終了ポイントインデックス
        
        "GridSearch_MultiThread" : 12,                               #スレッド数（0で逐次処理）初期値:2, 自宅PC:4, 研究室PC, 12
        "GridSearch_MaxNumberInOneFile" : 10,                        #１ファイルの最大のポイント数 初期値:5
        "GridSearch_MaxNumberInOnePool" : 1000,                       #１プール（並列する）最大のポイント数（この分メモリを消費） 初期値:50
        "GridSearch_NumberOfSamples" : 10,                           #サンプル数
        "GridSearch_ProjectName" : "EMChialvo",                    #プロジェクト名
        "GridSearch_ProjectDate" : "2025_04_08_20_30",              #プロジェクト日時
        "GridSearch_T_Process" : Process_SishuESN_GridSearch,     #GS処理指定
        "GridSearch_T_Output" : OutputLog_SishuESN_2024_06_01_10_30             #GS出力処理指定
        })()


def Project_GridSearch_NoChialvo_kOnly():
    """ 
    Sishu提案モデル（結合形態のみ指定、強度は乱数）のNRMSEやMC測定
    """
    GridSearch_No.Normal({
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        "Project_F_NRMSE" : True,                                   #NRMSEを調査するか
        "Project_F_MemoryCapacity" : True,                          #MCを調査するか
        "Project_F_MLE" : True,                                     #MLEを調査するか
        "Project_F_CovMatrixRank" : True,                  #Covariance Matrix Rankを調査するか
        "Project_F_DelayCapacity" : True,                   #Delay Capacityを調査するか

        "Project_F_OutputResults" : True,                           #各評価地点でパラメータを出力するか

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #軸パラメータ（41）
        #"_Param_k" : [-5.0, -4.75, -4.5, -4.25, -4.0, -3.75, -3.5, -3.25, -3.0, -2.75, -2.5, -2.25, -2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0]
        #狭い範囲の軸パラメータ（41）
        #"_Param_k" : [-5.0, -4.9, -4.8, -4.7, -4.6, -4.5, -4.4, -4.3, -4.2, -4.1, -4.0, -3.9, -3.8, -3.7, -3.6, -3.5, -3.4, -3.3, -3.2, -3.1, -3.0, -2.9, -2.8, -2.7, -2.6, -2.5, -2.4, -2.3, -2.2, -2.1, -2.0, -1.9, -1.8, -1.7, -1.6, -1.5, -1.4, -1.3, -1.2, -1.1, -1.0],
        #クソ長(401)
        #"_Param_k" : [-5.00,-4.99,-4.98,-4.97,-4.96,-4.95,-4.94,-4.93,-4.92,-4.91,-4.90,-4.89,-4.88,-4.87,-4.86,-4.85,-4.84,-4.83,-4.82,-4.81,-4.80,-4.79,-4.78,-4.77,-4.76,-4.75,-4.74,-4.73,-4.72,-4.71,-4.70,-4.69,-4.68,-4.67,-4.66,-4.65,-4.64,-4.63,-4.62,-4.61,-4.60,-4.59,-4.58,-4.57,-4.56,-4.55,-4.54,-4.53,-4.52,-4.51,-4.50,-4.49,-4.48,-4.47,-4.46,-4.45,-4.44,-4.43,-4.42,-4.41,-4.40,-4.39,-4.38,-4.37,-4.36,-4.35,-4.34,-4.33,-4.32,-4.31,-4.30,-4.29,-4.28,-4.27,-4.26,-4.25,-4.24,-4.23,-4.22,-4.21,-4.20,-4.19,-4.18,-4.17,-4.16,-4.15,-4.14,-4.13,-4.12,-4.11,-4.10,-4.09,-4.08,-4.07,-4.06,-4.05,-4.04,-4.03,-4.02,-4.01,-4.00,-3.99,-3.98,-3.97,-3.96,-3.95,-3.94,-3.93,-3.92,-3.91,-3.90,-3.89,-3.88,-3.87,-3.86,-3.85,-3.84,-3.83,-3.82,-3.81,-3.80,-3.79,-3.78,-3.77,-3.76,-3.75,-3.74,-3.73,-3.72,-3.71,-3.70,-3.69,-3.68,-3.67,-3.66,-3.65,-3.64,-3.63,-3.62,-3.61,-3.60,-3.59,-3.58,-3.57,-3.56,-3.55,-3.54,-3.53,-3.52,-3.51,-3.50,-3.49,-3.48,-3.47,-3.46,-3.45,-3.44,-3.43,-3.42,-3.41,-3.40,-3.39,-3.38,-3.37,-3.36,-3.35,-3.34,-3.33,-3.32,-3.31,-3.30,-3.29,-3.28,-3.27,-3.26,-3.25,-3.24,-3.23,-3.22,-3.21,-3.20,-3.19,-3.18,-3.17,-3.16,-3.15,-3.14,-3.13,-3.12,-3.11,-3.10,-3.09,-3.08,-3.07,-3.06,-3.05,-3.04,-3.03,-3.02,-3.01,-3.00,-2.99,-2.98,-2.97,-2.96,-2.95,-2.94,-2.93,-2.92,-2.91,-2.90,-2.89,-2.88,-2.87,-2.86,-2.85,-2.84,-2.83,-2.82,-2.81,-2.80,-2.79,-2.78,-2.77,-2.76,-2.75,-2.74,-2.73,-2.72,-2.71,-2.70,-2.69,-2.68,-2.67,-2.66,-2.65,-2.64,-2.63,-2.62,-2.61,-2.60,-2.59,-2.58,-2.57,-2.56,-2.55,-2.54,-2.53,-2.52,-2.51,-2.50,-2.49,-2.48,-2.47,-2.46,-2.45,-2.44,-2.43,-2.42,-2.41,-2.40,-2.39,-2.38,-2.37,-2.36,-2.35,-2.34,-2.33,-2.32,-2.31,-2.30,-2.29,-2.28,-2.27,-2.26,-2.25,-2.24,-2.23,-2.22,-2.21,-2.20,-2.19,-2.18,-2.17,-2.16,-2.15,-2.14,-2.13,-2.12,-2.11,-2.10,-2.09,-2.08,-2.07,-2.06,-2.05,-2.04,-2.03,-2.02,-2.01,-2.00,-1.99,-1.98,-1.97,-1.96,-1.95,-1.94,-1.93,-1.92,-1.91,-1.90,-1.89,-1.88,-1.87,-1.86,-1.85,-1.84,-1.83,-1.82,-1.81,-1.80,-1.79,-1.78,-1.77,-1.76,-1.75,-1.74,-1.73,-1.72,-1.71,-1.70,-1.69,-1.68,-1.67,-1.66,-1.65,-1.64,-1.63,-1.62,-1.61,-1.60,-1.59,-1.58,-1.57,-1.56,-1.55,-1.54,-1.53,-1.52,-1.51,-1.50,-1.49,-1.48,-1.47,-1.46,-1.45,-1.44,-1.43,-1.42,-1.41,-1.40,-1.39,-1.38,-1.37,-1.36,-1.35,-1.34,-1.33,-1.32,-1.31,-1.30,-1.29,-1.28,-1.27,-1.26,-1.25,-1.24,-1.23,-1.22,-1.21,-1.20,-1.19,-1.18,-1.17,-1.16,-1.15,-1.14,-1.13,-1.12,-1.11,-1.10,-1.09,-1.08,-1.07,-1.06,-1.05,-1.04,-1.03,-1.02,-1.01,-1.00],
        
        "_Param_k" : [-5.00,-4.99,-4.98,-4.97,-4.96,-4.95,-4.94,-4.93,-4.92,-4.91,-4.90,-4.89,-4.88,-4.87,-4.86,-4.85,-4.84,-4.83,-4.82,-4.81,-4.80,-4.79,-4.78,-4.77,-4.76,-4.75,-4.74,-4.73,-4.72,-4.71,-4.70,-4.69,-4.68,-4.67,-4.66,-4.65,-4.64,-4.63,-4.62,-4.61,-4.60,-4.59,-4.58,-4.57,-4.56,-4.55,-4.54,-4.53,-4.52,-4.51,-4.50,-4.49,-4.48,-4.47,-4.46,-4.45,-4.44,-4.43,-4.42,-4.41,-4.40,-4.39,-4.38,-4.37,-4.36,-4.35,-4.34,-4.33,-4.32,-4.31,-4.30,-4.29,-4.28,-4.27,-4.26,-4.25,-4.24,-4.23,-4.22,-4.21,-4.20,-4.19,-4.18,-4.17,-4.16,-4.15,-4.14,-4.13,-4.12,-4.11,-4.10,-4.09,-4.08,-4.07,-4.06,-4.05,-4.04,-4.03,-4.02,-4.01,-4.00,-3.99,-3.98,-3.97,-3.96,-3.95,-3.94,-3.93,-3.92,-3.91,-3.90,-3.89,-3.88,-3.87,-3.86,-3.85,-3.84,-3.83,-3.82,-3.81,-3.80,-3.79,-3.78,-3.77,-3.76,-3.75,-3.74,-3.73,-3.72,-3.71,-3.70,-3.69,-3.68,-3.67,-3.66,-3.65,-3.64,-3.63,-3.62,-3.61,-3.60,-3.59,-3.58,-3.57,-3.56,-3.55,-3.54,-3.53,-3.52,-3.51,-3.50,-3.49,-3.48,-3.47,-3.46,-3.45,-3.44,-3.43,-3.42,-3.41,-3.40,-3.39,-3.38,-3.37,-3.36,-3.35,-3.34,-3.33,-3.32,-3.31,-3.30,-3.29,-3.28,-3.27,-3.26,-3.25,-3.24,-3.23,-3.22,-3.21,-3.20,-3.19,-3.18,-3.17,-3.16,-3.15,-3.14,-3.13,-3.12,-3.11,-3.10,-3.09,-3.08,-3.07,-3.06,-3.05,-3.04,-3.03,-3.02,-3.01,-3.00,-2.99,-2.98,-2.97,-2.96,-2.95,-2.94,-2.93,-2.92,-2.91,-2.90,-2.89,-2.88,-2.87,-2.86,-2.85,-2.84,-2.83,-2.82,-2.81,-2.80,-2.79,-2.78,-2.77,-2.76,-2.75,-2.74,-2.73,-2.72,-2.71,-2.70,-2.69,-2.68,-2.67,-2.66,-2.65,-2.64,-2.63,-2.62,-2.61,-2.60,-2.59,-2.58,-2.57,-2.56,-2.55,-2.54,-2.53,-2.52,-2.51,-2.50,-2.49,-2.48,-2.47,-2.46,-2.45,-2.44,-2.43,-2.42,-2.41,-2.40,-2.39,-2.38,-2.37,-2.36,-2.35,-2.34,-2.33,-2.32,-2.31,-2.30,-2.29,-2.28,-2.27,-2.26,-2.25,-2.24,-2.23,-2.22,-2.21,-2.20,-2.19,-2.18,-2.17,-2.16,-2.15,-2.14,-2.13,-2.12,-2.11,-2.10,-2.09,-2.08,-2.07,-2.06,-2.05,-2.04,-2.03,-2.02,-2.01,-2.00,-1.99,-1.98,-1.97,-1.96,-1.95,-1.94,-1.93,-1.92,-1.91,-1.90,-1.89,-1.88,-1.87,-1.86,-1.85,-1.84,-1.83,-1.82,-1.81,-1.80,-1.79,-1.78,-1.77,-1.76,-1.75,-1.74,-1.73,-1.72,-1.71,-1.70,-1.69,-1.68,-1.67,-1.66,-1.65,-1.64,-1.63,-1.62,-1.61,-1.60,-1.59,-1.58,-1.57,-1.56,-1.55,-1.54,-1.53,-1.52,-1.51,-1.50,-1.49,-1.48,-1.47,-1.46,-1.45,-1.44,-1.43,-1.42,-1.41,-1.40,-1.39,-1.38,-1.37,-1.36,-1.35,-1.34,-1.33,-1.32,-1.31,-1.30,-1.29,-1.28,-1.27,-1.26,-1.25,-1.24,-1.23,-1.22,-1.21,-1.20,-1.19,-1.18,-1.17,-1.16,-1.15,-1.14,-1.13,-1.12,-1.11,-1.10,-1.09,-1.08,-1.07,-1.06,-1.05,-1.04,-1.03,-1.02,-1.01,-1.00],
        "_Param_Model" : ["EMChialvo"],

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        "GridSearch_MachineName" : "EMChialvo",                            #計算機名
        "GridSearch_StartPoint" : 0,                                #担当開始ポイントインデックス
        "GridSearch_EndPoint" : 40100,                                #担当終了ポイントインデックス
        
        "GridSearch_MultiThread" : 12,                               #スレッド数（0で逐次処理）初期値:2, 自宅PC:4, 研究室PC, 12
        "GridSearch_MaxNumberInOneFile" : 10,                        #１ファイルの最大のポイント数 初期値:5
        "GridSearch_MaxNumberInOnePool" : 1000,                       #１プール（並列する）最大のポイント数（この分メモリを消費） 初期値:50
        "GridSearch_NumberOfSamples" : 100,                           #サンプル数
        "GridSearch_ProjectName" : "EMChialvo_kOnly",                    #プロジェクト名
        "GridSearch_ProjectDate" : "2025_04_09_16_00",              #プロジェクト日時
        "GridSearch_T_Process" : Process_SishuESN_Only_k_GridSearch,     #GS処理指定
        "GridSearch_T_Output" : OutputLog_SishuESN_Only_k_2024_06_01_10_30             #GS出力処理指定
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
                + "NRMSE : %f, LogNRMSE : %f, TimeForTraining : %f, TimeForTesting : %f, MC : %f, MLE : %f, CovRank : %f, DC : %f\n")
                    %(Results["IndexInChank"], Results["NumPointsInChank"], Results["Index"], Results["Samples"],
                    self.getTag(chank_i),
                    Results["NRMSE_R_NRMSE"] if "NRMSE_R_NRMSE" in Results else 0,
                    Results["NRMSE_R_LogNRMSE"] if "NRMSE_R_LogNRMSE" in Results else 0,
                    Results["NRMSE_R_TimeForTraining"] if "NRMSE_R_TimeForTraining" in Results else 0,
                    Results["NRMSE_R_TimeForTesting"] if "NRMSE_R_TimeForTesting" in Results else 0,
                    Results["MemoryCapacity_R_MC"] if "MemoryCapacity_R_MC" in Results else 0,
                    Results["MLE_R_MLE"] if "MLE_R_MLE" in Results else 0,
                    Results["CovMatrixRank_R_CovMatrixRank"] if "CovMatrixRank_R_CovMatrixRank" in Results else 0,
                    Results["DelayCapacity_R_DelayCapacity"] if "DelayCapacity_R_DelayCapacity" in Results else 0
                    ))
    
        return Results

    #ポイントのパラメータ設定
    def Exp_GridWorld2D(self, chank_i, gs_param):
        #各軸パラメータ
        Param_Rho = chank_i["Rho"]
        Param_k = chank_i["k"]

        if chank_i["Model"] == "EMChialvo":
            Param_Model = Model_No.Model_EMChialvo
            
        #共通パラメータ
        Param = {
            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            "Task_SinCurve_RK_h" : 0.01,                         #ルンゲクッタ法刻み幅
            
            "Task_MC_Tau" : 10,                                  #遅延量，MCのτ
            
            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            #ルンゲクッタ法通常レスラー方程式
            "Task_NormalRosslor_Scale" : 1 / 30,
            "Task_NormalRosslor_Dt" : 0.05,
            "Task_NormalRosslor_Tau" : 5,
            "Task_NormalRosslor_InitTerm" : 1000,

            "Task_NormalRosslor_a" : 0.2,
            "Task_NormalRosslor_b" : 0.2,
            "Task_NormalRosslor_c" : 5.7,

            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            #ルンゲクッタ法通常ローレンツ方程式
            "Task_NormalLorenz_Scale" : 1/50,
            "Task_NormalLorenz_Dt" : 0.01,
            "Task_NormalLorenz_Tau" : 5,
            "Task_NormalLorenz_InitTerm" : 1000,

            "Task_NormalLorenz_Sigma" : 10,
            "Task_NormalLorenz_Beta" : 8/3,
            "Task_NormalLorenz_Rho" : 28,

            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            #ロジスティック写像のパラメータ
            "Task_LogisticEquation_A" : 4,                      #ロジスティック写像（離散）の大きさ
            "Task_LogisticEquation_Tau" : 1,                    #どれくらい先を予測するか

            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            #ローレンツ方程式96のパラメータ
            "Task_Lorenz96_Scale" : 1/50,                       #ローレンツ方程式96の大きさ
            "Task_Lorenz96_Dt" : 0.01,                          #時間刻み幅
            "Task_Lorenz96_Tau" : 5,                            #どれくらい先を予測するか
            "Task_Lorenz96_InitTerm" : 1000,                    #初期状態排除期間
            "Task_Lorenz96_N" : 10,                             #ニューロン数
            "Task_Lorenz96_F" : 8,                              #大きさ？

            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            #離散時間のマッキー・グラス方程式のパラメータ
            "Task_MackeyGlass_Scale" : 1/ 50,                   #信号のスケール
            "Task_Predict_Tau" : 2,                             #どれくらい先を予測するか
            "Task_MackeyGlass_Tau": 0,                          #マッキー・グラスの遅延量
            "Task_MackeyGlass_InitTerm": 1000,                  #初期状態排除期間

            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            #連続時間のマッキー・グラス方程式のパラメータ
            "Task_PredictDDE_Tau" : 5,                          #どれくらい先を予測するか
            
            "Task_MackeyGlassDDE_Scale" : 1 / 30,                 #信号のスケール
            "Task_MackeyGlassDDE_Dt" : 0.5,                    #時間刻み幅
            
            "Task_MackeyGlassDDE_Beta" : 0.2,                       #γ（？）:0.1
            "Task_MackeyGlassDDE_Gamma" : 0.1,                     #β（？）:0.2
            "Task_MackeyGlassDDE_N" : 10,                       #乗数: 10
            "Task_MackeyGlassDDE_Tau" : 17,                      #マッキー・グラスの遅延量 :17
            "Task_MackeyGlassDDE_InitTerm" : 1000,              #初期状態排除期間

            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            "Model_EMChialvo_D_u" : 1,                          #入力信号次元
            "Model_EMChialvo_D_x" : 100,                        #ニューロン数
            "Model_EMChialvo_D_y" : 1,                          #出力信号次元

            "Model_Reservoir_Neurons" : 10,                     #描写するリザバー層のニューロン数
            "Model_EMChialvo_InputScale" : 0.1,                 #入力スケーリング
            
            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            "Model_EMChialvo_a" : 0.89,                         #変数:a
            "Model_EMChialvo_b" : 0.6,                          #変数:b
            "Model_EMChialvo_c" : 0.28,                         #変数:c
            "Model_EMChialvo_k0" : 0.04,                        #変数:k0

            "Model_EMChialvo_k1" : 0.1,                         #変数:k1
            "Model_EMChialvo_k2" : 0.2,                         #変数:k2
            "Model_EMChialvo_alpha" : 0.1,                      #変数:alpha
            "Model_EMChialvo_beta" : 0.2,                       #変数:beta

            "Model_EMChialvo_k" : Param_k,                         #変数:k
            "Model_EMChialvo_Rho" : Param_Rho,                      #スペクトル半径

            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # "Module_Reservoir" に直接渡す
            "EMChialvo_Reservoir_Density" : 1,                          #結合密度
            
            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            "LinerTransformer_Beta" : 2.0 * 1e-01,                      #正規化係数
            }
    
        Results = dict()
        #NRMSE評価
        if gs_param["Project_F_NRMSE"]:
            param = Param.copy()
            param.update({
                #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                "NRMSE_F_OutputLog" : False,                        #経過の出力を行うか

                "NRMSE_D_u" : 1,                            #入力信号次元
                "NRMSE_D_x" : 100,
                "NRMSE_D_y" : 1,                            #出力信号次元

                "NRMSE_Length_Burnin" : 1000,                       #空走用データ時間長
                "NRMSE_Length_Train" : 20000,                       #学習用データ時間長
                "NRMSE_Length_Test" : 5000,                         #評価用データ時間長
                "NRMSE_T_Task" : Task_No.Task_NormalLorenz,                                #評価用タスク（Type型）
                "NRMSE_T_Model" : Param_Model,                                      #モデル（Type型）
                "NRMSE_T_Output" : Output_No.Output_Single_NRMSE_2023_04_19_15_25,     #作図出力（Type型）
        
                #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                "DirPath_Project" : self.GridSearch.Dir_Points_Branch,              #GSのパスを使用
                "AxisTag" : self.getTag(chank_i),                                   #GSのパラメータを図題に使用

                "NRMSE_F_OutputCharts" : False,                      #図の出力フラグ
                "NRMSE_F_OutputCharts_UYYdEWaves" : False,           #入出力＆誤差波形図の出力フラグ

                })
            Results.update(Evaluation_No.Evaluation_NRMSE(param)())
    
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
                "MemoryCapacity_T_Task" : Task_No.Task_MC,                                     #評価用タスク（Type型）
                "MemoryCapacity_T_Model" : Param_Model,                                     #モデル（Type型）
                "MemoryCapacity_T_Output" : Output_No.Output_Single_MC_2023_05_25_13_28,       #作図出力（Type型）
        
                #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                "DirPath_Project" : self.GridSearch.Dir_Points_Branch,                      #GSのパスを使用
                "AxisTag" : self.getTag(chank_i),                                           #GSのパラメータを図題に使用

                "MemoryCapacity_F_OutputCharts" : False,             #図の出力フラグ
                "MemoryCapacity_F_OutputCharts_MCGraph" : False,     #MC曲線の出力フラグ

                })
            Results.update(Evaluation_No.Evaluation_MC(param)())

        #MLE評価
        if gs_param["Project_F_MLE"]:
            param = Param.copy()
            param.update({
                #==========================================================================================
                "MLE_F_OutputLog" : False,                        #経過の出力を行うか

                "MLE_D_u" : 1,                                   #入力信号次元
                "MLE_D_x" : 100,                                 #リザバー層次元
                "MLE_D_y" : 1,                                   #出力信号次元

                "MLE_Length_Burnin" : 1000,                      #空走用データ時間長
                "MLE_Length_Test" : 5000,                        #評価用データ時間長

                "MLE_Epsilon" : 1e-08,                            #摂動の大きさ

                "MLE_T_Task" : Task_No.Task_NormalLorenz,         #評価用タスク（Type型）
                "MLE_T_Model" : Param_Model,
                "MLE_T_Output" : Output_No.Output_Single_MLE_2023_07_08_17_12,

                #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                "DirPath_Project" : self.GridSearch.Dir_Points_Branch,              #GSのパスを使用
                "AxisTag" : self.getTag(chank_i),                                   #GSのパラメータを図題に使用

                "MLE_F_OutputCharts" : False,                      #図の出力フラグ
                "MLE_F_OutputCharts_MLEWaves" : False,             #MLE曲線の出力フラグ

                })
            Results.update(Evaluation_No.Evaluation_MLE(param)())

        #CovMatrixRank評価
        if gs_param["Project_F_CovMatrixRank"]:
            param = Param.copy()
            param.update({
                #==========================================================================================
                "CovMatrixRank_F_OutputLog" : False,                        #経過の出力を行うか

                "CovMatrixRank_D_u" : 1,                                   #入力信号次元
                "CovMatrixRank_D_x" : 100,                                 #リザバー層次元
                "CovMatrixRank_D_y" : 1,                                   #出力信号次元

                "CovMatrixRank_Length_Burnin" : 1000,                      #空走用データ時間長
                "CovMatrixRank_Length_Test" : 5000,                        #評価用データ時間長

                "CovMatrixRank_T_Task" : Task_No.Task_NormalLorenz,         #評価用タスク（Type型）
                "CovMatrixRank_T_Model" : Param_Model,
                "CovMatrixRank_T_Output" : Output_No.Output_Single_CovMatrixRank_2025_03_15_15_32,

                #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                "DirPath_Project" : self.GridSearch.Dir_Points_Branch,              #GSのパスを使用
                "AxisTag" : self.getTag(chank_i),                                   #GSのパラメータを図題に使用

                "CovMatrixRank_F_OutputCharts" : False,                      #図の出力フラグ
                "CovMatrixRank_F_OutputCharts_CovMatrixRankWaves" : False,    #CovMatrixRank曲線の出力フラグ

                })
            Results.update(Evaluation_No.Evaluation_CovMatrixRank(param)())

        #DC評価
        if gs_param["Project_F_DelayCapacity"]:
            param = Param.copy()
            param.update({
                #==========================================================================================
                "DelayCapacity_F_OutputLog" : False,                        #経過の出力を行うか

                "DelayCapacity_D_u" : 1,                                   #入力信号次元
                "DelayCapacity_D_x" : 100,                                 #リザバー層次元
                "DelayCapacity_D_y" : 1,                                   #出力信号次元

                "DelayCapacity_Length_Burnin" : 1000,                      #空走用データ時間長
            
                "DelayCapacity_Length_Tdc" : 5000,                        #評価用データ時間長
                "DelayCapacity_Length_Taumax" : 50,

                "DelayCapacity_T_Task" : Task_No.Task_NormalLorenz,         #評価用タスク（Type型）
                "DelayCapacity_T_Model" : Param_Model,
                "DelayCapacity_T_Output" : Output_No.Output_Single_DelayCapacity_2025_03_15_15_32,

                #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                "DirPath_Project" : self.GridSearch.Dir_Points_Branch,              #GSのパスを使用
                "AxisTag" : self.getTag(chank_i),                                   #GSのパラメータを図題に使用

                "DelayCapacity_F_OutputCharts" : False,             #図の出力フラグ
                "DelayCapacity_F_OutputCharts_DCGraph" : False,     #DC曲線の出力フラグ
                
                })
            Results.update(Evaluation_No.Evaluation_DelayCapacity(param)())

        #地点のパラメータを改めて設定
        if gs_param["Project_F_OutputResults"]:
            self.GridSearch.CSV_Point_Param.Save(Results)

        return Results

    #軸パラメータを文字列化
    def getTag(self, param: dict) -> str:
        return "Rho : " + str(param["Rho"]) + " k : " + str(param["k"]) + " Model : " + param["Model"]


class Process_SishuESN_Only_k_GridSearch:
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
                + "NRMSE : %f, LogNRMSE : %f, TimeForTraining : %f, TimeForTesting : %f, MC : %f, MLE : %f, CovRank : %f, DC : %f\n")
                    %(Results["IndexInChank"], Results["NumPointsInChank"], Results["Index"], Results["Samples"],
                    self.getTag(chank_i),
                    Results["NRMSE_R_NRMSE"] if "NRMSE_R_NRMSE" in Results else 0,
                    Results["NRMSE_R_LogNRMSE"] if "NRMSE_R_LogNRMSE" in Results else 0,
                    Results["NRMSE_R_TimeForTraining"] if "NRMSE_R_TimeForTraining" in Results else 0,
                    Results["NRMSE_R_TimeForTesting"] if "NRMSE_R_TimeForTesting" in Results else 0,
                    Results["MemoryCapacity_R_MC"] if "MemoryCapacity_R_MC" in Results else 0,
                    Results["MLE_R_MLE"] if "MLE_R_MLE" in Results else 0,
                    Results["CovMatrixRank_R_CovMatrixRank"] if "CovMatrixRank_R_CovMatrixRank" in Results else 0,
                    Results["DelayCapacity_R_DelayCapacity"] if "DelayCapacity_R_DelayCapacity" in Results else 0
                    ))
    
        return Results

    #ポイントのパラメータ設定
    def Exp_GridWorld2D(self, chank_i, gs_param):
        #各軸パラメータ
        Param_k = chank_i["k"]
        if chank_i["Model"] == "EMChialvo":
            Param_Model = Model_No.Model_EMChialvo
            
        #共通パラメータ
        Param = {
            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            "Task_SinCurve_RK_h" : 0.01,                         #ルンゲクッタ法刻み幅
            
            "Task_MC_Tau" : 10,                                  #遅延量，MCのτ
            
            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            #ルンゲクッタ法通常レスラー方程式
            "Task_NormalRosslor_Scale" : 1 / 30,
            "Task_NormalRosslor_Dt" : 0.05,
            "Task_NormalRosslor_Tau" : 5,
            "Task_NormalRosslor_InitTerm" : 1000,

            "Task_NormalRosslor_a" : 0.2,
            "Task_NormalRosslor_b" : 0.2,
            "Task_NormalRosslor_c" : 5.7,

            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            #ルンゲクッタ法通常ローレンツ方程式
            "Task_NormalLorenz_Scale" : 1/50,
            "Task_NormalLorenz_Dt" : 0.01,
            "Task_NormalLorenz_Tau" : 5,
            "Task_NormalLorenz_InitTerm" : 1000,

            "Task_NormalLorenz_Sigma" : 10,
            "Task_NormalLorenz_Beta" : 8/3,
            "Task_NormalLorenz_Rho" : 28,

            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            #ロジスティック写像のパラメータ
            "Task_LogisticEquation_A" : 4,                      #ロジスティック写像（離散）の大きさ
            "Task_LogisticEquation_Tau" : 1,                    #どれくらい先を予測するか

            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            #ローレンツ方程式96のパラメータ
            "Task_Lorenz96_Scale" : 1/50,                       #ローレンツ方程式96の大きさ
            "Task_Lorenz96_Dt" : 0.01,                          #時間刻み幅
            "Task_Lorenz96_Tau" : 5,                            #どれくらい先を予測するか
            "Task_Lorenz96_InitTerm" : 1000,                    #初期状態排除期間
            "Task_Lorenz96_N" : 10,                             #ニューロン数
            "Task_Lorenz96_F" : 8,                              #大きさ？

            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            #離散時間のマッキー・グラス方程式のパラメータ
            "Task_MackeyGlass_Scale" : 1/ 50,                   #信号のスケール
            "Task_Predict_Tau" : 2,                             #どれくらい先を予測するか
            "Task_MackeyGlass_Tau": 0,                          #マッキー・グラスの遅延量
            "Task_MackeyGlass_InitTerm": 1000,                  #初期状態排除期間

            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            #連続時間のマッキー・グラス方程式のパラメータ
            "Task_PredictDDE_Tau" : 5,                          #どれくらい先を予測するか
            
            "Task_MackeyGlassDDE_Scale" : 1 / 30,                 #信号のスケール
            "Task_MackeyGlassDDE_Dt" : 0.5,                    #時間刻み幅
            
            "Task_MackeyGlassDDE_Beta" : 0.2,                       #γ（？）:0.1
            "Task_MackeyGlassDDE_Gamma" : 0.1,                     #β（？）:0.2
            "Task_MackeyGlassDDE_N" : 10,                       #乗数: 10
            "Task_MackeyGlassDDE_Tau" : 17,                      #マッキー・グラスの遅延量 :17
            "Task_MackeyGlassDDE_InitTerm" : 1000,              #初期状態排除期間
        
            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            "Model_EMChialvo_D_u" : 1,                          #入力信号次元
            "Model_EMChialvo_D_x" : 100,                        #ニューロン数
            "Model_EMChialvo_D_y" : 1,                          #出力信号次元

            "Model_Reservoir_Neurons" : 10,                     #描写するリザバー層のニューロン数
            "Model_EMChialvo_InputScale" : 0.1,                 #入力スケーリング
            
            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            "Model_EMChialvo_a" : 0.89,                         #変数:a
            "Model_EMChialvo_b" : 0.6,                          #変数:b
            "Model_EMChialvo_c" : 0.28,                         #変数:c
            "Model_EMChialvo_k0" : 0.04,                        #変数:k0

            "Model_EMChialvo_k1" : 0.1,                         #変数:k1
            "Model_EMChialvo_k2" : 0.2,                         #変数:k2
            "Model_EMChialvo_alpha" : 0.1,                      #変数:alpha
            "Model_EMChialvo_beta" : 0.2,                       #変数:beta

            "Model_EMChialvo_k" : Param_k,                         #変数:k
            "Model_EMChialvo_Rho" : 0.01,                      #スペクトル半径

            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # "Module_Reservoir" に直接渡す
            "EMChialvo_Reservoir_Density" : 1,                          #結合密度
            
            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            "LinerTransformer_Beta" : 2.0 * 1e-01,                      #正規化係数
            }
    
        Results = dict()
        #NRMSE評価
        if gs_param["Project_F_NRMSE"]:
            param = Param.copy()
            param.update({
                #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                "NRMSE_F_OutputLog" : False,                        #経過の出力を行うか

                "NRMSE_D_u" : 1,                            #入力信号次元
                "NRMSE_D_x" : 100,
                "NRMSE_D_y" : 1,                            #出力信号次元

                "NRMSE_Length_Burnin" : 1000,                       #空走用データ時間長
                "NRMSE_Length_Train" : 20000,                       #学習用データ時間長
                "NRMSE_Length_Test" : 5000,                         #評価用データ時間長
                "NRMSE_T_Task" : Task_No.Task_NormalLorenz,                                #評価用タスク（Type型）
                "NRMSE_T_Model" : Param_Model,                                      #モデル（Type型）
                "NRMSE_T_Output" : Output_No.Output_Single_NRMSE_2023_04_19_15_25,     #作図出力（Type型）
        
                #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                "DirPath_Project" : self.GridSearch.Dir_Points_Branch,              #GSのパスを使用
                "AxisTag" : self.getTag(chank_i),                                   #GSのパラメータを図題に使用

                "NRMSE_F_OutputCharts" : False,                      #図の出力フラグ
                "NRMSE_F_OutputCharts_UYYdEWaves" : False,           #入出力＆誤差波形図の出力フラグ

                })
            Results.update(Evaluation_No.Evaluation_NRMSE(param)())
    
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
                "MemoryCapacity_T_Task" : Task_No.Task_MC,                                     #評価用タスク（Type型）
                "MemoryCapacity_T_Model" : Param_Model,                                     #モデル（Type型）
                "MemoryCapacity_T_Output" : Output_No.Output_Single_MC_2023_05_25_13_28,       #作図出力（Type型）
        
                #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                "DirPath_Project" : self.GridSearch.Dir_Points_Branch,                      #GSのパスを使用
                "AxisTag" : self.getTag(chank_i),                                           #GSのパラメータを図題に使用

                "MemoryCapacity_F_OutputCharts" : False,             #図の出力フラグ
                "MemoryCapacity_F_OutputCharts_MCGraph" : False,     #MC曲線の出力フラグ

                })
            Results.update(Evaluation_No.Evaluation_MC(param)())

        #MLE評価
        if gs_param["Project_F_MLE"]:
            param = Param.copy()
            param.update({
                #==========================================================================================
                "MLE_F_OutputLog" : False,                        #経過の出力を行うか

                "MLE_D_u" : 1,                                   #入力信号次元
                "MLE_D_x" : 100,                                 #リザバー層次元
                "MLE_D_y" : 1,                                   #出力信号次元

                "MLE_Length_Burnin" : 1000,                      #空走用データ時間長
                "MLE_Length_Test" : 5000,                        #評価用データ時間長

                "MLE_Epsilon" : 1e-08,                            #摂動の大きさ

                "MLE_T_Task" : Task_No.Task_NormalLorenz,         #評価用タスク（Type型）
                "MLE_T_Model" : Param_Model,
                "MLE_T_Output" : Output_No.Output_Single_MLE_2023_07_08_17_12,

                #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                "DirPath_Project" : self.GridSearch.Dir_Points_Branch,              #GSのパスを使用
                "AxisTag" : self.getTag(chank_i),                                   #GSのパラメータを図題に使用

                "MLE_F_OutputCharts" : False,                      #図の出力フラグ
                "MLE_F_OutputCharts_MLEWaves" : False,             #MLE曲線の出力フラグ

                })
            Results.update(Evaluation_No.Evaluation_MLE(param)())

        #CovMatrixRank評価
        if gs_param["Project_F_CovMatrixRank"]:
            param = Param.copy()
            param.update({
                #==========================================================================================
                "CovMatrixRank_F_OutputLog" : False,                        #経過の出力を行うか

                "CovMatrixRank_D_u" : 1,                                   #入力信号次元
                "CovMatrixRank_D_x" : 100,                                 #リザバー層次元
                "CovMatrixRank_D_y" : 1,                                   #出力信号次元

                "CovMatrixRank_Length_Burnin" : 1000,                      #空走用データ時間長
                "CovMatrixRank_Length_Test" : 5000,                        #評価用データ時間長

                "CovMatrixRank_T_Task" : Task_No.Task_NormalLorenz,         #評価用タスク（Type型）
                "CovMatrixRank_T_Model" : Param_Model,
                "CovMatrixRank_T_Output" : Output_No.Output_Single_CovMatrixRank_2025_03_15_15_32,

                #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                "DirPath_Project" : self.GridSearch.Dir_Points_Branch,              #GSのパスを使用
                "AxisTag" : self.getTag(chank_i),                                   #GSのパラメータを図題に使用

                "CovMatrixRank_F_OutputCharts" : False,                      #図の出力フラグ
                "CovMatrixRank_F_OutputCharts_CovMatrixRankWaves" : False,    #CovMatrixRank曲線の出力フラグ

                })
            Results.update(Evaluation_No.Evaluation_CovMatrixRank(param)())

        #DC評価
        if gs_param["Project_F_DelayCapacity"]:
            param = Param.copy()
            param.update({
                #==========================================================================================
                "DelayCapacity_F_OutputLog" : False,                        #経過の出力を行うか

                "DelayCapacity_D_u" : 1,                                   #入力信号次元
                "DelayCapacity_D_x" : 100,                                 #リザバー層次元
                "DelayCapacity_D_y" : 1,                                   #出力信号次元

                "DelayCapacity_Length_Burnin" : 1000,                      #空走用データ時間長
            
                "DelayCapacity_Length_Tdc" : 5000,                        #評価用データ時間長
                "DelayCapacity_Length_Taumax" : 50,

                "DelayCapacity_T_Task" : Task_No.Task_NormalLorenz,         #評価用タスク（Type型）
                "DelayCapacity_T_Model" : Param_Model,
                "DelayCapacity_T_Output" : Output_No.Output_Single_DelayCapacity_2025_03_15_15_32,

                #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                "DirPath_Project" : self.GridSearch.Dir_Points_Branch,              #GSのパスを使用
                "AxisTag" : self.getTag(chank_i),                                   #GSのパラメータを図題に使用

                "DelayCapacity_F_OutputCharts" : False,             #図の出力フラグ
                "DelayCapacity_F_OutputCharts_DCGraph" : False,     #DC曲線の出力フラグ
                
                })
            Results.update(Evaluation_No.Evaluation_DelayCapacity(param)())

        #地点のパラメータを改めて設定
        if gs_param["Project_F_OutputResults"]:
            self.GridSearch.CSV_Point_Param.Save(Results)

        return Results

    #軸パラメータを文字列化
    def getTag(self, param: dict) -> str:
        return " k : " + str(param["k"]) + " Model : " + param["Model"]


#********************************************************************
#********************************************************************
#GS出力
#====================================================================
class OutputLog_SishuESN_2024_06_01_10_30(Output_No.Output):
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
                ["Memory Capacity", "MemoryCapacity_R_MC", "MC"],
                ["MLE", "MLE_R_MLE", "MLE"],
                ["CovMatrixRank", "CovMatrixRank_R_CovMatrixRank", "CovMatrixRank"],
                ["DelayCapacity", "DelayCapacity_R_DelayCapacity", "DelayCapacity"]]
        
        #指標でループ
        for score in L_Score:
            #データ抽出
            ScoreData = self.MakeData(AllData, score[1])
            SishuESN_Score = ScoreData[:, :, :, 0]
            #指標の文字列一覧（タイトル用，データ，ファイル名用）
            L_Chart = [["SishuESN", SishuESN_Score, "SishuESN"]]
            #フォルダパスを作成
            Plt_Score = self.Parent.Dir_Charts_Branch.AddChild(
                FileAndDir_No.DirNode(score[1]))

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
                        FileAndDir_No.FileNode_plt(chart[2] + "_" + stats[2]))
                    Plt_Chart.Save()

                    #図の設定
                    LabelsX = "Strength of electromagnetic flux : k"                     #x軸ラベル
                    LabelsY = "Scaling size : \u03c3"                     #y軸ラベル
                    FigSize = (16, 9)                   #図の大きさとアスペクト比

                    FontSize_Label = 50                 #ラベルのフォントサイズ 初期36
                    FontSize_Title = 36                 #図題のフォントサイズ 初期36
                    FontSize_Tick = 40                  #目盛のフォントサイズ 初期36

                    FontSize_ColorBarLabel = 45         #カラーバーのフォントサイズ 初期36
                    FontSize_ColorBarTick = 40          #カラーバーの目盛のフォントサイズ 初期36

                    LineWidth = 0                     #線の太さ
                    FileFormat = ".png"                 #ファイル拡張子
                    MapCode = "jet"                     #カラーマップ
                    Annot = False#True                  #数値を表示するか

                    #図題
                    Title = None
                    #作図
                    fig = plt.figure(figsize = FigSize)
                    sub = fig.add_subplot(1, 1, 1)
                    plt.tick_params(labelsize = FontSize_Tick)
                    plt.yscale('log')

                    body = sns.heatmap(stats[1], xticklabels = [-5.0, None, None, None, None, -4.5, None, None, None, None, -4.0, None, None, None, None, -3.5, None, None, None, None, -3.0, None, None, None, None, -2.5, None, None, None, None, -2.0, None, None, None, None, -1.5, None, None, None, None, -1.0], yticklabels = [0.1, None, None, None, None, 0.05, None, None, None, 0.01, None, None, None, None, 0.005, None, None, None, 0.001, None, None, None, None, None, None, None, None, 0.0001], cmap = MapCode, linewidth = LineWidth, annot = Annot, ax = sub)
                    body.collections[0].colorbar.set_label(f'Average {score[0]}', fontsize = FontSize_ColorBarLabel)
                    body.collections[0].colorbar.ax.tick_params(labelsize = FontSize_ColorBarTick)
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

class OutputLog_SishuESN_Only_k_2024_06_01_10_30(Output_No.Output):
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
                ["Memory Capacity", "MemoryCapacity_R_MC", "MC"],
                ["MLE", "MLE_R_MLE", "MLE"],
                ["CovMatrixRank", "CovMatrixRank_R_CovMatrixRank", "CovMatrixRank"],
                ["DelayCapacity", "DelayCapacity_R_DelayCapacity", "DelayCapacity"]]
        
        Count = 0

        #指標でループ
        for score in L_Score:
            #データ抽出
            ScoreData = self.MakeData(AllData, score[1])
            SishuESN_Score = ScoreData[:, :, 0]
            #指標の文字列一覧（タイトル用，データ，ファイル名用）
            L_Chart = [["SishuESN", SishuESN_Score, "SishuESN"]]
            #フォルダパスを作成
            Plt_Score = self.Parent.Dir_Charts_Branch.AddChild(
                FileAndDir_No.DirNode(score[1]))

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
                    AxisY = np.array(stats[0])

                    #ファイル作成
                    Plt_Chart = Plt_Score.AddChild(
                        FileAndDir_No.FileNode_plt(chart[2] + "_" + stats[2]))
                    Plt_Chart.Save()

                    #図の設定
                    LabelsX = "Strength of electromagnetic flux : k"                     #x軸ラベル
                    LabelsY = L_Score[Count][0]                     #y軸ラベル
                    FigSize = (16, 9)                   #図の大きさとアスペクト比
                    FontSize_Label = 45                 #ラベルのフォントサイズ
                    FontSize_Title = 36                 #図題のフォントサイズ
                    FontSize_Tick = 33                  #目盛のフォントサイズ 初期36

                    LineWidth = 3                     #線の太さ
                    FileFormat = ".png"                 #ファイル拡張子
                    MapCode = "jet"                     #カラーマップ
                    Annot = False#True                  #数値を表示するか

                    #図題
                    Title = None
                    #作図
                    fig  = plt.figure(figsize = FigSize)
                    ax = fig.add_subplot(1, 1, 1)

                    ax.plot(self.Param["_Param_k"],stats[1], lw = LineWidth)
                    #plt.gca().invert_yaxis()
                    ax.set_xlabel(LabelsX, fontsize = FontSize_Label)
                    ax.set_ylabel(f"Average {LabelsY}", fontsize = FontSize_Label)
                    plt.tick_params(labelsize = FontSize_Tick)
                    
                    #以下はNRMSE用の設定
                    """
                    ax.set_xticks(np.arange(-5,-0.9,0.25))
                    ax.set_xticks(np.arange(-5,-0.9,0.1),minor=True)
                    ax.set_yticks(np.arange(0.08,0.17,0.01))
                    ax.set_yticks(np.arange(0.08,0.17,0.005),minor=True)
                    """
                    #以下はMLE用の設定
                    """
                    ax.set_xticks(np.arange(-5,-0.9,0.25))
                    ax.set_xticks(np.arange(-5,-0.9,0.1),minor=True)
                    ax.set_yticks(np.arange(-0.1,0.02,0.01))
                    ax.set_yticks(np.arange(-0.1,0.015,0.005),minor=True)
                    """
                    #以下はCovMatrix用の設定
                    """
                    ax.set_xticks(np.arange(-5,-0.9,0.25))
                    ax.set_xticks(np.arange(-5,-0.9,0.1),minor=True)
                    ax.set_yticks(np.arange(84,103,2))
                    ax.set_yticks(np.arange(84,103,1),minor=True)
                    """
                    #以下はDelay Capacity用の設定
                    """
                    ax.set_xticks(np.arange(-5,-0.9,0.25))
                    ax.set_xticks(np.arange(-5,-0.9,0.1),minor=True)
                    ax.set_yticks(np.arange(4,18,2))
                    ax.set_yticks(np.arange(4,17,1),minor=True)
                    """

                    #ax.set_title(Title, fontsize = FontSize_Title)
                    ax.grid()
                    ax.legend()
                    plt.tight_layout()
                    fig.savefig(Plt_Chart.Path + FileFormat)
            
                    #表示せず図を閉じる
                    plt.close()
            Count += 1
    
    #DataFrameをNdarrayに変換し，結果のキーから結果を抽出
    def MakeData(self, data, key):
        return data[key].values.reshape([axis.NumberOfPoints for axis in self.Parent.GridSearch_AxisList])

    #以下フォルダ構造
    def ConstractFileTree_Charts(self):
        pass


#********************************************************************
#********************************************************************