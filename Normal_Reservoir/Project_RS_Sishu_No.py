#####################################################################
#信川研ESNプロジェクト用
#制作者：中村亮介 吉原悠斗 吉田聡太 飯沼貴大 田中勝規
#作成日：2024/03/26
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

import numpy as np

import Task_No
import Model_No
import Output_No
import Evaluation_No

import Random_Search_No

#********************************************************************
#Sishu提案モデル（結合形態のみ指定、結合強度は乱数）
def Project_RandomSearch_NRMSE_EMChialvo():
    
    RS_Param = {
            "Model_EMChialvo_k__01" : [-5.0,5.0],
            "Model_EMChialvo_Rho__01" : [0.001,0.01]
    }
    
    Param = {
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            "Use_model" : "EMChialvo",                         #モデル
            "Evaluation" : "NRMSE",                             #評価方法(現在NRMSEのみ対応)
            
            "RandomSearch_target_num_sample" : 300,           #おおよそのサンプル数
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
            "Task_Lorenz_InitTerm" : 1000,                     #初期状態排除期間
            
            "Task_vandelPol_Mu" : [5, 5],                       #振動の係数
            "Task_vandelPol_c" : [0.001, 1],                    #結合の係数
            "Task_vandelPol_TimeScale" : [0.05, 0.5],           #周期の設定係数
            "Task_vandelPol_Dt" : 0.001,                        #時間スケール
            "Task_vandelPol_Init" : [0., 0., 0., 0.001],        #初期状態
            "Task_vandelPol_Tau" : 5,                           #どれくらい先を予測するか
            "Task_vandelPol_InitTerm" : 1000,                   #初期状態排除期間
            
            "Task_LogisticEquation_A" : 4,
            "Task_LogisticEquation_Tau" : 1,

            "Task_Lorenz96_Scale" : 1/50,
            "Task_Lorenz96_Dt" : 0.01,
            "Task_Lorenz96_Tau" : 5,
            "Task_Lorenz96_InitTerm" : 1000,

            "Task_Lorenz96_N" : 10,
            "Task_Lorenz96_F" : 8,
            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            "Model_EMChialvo_D_u" : 1,                          #入力信号次元
            "Model_EMChialvo_D_x" : 100,                        #ニューロン数
            "Model_EMChialvo_D_y" : 1,                          #出力信号次元

            "Model_EMChialvo_Ring" : False,                     #結合の形態を指定するか（値は乱数）
            "Model_EMChialvo_Star" : False,                     #結合の形態を指定するか（値は乱数）
            
            "Model_EMChialvo_InputScale" : 0.1,                 #入力スケーリング
            
            "Model_EMChialvo_a" : 0.89,                         #変数:a
            "Model_EMChialvo_b" : 0.6,                          #変数:b
            "Model_EMChialvo_c" : 0.28,                         #変数:c
            "Model_EMChialvo_k0" : 0.04,                        #変数:k0

            "Model_EMChialvo_k1" : 0.1,                         #変数:k1
            "Model_EMChialvo_k2" : 0.2,                         #変数:k2
            "Model_EMChialvo_alpha" : 0.1,                      #変数:alpha
            "Model_EMChialvo_beta" : 0.2,                       #変数:beta

            "Model_EMChialvo_k" : -3.2,                         #変数:k
            
            "Model_EMChialvo_Rho" : 0.003,                      #スペクトル半径

            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # "Module_Reservoir" に直接渡す
            "EMChialvo_Reservoir_Density" : 1,                          #結合密度
            
            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            "LinerTransformer_Beta" : 0.2,                      #正規化係数
    }
    
    if Param["Use_model"] == "EMChialvo":
        Param_Model = Model_No.Model_EMChialvo
    
    #NRMSE評価
    if Param["Evaluation"] == "NRMSE":
        NRMSE_param = Param.copy()
        NRMSE_param.update({
            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            "NRMSE_F_OutputLog" : True,                        #経過の出力を行うか
            "NRMSE_D_u" : 1,                            #入力信号次元
            "NRMSE_D_x" : 100,
            "NRMSE_D_y" : 1,                            #出力信号次元

            "NRMSE_Length_Burnin" : 1000,                       #空走用データ時間長
            "NRMSE_Length_Train" : 20000,                       #学習用データ時間長
            "NRMSE_Length_Test" : 5000,                         #評価用データ時間長
            "NRMSE_Reservoir_Neurons" : 10,                     #描写するリザバー層のニューロン数            
            
            "NRMSE_T_Task" : Task_No.Task_Lorenz96,                                #評価用タスク（Type型）
            "NRMSE_T_Model" : Param_Model,                                      #モデル（Type型）
            "NRMSE_T_Output" : Output_No.Output_Single_NRMSE_2023_04_19_15_25,     #作図出力（Type型）
    
            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            "DirPath_Project" : "./Results/Project_RS_NRMSE_SishuESN_2024_06_23_01_12",
            
            "NRMSE_F_OutputCharts" : False,                      #図の出力フラグ
            "NRMSE_F_OutputCharts_UYYdEWaves" : False,           #入出力＆誤差波形図の出力フラグ

            })
        Random = Random_Search_No.RandomSearch(NRMSE_param, RS_Param)
    
    Random.random_search()

#********************************************************************
