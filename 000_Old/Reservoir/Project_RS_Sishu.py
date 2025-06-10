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

import Task
import Model
import Output
import Evaluation

import Random_Search

#********************************************************************
#Sishu提案モデル（結合形態のみ指定、結合強度は乱数）
def Project_RandomSearch_NRMSE_SishuESN():
    
    RS_Param = {
            "Model_SishuESN_k__01" : [-5.0,-2.0],
            "Model_SishuESN_Rho__01" : [0.001,0.05]
    }
    
    Param = {
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            "Use_model" : "SishuESN",                         #モデル
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

            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            "Model_SishuESN_D_u" : 1,                          #入力信号次元
            "Model_SishuESN_D_x" : 100,                        #ニューロン数
            "Model_SishuESN_D_y" : 1,                          #出力信号次元

            "Model_SishuESN_InputScale" : 0.1,                 #入力スケーリング
            "Model_SishuESN_sigma" : False,    #リング・ネットワークの有無
            "Model_SishuESN_mu" : False,        #スター・ネットワークの有無
            "Model_SishuESN_k" : -3,           #Chialvoの変数：k

            "Model_SishuESN_Rho" : 0.03,                             #スペクトル半径
            "SishuReservoir_Density" : 1,                         #結合密度
            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            "LinerTransformer_Beta" : 0.2,                      #正規化係数
    }
    
    if Param["Use_model"] == "SishuESN":
        Param_Model = Model.Model_SishuESN
    
    #NRMSE評価
    if Param["Evaluation"] == "NRMSE":
        NRMSE_param = Param.copy()
        NRMSE_param.update({
            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            "NRMSE_F_OutputLog" : True,                        #経過の出力を行うか
            "NRMSE_D_u" : 1,                            #入力信号次元
            "NRMSE_D_y" : 1,                            #出力信号次元
            "NRMSE_Length_Burnin" : 1000,                       #空走用データ時間長
            "NRMSE_Length_Train" : 20000,                       #学習用データ時間長
            "NRMSE_Length_Test" : 5000,                         #評価用データ時間長
            "NRMSE_T_Task" : Task.Task_NDLorenz,                                #評価用タスク（Type型）
            "NRMSE_T_Model" : Param_Model,                                      #モデル（Type型）
            "NRMSE_T_Output" : Output.Output_Single_NRMSE_2023_04_19_15_25,     #作図出力（Type型）
    
            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            "DirPath_Project" : "./Results/Project_RS_NRMSE_SishuESN_2024_06_23_01_12",
            
            "NRMSE_F_OutputCharts" : False,                      #図の出力フラグ
            "NRMSE_F_OutputCharts_UYYdEWaves" : False,           #入出力＆誤差波形図の出力フラグ

            })
        Random = Random_Search.RandomSearch(NRMSE_param, RS_Param)
    
    Random.random_search()

#********************************************************************
