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

import Evaluation_No
import Task_No
import Model_No
import Output_No

#====================================================================
#プロジェクト

#********************************************************************
#ESNプロジェクト用
def Project_ESN_2025_04_18_14_51():
    """
    ESNプロジェクト用のプロジェクト
    """
    #共通パラメータ
    Param = {
        #==========================================================================================
        "Project_F_NRMSE" : True,                           #NRMSEを調査するか
        "Project_F_MemoryCapacity" : False,                  #MCを調査するか
        "Project_F_MLE" : True,                             #MLE（最大リアプノフ指数）を調査するか
        
        "Project_F_CovMatrixRank" : True,                  #Covariance Matrix Rankを調査するか
        "Project_F_DelayCapacity" : True,                   #Delay Capacityを調査するか

        #------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------
        "Task_SinCurve_RK_h" : 0.01,                         #ルンゲクッタ法刻み幅
        
        "Task_MC_Tau" : 10,                                  #遅延量，MCのτ
        
        #------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------
        #ルンゲクッタ法通常レスラー方程式
        "Task_NormalRosslor_Scale" : 1 / 10,
        "Task_NormalRosslor_Dt" : 0.02,
        "Task_NormalRosslor_Tau" : 5,
        "Task_NormalRosslor_InitTerm" : 1000,

        "Task_NormalRosslor_a" : 0.2,
        "Task_NormalRosslor_b" : 0.2,
        "Task_NormalRosslor_c" : 5.7,

        #------------------------------------------------------------------------------------------
        #ルンゲクッタ法通常ローレンツ方程式
        "Task_NormalLorenz_Scale" : 1 / 20,
        "Task_NormalLorenz_Dt" : 0.005,
        "Task_NormalLorenz_Tau" : 5,
        "Task_NormalLorenz_InitTerm" : 1000,

        "Task_NormalLorenz_Sigma" : 10,
        "Task_NormalLorenz_Beta" : 8/3,
        "Task_NormalLorenz_Rho" : 28,

        #------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------
        #ロジスティック写像のパラメータ
        "Task_LogisticEquation_A" : 4,                      #ロジスティック写像（離散）の大きさ
        "Task_LogisticEquation_Tau" : 1,                    #どれくらい先を予測するか

        #------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------
        #ローレンツ方程式96のパラメータ
        "Task_Lorenz96_Scale" : 1 / 30,                       #ローレンツ方程式96の大きさ
        "Task_Lorenz96_Dt" : 0.01,                          #時間刻み幅
        "Task_Lorenz96_Tau" : 5,                            #どれくらい先を予測するか
        "Task_Lorenz96_InitTerm" : 1000,                    #初期状態排除期間
        "Task_Lorenz96_N" : 10,                             #ニューロン数
        "Task_Lorenz96_F" : 8,                              #大きさ？

        #------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------
        #離散時間のマッキー・グラス方程式のパラメータ
        "Task_MackeyGlass_Scale" : 1/ 50,                   #信号のスケール
        "Task_Predict_Tau" : 2,                             #どれくらい先を予測するか
        "Task_MackeyGlass_Tau": 0,                          #マッキー・グラスの遅延量
        "Task_MackeyGlass_InitTerm": 1000,                  #初期状態排除期間

        #------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------
        #連続時間のマッキー・グラス方程式のパラメータ
        "Task_PredictDDE_Tau" : 5,                          #どれくらい先を予測するか
        
        "Task_MackeyGlassDDE_Scale" : 1 / 30,                 #信号のスケール
        "Task_MackeyGlassDDE_Dt" : 0.2,                    #時間刻み幅
        
        "Task_MackeyGlassDDE_Beta" : 0.2,                       #β:0.2
        "Task_MackeyGlassDDE_Gamma" : 0.1,                     #γ:0.1
        "Task_MackeyGlassDDE_N" : 10,                       #乗数: 10
        "Task_MackeyGlassDDE_Tau" : 22,                      #マッキー・グラスの遅延量 :17
        "Task_MackeyGlassDDE_InitTerm" : 1000,              #初期状態排除期間

        #------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------
        #Van der Pol振動子のパラメータ
        "Task_vandelPol_Mu" : [5, 5],                       #振動の係数
        "Task_vandelPol_c" : [0.001, 1],                    #結合の係数
        "Task_vandelPol_TimeScale" : [0.05, 0.5],           #周期の設定係数
        "Task_vandelPol_Dt" : 0.001,                        #時間スケール
        "Task_vandelPol_Init" : [0., 0., 0., 0.001],        #初期状態
        "Task_vandelPol_Tau" : 5,                           #どれくらい先を予測するか
        "Task_vandelPol_InitTerm" : 1000,                   #初期状態排除期間

        #==========================================================================================
        #==========================================================================================
        "Model_NormalESN_D_u" : 1,                          #入力信号次元
        "Model_NormalESN_D_x" : 100,                        #ニューロン数
        "Model_NormalESN_D_y" : 1,                          #出力信号次元

        "Model_Reservoir_Neurons" : 10,                     #描写するリザバー層のニューロン数
        "Model_NormalESN_InputScale" : 0.1,                 #入力スケーリング

        #------------------------------------------------------------------------------------------
        "SubReservoir_LeakingRate" : 1,                     #リーク率
        "SubReservoir_InputScale" : 0.1,                    #入力スケーリング
        "SubReservoir_Rho" : 0.8,                             #スペクトル半径
        "SubReservoir_Density" : 0.8,                         #結合密度
        "SubReservoir_ActivationFunc" : np.tanh,            #活性化関数

        #------------------------------------------------------------------------------------------
        "LinerTransformer_Beta" : 0.2,                      #正則化係数
        }

    #NRMSE評価
    if Param["Project_F_NRMSE"]:
        param = Param.copy()
        param.update({
            #==========================================================================================
            "NRMSE_F_OutputLog" : True,                         #経過の出力を行うか

            "NRMSE_D_u" : 1,                                    #入力信号次元
            "NRMSE_D_x" : 100,                                  #リザバー層次元
            "NRMSE_D_y" : 1,                                    #出力信号次元

            #------------------------------------------------------------------------------------------
            "NRMSE_Length_Burnin" : 1000,                       #空走用データ時間長
            "NRMSE_Length_Train" : 20000,                       #学習用データ時間長
            "NRMSE_Length_Test" : 5000,                         #評価用データ時間長

            #------------------------------------------------------------------------------------------
            "NRMSE_T_Task" : Task_No.Task_NormalRosslor,                                #評価用タスク（Type型）
            "Task_Noise" : False,                               #タスクにノイズを加えるか
            "Task_Noise_Scale" : 0.025,                       #ノイズのスケール
            "NRMSE_T_Model" : Model_No.Model_NormalESN,                 #モデル（Type型）
            "NRMSE_T_Output" : Output_No.Output_Single_NRMSE_2023_04_19_15_25,     #作図出力（Type型）
        
            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            "DirPath_Project" : "./Normal_Reservoir/Results/Single_Task/NRMSE",
        
            "NRMSE_F_OutputCharts" : True,                      #図の出力フラグ
            "NRMSE_F_OutputCharts_UYYdEWaves" : True,           #入出力＆誤差波形図の出力フラグ

            })
        Evaluation_No.Evaluation_NRMSE(param)()
    
    #MC評価
    if Param["Project_F_MemoryCapacity"]:
        param = Param.copy()
        param.update({
            #==========================================================================================
            "MemoryCapacity_F_OutputLog" : True,                #経過の出力を行うか
            "MemoryCapacity_D_u" : 1,                           #入力信号次元
            "MemoryCapacity_D_y" : 1,                           #出力信号次元
            "MemoryCapacity_Length_Burnin" : 1000,              #空走用データ時間長
            "MemoryCapacity_Length_Train" : 5000,              #学習用データ時間長
            "MemoryCapacity_Length_Test" : 1000,                #評価用データ時間長
            "MemoryCapacity_MaxTau" : 100,                       #評価する最大遅延
            "MemoryCapacity_T_Task" : Task_No.Task_MC,                                     #評価用タスク（Type型）
            "Task_Noise" : False,                               #タスクにノイズを加えるか
            "Task_Noise_Scale" : 0.025,                       #ノイズのスケール
            "MemoryCapacity_T_Model" : Model_No.Model_NormalESN,                           #モデル（Type型）
            "MemoryCapacity_T_Output" : Output_No.Output_Single_MC_2023_05_25_13_28,       #作図出力（Type型）
        
            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            "DirPath_Project" : "./Normal_Reservoir/Results/Single_Task/MC",
        
            "MemoryCapacity_F_OutputCharts" : True,             #図の出力フラグ
            "MemoryCapacity_F_OutputCharts_MCGraph" : True,     #MC曲線の出力フラグ

            })
        Evaluation_No.Evaluation_MC(param)()

    #MLE評価
    if Param["Project_F_MLE"]:
        param = Param.copy()
        param.update({
            #==========================================================================================
            "MLE_F_OutputLog" : True,                        #経過の出力を行うか

            "MLE_D_u" : 1,                                   #入力信号次元
            "MLE_D_x" : 100,                                 #リザバー層次元
            "MLE_D_y" : 1,                                   #出力信号次元

            "MLE_Length_Burnin" : 1000,                      #空走用データ時間長
            "MLE_Length_Test" : 5000,                        #評価用データ時間長

            "MLE_Epsilon" : 1e-06,                            #摂動の大きさ

            "MLE_T_Task" : Task_No.Task_NormalRosslor,
            "Task_Noise" : False,                        #タスクにノイズを加えるか
            "Task_Noise_Scale" : 0.025,                       #ノイズのスケール
            "MLE_T_Model" : Model_No.Model_NormalESN,
            "MLE_T_Output" : Output_No.Output_Single_MLE_2023_07_08_17_12,

            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            "DirPath_Project" : "./Normal_Reservoir/Results/Single_Task/MLE",
        
            "MLE_F_OutputCharts" : True,             #図の出力フラグ
            "MLE_F_OutputCharts_MLEWaves" : True,     #MC曲線の出力フラグ
        })
        Evaluation_No.Evaluation_MLE(param)()

    #CovarianceRank評価
    if Param["Project_F_CovMatrixRank"]:
        param = Param.copy()
        param.update({
            #==========================================================================================
            "CovMatrixRank_F_OutputLog" : True,                        #経過の出力を行うか

            "CovMatrixRank_D_u" : 1,                                   #入力信号次元
            "CovMatrixRank_D_x" : 100,                                 #リザバー層次元
            "CovMatrixRank_D_y" : 1,                                   #出力信号次元

            "CovMatrixRank_Length_Burnin" : 1000,                      #空走用データ時間長
            "CovMatrixRank_Length_Test" : 5000,                        #評価用データ時間長
            
            "CovMatrixRank_T_Task" : Task_No.Task_NormalRosslor,
            "Task_Noise" : False,                        #タスクにノイズを加えるか
            "Task_Noise_Scale" : 0.025,                       #ノイズのスケール
            "CovMatrixRank_T_Model" : Model_No.Model_NormalESN,
            "CovMatrixRank_T_Output" : Output_No.Output_Single_CovMatrixRank_2025_03_15_15_32,

            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            "DirPath_Project" : "./Normal_Reservoir/Results/Single_Task/CovarianceMatrixRank",
        
            "CovMatrixRank_F_OutputCharts" : True,             #図の出力フラグ
            "CovMatrixRank_F_OutputCharts_CovMatrixRankWaves" : True,     #MC曲線の出力フラグ
        })
        Evaluation_No.Evaluation_CovMatrixRank(param)()

    #Delay Capacity評価
    if Param["Project_F_DelayCapacity"]:
        param = Param.copy()
        param.update({
            #==========================================================================================
            "DelayCapacity_F_OutputLog" : True,                        #経過の出力を行うか

            "DelayCapacity_D_u" : 1,                                   #入力信号次元
            "DelayCapacity_D_x" : 100,                                 #リザバー層次元
            "DelayCapacity_D_y" : 1,                                   #出力信号次元

            "DelayCapacity_Length_Burnin" : 1000,                      #空走用データ時間長
            
            "DelayCapacity_Length_Tdc" : 5000,                        #評価用データ時間長
            "DelayCapacity_Length_Taumax" : 50,

            "DelayCapacity_T_Task" : Task_No.Task_NormalRosslor,
            "Task_Noise" : False,                        #タスクにノイズを加えるか
            "Task_Noise_Scale" : 0.025,                       #ノイズのスケール
            "DelayCapacity_T_Model" : Model_No.Model_NormalESN,
            "DelayCapacity_T_Output" : Output_No.Output_Single_DelayCapacity_2025_03_15_15_32,

            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            "DirPath_Project" : "./Normal_Reservoir/Results/Single_Task/DelayCapacity",
        
            "DelayCapacity_F_OutputCharts" : True,             #図の出力フラグ
            "DelayCapacity_F_OutputCharts_DCGraph" : True,     #DC曲線の出力フラグ
        })
        Evaluation_No.Evaluation_DelayCapacity(param)()


#********************************************************************