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

import Evaluation_EM
import Task_EM
import Model_EM
import Output_EM

#====================================================================
#プロジェクト

#********************************************************************
#Sishu提案モデル
def Project_EMChialvo_2025_01_28_12_34():
    """
    Sishu提案モデル（結合形態のみ指定、強度は乱数）のNRMSEやMC測定
    """
    #共通パラメータ
    Param = {
        #==========================================================================================
        "Project_F_NRMSE" : True,                           #NRMSEを調査するか
        "Project_F_MemoryCapacity" : True,                  #MCを調査するか
        "Project_F_MLE" : True,                             #MLE（最大リアプノフ指数）を調査するか
        
        "Project_F_CovMatrixRank" : True,                  #Covariance Matrix Rankを調査するか
        "Project_F_DelayCapacity" : True,                   #Delay Capacityを調査するか

        #------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------
        #Sin波のタスク
        "Task_SinCurve_RK_h" : 0.01,                         #ルンゲクッタ法刻み幅
        "Task_SinCurve_Tau" : 30,

        #------------------------------------------------------------------------------------------
        #MCタスク
        "Task_MC_Tau" : 10,                                  #遅延量，MCのτ
        
        #------------------------------------------------------------------------------------------
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
        "Task_NormalLorenz_Scale" : 1 / 50,
        "Task_NormalLorenz_Dt" : 0.005,
        "Task_NormalLorenz_Tau" : 5,
        "Task_NormalLorenz_InitTerm" : 1000,

        "Task_NormalLorenz_Sigma" : 10,
        "Task_NormalLorenz_Beta" : 8/3,
        "Task_NormalLorenz_Rho" : 28,

        #------------------------------------------------------------------------------------------
        #ローレンツ方程式96のパラメータ
        "Task_Lorenz96_Scale" : 1 / 10,                       #ローレンツ方程式96の大きさ
        "Task_Lorenz96_Dt" : 0.005,                          #時間刻み幅
        "Task_Lorenz96_Tau" : 5,                            #どれくらい先を予測するか
        "Task_Lorenz96_InitTerm" : 1000,                    #初期状態排除期間
        "Task_Lorenz96_N" : 10,                             #ニューロン数
        "Task_Lorenz96_F" : 8,                              #大きさ？


        #------------------------------------------------------------------------------------------
        #連続時間のマッキー・グラス方程式のパラメータ
        "Task_PredictDDE_Tau" : 5,                          #どれくらい先を予測するか
        "Task_MackeyGlassDDE_Dt" : 0.2,                    #時間刻み幅
        
        "Task_MackeyGlassDDE_Beta" : 0.2,                       #β:0.2
        "Task_MackeyGlassDDE_Gamma" : 0.1,                     #γ:0.1
        "Task_MackeyGlassDDE_N" : 10,                       #乗数: 10
        "Task_MackeyGlassDDE_Tau" : 22,                      #マッキー・グラスの遅延量 :17
        "Task_MackeyGlassDDE_InitTerm" : 1000,              #初期状態排除期間

        #==========================================================================================
        #==========================================================================================
        "Model_EMChialvo_D_u" : 1,                          #入力信号次元
        "Model_EMChialvo_D_x" : 100,                        #ニューロン数
        "Model_EMChialvo_D_y" : 1,                          #出力信号次元

        "Model_Reservoir_Neurons" : 10,                     #描写するリザバー層のニューロン数
        "Model_EMChialvo_InputScale" : 0.1,                 #入力スケーリング

        #------------------------------------------------------------------------------------------
        "Model_EMChialvo_a" : 0.89,                         #変数:a
        "Model_EMChialvo_b" : 0.6,                          #変数:b
        "Model_EMChialvo_c" : 0.28,                         #変数:c
        "Model_EMChialvo_k0" : 0.04,                        #変数:k0

        "Model_EMChialvo_k1" : 0.1,                         #変数:k1
        "Model_EMChialvo_k2" : 0.2,                         #変数:k2
        "Model_EMChialvo_alpha" : 0.1,                      #変数:alpha
        "Model_EMChialvo_beta" : 0.2,                       #変数:beta

        "Model_EMChialvo_k" : -5,                         #変数:k
        "Model_EMChialvo_Rho" : 0.01,                      #スペクトル半径

        #------------------------------------------------------------------------------------------
        "EMChialvo_Reservoir_Density" : 1,                          #結合密度
        "LinerTransformer_Beta" : 0.2,                      #正則化係数
        }
    
    #一括でタスク変更用
    TaskSignal = Task_EM.Task_NormalLorenz

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
            "NRMSE_T_Task" : TaskSignal,                                #評価用タスク（Type型）
            "Task_Noise" : False,                               #タスクにノイズを加えるか
            "Task_Noise_Scale" : 0.025,                       #ノイズのスケール
            "NRMSE_T_Model" : Model_EM.Model_EMChialvo,                 #モデル（Type型）
            "NRMSE_T_Output" : Output_EM.Output_Single_NRMSE_2023_04_19_15_25,     #作図出力（Type型）
        
            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            "DirPath_Project" : "./EMChialvo_Reservoir/Results/Single_Task/NRMSE",
        
            "NRMSE_F_OutputCharts" : True,                      #図の出力フラグ
            "NRMSE_F_OutputCharts_UYYdEWaves" : True,           #入出力＆誤差波形図の出力フラグ

            })
        Evaluation_EM.Evaluation_NRMSE(param)()
    
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
            "MemoryCapacity_T_Task" : Task_EM.Task_MC,                                     #評価用タスク（Type型）
            "Task_Noise" : False,                               #タスクにノイズを加えるか
            "Task_Noise_Scale" : 0.025,                       #ノイズのスケール
            "MemoryCapacity_T_Model" : Model_EM.Model_EMChialvo,                           #モデル（Type型）
            "MemoryCapacity_T_Output" : Output_EM.Output_Single_MC_2023_05_25_13_28,       #作図出力（Type型）
        
            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            "DirPath_Project" : "./EMChialvo_Reservoir/Results/Single_Task/MC",
        
            "MemoryCapacity_F_OutputCharts" : True,             #図の出力フラグ
            "MemoryCapacity_F_OutputCharts_MCGraph" : True,     #MC曲線の出力フラグ

            })
        Evaluation_EM.Evaluation_MC(param)()

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

            "MLE_Epsilon" : 1e-08,                            #摂動の大きさ

            "MLE_T_Task" : TaskSignal,
            "Task_Noise" : False,                        #タスクにノイズを加えるか
            "Task_Noise_Scale" : 0.025,                       #ノイズのスケール
            "MLE_T_Model" : Model_EM.Model_EMChialvo,
            "MLE_T_Output" : Output_EM.Output_Single_MLE_2023_07_08_17_12,

            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            "DirPath_Project" : "./EMChialvo_Reservoir/Results/Single_Task/MLE",
        
            "MLE_F_OutputCharts" : True,             #図の出力フラグ
            "MLE_F_OutputCharts_MLEWaves" : True,     #MC曲線の出力フラグ
        })
        Evaluation_EM.Evaluation_MLE(param)()

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
            
            "CovMatrixRank_T_Task" : TaskSignal,
            "Task_Noise" : False,                        #タスクにノイズを加えるか
            "Task_Noise_Scale" : 0.025,                       #ノイズのスケール
            "CovMatrixRank_T_Model" : Model_EM.Model_EMChialvo,
            "CovMatrixRank_T_Output" : Output_EM.Output_Single_CovMatrixRank_2025_03_15_15_32,

            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            "DirPath_Project" : "./EMChialvo_Reservoir/Results/Single_Task/CovarianceMatrixRank",
        
            "CovMatrixRank_F_OutputCharts" : True,             #図の出力フラグ
            "CovMatrixRank_F_OutputCharts_CovMatrixRankWaves" : True,     #MC曲線の出力フラグ
        })
        Evaluation_EM.Evaluation_CovMatrixRank(param)()

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
            "DelayCapacity_Length_Taumax" : 100,

            "DelayCapacity_T_Task" : TaskSignal,
            "Task_Noise" : False,                        #タスクにノイズを加えるか
            "Task_Noise_Scale" : 0.025,                       #ノイズのスケール
            "DelayCapacity_T_Model" : Model_EM.Model_EMChialvo,
            "DelayCapacity_T_Output" : Output_EM.Output_Single_DelayCapacity_2025_03_15_15_32,

            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            "DirPath_Project" : "./EMChialvo_Reservoir/Results/Single_Task/DelayCapacity",
        
            "DelayCapacity_F_OutputCharts" : True,             #図の出力フラグ
            "DelayCapacity_F_OutputCharts_DCGraph" : True,     #DC曲線の出力フラグ
        })
        Evaluation_EM.Evaluation_DelayCapacity(param)()


#********************************************************************