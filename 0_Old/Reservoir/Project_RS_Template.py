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

def Project_RandomSearch_NRMSE_Sample():
    
    RS_Param = {
            "Model_DifferentUpdateAESN_LeakingRate__01" : [0.01, 1.0],
            "Model_DifferentUpdateAESN_LeakingRate__02" : [0.01, 1.0],
            "Model_DifferentUpdateAESN_LeakingRate__03" : [0.01, 1.0],
            "Model_DifferentUpdateAESN_LeakingRate__04" : [0.01, 1.0],
            "Model_DifferentUpdateAESN_InputScale__01" : [0.01, 1.0],
            "Model_DifferentUpdateAESN_InputScale__02" : [0.01, 1.0],
            "Model_DifferentUpdateAESN_InputScale__03" : [0.01, 1.0],
            "Model_DifferentUpdateAESN_InputScale__04" : [0.01, 1.0],
            "Model_DifferentUpdateAESN_Rho__01" : [0.01, 2.0],
            "Model_DifferentUpdateAESN_Rho__02" : [0.01, 2.0],
            "Model_DifferentUpdateAESN_Rho__03" : [0.01, 2.0],
            "Model_DifferentUpdateAESN_Rho__04" : [0.01, 2.0],
    }
    
    Param = {
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            "Use_model" : "HetAESN",                                #モデル
            "Evaluation" : "NRMSE",                             #評価方法(現在NRMSEのみ対応)
            
            "RandomSearch_target_num_sample" : 10,           #おおよそのサンプル数
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
            "Model_NormalESN_D_u" : 4,                  #入力信号次元
            "Model_NormalESN_D_x" : 10,      #ニューロン数
            "Model_NormalESN_D_y" : 4,                  #出力信号次元
        
            "Model_NormalAESN_D_u" : 5,                 #入力信号次元
            "Model_NormalAESN_D_x" : 100,                 #サブリザバーニューロン数
            "Model_NormalAESN_D_y" : 5,                 #出力信号次元
        
            "Model_DifferentUpdateESN_D_u" : 4,                         #入力信号次元
            "Model_DifferentUpdateESN_D_x" : 10,                       #サブリザバーニューロン数（リスト型可能）
            "Model_DifferentUpdateESN_D_y" : 4,                         #出力信号次元
            "Model_DifferentUpdateESN_LeakingRate" : [0.01, 0.1, 1.0],  #リーク率配列->サブリザバーの数(必ずリスト型)
            "Model_DifferentUpdateESN_InputScale" : None,               #入力スケーリング配列（リスト型可能，None可能）
            "Model_DifferentUpdateESN_Rho" : None,                      #スペクトル半径配列（リスト型可能，None可能）
            "Model_DifferentUpdateESN_Density" : None,                  #結合密度配列（リスト型可能，None可能）
            
            "Model_DifferentUpdateAESN_D_u" : 4,                         #入力信号次元
            "Model_DifferentUpdateAESN_D_x" : 10,                       #サブリザバーニューロン数（リスト型可能）
            "Model_DifferentUpdateAESN_D_y" : 4,                         #出力信号次元
            "Model_DifferentUpdateAESN_LeakingRate" : [0.01, 0.1, 1.0],  #リーク率配列->サブリザバーの数(必ずリスト型)
            "Model_DifferentUpdateAESN_InputScale" : None,               #入力スケーリング配列（リスト型可能，None可能）
            "Model_DifferentUpdateAESN_Rho" : None,                      #スペクトル半径配列（リスト型可能，None可能）
            "Model_DifferentUpdateAESN_Density" : None,                  #結合密度配列（リスト型可能，None可能）
        
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
            "LinerTransformer_Beta" : 0.2,                      #正規化係数

            "DNN_LearningRate" : 0.001,                         #学習率
            "DNN_MaxLearningLoop" : 1000,                       #最大学習ループ数（使わない場合は0にする）
            "DNN_AimingError" : 0.001,                          #目標（最小）誤差（使わない場合は0にする）
    }
    
    if Param["Use_model"] == "ESN":
            Param_Model = Model.Model_NormalESN
    elif Param["Use_model"] == "AESN":
        Param_Model = Model.Model_NormalAESN
    elif Param["Use_model"] == "HetAESN":
        Param_Model = Model.Model_DifferentUpdateAESN
    elif Param["Use_model"] == "HubAESN":
        Param_Model = Model.Model_AESNwithHubReservoir
    
    #NRMSE評価
    if Param["Evaluation"] == "NRMSE":
        NRMSE_param = Param.copy()
        NRMSE_param.update({
            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            "NRMSE_F_OutputLog" : False,                        #経過の出力を行うか
            "NRMSE_D_u" : 4,                            #入力信号次元
            "NRMSE_D_y" : 4,                            #出力信号次元
            "NRMSE_Length_Burnin" : 1000,                       #空走用データ時間長
            "NRMSE_Length_Train" : 20000,                       #学習用データ時間長
            "NRMSE_Length_Test" : 5000,                         #評価用データ時間長
            "NRMSE_T_Task" : Task.Task_tcVDP,                                #評価用タスク（Type型）
            "NRMSE_T_Model" : Param_Model,                                      #モデル（Type型）
            "NRMSE_T_Output" : Output.Output_Single_NRMSE_2023_04_19_15_25,     #作図出力（Type型）
    
            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            "DirPath_Project" : "./Results/Project_RS_NRMSE_2024_03_18_16_22",
            
            "NRMSE_F_OutputCharts" : False,                      #図の出力フラグ
            "NRMSE_F_OutputCharts_UYYdEWaves" : False,           #入出力＆誤差波形図の出力フラグ

            })
        Random = Random_Search.RandomSearch(NRMSE_param, RS_Param)
    
    Random.random_search()
#********************************************************************
#Chialvo Neuron Mapを用いたESNモデル
def Project_RandomSearch_NRMSE_ChialvoESN():
    
    RS_Param = {
            "Model_ChialvoESN_Rho__01" : [0.01,1.5]
    }
    
    Param = {
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            "Use_model" : "ChialvoESN",                         #モデル
            "Evaluation" : "NRMSE",                             #評価方法(現在NRMSEのみ対応)
            
            "RandomSearch_target_num_sample" : 200,           #おおよそのサンプル数
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
            "Model_ChialvoESN_D_u" : 1,                          #入力信号次元
            "Model_ChialvoESN_D_x" : 100,                        #ニューロン数
            "Model_ChialvoESN_D_y" : 1,                          #出力信号次元

            "Model_ChialvoESN_InputScale" : 0.1,                 #入力スケーリング

            "Model_ChialvoESN_Rho" : 0.99,                             #スペクトル半径
            
            "ChialvoReservoir_ActivationFunc" : np.tanh,            #活性化関数

            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            "LinerTransformer_Beta" : 0.2,                      #正規化係数
    }
    
    if Param["Use_model"] == "ChialvoESN":
        Param_Model = Model.Model_ChialvoESN
    
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
            "DirPath_Project" : "./Results/Project_RS_NRMSE_ChialvoESN_2024_06_23_01_12",
            
            "NRMSE_F_OutputCharts" : False,                      #図の出力フラグ
            "NRMSE_F_OutputCharts_UYYdEWaves" : False,           #入出力＆誤差波形図の出力フラグ

            })
        Random = Random_Search.RandomSearch(NRMSE_param, RS_Param)
    
    Random.random_search()

#電磁束の影響を与えたChialvo Neuron Mapを用いたESNモデル
def Project_RandomSearch_NRMSE_EMChialvoESN():
    
    RS_Param = {
            "Model_EMChialvoESN_Rho__01" : [0.01,1.5],
            "Model_EMChialvoESN_k__01" : [-5,5]
    }
    
    Param = {
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            "Use_model" : "EMChialvoESN",                         #モデル
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
            "Model_EMChialvoESN_D_u" : 1,                          #入力信号次元
            "Model_EMChialvoESN_D_x" : 200,                        #ニューロン数
            "Model_EMChialvoESN_D_y" : 1,                          #出力信号次元

            "Model_EMChialvoESN_InputScale" : 0.1,                 #入力スケーリング

            "Model_EMChialvoESN_Rho" : 0.99,                             #スペクトル半径
            "Model_EMChialvoESN_k" : 3,
            
            "EMChialvoReservoir_ActivationFunc" : np.tanh,            #活性化関数

            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            "LinerTransformer_Beta" : 0.2,                      #正規化係数
    }
    
    if Param["Use_model"] == "EMChialvoESN":
        Param_Model = Model.Model_EMChialvoESN
    
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
            "DirPath_Project" : "./Results/Project_RS_NRMSE_EMChialvoESN_2024_06_23_01_12",
            
            "NRMSE_F_OutputCharts" : False,                      #図の出力フラグ
            "NRMSE_F_OutputCharts_UYYdEWaves" : False,           #入出力＆誤差波形図の出力フラグ

            })
        Random = Random_Search.RandomSearch(NRMSE_param, RS_Param)
    
    Random.random_search()

#********************************************************************

def Project_RandomSearch_NRMSE_NewChialvo():
    
    RS_Param = {
            "Model_ChialvoESN_Rho__01" : [0.001,0.1],
            "Model_ChialvoESN_k__01" : [-3.5,-2.5],
    }
    
    Param = {
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            "Use_model" : "NewChialvoESN",                         #モデル
            "Evaluation" : "NRMSE",                             #評価方法(現在NRMSEのみ対応)
            
            "RandomSearch_target_num_sample" : 200,           #おおよそのサンプル数
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
            "Model_ChialvoESN_D_u" : 1,                          #入力信号次元
            "Model_ChialvoESN_D_x" : 100,                        #ニューロン数
            "Model_ChialvoESN_D_y" : 1,                          #出力信号次元

            "Model_ChialvoESN_InputScale" : 0.1,                 #入力スケーリング
            "Model_ChialvoESN_sigma" : False,                   #リング・ネットワークの結合強度（Trueでリングネットワーク）
            "Model_ChialvoESN_mu" : True,                           #スター・ネットワークの結合強度（Trueでスターネットワーク）
            "Model_ChialvoESN_k" : -3,                          #Chialvoの変数：k

            "Model_ChialvoESN_Rho" : 0.002,                             #スペクトル半径
            "ChialvoReservoir_ActivationFunc" : np.tanh,            #活性化関数

            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            "LinerTransformer_Beta" : 0.2,                      #正規化係数
    }
    
    if Param["Use_model"] == "NewChialvoESN":
        Param_Model = Model.Model_NewChialvoESN
    
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
            "DirPath_Project" : "./Results/Project_RS_NRMSE_2024_05_18_16_22",
            
            "NRMSE_F_OutputCharts" : False,                      #図の出力フラグ
            "NRMSE_F_OutputCharts_UYYdEWaves" : False,           #入出力＆誤差波形図の出力フラグ

            })
        Random = Random_Search.RandomSearch(NRMSE_param, RS_Param)
    
    Random.random_search()