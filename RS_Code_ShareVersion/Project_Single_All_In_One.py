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

import Evaluation_Performance
import Task
import Model
import Output

import PreAndPostProcess

#====================================================================
#定数
T = True
F = False

#====================================================================
#プロジェクト

#********************************************************************
"""
NRMSEとMCの評価．
"""
if __name__ == '__main__':
    PreAndPostProcess.PreProcess()

    Param = {
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        "Project_F_UsePytorch" : False,                     #Pytorchを使うか（多層リードアウトでは強制的に使用）
        "Project_DeviceCode" : "cpu",                       #CPU/GPUを使うか（CPU -> cpu, GPU -> cuda:n（nはデバイス番号，無くてもいい））
        "Project_DataType" : torch.float,                   #Pytorchのデータ型
        
        "Project_F_NRMSE" : True,                           #NRMSEを調査するか
        "Project_F_MemoryCapacity" : True,                  #MCを調査するか

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
        "Task_Rosslor_InitTerm" : 1000,                     #初期状態排除期間
        
        "Task_SLorenz_SelectedInput" :  [T, T, F, F, T,
                                         F, T, F, F, T],    #入力に使用する成分（Tの数がD_u）
        "Task_SLorenz_SelectedOutput" : [F, F, T, T, F,
                                         T, F, T, T, F],    #出力に使用する成分（Tの数がD_y）

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        "Model_NormalESN_D_u" : 5,                          #入力信号次元
        "Model_NormalESN_D_x" : 500,                        #ニューロン数
        "Model_NormalESN_D_y" : 5,                          #出力信号次元
        
        "Model_NormalAESN_D_u" : 5,                         #入力信号次元
        "Model_NormalAESN_D_x" : 100,                       #サブリザバーニューロン数
        "Model_NormalAESN_D_y" : 5,                         #出力信号次元
        
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
        "LinearTransformer_Beta" : 0.2,                      #正規化係数

        "DNN_LearningRate" : 0.001,                         #学習率
        "DNN_MaxLearningLoop" : 1000,                       #最大学習ループ数（使わない場合は0にする）
        "DNN_AimingError" : 0.001,                          #目標（最小）誤差（使わない場合は0にする）
        
        }

    #NRMSE評価
    if Param["Project_F_NRMSE"]:
        param = Param.copy()
        param.update({
            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            "NRMSE_F_OutputLog" : True,                         #経過の出力を行うか
            "NRMSE_D_u" : 5,                                    #入力信号次元
            "NRMSE_D_y" : 5,                                    #出力信号次元
            "NRMSE_Length_Burnin" : 1000,                       #空走用データ時間長
            "NRMSE_Length_Train" : 20000,                       #学習用データ時間長
            "NRMSE_Length_Test" : 5000,                         #評価用データ時間長
            "NRMSE_T_Task" : Task.Task_NDLorenz,                                #評価用タスク（Type型）
            "NRMSE_T_Model" : Model.Model_ModifiedDeepESN,                 #モデル（Type型）
            "NRMSE_T_Output" : Output.Output_Single_NRMSE_2023_04_19_15_25,     #作図出力（Type型）
        
            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            "DirPath_Project" : "./Results/Project_Single_2023_06_12_07_38/NRMSE",
        
            "NRMSE_F_OutputCharts" : True,                      #図の出力フラグ
            "NRMSE_F_OutputCharts_UYYdEWaves" : True,           #入出力＆誤差波形図の出力フラグ

            })
        Evaluation_Performance.Evaluation_NRMSE(param)()
    
    #MC評価
    if Param["Project_F_MemoryCapacity"]:
        param = Param.copy()
        param.update({
            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            "MemoryCapacity_F_OutputLog" : True,                #経過の出力を行うか
            "MemoryCapacity_D_u" : 5,                           #入力信号次元
            "MemoryCapacity_D_y" : 5,                           #出力信号次元
            "MemoryCapacity_Length_Burnin" : 1000,              #空走用データ時間長
            "MemoryCapacity_Length_Train" : 5000,              #学習用データ時間長
            "MemoryCapacity_Length_Test" : 1000,                #評価用データ時間長
            "MemoryCapacity_MaxTau" : 100,                       #評価する最大遅延
            "MemoryCapacity_T_Task" : Task.Task_MC,                                     #評価用タスク（Type型）
            "MemoryCapacity_T_Model" : Model.Model_NormalAESN,                           #モデル（Type型）
            "MemoryCapacity_T_Output" : Output.Output_Single_MC_2023_05_25_13_28,       #作図出力（Type型）
        
            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            "DirPath_Project" : "./Results/Project_Single_2023_06_12_07_38/MC",
        
            "MemoryCapacity_F_OutputCharts" : True,             #図の出力フラグ
            "MemoryCapacity_F_OutputCharts_MCGraph" : True,     #MC曲線の出力フラグ

            })
        Evaluation_Performance.Evaluation_MC(param)()

    PreAndPostProcess.PostProcess()
    