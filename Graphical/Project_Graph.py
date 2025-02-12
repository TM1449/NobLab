#####################################################################
#ESNプロジェクト，Chialvoニューロンマップのグラフ描写用
#制作者：田中勝規
#作成日：2024/09/24
#####################################################################

#====================================================================
#＜このプログラムのTodoメモ＞
#・
#・
#・
#====================================================================

#====================================================================
#外部ライブラリ
import numpy as np
import random

#====================================================================
#内部プログラム
import Output_Graph
import Model_Graph

#====================================================================
def Project_Chialvo_TimeLine_2024_09_24_17_25():
    """
    変数1つに対して、時間経過における変数の値を描写する関数。
    """
    #共通パラメータ
    Param = {
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #時系列の描写
        "Project_Plot_TimeLine_X"               : True,
        "Project_Plot_TimeLine_Y"               : True,
        "Project_Plot_TimeLine_Phi"             : True,

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #従来のChialvoパラメータ
        "Chialvo_a"                 : 0.89,
        "Chialvo_b"                 : 0.16,
        "Chialvo_c"                 : 0.28,
        "Chialvo_k0"                : 0.04,

        #電磁束下のChialvoパラメータ
        "Chialvo_k"                 : -3.2,
        "Chialvo_k1"                : 0.1,
        "Chialvo_k2"                : 0.2,
        "Chialvo_alpha"             : 0.1,
        "Chialvo_beta"              : 0.2,

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #空走時間
        "Length_Burnin" : 1000,
        #評価時間
        "Length_Eva" : 10000,
        #プロット時間
        "Length_Plot" : 2000,

        #x軸のスケールをlogにするかいなか
        "x_Log_scale" : False,
        
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #入力信号の振幅について
        "Input_Signal_Amp" : 0,

        #恒等関数：None, sin関数：np.sin, 離散信号：random.randint
        "Input_Signal_Def" : np.sin,

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #時系列描写の初期化指定
        "Initial_Value_X" : None,
        "Initial_Value_Y" : None,
        "Initial_Value_Phi" : None,
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    }
    
    param = Param.copy()
    param.update({
        "Model" : Model_Graph.Model_Chialvo
    })
    Output_Graph.Output_TimeLine(param)()

def Project_Chialvo_PhaseSpace_2024_09_26_12_34():
    """
    変数2つに対して、時間経過における変数の値の組み合わせを描写する関数。
    """

    #共通パラメータ
    Param = {
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #相平面の描写
        "Project_Plot_PhaseSpace_XandY"         : True,
        "Project_Plot_PhaseSpace_XandPhi"       : True,
        "Project_Plot_PhaseSpace_YandPhi"       : True,

        #3次元相平面の描写
        "Project_Plot_PhaseSpace_3D"            : True,

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #従来のChialvoパラメータ
        "Chialvo_a"                 : 0.89,
        "Chialvo_b"                 : 0.16,
        "Chialvo_c"                 : 0.28,
        "Chialvo_k0"                : 0.04,

        #電磁束下のChialvoパラメータ
        "Chialvo_k"                 : -3.2,
        "Chialvo_k1"                : 0.1,
        "Chialvo_k2"                : 0.2,
        "Chialvo_alpha"             : 0.1,
        "Chialvo_beta"              : 0.2,
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #空走時間
        "Length_Burnin" : 1000,
        #評価時間
        "Length_Eva" : 10000,
        #プロット時間
        "Length_Plot" : 2000,

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #入力信号の振幅について
        "Input_Signal_Amp" : 0,

        #恒等関数：None, sin関数：np.sin, 離散信号：random.randint
        "Input_Signal_Def" : np.sin,
        
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #時系列描写の初期化指定
        "Initial_Value_X" : None,
        "Initial_Value_Y" : None,
        "Initial_Value_Phi" : None,
        
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    }
    
    param = Param.copy()
    param.update({
        "Model" : Model_Graph.Model_Chialvo
    })
    Output_Graph.Output_PhaseSpace(param)()

def Project_Chialvo_OldNullcline_2024_09_27_15_07():
    """
    2次元Chialvo Neuron Mapにおけるベクトル場とNullclineを描写する関数。
    """

    #共通パラメータ
    Param = {
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #Nullclineの描写
        "Project_Plot_OldNullcline"         : True,
        #相平面における解の軌道の描写
        "Project_Plot_PhaseSpace"           : True,

        #+++++++++++++++++++++++++s+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #従来のChialvoパラメータ
        "Chialvo_a"                 : 0.89,
        "Chialvo_b"                 : 0.18,
        "Chialvo_c"                 : 0.28,
        "Chialvo_k0"                : 0.03,

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #時系列描写の実行時間
        "RunTime" : 300,
        "Plot_Start" : 0,
        "Plot_End" : 300,

        #入力信号：uについて
        "Input_Signal"  : 0,

        #ベクトル場の間隔
        "Vdt" : 0.05,
        #Nullclineの間隔
        "dt" : 0.0001,

        #どこからどこまでプロット図の点を作成するか
        "Plot_x_Start" : -0.5,
        "Plot_x_End" : 3.5,

        "Plot_y_Start" : -0.5,
        "Plot_y_End" : 3.5,

        #時系列描写の初期化指定
        "Initial_Value_X" : 1,
        "Initial_Value_Y" : 3,
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    }
    
    param = Param.copy()
    param.update({
        "Model" : Model_Graph.Model_Chialvo_OldNullcline
    })
    Output_Graph.Output_OldNullcline(param)()

def Project_Chialvo_NewNullcline_2024_10_14_18_17():
    """
    3次元Chialvo Neuron Mapにおけるベクトル場とNullclineを描写する関数。
    """

    #共通パラメータ
    Param = {
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #Nullclineの描写
        "Project_Plot_NewNullcline"         : True,
        #相平面における解の軌道の描写
        "Project_Plot_PhaseSpace"           : False,

        #+++++++++++++++++++++++++s+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #従来のChialvoパラメータ
        "Chialvo_a"                 : 0.89,
        "Chialvo_b"                 : 0.6,
        "Chialvo_c"                 : 0.28,
        "Chialvo_k0"                : 0.04,

        #電磁束下のChialvoパラメータ
        "Chialvo_k"                 : -1.1,

        "Chialvo_k1"                : 0.1,
        "Chialvo_k2"                : 0.2,
        "Chialvo_alpha"             : 0.1,
        "Chialvo_beta"              : 0.2,
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #時系列描写の実行時間
        "RunTime" : 1000,
        "Plot_Start" : 0,
        "Plot_End" : 1000,

        #入力信号：uについて
        "Input_Signal"  : 0,

        #ベクトル場の間隔
        "Vdt" : 0.2,
        #Nullclineの間隔
        "dt" : 0.0001,

        #交点の算出

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #どこからどこまでプロット図の点を作成するか
        "Plot_x_Start" : -5,
        "Plot_x_End" : 5,

        "Plot_y_Start" : -5,
        "Plot_y_End" : 5,

        #時系列描写の初期化指定
        "Initial_Value_X" : None,
        "Initial_Value_Y" : None,
        "Initial_Value_Phi" : None,
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    }
    
    param = Param.copy()
    param.update({
        "Model" : Model_Graph.Model_Chialvo_NewNullcline
    })
    Output_Graph.Output_NewNullcline(param)()


#====================================================================
def Project_ChialvoNeuronMap_TimeLine_2025_02_09_01_32():
    """
    変数1つに対して、時間経過における変数の値を描写する関数。
    """
    #共通パラメータ
    Param = {
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #時系列の描写
        "Project_Plot_TimeLine_X"               : True,
        "Project_Plot_TimeLine_Y"               : True,
        "Project_Plot_TimeLine_Phi"             : True,

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #従来のChialvoパラメータ
        "Chialvo_a"                 : 0.89,
        "Chialvo_b"                 : 0.6,
        "Chialvo_c"                 : 0.28,
        "Chialvo_k0"                : 0.04,

        #電磁束下のChialvoパラメータ
        "Chialvo_k"                 : -3.2,
        "Chialvo_k1"                : 0.1,
        "Chialvo_k2"                : 0.2,
        "Chialvo_alpha"             : 0.1,
        "Chialvo_beta"              : 0.2,

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #電磁束下Chialvoニューロンのネットワークパラメータ
        "Chialvo_Neurons"           : 200,
        
        #Chialvo Neuron Map内の結合
        "Chialvo_Rho" : 0.001,
        #入力信号の結合形態
        "W_in_Scale" : 0.1,

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #空走時間
        "Length_Burnin" : 1000,
        #評価時間
        "Length_Eva" : 10000,
        #プロット時間
        "Length_Plot" : 2000,

        #x軸のスケールをlogにするかいなか
        "x_Log_scale" : False,
        
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #入力信号の振幅について
        "Input_Signal_Amp" : 0.1,

        #恒等関数：None, sin関数：np.sin, 離散信号：random.randint
        "Input_Signal_Def" : np.sin,

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #時系列描写の初期化指定
        "Initial_Value_X" : None,
        "Initial_Value_Y" : None,
        "Initial_Value_Phi" : None,
        
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    }
    
    param = Param.copy()
    param.update({
        "Model" : Model_Graph.Model_ChialvoNeuronMap
    })
    Output_Graph.Output_TimeLine_NeuronMap(param)()


def Project_ChialvoNeuronMap_PhaseSpace_2025_02_09_01_32():
    """
    変数2つに対して、時間経過における変数の値の組み合わせを描写する関数。
    Chialvo Neurons Neworkである。
    """

    #共通パラメータ
    Param = {
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #相平面の描写
        "Project_Plot_PhaseSpace_XandY"         : True,
        "Project_Plot_PhaseSpace_XandPhi"       : True,
        "Project_Plot_PhaseSpace_YandPhi"       : True,

        #3次元相平面の描写
        "Project_Plot_PhaseSpace_3D"            : True,

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #従来のChialvoパラメータ
        "Chialvo_a"                 : 0.89,
        "Chialvo_b"                 : 0.6,
        "Chialvo_c"                 : 0.28,
        "Chialvo_k0"                : 0.04,

        #電磁束下のChialvoパラメータ
        "Chialvo_k"                 : -3.2,
        "Chialvo_k1"                : 0.1,
        "Chialvo_k2"                : 0.2,
        "Chialvo_alpha"             : 0.1,
        "Chialvo_beta"              : 0.2,

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #電磁束下Chialvoニューロンのネットワークパラメータ
        "Chialvo_Neurons"           : 100,

        #Chialvo Neuron Map内の結合
        "Chialvo_Rho" : 0.001,
        #入力信号の結合形態
        "W_in_Scale" : 0.1,

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #空走時間
        "Length_Burnin" : 1000,
        #評価時間
        "Length_Eva" : 10000,
        #プロット時間
        "Length_Plot" : 2000,
        
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #入力信号の振幅について
        "Input_Signal_Amp" : 0.2,

        #恒等関数：None, sin関数：np.sin, 離散信号：random.randint
        "Input_Signal_Def" : np.sin,

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #時系列描写の初期化指定
        "Initial_Value_X" : None,
        "Initial_Value_Y" : None,
        "Initial_Value_Phi" : None,
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    }
    
    param = Param.copy()
    param.update({
        "Model" : Model_Graph.Model_ChialvoNeuronMap
    })
    Output_Graph.Output_PhaseSpace_NeuronMap(param)()


def Project_ChialvoNeuronMap_MaximumLyapunov_2025_02_09_01_32():
    """

    """

    #共通パラメータ
    Param = {
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #従来のChialvoパラメータ
        "Chialvo_a"                 : 0.89,
        "Chialvo_b"                 : 0.6,
        "Chialvo_c"                 : 0.28,
        "Chialvo_k0"                : 0.04,

        #電磁束下のChialvoパラメータ
        "Chialvo_k"                 : -3.2,
        "Chialvo_k1"                : 0.1,
        "Chialvo_k2"                : 0.2,
        "Chialvo_alpha"             : 0.1,
        "Chialvo_beta"              : 0.2,

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #電磁束下Chialvoニューロンのネットワークパラメータ
        "Chialvo_Neurons"           : 100,        
        #Chialvo Neuron Map内の結合
        "Chialvo_Rho" : 0.001,
        #入力信号の結合形態
        "W_in_Scale" : 0.1,

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #空走時間
        "Length_Burnin" : 1000,
        #評価時間
        "Length_Eva" : 10000,
        #プロット時間
        "Length_Plot" : 2000,
        
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #入力信号の振幅について
        "Input_Signal_Amp" : 0.2,

        #恒等関数：None, sin関数：np.sin, 離散信号：random.randint
        "Input_Signal_Def" : np.sin,

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #時系列描写の初期化指定
        "Initial_Value_X" : None,
        "Initial_Value_Y" : None,
        "Initial_Value_Phi" : None,
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    }
    
    param = Param.copy()
    param.update({
        "Model" : Model_Graph.Model_ChialvoNeuronMap_MaximumLyapunov
    })
    Output_Graph.OutputA(param)()




#====================================================================