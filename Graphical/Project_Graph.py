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
        "Chialvo_b"                 : 0.6,
        "Chialvo_c"                 : 0.28,
        "Chialvo_k0"                : 0.04,

        #電磁束下のChialvoパラメータ
        "Chialvo_k"                 : -0.8,
        "Chialvo_k1"                : 0.1,
        "Chialvo_k2"                : 0.2,
        "Chialvo_alpha"             : 0.1,
        "Chialvo_beta"              : 0.2,

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #空走時間
        "Length_Burnin" : 100000,
        #描写時間
        "Length_Plot" : 10000,

        #x軸のスケールをlogにするかいなか
        "x_Log_scale" : False,
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #入力信号の振幅について
        "Input_Signal" : 0,

        #入力信号の間隔
        #（ None は通常の信号。数値を指定するとその間隔で信号が生成）
        "Input_Signal_Interval" : None,

        #入力信号の長さ
        #( None は通常の信号。Interval の数値を指定するとその長さで信号が生成)
        "Input_Signal_Line" : None,

        #恒等関数：None, sin関数：np.sin, 離散信号：random.randint
        "Input_Signal_def" : None,

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
        "Project_Plot_PhaseSpace_XandY"         : False,
        "Project_Plot_PhaseSpace_XandPhi"       : False,
        "Project_Plot_PhaseSpace_YandPhi"       : False,

        #3次元相平面の描写
        "Project_Plot_PhaseSpace_3D"            : True,

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
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
        "RunTime" : 6000,
        "Plot_Start" : 1000,
        "Plot_End" : 6000,

        #入力信号：uについて
        "Input_Signal"  : 0,

        #恒等関数：None
        #sin関数：np.sin
        #離散信号：random.randint
        "Input_Signal_def" : None,

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

def Project_Chialvo_Intersection_2024_09_24_17_25():
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
        "Chialvo_k"                 : -1.1,
        "Chialvo_k1"                : 0.1,
        "Chialvo_k2"                : 0.2,
        "Chialvo_alpha"             : 0.1,
        "Chialvo_beta"              : 0.2,
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #時系列描写の実行時間
        "RunTime" : 6000,
        "Plot_Start" : 1000,
        "Plot_End" : 6000,

        #入力信号：uについて
        "Input_Signal"  : 0,

        #恒等関数：None
        #sin関数：np.sin
        #離散信号：random.randint
        "Input_Signal_def" : None,

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

#====================================================================
def Project_Chialvo_Neurons_Network_PhaseSpace_2025_01_07_16_00():
    """
    変数2つに対して、時間経過における変数の値の組み合わせを描写する関数。
    ただし、Chialvo Neurons Neworkである。
    """

    #共通パラメータ
    Param = {
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #相平面の描写
        "Project_Plot_PhaseSpace_XandY"         : False,
        "Project_Plot_PhaseSpace_XandPhi"       : False,
        "Project_Plot_PhaseSpace_YandPhi"       : False,

        #3次元相平面の描写
        "Project_Plot_PhaseSpace_3D"            : True,

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
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

        #電磁束下Chialvoニューロンのネットワークパラメータ
        "Chialvo_Neurons"           : 100,
        "Chialvo_Mu"                : 0.1, #中心ニューロン間
        "Chialvo_Sigma"             : 0.1, #隣接ニューロン間
        "Chialvo_R"                 : 10,  #ニューロン結合数

        #電磁束下Chialvoニューロンネットワークの追加パラメータ
        "Chialvo_Xi_mu"             : 0.001,
        "Chialvo_Xi_sigma"          : 0.002,
        "Chialvo_D_mu"              : 0.1,
        "Chialvo_D_sigma"           : 0.1,

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #時系列描写の実行時間
        "RunTime" : 20000,
        "Plot_Start" : 10000,
        "Plot_End" : 20000,

        #入力信号：uについて
        "Input_Signal"  : 0,

        #恒等関数：None
        #sin関数：np.sin
        #離散信号：random.randint
        "Input_Signal_def" : None,
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    }
    
    param = Param.copy()
    param.update({
        "Model" : Model_Graph.Model_Chialvo_Neurons_Network
    })
    Output_Graph.Output_PhaseSpace(param)()

#====================================================================