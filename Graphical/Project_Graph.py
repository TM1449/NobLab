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
def Project_Chialvo_All_2024_09_24_14_17():
    """
    プロジェクト例
    ・命名規則は Project_[手法]_[タスク]_YYYY_MM_DD_HH_MM

    ・昔の有意有用な結果のプロジェクトは全て残す
    ・新しいものは別の関数で作る．
    """
    
    """
    Chialvoニューロンマップを用いた様々なグラフの描写プロジェクト
    """

    #共通パラメータ
    Param = {
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #時系列の描写をするかどうか（これがオフだと描写しない）
        "Project_Plot_TimeLine_ALL" : True,
        
        #--------------------------------------------------------------------
        #時系列の描写
        "Project_Plot_TimeLine_X"               : True,
        "Project_Plot_TimeLine_Y"               : True,
        "Project_Plot_TimeLine_Phi"             : True,
        
        #==========================================================================================
        #相平面の描写をするかどうか（これがオフだと描写しない）
        "Project_Plot_PhaseSpace_ALL" : True,
        
        #--------------------------------------------------------------------
        #相平面の描写
        "Project_Plot_PhaseSpace_XandY"         : False,
        "Project_Plot_PhaseSpace_XandPhi"       : False,
        "Project_Plot_PhaseSpace_YandPhi"       : False,

        #3次元相平面の描写
        "Project_Plot_PhaseSpace_3D"            : True,
        
        #==========================================================================================
        #従来ChialvoのNullclineらの描写をするかどうか（これがオフだと描写しない）
        "Project_Plot_OldNullcline_ALL" : False,
        
        #--------------------------------------------------------------------
        #従来ChialvoのNullcLineの描写
        "Project_Plot_OldNullcline"             : False,
        #従来Chialvoの相平面における解の軌道の描写
        "Project_Plot_PhaseSpace"           : False,

        #==========================================================================================
        #NewChialvoのNullclineらの描写をするかどうか（これがオフだと描写しない）
        "Project_Plot_NewNullcline_ALL" : True,
        
        #--------------------------------------------------------------------
        #電磁束下ChialvoのNullcLineの描写
        "Project_Plot_NewNullcline"             : True,
        #電磁束下Chialvoにおける相平面における解の軌道の描写
        "Project_Plot_PhaseSpace"           : False,

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #NewChialvoの3D-Nullclineらの描写をするかどうか（これがオフだと描写しない）
        "Project_Plot_NewNullcline3D_ALL" : False,
        
        #--------------------------------------------------------------------
        #Nullclineの描写
        "Project_Plot_NewNullcline3D"         : False,
        #相平面における解の軌道の描写
        "Project_Plot_PhaseSpace3D"           : False,

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #従来のChialvoパラメータ
        "Chialvo_a"                 : 0.89,
        "Chialvo_b"                 : 0.6,
        "Chialvo_c"                 : 0.28,
        "Chialvo_k0"                : 0.04,

        #電磁束下のChialvoパラメータ
        "Chialvo_k"                 : -22,
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

        #恒等関数：None
        #sin関数：np.sin
        #離散信号：random.randint
        "Input_Signal_def" : None,

        #ベクトル場の間隔
        "Vdt" : 0.05,
        #Nullclineの間隔
        "dt" : 0.0001,

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #どこからどこまでプロット図の点を作成するか
        "Plot_x_Start" : -2,
        "Plot_x_End" : 2,

        "Plot_y_Start" : -2,
        "Plot_y_End" : 2,

        "Plot_phi_Start" : -3,
        "Plot_phi_End" : 3,

        #時系列描写の初期化指定
        "Initial_Value_X" : 0.013131072,
        "Initial_Value_Y" : 2.4738305,
        "Initial_Value_Phi" : 0.0010943,
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    }

    if Param["Project_Plot_TimeLine_ALL"]:
        param = Param.copy()
        param.update({
            "Model" : Model_Graph.Model_Chialvo
        })
        Output_Graph.Output_TimeLine(param)()

    if Param["Project_Plot_PhaseSpace_ALL"]:
        param = Param.copy()
        param.update({
            "Model" : Model_Graph.Model_Chialvo
        })
        Output_Graph.Output_PhaseSpace(param)()

    if Param["Project_Plot_OldNullcline_ALL"]:
        param = Param.copy()
        param.update({
            "Model" : Model_Graph.Model_Chialvo_OldNullcline
        })
        Output_Graph.Output_OldNullcline(param)()

    if Param["Project_Plot_NewNullcline_ALL"]:
        param = Param.copy()
        param.update({
            "Model" : Model_Graph.Model_Chialvo_NewNullcline
        })
        Output_Graph.Output_NewNullcline(param)()

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
        "Chialvo_k"                 : -3.2,
        "Chialvo_k1"                : 0.1,
        "Chialvo_k2"                : 0.2,
        "Chialvo_alpha"             : 0.1,
        "Chialvo_beta"              : 0.2,
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #時系列描写の実行時間
        "RunTime" : 6000,

        #描写開始時間と描写終了時間
        "Plot_Start" : 1800,
        "Plot_End" : 2200,

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