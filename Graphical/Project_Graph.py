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
        "Project_Plot_TimeLine" : True,
        
        #--------------------------------------------------------------------
        #時系列の描写
        "Project_Plot_TimeLine_X"               : False,
        "Project_Plot_TimeLine_Y"               : False,
        "Project_Plot_TimeLine_Phi"             : False,
        
        #==========================================================================================
        #相平面の描写をするかどうか（これがオフだと描写しない）
        "Project_Plot_PhaseSpace" : True,
        
        #--------------------------------------------------------------------
        #相平面の描写
        "Project_Plot_PhaseSpace_XandY"         : True,
        "Project_Plot_PhaseSpace_XandPhi"       : True,
        "Project_Plot_PhaseSpace_YandPhi"       : True,

        #3次元相平面の描写
        "Project_Plot_PhaseSpace_3D"            : True,
        
        #==========================================================================================
        #従来ChialvoのNullclineらの描写をするかどうか（これがオフだと描写しない）
        "Project_Plot_OldNullcline" : True,
        
        #--------------------------------------------------------------------
        #従来ChialvoのNullcLineの描写
        "Project_Plot_OldNullcLine"             : True,
        #従来Chialvoの相平面における解の軌道の描写
        "Project_Plot_NewPhaseSpace"           : False,

        #==========================================================================================
        #NewChialvoのNullclineらの描写をするかどうか（これがオフだと描写しない）
        "Project_Plot_NewNullcline" : True,
        
        #--------------------------------------------------------------------
        #電磁束下ChialvoのNullcLineの描写
        "Project_Plot_NewNullcLine"             : True,
        #電磁束下Chialvoにおける相平面における解の軌道の描写
        "Project_Plot_NewPhaseSpace"           : False,

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #NewChialvoの3D-Nullclineらの描写をするかどうか（これがオフだと描写しない）
        "Project_Plot_NewNullcline3D" : True,
        
        #--------------------------------------------------------------------
        #Nullclineの描写
        "Project_Plot_NewNullcline3D"         : True,
        #相平面における解の軌道の描写
        "Project_Plot_PhaseSpace3D"           : True,

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #従来のChialvoパラメータ
        "Chialvo_a"                 : 0.89,
        "Chialvo_b"                 : 0.6,
        "Chialvo_c"                 : 0.28,
        "Chialvo_k0"                : 0.06,

        #電磁束下のChialvoパラメータ
        "Chialvo_k"                 : 2.2,
        "Chialvo_k1"                : 0.1,
        "Chialvo_k2"                : 0.2,
        "Chialvo_alpha"             : 0.1,
        "Chialvo_beta"              : 0.2,
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #時系列描写の実行時間
        "RunTime" : 3000,
        "Plot_Start" : 2500,
        "Plot_End" : 3000,

        #ベクトル場の間隔
        "Vdt" : 0.5,
        #Nullclineの間隔
        "dt" : 0.0001,

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #どこからどこまでプロット図の点を作成するか
        "Plot_x_Start" : -3,
        "Plot_x_End" : 3,

        "Plot_y_Start" : -3,
        "Plot_y_End" : 3,

        "Plot_phi_Start" : -3,
        "Plot_phi_End" : 3,

        #時系列描写の初期化指定
        "Initial_Value_X" : None,
        "Initial_Value_Y" : None,
        "Initial_Value_Phi" : None,
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    }

    if Param["Project_Plot_TimeLine"]:
        param = Param.copy()
        param.update({
            "Model" : Model_Graph.Model_Chialvo
        })
        Output_Graph.Output_TimeLine(param)()

    if Param["Project_Plot_PhaseSpace"]:
        param = Param.copy()
        param.update({
            "Model" : Model_Graph.Model_Chialvo
        })
        Output_Graph.Output_PhaseSpace(param)()

    if Param["Project_Plot_OldNullcLine"]:
        param = Param.copy()
        param.update({
            "Model" : Model_Graph.Model_Chialvo_OldNullcline
        })
        Output_Graph.Output_OldNullcline(param)()

    if Param["Project_Plot_NewNullcLine"]:
        param = Param.copy()
        param.update({
            "Model" : Model_Graph.Model_Chialvo_NewNullcline
        })
        Output_Graph.Output_NewNullcline(param)()

    if Param["Project_Plot_NewNullcline3D"]:
        param = Param.copy()
        param.update({
            "Model" : Model_Graph.Model_Chialvo_NewNullcline3D
        })
        Output_Graph.Output_NewNullcline3D(param)()

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
        "Chialvo_k"                 : -2.2,
        "Chialvo_k1"                : 0.1,
        "Chialvo_k2"                : 0.2,
        "Chialvo_alpha"             : 0.1,
        "Chialvo_beta"              : 0.2,
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #時系列描写の実行時間
        "RunTime" : 3500,
        "Plot_Start" : 2500,
        "Plot_End" : 3500,

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
        "Chialvo_b"                 : 0.6,
        "Chialvo_c"                 : 0.28,
        "Chialvo_k0"                : 0.04,

        #電磁束下のChialvoパラメータ
        "Chialvo_k"                 : -2.2,
        "Chialvo_k1"                : 0.1,
        "Chialvo_k2"                : 0.2,
        "Chialvo_alpha"             : 0.1,
        "Chialvo_beta"              : 0.2,
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #時系列描写の実行時間
        "RunTime" : 3500,
        "Plot_Start" : 2500,
        "Plot_End" : 3500,

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
        "Project_Plot_PhaseSpace"           : False,

        #+++++++++++++++++++++++++s+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #従来のChialvoパラメータ
        "Chialvo_a"                 : 0.89,
        "Chialvo_b"                 : 0.18,
        "Chialvo_c"                 : 0.28,
        "Chialvo_k0"                : 0.025,

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #時系列描写の実行時間
        "RunTime" : 3000,
        "Plot_Start" : 0,
        "Plot_End" : 3000,

        #ベクトル場の間隔
        "Vdt" : 0.05,
        #Nullclineの間隔
        "dt" : 0.0001,

        #どこからどこまでプロット図の点を作成するか
        "Plot_x_Start" : 0,
        "Plot_x_End" : 3,

        "Plot_y_Start" : 0,
        "Plot_y_End" : 3,

        #時系列描写の初期化指定
        "Initial_Value_X" : None,
        "Initial_Value_Y" : None,

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
        "Chialvo_b"                 : 0.18,
        "Chialvo_c"                 : 0.28,
        "Chialvo_k0"                : 0.025,

        #電磁束下のChialvoパラメータ
        "Chialvo_k"                 : -0.25,
        "Chialvo_k1"                : 0.1,
        "Chialvo_k2"                : 0.2,
        "Chialvo_alpha"             : 0.1,
        "Chialvo_beta"              : 0.2,
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #時系列描写の実行時間
        "RunTime" : 3000,
        "Plot_Start" : 0,
        "Plot_End" : 3000,

        #ベクトル場の間隔
        "Vdt" : 0.05,
        #Nullclineの間隔
        "dt" : 0.0001,

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #どこからどこまでプロット図の点を作成するか
        "Plot_x_Start" : -1,
        "Plot_x_End" : 1,

        "Plot_y_Start" : -1,
        "Plot_y_End" : 1,

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

def Project_Chialvo_NewNullcline3D_2024_10_20_12_17():
    """
    3次元Chialvo Neuron Mapの3Dマップにおけるベクトル場とNullclineを描写する関数。
    """

    #共通パラメータ
    Param = {
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #Nullclineの描写
        "Project_Plot_NewNullcline3D"         : True,
        #相平面における解の軌道の描写
        "Project_Plot_PhaseSpace3D"           : True,

        #+++++++++++++++++++++++++s+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
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
        "RunTime" : 3000,
        "Plot_Start" : 2900,
        "Plot_End" : 3000,

        #ベクトル場の間隔
        "Vdt" : 1,
        #Nullclineの間隔
        "dt" : 0.0001,

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #どこからどこまでプロット図の点を作成するか
        "Plot_x_Start" : -3,
        "Plot_x_End" : 3,

        "Plot_y_Start" : -3,
        "Plot_y_End" : 3,

        "Plot_phi_Start" : -3,
        "Plot_phi_End" : 3,

        #時系列描写の初期化指定
        "Initial_Value_X" : None,
        "Initial_Value_Y" : None,
        "Initial_Value_Phi" : None,
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    }
    
    param = Param.copy()
    param.update({
        "Model" : Model_Graph.Model_Chialvo_NewNullcline3D
    })
    Output_Graph.Output_NewNullcline3D(param)()