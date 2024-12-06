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
import ctypes
import os

import matplotlib

import Project


import Project_GS_Template
import Project_GS_Sishu

import Project_RS_Template
import Project_RS_Sishu

import Evaluation

#====================================================================
#並列処理数（コメントアウトでデフォルト，グリッドサーチで並列化するときは1にする．）
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_NUM_THREADS"] = "1"

#スリープ対策定数
ES_CONTINUOUS = 0x80000000
ES_AWAYMODE_REQUIRED = 0x00000040
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002

#====================================================================
#メイン
if __name__ == '__main__':
    #+++合図+++
    print("###Program has started.###")

    #フォント埋め込み
    matplotlib.rc('pdf', fonttype=42)
    #スリープ対策
    ctypes.windll.kernel32.SetThreadExecutionState(ES_SYSTEM_REQUIRED | ES_CONTINUOUS)
    
    #+++実行+++
    print("###Main Program has started###")

    #---試行---
#====================================================================
    """見本用"""
    #Project.Project_Single_NRMSE_2023_04_20_03_08()
    #Project.Project_Single_MC_2023_05_25_13_28()
    #Project.Project_Single_2023_06_12_07_38()

    #Project_GS_Template.Project_GridSearch_NRMSEAndMC()
    
    #Project_RS_Template.Project_RandomSearch_NRMSE_Sample()
#====================================================================
    """通常のESNモデルに関連するプロジェクト"""
    #Project.Project_ESN_NRMSE_MC_2024_04_16_13_58()

#====================================================================
    """Sishu提案モデル（結合形態、結合強度を指定）に関連するプロジェクト"""
    #Project.Project_SishuModel_NRMSE_MC_2024_06_23_00_12()

    #Project_RS_Sishu.Project_RandomSearch_NRMSE_SishuModel()

    #Project_GS_Sishu.Project_GridSearch_SishuModel_RingNetwork_NRMSEAndMC()
    #Project_GS_Sishu.Project_GridSearch_SishuModel_StarNetwork_NRMSEAndMC()
    #Project_GS_Sishu.Project_GridSearch_SishuModel_RingStarNetwork_NRMSEAndMC()
#====================================================================
    """Sishu提案モデル（結合形態のみ指定、結合強度は乱数）に関連するプロジェクト"""
    #Project.Project_SishuESN_NRMSE_MC_2024_06_23_00_12()

    #Project_RS_Sishu.Project_RandomSearch_NRMSE_SishuESN()

    Project_GS_Sishu.Project_GridSearch_SishuESN_NRMSEAndMC()
    #Project_GS_Sishu.Project_GridSearch_SishuESN_NRMSEAndMC_Tau()
#====================================================================
#====================================================================
    #+++終了+++
    print("###All Processes have finished###")