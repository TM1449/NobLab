#####################################################################
#信川研EM_Chialvoプロジェクト用
#制作者：田中勝規
#作成日：2025/01/28
"""
本体
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

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import Project_No
import Project_GS_Sishu_No
import Project_RS_Sishu_No

import Evaluation_No

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

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
    """Sishu提案モデルに関連するプロジェクト"""
    Project_No.Project_ESN_2025_04_18_14_51()
    
    #Project_RS_Sishu_EM.Project_RandomSearch_NRMSE_EMChialvo()llllllllllll

    #Project_GS_Sishu_EM.Project_GridSearch_EMChialvo()
    #Project_GS_Sishu_EM.Project_GridSearch_EMChialvo_kOnly()
    
#====================================================================
#====================================================================
    #+++終了+++
    print("###All Processes have finished###")