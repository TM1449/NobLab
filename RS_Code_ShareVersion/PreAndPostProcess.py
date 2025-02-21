#####################################################################
#信川研ESNプロジェクト用
#制作者：中村亮介 吉原悠斗 吉田聡太 飯沼貴大
#作成日：2023/05/24
#####################################################################

#====================================================================
#＜このプログラムのTodoメモ＞
#・
#・
#・

#====================================================================
import ctypes
import matplotlib
import Project_Single_MC
#====================================================================
#事前処理
def PreProcess():
    #並列処理数（コメントアウトでデフォルト，グリッドサーチで並列化するときは1にする．）
    #os.environ["OPENBLAS_NUM_THREADS"] = "1"
    #os.environ["MKL_NUM_THREADS"] = "1"
    #os.environ["VECLIB_NUM_THREADS"] = "1"

    #スリープ対策定数
    ES_CONTINUOUS = 0x80000000
    ES_AWAYMODE_REQUIRED = 0x00000040
    ES_SYSTEM_REQUIRED = 0x00000001
    ES_DISPLAY_REQUIRED = 0x00000002

    #+++合図+++
    print("###Program has started.###")

    #フォント埋め込み
    matplotlib.rc('pdf', fonttype=42)
    #スリープ対策
    ctypes.windll.kernel32.SetThreadExecutionState(ES_SYSTEM_REQUIRED | ES_CONTINUOUS)
    
    #+++実行+++
    print("###Main Program has started###")

#====================================================================
#事後処理
def PostProcess():
    #+++終了+++
    print("###All Processes have finished###")


#====================================================================
#並列処理数（コメントアウトでデフォルト，グリッドサーチで並列化するときは1にする．）
#os.environ["OPENBLAS_NUM_THREADS"] = "1"
#os.environ["MKL_NUM_THREADS"] = "1"
#os.environ["VECLIB_NUM_THREADS"] = "1"

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
    #Project.Project_Single_NRMSE_2023_04_20_03_08()
    Project_Single_MC.Project_Single_MC_2023_05_25_13_28()
    #Project.Project_SingleMLE_2023_07_08_17_15()

    #Project.Project_Single_2023_06_12_07_38()

    #Project_GS_Template.Project_GridSearch_NRMSEAndMC()
       
    #+++終了+++
    print("###All Processes have finished###")
