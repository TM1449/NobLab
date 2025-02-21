#####################################################################
#信川研ESNプロジェクト用
#制作者：中村亮介 吉原悠斗 吉田聡太 飯沼貴大
#作成日：2023/05/26
#####################################################################

#====================================================================
#＜このプログラムのTodoメモ＞
#・
#・
#・

#====================================================================
import datetime
import re

import numpy as np
import pandas as pd
from pathos.multiprocessing import ProcessingPool

import FileAndDir

#====================================================================
#汎用グリッドサーチ

#********************************************************************
#変化軸
class Axis:
    """
    変化軸
    """
    #コンストラクタ
    def __init__(self, param: dict, key: str, value: np.ndarray):
        self.Param = dict()

        #パラメータ設定
        self.Key = key                                  #軸名
        self.Value = value                              #軸値（ndarray型）

        self.NumberOfPoints = len(self.Value)           #刻み数

#********************************************************************
#汎用グリッドサーチ
class Normal:
    """
    汎用グリッドサーチ
    """
    #コンストラクタ
    def __init__(self, param: dict):
        self.Param = param

        #パラメータ取得
        self.MachineName = self.Param["GridSearch_MachineName"]                     #計算機名
        self.MultiThread = self.Param["GridSearch_MultiThread"]                     #スレッド数
        self.MaxNumberInOneFile = self.Param["GridSearch_MaxNumberInOneFile"]       #１ファイルの最大のポイント数
        self.MaxNumberInOnePool = self.Param["GridSearch_MaxNumberInOnePool"]       #１プール（並列する）最大のポイント数
        self.NumberOfSamples = self.Param["GridSearch_NumberOfSamples"]             #サンプル数
        self.ProjectName = self.Param["GridSearch_ProjectName"]                     #プロジェクト名
        self.ProjectDate = self.Param["GridSearch_ProjectDate"]                     #プロジェクト日時

        self.StartPoint = self.Param["GridSearch_StartPoint"]                       #担当開始ポイントインデックス
        self.EndPoint = self.Param["GridSearch_EndPoint"]                           #担当終了ポイントインデックス
        self.NumPoints = self.EndPoint - self.StartPoint                            #担当ポイント数
        
        #フォルダ構造
        self.ConstractFileTree_Root()

        #プロジェクト（Type型）
        self.T_GS_Process = self.Param["GridSearch_T_Process"]
        #プロジェクトのインスタンス
        self.GS_Process = self.T_GS_Process(self.Param.copy(), self)

        #出力処理（Type型）
        self.T_GS_Output = self.Param["GridSearch_T_Output"]
        #出力処理のインスタンス
        self.GS_Output = self.T_GS_Output(self.Param.copy(), self)

    #実行
    def __call__(self):
        #初期処理
        print("### Started GS_" + self.ProjectName + " ###")
        
        #Paramから軸抽出
        self.GridSearch_AxisList = [Axis(self.Param, "Samples", list(range(self.NumberOfSamples)))]
        for key, value in self.Param.items():
            check = re.match(r"_Param_([a-zA-Z0-9_]+)", key)
            if check is not None:
                self.GridSearch_AxisList.append(Axis(self.Param, check.groups()[0], value))
        
        #評価点の数
        self.NumberOfAllPoints = 1
        for axis in self.GridSearch_AxisList:
            self.NumberOfAllPoints *= axis.NumberOfPoints
        
        #進行状況表示
        self.NumFinishedPoints = self.GetNumFinishedPoints()
        self.F_Finished = (self.NumFinishedPoints == self.NumberOfAllPoints)
        print("*** Remained %d / %d Points ( %d %%) ***"\
            %(self.NumPoints - self.NumFinishedPoints, self.NumPoints,\
             (self.NumberOfAllPoints - self.NumFinishedPoints) * 100. / self.NumberOfAllPoints) if self.NumberOfAllPoints != 0 else 0.)
        
        #終了判定
        self.F_Finished = (self.NumFinishedPoints == self.NumPoints)

        #処理本体ループ
        print("*** Started Grid Search Loop ***")
        while not self.F_Finished:
            Now = datetime.datetime.now()
            self.Date_Now = Now.strftime('%Y_%m_%d_%H_%M_%S')

            #未処理ポイント検索（チャンク作成）
            print("+++ Started Searching Chank +++")
            ChankIndex = self.GetNextChankIndex()
            print("+++ Number of Points in this Chank : %d +++"%(len(ChankIndex)))
            
            #進行状況表示
            Chank = [self.GetParamsFromIndex(i, len(ChankIndex), index) for i, index in enumerate(ChankIndex)]
            print("+++ Processing Samples from No.%d +++"%(Chank[0]["Samples"]))
            
            #チャンクの処理
            self.Process(Chank)

            #進行状況表示
            self.NumFinishedPoints = self.GetNumFinishedPoints()
            print("*** Remain %d / %d Points ( %d %%) ***"\
            %(self.NumFinishedPoints, self.NumberOfAllPoints,\
             (self.NumberOfAllPoints - self.NumFinishedPoints) * 100. / self.NumberOfAllPoints) if self.NumberOfAllPoints != 0 else 0.)
            
            #終了判定
            self.F_Finished = (self.NumFinishedPoints == self.NumPoints)

        #終了処理
        Now = datetime.datetime.now()
        self.Date_Now = Now.strftime('%Y_%m_%d_%H_%M_%S')
        print("*** Finished Grid Search Loop ***")
        if self.NumFinishedPoints == self.NumberOfAllPoints:
            print("*** Finished Grid Search Loop ALL ***")
            self.OutputResults()
            
        #終了
        print("### Finished Grid Search ###")
        
    #未処理のポイント数取得
    def GetNumFinishedPoints(self) -> int:
        print("*** Counting Finished Points ***")
        Data = self.LoadAll()
        if Data is not None:
            return sum((log is not None) if self.StartPoint <= i and i < self.EndPoint else False for i, log in enumerate(Data))
        else:
            return 0

    #未処理のチャンク作成
    def GetNextChankIndex(self) -> list:
        Chank = []
        for i, log in enumerate(self.LoadAll()):
            if self.MaxNumberInOnePool <= len(Chank):
                break
            if self.StartPoint <= i and i < self.EndPoint:
                if log == None:
                    Chank += [i]

        return Chank

    #データ保存
    def SaveData(self, data: dict):
        self.Save_Logs(data, "_AS")

    #全ロード
    def LoadAll(self) -> list:
        return self.Load_AllLogs("_AS")
    
    #インデックスリストからパラメータ収集
    def GetParamsFromIndex(self, index_in_chank: int, num_point_in_chank: int, index: int) -> dict:
        Params = {
            "IndexInChank" : index_in_chank, 
            "NumPointsInChank" : num_point_in_chank, 
            "Index" : index}
        L_IndexInAxis = []
        for axis in reversed(self.GridSearch_AxisList):
            axis_index = index % axis.NumberOfPoints
            index //= axis.NumberOfPoints

            Params.update({axis.Key : axis.Value[axis_index]})
            L_IndexInAxis += [axis_index]

        Params.update({"L_IndexInAxis" : reversed(L_IndexInAxis)})

        return Params

    #結果表示
    def OutputResults(self):
        print("*** Outputing Results ***")
        self.GS_Output(self.LoadAll())

    #処理本体
    def Process(self, chank: list):
        if self.MultiThread == 0 :
            for point in chank:
                point.update(self.Thread(point, self.Param))
                self.SaveData([point])
        else:
            Pool = ProcessingPool(self.MultiThread)
            thread = [Pool.apipe(self.Thread, point, self.Param) for point in chank]

            F_NotFinished = True
            F_NotSaved = [True for _ in range(len(thread))]
            while F_NotFinished:
                F_NotFinished = False
                for i in range(len(thread)):
                    if thread[i].ready():
                        if F_NotSaved[i]:
                            F_NotSaved[i] = False
                            chank[i].update(thread[i].get())
                            self.SaveData([chank[i]])
                    else:
                        F_NotFinished = True
            
    #並列処理内部
    def Thread(self, chank_i: dict, param: dict) -> dict:
        self.ConstractFileTree_Samples_Branch(chank_i["Samples"])
        self.ConstractFileTree_Points_Branch(chank_i["Index"])
        chank_i.update({"GridSearch_RootDir" : self.Dir_Points_Branch})
        return self.GS_Process(chank_i, param)
    
    #以下ファイルとフォルダ単位のセーブとロード
    def Load_Root(self, save_type: str):
        pass

        self.Load_Project(save_type)

    def Save_Root(self, save_type: str):
        pass

        self.Save_Project(save_type)
        
    def Load_Project(self, save_type: str):
        self.Load_Project_Param(save_type)

        self.Load_Samples(save_type)
        self.Load_Results(save_type)
        
    def Save_Project(self, save_type: str):
        self.Save_Project_Param(save_type)

        self.Save_Samples(save_type)
        self.Save_Results(save_type)
        
    def Load_Project_Param(self, save_type: str):
        pass

    def Save_Project_Param(self, save_type: str):
        self.CSV_Project_Param.Save(self.Param)
        
    def Load_AllLogs(self, save_type: str) -> list:
        Paths = self.Dir_Logs.SearchChildName("Log")

        Logs = []
        if Paths is not None:
            for path in Paths:
                Dir_Logs_Branch = self.Dir_Logs.AddTemporaryChild(FileAndDir.DirNode(path))
                Dir_Logs_Branch_Log = Dir_Logs_Branch.AddTemporaryChild(FileAndDir.FileNode_df("Log"))
                Logs += Dir_Logs_Branch_Log.Load().to_dict(orient='records')
            
        Data = [None for _ in range(self.NumberOfAllPoints)]
        for log in Logs:
            Data[log["Index"]] = log

        return Data

    def Save_Logs(self, data: list, save_type: str):
        Logs_prev = []
        if self.Dir_Logs.CheckLoadableLast("Log"):
            self.ConstractFileTree_Logs_Branch(False, "_" + self.MachineName)
            Logs_prev = self.CSV_Logs_Log.Load().to_dict(orient='records')
        else:
            self.ConstractFileTree_Logs_Branch(True, "_" + self.MachineName)
        
        Logs = pd.DataFrame(Logs_prev + data)
        if len(Logs) != 0:
            while len(Logs) - 1 > self.MaxNumberInOneFile:
                self.CSV_Logs_Log.Save(Logs[0 : self.MaxNumberInOneFile])
                Logs = Logs[self.MaxNumberInOneFile:]
                self.ConstractFileTree_Logs_Branch(True, "_" + self.MachineName)
                
            self.CSV_Logs_Log.Save(Logs)

    def Load_Samples(self, save_type: str):
        pass

    def Save_Samples(self, save_type: str):
        pass
    
    def Load_Results(self, save_type: str):
        self.Load_Results_Log(save_type)

    def Save_Results(self, save_type: str):
        self.Save_Results_Log(save_type)
        
    def Load_Charts_Param(self, save_type: str):
        pass

    def Save_Charts_Param(self, save_type: str):
        self.CSV_Charts_Param.Save(self.Param)
        
    #以下フォルダ構造
    def ConstractFileTree_Root(self):
        self.Dir_Root = FileAndDir.RootNode(".")
        self.Dir_Results = self.Dir_Root.AddChild(FileAndDir.DirNode("Results"))
        
        self.ConstractFileTree_Project()
        
    def ConstractFileTree_Project(self):
        self.Dir_Project = self.Dir_Results.AddChild(FileAndDir.DirNode("Prj" + self.ProjectName + self.ProjectDate))
        self.CSV_Project_Param = self.Dir_Project.AddChild(FileAndDir.FileNode_dict("Param"))
        
        self.ConstractFileTree_Samples()
        self.ConstractFileTree_Logs()
        self.ConstractFileTree_Charts()
       
    def ConstractFileTree_Samples(self):
        self.Dir_Samples = self.Dir_Project.AddChild(FileAndDir.DirNode("Smp"))
        
    def ConstractFileTree_Samples_Branch(self, index: int):
        self.Dir_Samples_Branch = self.Dir_Samples.AddChild(FileAndDir.DirNode("Smp" + str(index)))
        
    def ConstractFileTree_Points_Branch(self, index: int):
        self.Dir_Points_Branch = self.Dir_Samples_Branch.AddChild(FileAndDir.DirNode("Pnt" + str(index)))
        self.CSV_Point_Param = self.Dir_Points_Branch.AddChild(FileAndDir.FileNode_dict("Param"))
        
    def ConstractFileTree_Logs(self):
        self.Dir_Logs = self.Dir_Project.AddChild(FileAndDir.LogDirNode("Log"))
        
    def ConstractFileTree_Logs_Branch(self, f_new: bool = False, tag: str = ""):
        if f_new: self.Dir_Logs_Branch = self.Dir_Logs.AddChild(FileAndDir.DirNode("Log"), FileAndDir.Manager.getDate(), tag)
        else: self.Dir_Logs_Branch = self.Dir_Logs.SearchLastChild("Log")

        self.CSV_Logs_Log = self.Dir_Logs_Branch.AddChild(FileAndDir.FileNode_df("Log"))
        
    def ConstractFileTree_Charts(self):
        self.Dir_Charts = self.Dir_Project.AddChild(FileAndDir.LogDirNode("Charts"))
        
    def ConstractFileTree_Charts_Branch(self, f_new: bool = False, tag: str = ""):
        if f_new: self.Dir_Charts_Branch = self.Dir_Charts.AddChild(FileAndDir.DirNode("Charts"), FileAndDir.Manager.getDate(), tag)
        else: self.Dir_Charts_Branch = self.Dir_Charts.SearchLastChild("Charts")

        self.CSV_Charts_Param = self.Dir_Charts_Branch.AddChild(FileAndDir.FileNode_dict("Param"))
        
        self.GS_Output.ConstractFileTree_Charts()
