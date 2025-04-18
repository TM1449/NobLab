#####################################################################
#信川研ESNプロジェクト用
#制作者：中村亮介 吉原悠斗 吉田聡太 飯沼貴大
#作成日：2023/06/01

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
import os
import glob
import re
import datetime

import numpy as np
import pandas as pd
import torch

#====================================================================
#ファイルとフォルダとその構造管理

#********************************************************************
#構造マネージャー
class Manager:
    """
    構造マネージャー
    """
    #現在時刻取得
    @staticmethod
    def getDate() -> str:
        Now = datetime.datetime.now()
        return Now.strftime('%Y_%m_%d_%H_%M_%S')

    #文字列＞不動点小数のリスト
    @staticmethod
    def StoL_Float(string: str) -> list:
        if type(string) is not str:
            return

        string = string.replace('[', '')
        string = string.replace(']', '')
        L_string = string.split(",")
        L_float = [float(s) for s in L_string]

        return L_float
    
    #文字列＞整数のリスト
    @staticmethod
    def StoL_Int(string: str) -> list:
        if type(string) is not str:
            return

        string = string.replace('[', '')
        string = string.replace(']', '')
        L_string = string.split(",")
        L_int = [int(s) for s in L_string]

        return L_int

#********************************************************************
#構造ノード
class Node:
    """
    構造ノード
    """
    #コンストラクタ
    def __init__(self, name: str,):
        self.Path = None
        self.Name = name
        self.Parent = None

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#ファイルノード
class FileNode(Node):
    """
    ファイルノード
    """
    #コンストラクタ
    def __init__(self, name: str):
        super().__init__(name)

    #保存
    def Save(self):
        if not self.CheckLoadable():
            self.Parent.Save()
    
    #読み込み
    def Load(self):
        pass
    
    #読み込み可能か
    def CheckLoadable(self) -> bool:
        return os.path.exists(self.Path + self.Extension)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#フォルダノード
class DirNode(Node):
    """
    フォルダノード
    """
    #コンストラクタ
    def __init__(self, name: str):
        super().__init__(name)
        
    #子ノード登録
    def AddChild(self, node: any) -> any:
        node.Parent = self
        
        node.Path = self.Path + "/" + node.Name
        
        return node
        
    #フォルダ作成
    def Save(self):
        if not os.path.exists(self.Path):
            os.makedirs(self.Path, exist_ok=True)
            
    #名前から子ノードのパスを検索＞リストを返す
    def SearchChildName(self, child_name: str) -> list:
        Names = glob.glob(self.Path + "/" + child_name + "*")
        if len(Names) == 0: return None
        Names.sort()
        Paths = []
        for name in Names:
            Paths += [os.path.basename(name)]
        return Paths
    
    #子ノードを登録はせずパスだけ生成
    def AddTemporaryChild(self, node: any) -> any:
        node.Parent = self
        node.Path = self.Path + "/" + node.Name

        return node
    
    #子ノードに少なくとも１つはロードできるものがあるか
    def CheckLoadableAny(self, child_name: str):
        return len(glob.glob(self.Path + "/" + child_name + "*")) != 0
    
class DuplicatingDirNode(DirNode):
    """
    重複して同名子ノード（名前を変えて管理）を持つ可能性があるフォルダノード
    """
    #コンストラクタ
    def __init__(self, name: str):
        super().__init__(name)
        
        self.Index = None
        self.ResetChildNameList()
        
    #子ノード登録
    def AddChild(self, node: any, f_duplicating: bool = False) -> any:
        node.Parent = self

        #重複している場合フォルダ/ファイル名にインデックスを付ける
        self.BookChildName(node.Name)
        node.Index = self.CountChildren(node.Name)
        if not f_duplicating or node.Index == 1:
            node.Path = self.Path + "/" + node.Name
        else:
            node.Path = self.Path + "/" + node.Name + "_(" + str(node.Index) + ")"
        
        return node
        
    #子ノード名を辞書に登録
    def BookChildName(self, child_name: str):
        if child_name in self.ChildNameList.keys():
            self.ChildNameList[child_name] += 1
        else:
            self.ChildNameList.update({child_name : 1})

    #辞書から同名の子ノード数を取得
    def CountChildren(self, child_name: str) -> int:
        if child_name in self.ChildNameList.keys():
            return self.ChildNameList[child_name]
        else:
            return 0
        
    #辞書初期化
    def ResetChildNameList(self):
        self.ChildNameList = dict()

class LogDirNode(DirNode):
    """
    子ノード名が同名で日時を付けて管理するフォルダノード
    """
    #コンストラクタ
    def __init__(self, name: str):
        super().__init__(name)

    #子ノード登録（tag：日時）
    def AddChild(self, node: any, date: str, tag: str) -> any:
        node.Parent = self
        node.Name = node.Name + date + tag
        node.Path = self.Path + "/" + node.Name
        
        return node
        
    #子ノード名で検索＞最期に見つかったパスを返す
    def SearchLastChildName(self, child_name: str) -> str:
        Names = glob.glob(self.Path + "/" + child_name + "*")
        if len(Names) == 0: return None
        Names.sort()
        return os.path.basename(Names[-1])
    
    #子ノードのパスからインデックスを得る
    def GetIndexFromLastChildName(self, child_name: str, match: str) -> str:
        return re.search(r"[0-9]+_[0-9]+_[0-9]+_[0-9]+_[0-9]+_[0-9]+" + match, self.SearchLastChildName(child_name)).groups()[0]
    
    #名前から検索，子ノードを生成
    def SearchLastChild(self, child_name: str) -> DirNode:
        return super().AddChild(DirNode(self.SearchLastChildName(child_name)))

    #最期の子ノードを読み込めるか
    def CheckLoadableLast(self, child_name: str) -> DirNode:
        return self.CheckLoadableAny(child_name)

class RootNode(DirNode):
    """
    ルートノード（ソースコードの場所）
    """
    #コンストラクタ
    def __init__(self, name: str):
        super().__init__(name)

        self.Path = name
          
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#ファイルノード
class FileNode_dict(FileNode):
    """
    辞書データノード
    """
    #コンストラクタ
    def __init__(self, name: str):
        super().__init__(name)
        self.Extension = ".csv"

    #保存
    def Save(self, dict_data: dict):
        super().Save()

        df = pd.DataFrame([dict_data])
        df.to_csv(self.Path + self.Extension, index = False)

    #読み込み
    def Load(self) -> dict:
        super().Load()

        df = pd.read_csv(self.Path + self.Extension)
        if "Unnamed: 0" in df:
            df = pd.read_csv(self.Path + self.Extension, index_col = 0)
        
        Dict = df.to_dict()
        for k, v in Dict.items():
            Dict[k] = v[0]

        return Dict
    
class FileNode_df(FileNode):
    """
    Pandasデータフレームノード
    """
    #コンストラクタ
    def __init__(self, name: str):
        super().__init__(name)
        self.Extension = ".csv"

    #保存
    def Save(self, df_data: pd.DataFrame):
        super().Save()

        df_data.to_csv(self.Path + self.Extension, index = False)

    #読み込み
    def Load(self) -> pd.DataFrame:
        super().Load()

        df = pd.read_csv(self.Path + self.Extension)
        if "Unnamed: 0" in df:
            df = pd.read_csv(self.Path + self.Extension, index_col = 0)
        
        return df
    
class FileNode_np(FileNode):
    """
    Numpy行列データノード
    """
    #コンストラクタ
    def __init__(self, name: str):
        super().__init__(name)
        self.Extension = ".npy"

    #保存
    def Save(self, array: np.ndarray):
        super().Save()

        np.save(self.Path + self.Extension, array)

    #読み込み
    def Load(self) -> np.ndarray:
        super().Load()

        return np.load(self.Path + self.Extension)
    
class FileNode_pth(FileNode):
    """
    Pytorchモデルデータノード
    """
    #コンストラクタ
    def __init__(self, name: str):
        super().__init__(name)
        self.Extension = ".pth"

    #保存
    def Save(self, model: any):
        super().Save()

        torch.save(model, self.Path + self.Extension)

    #読み込み
    def Load(self) -> any:
        super().Load()

        return torch.load(self.Path + self.Extension)

class FileNode_txt(FileNode):##########################
    """
    テキストデータノード（！！！未作成！！！）
    """
    #コンストラクタ
    def __init__(self, name: str):
        super().__init__(name)
        self.Extension = ".txt"

    #保存
    def Save(self, string: str):
        super().Save()

        pass

    #読み込み
    def Load(self) -> str:
        super().Load()

        return None
    
class FileNode_plt(FileNode):
    """
    Matplotlib/Seabornの図ノード
    （パスだけ利用し，保存は別で行う）
    """
    #コンストラクタ
    def __init__(self, name: str):
        super().__init__(name)
        
    #保存
    def Save(self):
        self.Parent.Save()
    