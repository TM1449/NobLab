#####################################################################
#信川研ESNプロジェクト用
#制作者：中村亮介 吉原悠斗 吉田聡太 飯沼貴大
#作成日：2023/05/24
#####################################################################

#====================================================================
#＜このプログラムのTodoメモ＞
#・固定重みCNN未移植＠Ver1
#・
#・

#====================================================================
import numpy as np
import torch

import Model

#====================================================================
#モジュール

#********************************************************************
#継承元
class Module:
    """
    モジュール
    リザバーもリードアウトもこれを継承
    """
    #コンストラクタ
    def __init__(self, param: dict, parent: any):
        self.Param = param
        
        self.Parent = parent                                #親オブジェクト（モジュール，もしくはモデル）
        self.ModuleName = self.Param["ModuleName"]          #モジュール名
        self.ModuleShowName = self.Param["ModuleShowName"]  #モジュール表示名

        #モジュールパス名
        #モジュールパス表示名
        if type(self.Parent) is Module:
            self.PathName = self.Parent.PathName + self.ModuleName
            self.PathShowName = self.Parent.PathShowName + self.ModuleShowName
        else:
            self.PathName = self.ModuleName
            self.PathShowName = self.ModuleShowName

    #初期化
    def reset(self): pass

    #ディープコピー
    def clone(self): pass

    def inputParam_NumOrList(self, index: int, input_param: any):
        if type(input_param) is list: 
            return input_param[index]
        else:
            return input_param
    
class Network(torch.nn.Module):
    """
    Pytorchネットワーク
    Pytorchを利用したモジュールを作成する際に継承
    """
    #コンストラクタ
    def __init__(self, param: dict, parent: Module):
        super().__init__()

        self.Param = param
        
        self.Parent = parent                        #親オブジェクト（モジュール）
        
    #初期化
    def reset(self): pass

    #ディープコピー
    def clone(self): pass
