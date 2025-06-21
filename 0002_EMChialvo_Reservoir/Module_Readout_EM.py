#####################################################################
#信川研ESNプロジェクト用
#制作者：中村亮介 吉原悠斗 吉田聡太 飯沼貴大
#作成日：2023/05/24
"""
本体

maru
"""

#####################################################################

#====================================================================
#＜このプログラムのTodoメモ＞
#・オンライン学習未移植＠Ver1
#・多層リードアウトは構造をパラメータとして指定できるようにする
#・多層リードアウトをクローンとリセットに対応させる

#====================================================================
import numpy as np
import torch

import Module_EM

#====================================================================
#モジュール

#********************************************************************
#リードアウト

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#継承元
class Module_Readout(Module_EM.Module):
    """
    リードアウトモジュール
    この後，多層NNのリードアウトを実装予定．
    全てのリードアウトは以下のインタフェイスを備える．
    """
    #コンストラクタ
    def __init__(self, param: dict, parent: any):
        super().__init__(param, parent)
    
    #順伝播
    def forward(self, u: np.ndarray) -> np.ndarray: pass
    
    #学習
    def fit(self, u: np.ndarray, x: np.ndarray, t: np.ndarray): pass
    
    #初期化
    def reset(self): pass

    #ディープコピー
    def clone(self): pass
    
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#利用可能モジュール

#--------------------------------------------------------------------
#線形変換器
class Module_Readout_LinerTransformer(Module_Readout):
    """
    線形変換器の読み出し層
    """
    def __init__(self, param: dict, parent = None) -> None:
        super().__init__(param, parent)

        #パラメータ取得
        self.D_z = self.Param["LinerTransformer_D_z"]       #特徴ベクトル次元
        self.D_y = self.Param["LinerTransformer_D_y"]       #出力信号次元
        self.Beta = self.Param["LinerTransformer_Beta"]     #正規化係数
        self.Bias = np.ones([1])                            #バイアス

        self.W_out = None
        
    #順伝播
    def forward(self, z: np.ndarray) -> np.ndarray:
        return np.dot(np.concatenate([z, self.Bias]), self.W_out)
    
    #現在ステップの誤差計算
    def RMSE(self, z: np.ndarray, y_d: np.ndarray) -> np.ndarray:
        return np.sqrt(np.sum((y_d - np.dot(np.concatenate([z, self.Bias]), self.W_out))**2) / self.D_y)
    
    #学習
    def fit(self, Z: np.ndarray, Y_d: np.ndarray):
        X = np.concatenate([Z, np.ones([Z.shape[0], 1]) * self.Bias], 1)
        self.W_out = np.linalg.inv(np.dot(X.T, X) + self.Beta
                                 * np.identity(self.D_z + 1)).dot(X.T).dot(Y_d)
        
    #初期化
    def reset(self): pass

    #ディープコピー
    def clone(self):
        new = Module_Readout_LinerTransformer(self.Param.copy(), self.Parent)
        new.W_out = self.W_out

        return new
