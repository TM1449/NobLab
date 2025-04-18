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
#・２次元リザバー未実装
#・リングAESN未実装
#・DeepAESN未移植＠Ver2
#・オンライン学習未移植＠Ver1

#====================================================================
import numpy as np
import random

import Module_Reservoir_No as Reservoir
import Module_Readout_No as Readout

#====================================================================
#モデル

#********************************************************************
#継承元
class Model:
    """
    モデル
    このクラスで，どんなモジュール（どんな複雑な構造）を使っても，
    評価のクラスでは同一のインタフェイスになるようにする．
    つまり，モデルの違いはここで全て吸収する．
    これにより，評価はモデルごとに作成しなくて良くなる（はず．）
    """
    #コンストラクタ
    def __init__(self, param: dict, evaluation: any):
        self.Param = param
        
        self.Evaluation = evaluation                            #親オブジェクト
        
    #順伝播（リザバーのみ）
    def forwardReservoir(self, u: np.ndarray) -> np.ndarray: pass
    
    #順伝播（リードアウトのみ）
    def forwardReadout(self, z: np.ndarray) -> np.ndarray: pass
    
    #順伝播
    def forward(self, u: np.ndarray) -> np.ndarray: pass
    
    #初期化
    def reset(self): pass

    #ディープコピー
    def clone(self): pass
    
#********************************************************************
#利用可能モデル

#--------------------------------------------------------------------
#通常ESN
class Model_NormalESN(Model):
    """
    通常ESNモデル
    """
    #コンストラクタ
    def __init__(self, param: dict, evaluation: any):
        super().__init__(param, evaluation)

        #パラメータ取得
        self.D_u = self.Param["Model_NormalESN_D_u"]            #入力信号次元
        self.D_x = self.Param["Model_NormalESN_D_x"]            #ニューロン数（リスト型）
        self.D_y = self.Param["Model_NormalESN_D_y"]            #出力信号次元
        
        self.RS_neuron = self.Param["Model_Reservoir_Neurons"]

        self.D_z = self.D_x + self.D_u                          #特徴ベクトル次元
        
        #リザバーインスタンス
        param = self.Param.copy()
        param.update({
            "SubReservoir_D_u" : self.D_u,
            "SubReservoir_D_x" : self.D_x
            })
        self.SubReservoir = Reservoir.Module_SubReservoir(param, self)

        #リードアウトインスタンス
        param = self.Param.copy()
        param.update({
            "LinerTransformer_D_z" : self.D_z,
            "LinerTransformer_D_y" : self.D_y
            })
        self.Readout_LinerTransformer = Readout.Module_Readout_LinerTransformer(param, self)

    #順伝播（リザバーのみ）〇
    def forwardReservoir(self, u: np.ndarray) -> np.ndarray:
        s = self.SubReservoir.forward(u)
        z = np.concatenate([s, u])
        self.SubReservoir.update()

        return z , s[0:self.RS_neuron], s
    
    #順伝播（リードアウトのみ）
    def forwardReadout(self, z: np.ndarray) -> np.ndarray:
        return self.Readout_LinerTransformer.forward(z)
    
    #最大リアプノフ指数（MLE: Maximum Lyapunov Exponent）
    """
    1行目：基準軌道の1時刻前のニューロンの値
    2行目：基準軌道の現時刻のニューロンの値
    3行目：摂動軌道の1時刻前のニューロンの値
    4行目：摂動軌道の現時刻のニューロンの値
    """
    def forwardReservoir_MLE(self, u: np.ndarray, Delta_x: np.array):
        xRo, xR, xPo, xP = self.SubReservoir.forward_MLE(u, Delta_x)
        self.SubReservoir.update_MLE()

        return xRo, xR, xPo, xP
    
    #順伝播〇
    def forward(self, u: np.ndarray) -> np.ndarray: 
        s = self.SubReservoir.forward(u)
        z = np.concatenate([s, u])
        y = self.Readout_LinerTransformer.forward(z)
        self.SubReservoir.update()
        return y
    
    #順伝播（エラー出力付き）〇
    def forwardWithRMSE(self, u: np.ndarray, y_d: np.ndarray) -> tuple: 
        s = self.SubReservoir.forward(u)
        z = np.concatenate([s, u])
        y = self.Readout_LinerTransformer.forward(z)
        e = self.Readout_LinerTransformer.RMSE(z, y_d)
        self.SubReservoir.update()

        return y, e, s, s[0:self.RS_neuron], s
    
    #学習
    def fit(self, Z: np.ndarray, Y_d: np.ndarray):
        self.Readout_LinerTransformer.fit(Z, Y_d)
        
    #初期化
    def reset(self): 
        self.SubReservoir.reset()
        self.Readout_LinerTransformer.reset()

    #ディープコピー
    def clone(self):
        new = Model_NormalESN(self.Param.copy(), self.Evaluation)
        new.SubReservoir = self.SubReservoir.clone()
        new.Readout_LinerTransformer = self.Readout_LinerTransformer.clone()
