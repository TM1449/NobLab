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

import Module_Reservoir_EM as Reservoir
import Module_Readout_EM as Readout

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

    #順伝播（リザバーのみ）
    def forwardReservoir(self, u: np.ndarray) -> np.ndarray:
        s = self.SubReservoir.forward(u)
        z = np.concatenate([s, u])
        self.SubReservoir.update()

        #リザバー層のランダムなニューロンを抜粋
        Random_Neuron = random.sample(range(len(s)), self.RS_neuron)
        RS_N = s[Random_Neuron]

        return z , s[0:self.RS_neuron]
    
    #順伝播（リードアウトのみ）
    def forwardReadout(self, z: np.ndarray) -> np.ndarray:
        return self.Readout_LinerTransformer.forward(z)
    
    #順伝播
    def forward(self, u: np.ndarray) -> np.ndarray: 
        s = self.SubReservoir.forward(u)
        z = np.concatenate([s, u])
        y = self.Readout_LinerTransformer.forward(z)
        self.SubReservoir.update()
        return y
    
    #順伝播（エラー出力付き）
    def forwardWithRMSE(self, u: np.ndarray, y_d: np.ndarray) -> tuple: 
        s = self.SubReservoir.forward(u)
        z = np.concatenate([s, u])
        y = self.Readout_LinerTransformer.forward(z)
        e = self.Readout_LinerTransformer.RMSE(z, y_d)
        self.SubReservoir.update()

        return y, e, s[0:self.RS_neuron], s[:]
    
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

#--------------------------------------------------------------------
#Sishu提案モデル
class Model_EMChialvo(Model):
    """
    Sishu提案モデル
    """
    #コンストラクタ
    def __init__(self, param: dict, evaluation: any):
        super().__init__(param, evaluation)

        #パラメータ取得
        #ニューロン数に関係するパラメータ
        self.D_u = self.Param["Model_EMChialvo_D_u"]            #入力信号次元
        self.D_x = self.Param["Model_EMChialvo_D_x"]            #ニューロン数（リスト型）
        self.D_y = self.Param["Model_EMChialvo_D_y"]            #出力信号次元

        self.RS_neuron = self.Param["Model_Reservoir_Neurons"]  #リザバー層の抜き出すニューロン数

        self.ImputScale = self.Param["Model_EMChialvo_InputScale"]   #入力スケーリング
        
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.a = self.Param["Model_EMChialvo_a"]
        self.b = self.Param["Model_EMChialvo_b"]
        self.c = self.Param["Model_EMChialvo_c"]
        self.k0 = self.Param["Model_EMChialvo_k0"]

        self.k1 = self.Param["Model_EMChialvo_k1"]
        self.k2 = self.Param["Model_EMChialvo_k2"]
        self.alpha = self.Param["Model_EMChialvo_alpha"]
        self.beta = self.Param["Model_EMChialvo_beta"]

        self.k = self.Param["Model_EMChialvo_k"]                #Chialvoの変数：k
        
        self.D_z = self.D_x + self.D_u                           #特徴ベクトル次元
        #self.Rand = random.sample(range(len(self.D_x)), self.RS_neuron)

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #リザバーインスタンス
        param = self.Param.copy()
        param.update({
            "EMChialvo_Reservoir_D_u" : self.D_u,
            "EMChialvo_Reservoir_D_x" : self.D_x,

            "EMChialvo_Reservoir_InputScale" : self.ImputScale,
            
            "EMChialvo_Reservoir_a" : self.a,
            "EMChialvo_Reservoir_b" : self.b,
            "EMChialvo_Reservoir_c" : self.c,
            "EMChialvo_Reservoir_k0" : self.k0,

            "EMChialvo_Reservoir_k1" : self.k1,
            "EMChialvo_Reservoir_k2" : self.k2,
            "EMChialvo_Reservoir_alpha" : self.alpha,
            "EMChialvo_Reservoir_beta" : self.beta,

            "EMChialvo_Reservoir_k" : self.k
            })
        self.EMChialvo_Reservoir = Reservoir.Module_EMChialvo_Reservoir(param, self)

        #リードアウトインスタンス
        param = self.Param.copy()
        param.update({
            "LinerTransformer_D_z" : self.D_z,
            "LinerTransformer_D_y" : self.D_y
            })
        self.Readout_LinerTransformer = Readout.Module_Readout_LinerTransformer(param, self)

    #順伝播（リザバーのみ）
    def forwardReservoir(self, u: np.ndarray) -> np.ndarray:
        xr, yr, phir = self.EMChialvo_Reservoir.forward(u)
        z = np.concatenate([xr, u])
        self.EMChialvo_Reservoir.update()

        return z, xr[0:self.RS_neuron], yr[0:self.RS_neuron], phir[0:self.RS_neuron], xr, yr, phir
    
    #最大リアプノフ指数（MLE: Maximum Lyapunov Exponent）
    def forwardReservoir_MLE(self, u: np.ndarray, Delta_x: np.array, Delta_y: np.array, Delta_phi: np.array):
        xRo, yRo, phiRo, \
            xR, yR, phiR, \
                xPo, yPo, phiPo, \
                    xP, yP, phiP = self.EMChialvo_Reservoir.forward_MLE(u, Delta_x, Delta_y, Delta_phi)
        self.EMChialvo_Reservoir.update_MLE()

        return xRo, yRo, phiRo, xR, yR, phiR, xPo, yPo, phiPo, xP, yP, phiP

    #順伝播（リードアウトのみ）
    def forwardReadout(self, z: np.ndarray) -> np.ndarray:
        return self.Readout_LinerTransformer.forward(z)
    
    #順伝播
    def forward(self, u: np.ndarray) -> np.ndarray: 
        xr, yr, phir = self.EMChialvo_Reservoir.forward(u)
        z = np.concatenate([xr, u])
        y = self.Readout_LinerTransformer.forward(z)
        self.EMChialvo_Reservoir.update()
        return y
    
    #順伝播（エラー出力付き）
    def forwardWithRMSE(self, u: np.ndarray, y_d: np.ndarray) -> tuple: 
        xr, yr, phir = self.EMChialvo_Reservoir.forward(u)
        z = np.concatenate([xr, u])
        y = self.Readout_LinerTransformer.forward(z)
        e = self.Readout_LinerTransformer.RMSE(z, y_d)
        self.EMChialvo_Reservoir.update()

        return y, e, xr, xr[0:self.RS_neuron], yr[0:self.RS_neuron], phir[0:self.RS_neuron], xr, yr, phir
    
    #学習
    def fit(self, Z: np.ndarray, Y_d: np.ndarray):
        self.Readout_LinerTransformer.fit(Z, Y_d)
        
    #初期化
    def reset(self): 
        self.EMChialvo_Reservoir.reset()
        self.Readout_LinerTransformer.reset()

    #ディープコピー
    def clone(self):
        new = Model_EMChialvo(self.Param.copy(), self.Evaluation)
        new.EMChialvo_Reservoir = self.EMChialvo_Reservoir.clone()
        new.Readout_LinerTransformer = self.Readout_LinerTransformer.clone()
