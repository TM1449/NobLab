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
#・DeepAESN未移植＠Ver2
#・IP未移植＠Ver1
#・ChESN未移植＠Ver1
#・リングAESN未実装
#・２次元リザバー未実装

#====================================================================
import numpy as np

import Module_No
import random

#====================================================================
#モジュール

#********************************************************************
#リザバー

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#継承元
class Module_Reservoir(Module_No.Module):
    """
    リザバーモジュール
    全てのリザバーは以下のインタフェイスを備える．
    """
    #コンストラクタ
    def __init__(self, param: dict, parent: any):
        super().__init__(param, parent)
    
    #順伝播
    def forward(self, u: np.ndarray) -> np.ndarray: pass

    #時間発展
    def update(self): pass
        
    #初期化
    def reset(self): pass

    #ディープコピー
    def clone(self): pass
    
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#利用可能モジュール
class Module_SubReservoir(Module_Reservoir):
    """
    サブリザバーモジュール
    どのモデルのリザバーもこのクラスを使うようにできたら良いなあ．
    """
    #コンストラクタ
    def __init__(self, param: dict, parent: any):
        super().__init__(param, parent)

        #パラメータ取得
        self.D_u = self.Param["SubReservoir_D_u"]                       #入力信号次元＋バイアス
        self.D_x = self.Param["SubReservoir_D_x"]                       #ニューロン数
        self.LeakingRate = self.Param["SubReservoir_LeakingRate"]       #リーク率
        self.InputScale = self.Param["SubReservoir_InputScale"]         #入力スケーリング
        self.Rho = self.Param["SubReservoir_Rho"]                       #スペクトル半径
        self.Density = self.Param["SubReservoir_Density"]               #結合密度
        self.ActivationFunc = self.Param["SubReservoir_ActivationFunc"] #活性化関数
        self.Bias = np.ones([1])                                        #バイアス

        #変数
        self.s = np.zeros([self.D_x])                           #状態ベクトル
        self.s_old = self.s                                     #1step前の状態ベクトル
        #重み初期化
        self.W_in = self._makeInputWeight()                     #入力重み
        self.W_rec = self._makeRecurrentWeight()                #再帰重み

    #入力重み生成
    def _makeInputWeight(self) -> np.ndarray:
        return np.concatenate([
            np.random.rand(1, self.D_x),                        #バイアスに掛かる重み（正）
            np.random.rand(self.D_u, self.D_x) * 2 - 1]         #入力信号に掛かる重み（正負）
                              ) * self.InputScale               #入力スケールをかける
    
    #再帰重み生成
    def _makeRecurrentWeight(self) -> np.ndarray:
        #np.random.seed(999)
        W = np.random.rand(self.D_x, self.D_x) * 2 - 1
        W = self._makeWSparse(W)
        w, v = np.linalg.eig(W)
        return self.Rho * W / np.amax(w.real)
        
    #疎行列化
    def _makeWSparse(self, w: np.ndarray) -> np.ndarray:
        s_w = w.reshape([-1])
        s_w[np.random.choice(len(s_w), (int)(len(s_w) * (1 - self.Density)), replace = False)] = 0.
        return s_w.reshape(w.shape[0], w.shape[1])
    
    #順伝播
    def forward(self, u: np.ndarray) -> np.ndarray: 
        self.s = (1 - self.LeakingRate) * self.s_old + \
            self.LeakingRate * self.ActivationFunc(np.dot(np.concatenate([self.Bias, u]), self.W_in) + np.dot(self.s_old, self.W_rec))
        return self.s
    
    #最大リアプノフ指数用
    def forward_MLE(self, u: np.ndarray, s_D: np.ndarray) -> np.ndarray:
        #基準軌道の計算
        self.s = (1 - self.LeakingRate) * self.s_old + \
            self.LeakingRate * self.ActivationFunc(np.dot(np.concatenate([self.Bias, u]), self.W_in) + np.dot(self.s_old, self.W_rec))
        
        #摂動軌道の計算
        sP = (1 - self.LeakingRate) * (self.s_old + s_D) + \
            self.LeakingRate * self.ActivationFunc(np.dot(np.concatenate([self.Bias, u]), self.W_in) + np.dot((self.s_old + s_D), self.W_rec))

        return self.s_old, self.s, (self.s_old + s_D), sP

    #時間発展
    def update(self):
        self.s_old = self.s

    #時間発展（最大リアプノフ指数用）
    def update_MLE(self):
        self.s_old = self.s
    
    #リザバーの初期化
    def reset(self):
        self.s = np.zeros([self.D_x])
        self.s_old = self.s
        
    #ディープコピー
    def clone(self):
        new = Module_SubReservoir(self.Param.copy(), self.Parent)
        
        new.s = self.s
        new.s_old = self.s_old
        new.W_in = self.W_in
        new.W_rec = self.W_rec

        return new
    
