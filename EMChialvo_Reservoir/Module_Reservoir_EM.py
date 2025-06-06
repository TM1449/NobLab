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

import Module_EM
import random

#====================================================================
#モジュール

#********************************************************************
#リザバー

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#継承元
class Module_Reservoir(Module_EM.Module):
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
#--------------------------------------------------------------------
#Sishu提案モデル
class Module_EMChialvo_Reservoir(Module_Reservoir):
    """
    Sishu提案モデル
    """
    #コンストラクタ
    def __init__(self, param: dict, parent: any):
        super().__init__(param, parent)

        #パラメータ取得
        self.D_u = self.Param["EMChialvo_Reservoir_D_u"]                       #入力信号次元＋バイアス
        self.D_x = self.Param["EMChialvo_Reservoir_D_x"]                       #ニューロン数
        self.InputScale = self.Param["EMChialvo_Reservoir_InputScale"]         #入力スケーリング
        
        self.a = self.Param["EMChialvo_Reservoir_a"]
        self.b = self.Param["EMChialvo_Reservoir_b"]
        self.c = self.Param["EMChialvo_Reservoir_c"]
        self.k0 = self.Param["EMChialvo_Reservoir_k0"]

        self.k1 = self.Param["EMChialvo_Reservoir_k1"]
        self.k2 = self.Param["EMChialvo_Reservoir_k2"]
        self.alpha = self.Param["EMChialvo_Reservoir_alpha"]
        self.beta = self.Param["EMChialvo_Reservoir_beta"]

        self.k = self.Param["EMChialvo_Reservoir_k"]                 #k
        
        self.Rho = self.Param["Model_EMChialvo_Rho"]                        #スペクトル半径
        
        self.Density = self.Param["EMChialvo_Reservoir_Density"]
        self.Bias = np.ones([1])                                        #バイアス

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #変数
        """
        self.x = np.ones([self.D_x]) * (np.random.rand(self.D_x) * 2 - 1)                     #xの状態ベクトル
        self.x_old = np.ones([self.D_x]) * (np.random.rand(self.D_x) * 2 - 1)                 #1step前のxの状態ベクトル
        self.y = np.ones([self.D_x]) * (np.random.rand(self.D_x) * 2 - 1)                     #yの状態ベクトル
        self.y_old = np.ones([self.D_x]) * (np.random.rand(self.D_x) * 2 - 1)                 #1step前のyの状態ベクトル
        self.phi = np.ones([self.D_x]) * (np.random.rand(self.D_x) * 2 - 1)                   #phiの状態ベクトル
        self.phi_old = np.ones([self.D_x]) * (np.random.rand(self.D_x) * 2 - 1)               #1step前のphiの状態ベクトル

        self.xP = np.ones([self.D_x]) * (np.random.rand(self.D_x) * 2 - 1)                    #xPの状態ベクトル
        self.yP = np.ones([self.D_x]) * (np.random.rand(self.D_x) * 2 - 1)                    #yPの状態ベクトル
        self.phiP = np.ones([self.D_x]) * (np.random.rand(self.D_x) * 2 - 1)                  #phiPの状態ベクトル
        """
        #np.random.seed(999)
        self.x = np.random.rand(self.D_x) * 2 - 1                     #xの状態ベクトル
        self.x_old = np.random.rand(self.D_x) * 2 - 1                 #1step前のxの状態ベクトル
        self.y = np.random.rand(self.D_x) * 2 - 1                     #yの状態ベクトル
        self.y_old = np.random.rand(self.D_x) * 2 - 1                 #1step前のyの状態ベクトル
        self.phi = np.random.rand(self.D_x) * 2 - 1                   #phiの状態ベクトル
        self.phi_old = np.random.rand(self.D_x) * 2 - 1               #1step前のphiの状態ベクトル

        self.xP = np.random.rand(self.D_x) * 2 - 1                    #xPの状態ベクトル
        self.yP = np.random.rand(self.D_x) * 2 - 1                    #yPの状態ベクトル
        self.phiP = np.random.rand(self.D_x) * 2 - 1                  #phiPの状態ベクトル
        
        #重み初期化
        self.W_in = self._makeInputWeight()                     #入力重み
        self.W_rec = self._makeRecurrentWeight()                #再帰重み

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    #入力重み生成
    def _makeInputWeight(self) -> np.ndarray:
        return np.concatenate([
            np.random.rand(1, self.D_x),                        #バイアスに掛かる重み（正）
            np.random.rand(self.D_u, self.D_x) * 2 - 1]         #入力信号に掛かる重み（正負）
                              ) * self.InputScale               #入力スケールをかける
    
    #疎行列化
    def _makeWSparse(self, w: np.ndarray) -> np.ndarray:
        s_w = w.reshape([-1])
        s_w[np.random.choice(len(s_w), (int)(len(s_w) * (1 - self.Density)), replace = False)] = 0.
        return s_w.reshape(w.shape[0], w.shape[1])
    
    #再帰重み生成
    def _makeRecurrentWeight(self) -> np.ndarray:
        
        #乱数で重み調整
        np.random.seed(seed=None)
        W = np.random.randn(self.D_x, self.D_x)
        W = self._makeWSparse(W)
        w , v = np.linalg.eig(W)
        Matrix = self.Rho * (W / np.max(np.abs(w)))

        return Matrix
    
    #順伝播
    def forward(self, u: np.ndarray) -> np.ndarray: 
        self.x = (pow(self.x_old, 2) * np.exp(self.y_old - self.x_old) + self.k0 \
            + self.k * self.x_old * (self.alpha + 3 * self.beta * pow(self.phi_old, 2)) \
                + np.dot(self.x_old, self.W_rec)) + np.dot(np.concatenate([self.Bias, u]), self.W_in)
        self.y = self.a * self.y_old - self.b * self.x_old + self.c
        self.phi = self.k1 * self.x_old - self.k2 * self.phi_old

        return self.x, self.y, self.phi
    
    #最大リアプノフ指数用
    def forward_MLE(self, u: np.array, x_D: np.array, y_D: np.array, phi_D: np.array) -> np.ndarray:
        #基準軌道の計算
        self.x = (pow(self.x_old, 2) * np.exp(self.y_old - self.x_old) + self.k0 \
            + self.k * self.x_old * (self.alpha + 3 * self.beta * pow(self.phi_old, 2)) \
                + np.dot(self.x_old, self.W_rec)) + np.dot(np.concatenate([self.Bias, u]), self.W_in)
        self.y = self.a * self.y_old - self.b * self.x_old + self.c
        self.phi = self.k1 * self.x_old - self.k2 * self.phi_old

        #摂動軌道の計算
        self.xP = (pow((self.x_old + x_D), 2) * np.exp((self.y_old + y_D) - (self.x_old + x_D)) + self.k0 \
            + self.k * (self.x_old + x_D) * (self.alpha + 3 * self.beta * pow((self.phi_old + phi_D), 2)) \
                + np.dot((self.x_old + x_D), self.W_rec)) + np.dot(np.concatenate([self.Bias, u]), self.W_in)
        self.yP = self.a * (self.y_old + y_D) - self.b * (self.x_old + x_D) + self.c
        self.phiP = self.k1 * (self.x_old + x_D) - self.k2 * (self.phi_old + phi_D)

        #returnの説明
        """
        1行目：基準軌道の1時刻前のニューロンの値
        2行目：基準軌道の現時刻のニューロンの値
        3行目：摂動軌道の1時刻前のニューロンの値
        4行目：摂動軌道の現時刻のニューロンの値
        """
        return self.x_old, self.y_old, self.phi_old, \
            self.x, self.y, self.phi, \
                (self.x_old + x_D), (self.y_old + y_D), (self.phi_old + phi_D), \
                    self.xP, self.yP, self.phiP

    #時間発展
    def update(self):
        self.x_old = self.x
        self.y_old = self.y
        self.phi_old = self.phi

    #時間発展（最大リアプノフ指数用）
    def update_MLE(self):
        self.x_old = self.x
        self.y_old = self.y
        self.phi_old = self.phi

    #リザバーの初期化
    def reset(self):
        self.x = np.zeros([self.D_x])
        self.x_old = self.x
        self.y = np.ones([self.D_x])
        self.y_old = self.y
        self.phi = np.ones([self.D_x])
        self.phi_old = self.phi

    #ディープコピー
    def clone(self):
        new = Module_EMChialvo_Reservoir(self.Param.copy(), self.Parent)
        
        new.x = self.x
        new.x_old = self.x_old
        new.y = self.y
        new.y_old = self.y_old
        new.phi = self.phi
        new.phi_old = self.phi_old

        new.W_in = self.W_in
        new.W_rec = self.W_rec

        return new