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

import Module_Reservoir as Reservoir
import Module_Readout as Readout

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
        return z, s[10:21]
    
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
        return y, e, s[10:21]
    
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
        
class Model_ESN_DNNReadout(Model):
    """
    多層リードアウトESNモデル
    """
    #コンストラクタ
    def __init__(self, param: dict, evaluation: any):
        super().__init__(param, evaluation)

        #パラメータ取得
        self.D_u = self.Param["Model_NormalESN_D_u"]            #入力信号次元
        self.D_x = self.Param["Model_NormalESN_D_x"]            #ニューロン数（リスト型）
        self.D_y = self.Param["Model_NormalESN_D_y"]            #出力信号次元
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
            "DNN_D_z" : self.D_z,
            "DNN_D_y" : self.D_y
            })
        self.Readout_DNN = Readout.Module_Readout_DNN(param, self)

    #順伝播（リザバーのみ）
    def forwardReservoir(self, u: np.ndarray) -> np.ndarray:
        s = self.SubReservoir.forward(u)
        z = np.concatenate([s, u])
        self.SubReservoir.update()
        return z
    
    #順伝播（リードアウトのみ）
    def forwardReadout(self, z: np.ndarray) -> np.ndarray:
        return self.Readout_DNN.forward(z)
    
    #順伝播
    def forward(self, u: np.ndarray) -> np.ndarray: 
        s = self.SubReservoir.forward(u)
        z = np.concatenate([s, u])
        y = self.Readout_DNN.forward(z)
        self.SubReservoir.update()
        return y
    
    #順伝播（エラー出力付き）
    def forwardWithRMSE(self, u: np.ndarray, y_d: np.ndarray) -> tuple: 
        s = self.SubReservoir.forward(u)
        z = np.concatenate([s, u])
        y = self.Readout_DNN.forward(z)
        e = self.Readout_DNN.RMSE(z, y_d)
        self.SubReservoir.update()
        return y, e
    
    #学習
    def fit(self, Z: np.ndarray, Y_d: np.ndarray):
        self.Readout_DNN.fit(Z, Y_d)
        
    #初期化
    def reset(self): 
        self.SubReservoir.reset()
        self.Readout_DNN.reset()

    #ディープコピー
    def clone(self):
        new = Model_NormalESN(self.Param.copy(), self.Evaluation)
        new.SubReservoir = self.SubReservoir.clone()
        new.Readout_DNN = self.Readout_DNN.clone()

#--------------------------------------------------------------------
#通常AESNモデル
class Model_NormalAESN(Model):
    """
    通常AESNモデル
    """
    #コンストラクタ
    def __init__(self, param: dict, evaluation: any):
        super().__init__(param, evaluation)

        #パラメータ取得
        self.D_u = self.Param["Model_NormalAESN_D_u"]           #入力信号次元
        self.D_x = self.Param["Model_NormalAESN_D_x"]           #サブリザバーニューロン数
        self.D_s = self.D_u * self.D_x                          #総ニューロン数
        self.D_y = self.Param["Model_NormalAESN_D_y"]           #出力信号次元
        self.D_z = self.D_s + self.D_u                          #特徴ベクトル次元
        
        #リザバーインスタンス
        param = self.Param.copy()
        param.update({
            "AESNReservoir_D_u" : self.D_u,
            "AESNReservoir_D_x" : self.D_x
            })
        self.AESNReservoir = Reservoir.Module_AESNReservoir(param, self)
        #リードアウトインスタンス
        param = self.Param.copy()
        param.update({
            "LinerTransformer_D_z" : self.D_z,
            "LinerTransformer_D_y" : self.D_y
            })
        self.Readout_LinerTransformer = Readout.Module_Readout_LinerTransformer(param, self)

    #順伝播（リザバーのみ）
    def forwardReservoir(self, u: np.ndarray) -> np.ndarray:
        s = self.AESNReservoir.forward(u)
        z = np.concatenate([s, u])
        self.AESNReservoir.update()
        return z
    
    #順伝播（リードアウトのみ）
    def forwardReadout(self, z: np.ndarray) -> np.ndarray:
        return self.Readout_LinerTransformer.forward(z)
    
    #順伝播
    def forward(self, u: np.ndarray) -> np.ndarray: 
        s = self.AESNReservoir.forward(u)
        z = np.concatenate([s, u])
        y = self.Readout_LinerTransformer.forward(z)
        self.AESNReservoir.update()
        return y
    
    #順伝播（エラー出力付き）
    def forwardWithRMSE(self, u: np.ndarray, y_d: np.ndarray) -> tuple: 
        s = self.AESNReservoir.forward(u)
        z = np.concatenate([s, u])
        y = self.Readout_LinerTransformer.forward(z)
        e = self.Readout_LinerTransformer.RMSE(z, y_d)
        self.AESNReservoir.update()
        return y, e
    
    #学習
    def fit(self, Z: np.ndarray, Y_d: np.ndarray):
        self.Readout_LinerTransformer.fit(Z, Y_d)
        
    #初期化
    def reset(self): 
        self.AESNReservoir.reset()
        self.Readout_LinerTransformer.reset()

    #ディープコピー
    def clone(self):
        new = Model_NormalESN(self.Param.copy(), self.Evaluation)
        new.AESNReservoir = self.AESNReservoir.clone()
        new.Readout_LinerTransformer = self.Readout_LinerTransformer.clone()
        
class Model_AESN_DNNReadout(Model):
    """
    多層リードアウトAESNモデル
    """
    #コンストラクタ
    def __init__(self, param: dict, evaluation: any):
        super().__init__(param, evaluation)

        #パラメータ取得
        self.D_u = self.Param["Model_NormalAESN_D_u"]           #入力信号次元
        self.D_x = self.Param["Model_NormalAESN_D_x"]           #サブリザバーニューロン数
        self.D_s = self.D_u * self.D_x                          #総ニューロン数
        self.D_y = self.Param["Model_NormalAESN_D_y"]           #出力信号次元
        self.D_z = self.D_s + self.D_u                          #特徴ベクトル次元
        
        #リザバーインスタンス
        param = self.Param.copy()
        param.update({
            "AESNReservoir_D_u" : self.D_u,
            "AESNReservoir_D_x" : self.D_x
            })
        self.AESNReservoir = Reservoir.Module_AESNReservoir(param, self)
        #リードアウトインスタンス
        param = self.Param.copy()
        param.update({
            "DNN_D_z" : self.D_z,
            "DNN_D_y" : self.D_y
            })
        self.Readout_DNN = Readout.Module_Readout_DNN(param, self)

    #順伝播（リザバーのみ）
    def forwardReservoir(self, u: np.ndarray) -> np.ndarray:
        s = self.AESNReservoir.forward(u)
        z = np.concatenate([s, u])
        self.AESNReservoir.update()
        return z
    
    #順伝播（リードアウトのみ）
    def forwardReadout(self, z: np.ndarray) -> np.ndarray:
        return self.Readout_DNN.forward(z)
    
    #順伝播
    def forward(self, u: np.ndarray) -> np.ndarray: 
        s = self.AESNReservoir.forward(u)
        z = np.concatenate([s, u])
        y = self.Readout_DNN.forward(z)
        self.AESNReservoir.update()
        return y
    
    #順伝播（エラー出力付き）
    def forwardWithRMSE(self, u: np.ndarray, y_d: np.ndarray) -> tuple: 
        s = self.AESNReservoir.forward(u)
        z = np.concatenate([s, u])
        y = self.Readout_DNN.forward(z)
        e = self.Readout_DNN.RMSE(z, y_d)
        self.AESNReservoir.update()
        return y, e
    
    #学習
    def fit(self, Z: np.ndarray, Y_d: np.ndarray):
        self.Readout_DNN.fit(Z, Y_d)
        
    #初期化
    def reset(self): 
        self.AESNReservoir.reset()
        self.Readout_DNN.reset()

    #ディープコピー
    def clone(self):
        new = Model_NormalESN(self.Param.copy(), self.Evaluation)
        new.AESNReservoir = self.AESNReservoir.clone()
        new.Readout_DNN = self.Readout_DNN.clone()
        
#--------------------------------------------------------------------
#更新速度の異なるESNモデル
class Model_DifferentUpdateESN(Model):
    """
    更新速度の異なるESNモデル
    """
    #コンストラクタ
    def __init__(self, param: dict, evaluation: any):
        super().__init__(param, evaluation)

        #パラメータ取得
        self.D_u = self.Param["Model_DifferentUpdateESN_D_u"]                   #入力信号次元
        self.D_x = self.Param["Model_DifferentUpdateESN_D_x"]                   #サブリザバーニューロン数
        self.LeakingRate = self.Param["Model_DifferentUpdateESN_LeakingRate"]   #リーク率配列->サブリザバーの数(必ずリスト型)
        self.N_SubReservoirs = len(self.LeakingRate)                            #サブリザバー数
        self.D_y = self.Param["Model_DifferentUpdateESN_D_y"]                   #出力信号次元
        self.D_z = self.N_SubReservoirs * self.D_x + self.D_u                   #特徴ベクトル次元
        self.InputScale = self.Param["Model_DifferentUpdateESN_InputScale"]     #入力スケーリング配列（リスト型可能，None可能）
        self.Rho = self.Param["Model_DifferentUpdateESN_Rho"]                   #スペクトル半径配列（リスト型可能，None可能）
        self.Density = self.Param["Model_DifferentUpdateESN_Density"]           #結合密度配列（リスト型可能，None可能）

        #リザバーインスタンス
        param = self.Param.copy()
        param.update({
            "DifferentUpdateESNReservoir_D_u" : self.D_u,
            "DifferentUpdateESNReservoir_D_x" : self.D_x,
            "DifferentUpdateESNReservoir_LeakingRate" : self.LeakingRate,
            "DifferentUpdateESNReservoir_InputScale" : self.InputScale,
            "DifferentUpdateESNReservoir_Rho" : self.Rho,
            "DifferentUpdateESNReservoir_Density" : self.Density
            })
        self.DifferentUpdateESNReservoir = Reservoir.Module_DifferentUpdateESNReservoir(param, self)
        
        #リードアウトインスタンス
        param = self.Param.copy()
        param.update({
            "LinerTransformer_D_z" : self.D_z,
            "LinerTransformer_D_y" : self.D_y
            })
        self.Readout_LinerTransformer = Readout.Module_Readout_LinerTransformer(param, self)

    #順伝播（リザバーのみ）
    def forwardReservoir(self, u: np.ndarray) -> np.ndarray:
        s = self.DifferentUpdateESNReservoir.forward(u)
        z = np.concatenate([s, u])
        self.DifferentUpdateESNReservoir.update()
        return z
    
    #順伝播（リードアウトのみ）
    def forwardReadout(self, z: np.ndarray) -> np.ndarray:
        return self.Readout_LinerTransformer.forward(z)
    
    #順伝播
    def forward(self, u: np.ndarray) -> np.ndarray: 
        s = self.DifferentUpdateESNReservoir.forward(u)
        z = np.concatenate([s, u])
        y = self.Readout_LinerTransformer.forward(z)
        self.DifferentUpdateESNReservoir.update()
        return y
    
    #順伝播（エラー出力付き）?
    def forwardWithRMSE(self, u: np.ndarray, y_d: np.ndarray) -> tuple: 
        s = self.DifferentUpdateESNReservoir.forward(u)
        z = np.concatenate([s, u])
        y = self.Readout_LinerTransformer.forward(z)
        e = self.Readout_LinerTransformer.RMSE(z, y_d)
        self.DifferentUpdateESNReservoir.update()
        return y, e
    
    #学習
    def fit(self, Z: np.ndarray, Y_d: np.ndarray):
        self.Readout_LinerTransformer.fit(Z, Y_d)
        
    #初期化
    def reset(self): 
        self.DifferentUpdateESNReservoir.reset()
        self.Readout_LinerTransformer.reset()

    #ディープコピー
    def clone(self):
        new = Model_DifferentUpdateESN(self.Param.copy(), self.Evaluation)
        new.DifferentUpdateESNReservoir = self.DifferentUpdateESNReservoir.clone()
        new.Readout_LinerTransformer = self.Readout_LinerTransformer.clone()

class Model_DifferentUpdateESN_DNNReadout(Model):
    """
    多層リードアウト更新速度の異なるESNモデル
    """
    #コンストラクタ
    def __init__(self, param: dict, evaluation: any):
        super().__init__(param, evaluation)

        #パラメータ取得
        self.D_u = self.Param["Model_DifferentUpdateESN_D_u"]                   #入力信号次元
        self.D_x = self.Param["Model_DifferentUpdateESN_D_x"]                   #サブリザバーニューロン数
        self.LeakingRate = self.Param["Model_DifferentUpdateESN_LeakingRate"]   #リーク率配列->サブリザバーの数(必ずリスト型)
        self.N_SubReservoirs = len(self.LeakingRate)                            #サブリザバー数
        self.D_y = self.Param["Model_DifferentUpdateESN_D_y"]                   #出力信号次元
        self.D_z = self.N_SubReservoirs * self.D_x + self.D_u                   #特徴ベクトル次元
        self.InputScale = self.Param["Model_DifferentUpdateESN_InputScale"]     #入力スケーリング配列（リスト型可能，None可能）
        self.Rho = self.Param["Model_DifferentUpdateESN_Rho"]                   #スペクトル半径配列（リスト型可能，None可能）
        self.Density = self.Param["Model_DifferentUpdateESN_Density"]           #結合密度配列（リスト型可能，None可能）

        #リザバーインスタンス
        param = self.Param.copy()
        param.update({
            "DifferentUpdateESNReservoir_D_u" : self.D_u,
            "DifferentUpdateESNReservoir_D_x" : self.D_x,
            "DifferentUpdateESNReservoir_LeakingRate" : self.LeakingRate,
            "DifferentUpdateESNReservoir_InputScale" : self.InputScale,
            "DifferentUpdateESNReservoir_Rho" : self.Rho,
            "DifferentUpdateESNReservoir_Density" : self.Density
            })
        self.DifferentUpdateESNReservoir = Reservoir.Module_DifferentUpdateESNReservoir(param, self)
        #リードアウトインスタンス
        param = self.Param.copy()
        param.update({
            "DNN_D_z" : self.D_z,
            "DNN_D_y" : self.D_y
            })
        self.Readout_DNN = Readout.Module_Readout_DNN(param, self)

    #順伝播（リザバーのみ）
    def forwardReservoir(self, u: np.ndarray) -> np.ndarray:
        s = self.DifferentUpdateESNReservoir.forward(u)
        z = np.concatenate([s, u])
        self.DifferentUpdateESNReservoir.update()
        return z
    
    #順伝播（リードアウトのみ）
    def forwardReadout(self, z: np.ndarray) -> np.ndarray:
        return self.Readout_DNN.forward(z)
    
    #順伝播
    def forward(self, u: np.ndarray) -> np.ndarray: 
        s = self.DifferentUpdateESNReservoir.forward(u)
        z = np.concatenate([s, u])
        y = self.Readout_DNN.forward(z)
        self.DifferentUpdateESNReservoir.update()
        return y
    
    #順伝播（エラー出力付き）?
    def forwardWithRMSE(self, u: np.ndarray, y_d: np.ndarray) -> tuple: 
        s = self.DifferentUpdateESNReservoir.forward(u)
        z = np.concatenate([s, u])
        y = self.Readout_DNN.forward(z)
        e = self.Readout_DNN.RMSE(z, y_d)
        self.DifferentUpdateESNReservoir.update()
        return y, e
    
    #学習
    def fit(self, Z: np.ndarray, Y_d: np.ndarray):
        self.Readout_DNN.fit(Z, Y_d)
        
    #初期化
    def reset(self): 
        self.DifferentUpdateESNReservoir.reset()
        self.Readout_DNN.reset()

    #ディープコピー
    def clone(self):
        new = Model_DifferentUpdateESN(self.Param.copy(), self.Evaluation)
        new.DifferentUpdateESNReservoir = self.DifferentUpdateESNReservoir.clone()
        new.Readout_DNN = self.Readout_DNN.clone()

#--------------------------------------------------------------------
#リーク率の異なるAESNモデル
class Model_DifferentUpdateAESN(Model):
    """
    更新速度の異なるAESNモデル
    """
    #コンストラクタ
    def __init__(self, param: dict, evaluation: any):
        super().__init__(param, evaluation)

        #パラメータ取得
        self.D_u = self.Param["Model_DifferentUpdateAESN_D_u"]                   #入力信号次元
        self.D_x = self.Param["Model_DifferentUpdateAESN_D_x"]                   #サブリザバーニューロン数
        self.LeakingRate = self.Param["Model_DifferentUpdateAESN_LeakingRate"]   #リーク率配列->サブリザバーの数(必ずリスト型)
        self.N_SubReservoirs = len(self.LeakingRate)                            #サブリザバー数
        self.D_y = self.Param["Model_DifferentUpdateAESN_D_y"]                   #出力信号次元
        self.D_z = self.N_SubReservoirs * self.D_x + self.D_u                   #特徴ベクトル次元
        self.InputScale = self.Param["Model_DifferentUpdateAESN_InputScale"]     #入力スケーリング配列（リスト型可能，None可能）
        self.Rho = self.Param["Model_DifferentUpdateAESN_Rho"]                   #スペクトル半径配列（リスト型可能，None可能）
        self.Density = self.Param["Model_DifferentUpdateAESN_Density"]           #結合密度配列（リスト型可能，None可能）

        #リザバーインスタンス
        param = self.Param.copy()
        param.update({
            "DifferentUpdateAESNReservoir_D_u" : self.D_u,
            "DifferentUpdateAESNReservoir_D_x" : self.D_x,
            "DifferentUpdateAESNReservoir_LeakingRate" : self.LeakingRate,
            "DifferentUpdateAESNReservoir_InputScale" : self.InputScale,
            "DifferentUpdateAESNReservoir_Rho" : self.Rho,
            "DifferentUpdateAESNReservoir_Density" : self.Density
            })
        self.DifferentUpdateAESNReservoir = Reservoir.Module_DifferentUpdateAESNReservoir(param, self)
        #リードアウトインスタンス
        param = self.Param.copy()
        param.update({
            "LinerTransformer_D_z" : self.D_z,
            "LinerTransformer_D_y" : self.D_y
            })
        self.Readout_LinerTransformer = Readout.Module_Readout_LinerTransformer(param, self)

    #順伝播（リザバーのみ）
    def forwardReservoir(self, u: np.ndarray) -> np.ndarray:
        s = self.DifferentUpdateAESNReservoir.forward(u)
        z = np.concatenate([s, u])
        self.DifferentUpdateAESNReservoir.update()
        return z
    
    #順伝播（リードアウトのみ）
    def forwardReadout(self, z: np.ndarray) -> np.ndarray:
        return self.Readout_LinerTransformer.forward(z)
    
    #順伝播
    def forward(self, u: np.ndarray) -> np.ndarray: 
        s = self.DifferentUpdateAESNReservoir.forward(u)
        z = np.concatenate([s, u])
        y = self.Readout_LinerTransformer.forward(z)
        self.DifferentUpdateAESNReservoir.update()
        return y
    
    #順伝播（エラー出力付き）?
    def forwardWithRMSE(self, u: np.ndarray, y_d: np.ndarray) -> tuple: 
        s = self.DifferentUpdateAESNReservoir.forward(u)
        z = np.concatenate([s, u])
        y = self.Readout_LinerTransformer.forward(z)
        e = self.Readout_LinerTransformer.RMSE(z, y_d)
        self.DifferentUpdateAESNReservoir.update()
        return y, e
    
    #学習
    def fit(self, Z: np.ndarray, Y_d: np.ndarray):
        self.Readout_LinerTransformer.fit(Z, Y_d)
        
    #初期化
    def reset(self): 
        self.DifferentUpdateAESNReservoir.reset()
        self.Readout_LinerTransformer.reset()

    #ディープコピー
    def clone(self):
        new = Model_DifferentUpdateAESN(self.Param.copy(), self.Evaluation)
        new.DifferentUpdateAESNReservoir = self.DifferentUpdateAESNReservoir.clone()
        new.Readout_LinerTransformer = self.Readout_LinerTransformer.clone()

#--------------------------------------------------------------------
#Hubを持つAESNモデル
class Model_AESNwithHubReservoir(Model):
    """
    Hubを持つAESNモデル
    """
    #コンストラクタ
    def __init__(self, param: dict, evaluation: any):
        super().__init__(param, evaluation)

        #パラメータ取得
        self.D_u = self.Param["Model_AESNwithHub_D_u"]          #入力信号次元
        self.D_h = self.Param["Model_AESNwithHub_D_h"]          #サブリザバーニューロン数
        self.D_H = self.D_u * self.D_h                          #総ニューロン数
        self.D_x = self.Param["Model_AESNwithHub_D_x"]          #ハブリザバーニューロン数
        self.D_y = self.Param["Model_AESNwithHub_D_y"]          #出力信号次元
        self.F_Use_U2HubConnection = self.Param["Model_AESNwithHub_F_Use_U2HubConnection"]   #Hubへの入力信号使用の有無
        self.F_Use_x2zConnection = self.Param["Model_AESNwithHub_F_Use_x2zConnection"]       #出力へのHub状態使用の有無
        self.F_Use_x2AESNConnection = self.Param["Model_AESNwithHub_F_Use_x2AESNConnection"] #AESNへのHub信号使用の有無
        self.F_Use_AverageHInHub = self.Param["Model_AESNwithHub_F_Use_AverageHInHub"]       #HubでHの平均をするか
        if self.F_Use_x2zConnection:
            self.D_z = self.D_H + self.D_x + self.D_u           #特徴ベクトル次元
        else:
            self.D_z = self.D_H + self.D_u                      #特徴ベクトル次元
        self.Hub_LeakingRate = self.Param["Model_AESNwithHub_Hub_LeakingRate"]  #Hubリーク率配列（None可能）
        self.Hub_InputScale = self.Param["Model_AESNwithHub_Hub_InputScale"]    #Hub入力スケーリング配列（None可能）
        self.Hub_Rho = self.Param["Model_AESNwithHub_Hub_Rho"]                  #Hubスペクトル半径配列（None可能）
        self.Hub_Density = self.Param["Model_AESNwithHub_Hub_Density"]          #Hub結合密度配列（None可能）

        #リザバーインスタンス
        param = self.Param.copy()
        param.update({
            "AESNwithHubReservoir_D_u" : self.D_u,
            "AESNwithHubReservoir_D_h" : self.D_h,
            "AESNwithHubReservoir_D_x" : self.D_x,
            "AESNwithHubReservoir_F_Use_U2HubConnection" : self.F_Use_U2HubConnection,
            "AESNwithHubReservoir_F_Use_x2zConnection" : self.F_Use_x2zConnection,
            "AESNwithHubReservoir_F_Use_x2AESNConnection" : self.F_Use_x2AESNConnection,
            "AESNwithHubReservoir_F_Use_AverageHInHub" : self.F_Use_AverageHInHub,
            "AESNwithHubReservoir_Hub_LeakingRate" : self.Hub_LeakingRate,
            "AESNwithHubReservoir_Hub_InputScale" : self.Hub_InputScale,
            "AESNwithHubReservoir_Hub_Rho" : self.Hub_Rho,
            "AESNwithHubReservoir_Hub_Density" : self.Hub_Density,
            })
        self.AESNwithHubReservoir = Reservoir.Module_AESNwithHubReservoir(param, self)
        #リードアウトインスタンス
        param = self.Param.copy()
        param.update({
            "LinerTransformer_D_z" : self.D_z,
            "LinerTransformer_D_y" : self.D_y
            })
        self.Readout_LinerTransformer = Readout.Module_Readout_LinerTransformer(param, self)

    #順伝播（リザバーのみ）
    def forwardReservoir(self, u: np.ndarray) -> np.ndarray:
        s = self.AESNwithHubReservoir.forward(u)
        z = np.concatenate([s, u])
        self.AESNwithHubReservoir.update()
        return z
    
    #順伝播（リードアウトのみ）
    def forwardReadout(self, z: np.ndarray) -> np.ndarray:
        return self.Readout_LinerTransformer.forward(z)
    
    #順伝播
    def forward(self, u: np.ndarray) -> np.ndarray: 
        s = self.AESNwithHubReservoir.forward(u)
        z = np.concatenate([s, u])
        y = self.Readout_LinerTransformer.forward(z)
        self.AESNwithHubReservoir.update()
        return y
    
    #順伝播（エラー出力付き）
    def forwardWithRMSE(self, u: np.ndarray, y_d: np.ndarray) -> tuple: 
        s = self.AESNwithHubReservoir.forward(u)
        z = np.concatenate([s, u])
        y = self.Readout_LinerTransformer.forward(z)
        e = self.Readout_LinerTransformer.RMSE(z, y_d)
        self.AESNwithHubReservoir.update()
        return y, e
    
    #学習
    def fit(self, Z: np.ndarray, Y_d: np.ndarray):
        self.Readout_LinerTransformer.fit(Z, Y_d)
        
    #初期化
    def reset(self): 
        self.AESNwithHubReservoir.reset()
        self.Readout_LinerTransformer.reset()

    #ディープコピー
    def clone(self):
        new = Model_NormalESN(self.Param.copy(), self.Evaluation)
        new.AESNwithHubReservoir = self.AESNwithHubReservoir.clone()
        new.Readout_LinerTransformer = self.Readout_LinerTransformer.clone()
        
class Model_AESNwithHubReservoir(Model):
    """
    多層リードアウトHubを持つAESNモデル
    """
    #コンストラクタ
    def __init__(self, param: dict, evaluation: any):
        super().__init__(param, evaluation)
        
        #パラメータ取得
        self.D_u = self.Param["Model_AESNwithHub_D_u"]          #入力信号次元
        self.D_h = self.Param["Model_AESNwithHub_D_h"]          #サブリザバーニューロン数
        self.D_H = self.D_u * self.D_h                          #総ニューロン数
        self.D_x = self.Param["Model_AESNwithHub_D_x"]          #ハブリザバーニューロン数
        self.D_y = self.Param["Model_AESNwithHub_D_y"]          #出力信号次元
        self.F_Use_U2HubConnection = self.Param["Model_AESNwithHub_F_Use_U2HubConnection"]   #Hubへの入力信号使用の有無
        self.F_Use_x2zConnection = self.Param["Model_AESNwithHub_F_Use_x2zConnection"]       #出力へのHub状態使用の有無
        self.F_Use_x2AESNConnection = self.Param["Model_AESNwithHub_F_Use_x2AESNConnection"] #AESNへのHub信号使用の有無
        self.F_Use_AverageHInHub = self.Param["Model_AESNwithHub_F_Use_AverageHInHub"]       #HubでHの平均をするか
        if self.F_Use_x2zConnection:
            self.D_z = self.D_H + self.D_x + self.D_u           #特徴ベクトル次元
        else:
            self.D_z = self.D_H + self.D_u                      #特徴ベクトル次元
        self.Hub_LeakingRate = self.Param["Model_AESNwithHub_Hub_LeakingRate"]  #Hubリーク率配列（None可能）
        self.Hub_InputScale = self.Param["Model_AESNwithHub_Hub_InputScale"]    #Hub入力スケーリング配列（None可能）
        self.Hub_Rho = self.Param["Model_AESNwithHub_Hub_Rho"]                  #Hubスペクトル半径配列（None可能）
        self.Hub_Density = self.Param["Model_AESNwithHub_Hub_Density"]          #Hub結合密度配列（None可能）

        #リザバーインスタンス
        param = self.Param.copy()
        param.update({
            "AESNwithHubReservoir_D_u" : self.D_u,
            "AESNwithHubReservoir_D_h" : self.D_h,
            "AESNwithHubReservoir_D_x" : self.D_x,
            "AESNwithHubReservoir_F_Use_U2HubConnection" : self.F_Use_U2HubConnection,
            "AESNwithHubReservoir_F_Use_x2zConnection" : self.F_Use_x2zConnection,
            "AESNwithHubReservoir_F_Use_x2AESNConnection" : self.F_Use_x2AESNConnection,
            "AESNwithHubReservoir_F_Use_AverageHInHub" : self.F_Use_AverageHInHub,
            "AESNwithHubReservoir_Hub_LeakingRate" : self.Hub_LeakingRate,
            "AESNwithHubReservoir_Hub_InputScale" : self.Hub_InputScale,
            "AESNwithHubReservoir_Hub_Rho" : self.Hub_Rho,
            "AESNwithHubReservoir_Hub_Density" : self.Hub_Density,
            })
        self.AESNwithHubReservoir = Reservoir.Module_AESNwithHubReservoir(param, self)
        #リードアウトインスタンス
        param = self.Param.copy()
        param.update({
            "DNN_D_z" : self.D_z,
            "DNN_D_y" : self.D_y
            })
        self.Readout_DNN = Readout.Module_Readout_DNN(param, self)

    #順伝播（リザバーのみ）
    def forwardReservoir(self, u: np.ndarray) -> np.ndarray:
        s = self.AESNwithHubReservoir.forward(u)
        z = np.concatenate([s, u])
        self.AESNwithHubReservoir.update()
        return z
    
    #順伝播（リードアウトのみ）
    def forwardReadout(self, z: np.ndarray) -> np.ndarray:
        return self.Readout_DNN.forward(z)
    
    #順伝播
    def forward(self, u: np.ndarray) -> np.ndarray: 
        s = self.AESNwithHubReservoir.forward(u)
        z = np.concatenate([s, u])
        y = self.Readout_DNN.forward(z)
        self.AESNwithHubReservoir.update()
        return y
    
    #順伝播（エラー出力付き）
    def forwardWithRMSE(self, u: np.ndarray, y_d: np.ndarray) -> tuple: 
        s = self.AESNwithHubReservoir.forward(u)
        z = np.concatenate([s, u])
        y = self.Readout_DNN.forward(z)
        e = self.Readout_DNN.RMSE(z, y_d)
        self.AESNwithHubReservoir.update()
        return y, e
    
    #学習
    def fit(self, Z: np.ndarray, Y_d: np.ndarray):
        self.Readout_DNN.fit(Z, Y_d)
        
    #初期化
    def reset(self): 
        self.AESNwithHubReservoir.reset()
        self.Readout_DNN.reset()

    #ディープコピー
    def clone(self):
        new = Model_NormalESN(self.Param.copy(), self.Evaluation)
        new.AESNwithHubReservoir = self.AESNwithHubReservoir.clone()
        new.Readout_DNN = self.Readout_DNN.clone()
        
#--------------------------------------------------------------------
#変形DeepESNモデル
class Model_ModifiedDeepESN(Model):
    """
    変形DeepESNモデル
    """
    #コンストラクタ
    def __init__(self, param: dict, evaluation: any):
        super().__init__(param, evaluation)

        #パラメータ取得
        self.D_u = self.Param["Model_ModifiedDeepESN_D_u"]      #入力信号次元
        self.D_x = self.Param["Model_ModifiedDeepESN_D_x"]      #各層ニューロン数（リスト型）
        self.D_l = len(self.D_x)                                #層の数
        self.D_y = self.Param["Model_ModifiedDeepESN_D_y"]      #出力信号次元
        self.D_s = sum(self.D_x)                                #状態ベクトル次元
        self.D_z = self.D_s + self.D_u                          #特徴ベクトル次元
        self.LeakingRate = self.Param["Model_ModifiedDeepESN_LeakingRate"]  #リーク率配列（リスト型可能，None可能）
        self.InputScale = self.Param["Model_ModifiedDeepESN_InputScale"]    #入力スケーリング配列（リスト型可能，None可能）
        self.Rho = self.Param["Model_ModifiedDeepESN_Rho"]                  #スペクトル半径配列（リスト型可能，None可能）
        self.Density = self.Param["Model_ModifiedDeepESN_Density"]          #結合密度配列（リスト型可能，None可能）

        #リザバーインスタンス
        param = self.Param.copy()
        param.update({
            "ModifiedDeepReservoir_D_u" : self.D_u,
            "ModifiedDeepReservoir_D_x" : self.D_x,
            "ModifiedDeepReservoir_LeakingRate" : self.LeakingRate,
            "ModifiedDeepReservoir_InputScale" : self.InputScale,
            "ModifiedDeepReservoir_Rho" : self.Rho,
            "ModifiedDeepReservoir_Density" : self.Density
            })
        self.ModifiedDeepReservoir = Reservoir.Module_ModifiedDeepReservoir(param, self)
        #リードアウトインスタンス
        param = self.Param.copy()
        param.update({
            "LinerTransformer_D_z" : self.D_z,
            "LinerTransformer_D_y" : self.D_y
            })
        self.Readout_LinerTransformer = Readout.Module_Readout_LinerTransformer(param, self)
        
    #順伝播（リザバーのみ）
    def forwardReservoir(self, u: np.ndarray) -> np.ndarray:
        s = self.ModifiedDeepReservoir.forward(u)
        z = np.concatenate([s, u])
        self.ModifiedDeepReservoir.update()
        return z
    
    #順伝播（リードアウトのみ）
    def forwardReadout(self, z: np.ndarray) -> np.ndarray:
        return self.Readout_LinerTransformer.forward(z)
    
    #順伝播
    def forward(self, u: np.ndarray) -> np.ndarray: 
        s = self.ModifiedDeepReservoir.forward(u)
        z = np.concatenate([s, u])
        y = self.Readout_LinerTransformer.forward(z)
        self.ModifiedDeepReservoir.update()
        return y
    
    #順伝播（エラー出力付き）
    def forwardWithRMSE(self, u: np.ndarray, y_d: np.ndarray) -> tuple: 
        s = self.ModifiedDeepReservoir.forward(u)
        z = np.concatenate([s, u])
        y = self.Readout_LinerTransformer.forward(z)
        e = self.Readout_LinerTransformer.RMSE(z, y_d)
        self.ModifiedDeepReservoir.update()
        return y, e
    
    #学習
    def fit(self, Z: np.ndarray, Y_d: np.ndarray):
        self.Readout_LinerTransformer.fit(Z, Y_d)
        
    #初期化
    def reset(self): 
        self.ModifiedDeepReservoir.reset()
        self.Readout_LinerTransformer.reset()

    #ディープコピー
    def clone(self):
        new = Model_ModifiedDeepESN(self.Param.copy(), self.Evaluation)
        new.ModifiedDeepReservoir = self.ModifiedDeepReservoir.clone()
        new.Readout_LinerTransformer = self.Readout_LinerTransformer.clone()

class Model_ModifiedDeepESN_DNNReadout(Model):
    """
    多層リードアウト変形DeepESNモデル
    """
    #コンストラクタ
    def __init__(self, param: dict, evaluation: any):
        super().__init__(param, evaluation)
        
        #パラメータ取得
        self.D_u = self.Param["Model_ModifiedDeepESN_D_u"]      #入力信号次元
        self.D_x = self.Param["Model_ModifiedDeepESN_D_x"]      #各層ニューロン数（リスト型）
        self.D_l = len(self.D_x)                                #層の数
        self.D_y = self.Param["Model_ModifiedDeepESN_D_y"]      #出力信号次元
        self.D_s = sum(self.D_x)                                #状態ベクトル次元
        self.D_z = self.D_s + self.D_u                          #特徴ベクトル次元
        self.LeakingRate = self.Param["Model_ModifiedDeepESN_LeakingRate"]  #リーク率配列（リスト型可能，None可能）
        self.InputScale = self.Param["Model_ModifiedDeepESN_InputScale"]    #入力スケーリング配列（リスト型可能，None可能）
        self.Rho = self.Param["Model_ModifiedDeepESN_Rho"]                  #スペクトル半径配列（リスト型可能，None可能）
        self.Density = self.Param["Model_ModifiedDeepESN_Density"]          #結合密度配列（リスト型可能，None可能）

        #リザバーインスタンス
        param = self.Param.copy()
        param.update({
            "ModifiedDeepReservoir_D_u" : self.D_u,
            "ModifiedDeepReservoir_D_x" : self.D_x,
            "ModifiedDeepReservoir_LeakingRate" : self.LeakingRate,
            "ModifiedDeepReservoir_InputScale" : self.InputScale,
            "ModifiedDeepReservoir_Rho" : self.Rho,
            "ModifiedDeepReservoir_Density" : self.Density
            })
        self.ModifiedDeepReservoir = Reservoir.Module_ModifiedDeepReservoir(param, self)
        #リードアウトインスタンス
        param = self.Param.copy()
        param.update({
            "DNN_D_z" : self.D_z,
            "DNN_D_y" : self.D_y
            })
        self.Readout_DNN = Readout.Module_Readout_DNN(param, self)
        
    #順伝播（リザバーのみ）
    def forwardReservoir(self, u: np.ndarray) -> np.ndarray:
        s = self.ModifiedDeepReservoir.forward(u)
        z = np.concatenate([s, u])
        self.ModifiedDeepReservoir.update()
        return z
    
    #順伝播（リードアウトのみ）
    def forwardReadout(self, z: np.ndarray) -> np.ndarray:
        return self.Readout_DNN.forward(z)
    
    #順伝播
    def forward(self, u: np.ndarray) -> np.ndarray: 
        s = self.ModifiedDeepReservoir.forward(u)
        z = np.concatenate([s, u])
        y = self.Readout_DNN.forward(z)
        self.ModifiedDeepReservoir.update()
        return y
    
    #順伝播（エラー出力付き）
    def forwardWithRMSE(self, u: np.ndarray, y_d: np.ndarray) -> tuple: 
        s = self.ModifiedDeepReservoir.forward(u)
        z = np.concatenate([s, u])
        y = self.Readout_DNN.forward(z)
        e = self.Readout_DNN.RMSE(z, y_d)
        self.ModifiedDeepReservoir.update()
        return y, e
    
    #学習
    def fit(self, Z: np.ndarray, Y_d: np.ndarray):
        self.Readout_DNN.fit(Z, Y_d)
        
    #初期化
    def reset(self): 
        self.ModifiedDeepReservoir.reset()
        self.Readout_DNN.reset()

    #ディープコピー
    def clone(self):
        new = Model_ModifiedDeepESN(self.Param.copy(), self.Evaluation)
        new.ModifiedDeepReservoir = self.ModifiedDeepReservoir.clone()
        new.Readout_DNN = self.Readout_DNN.clone()

#--------------------------------------------------------------------
#Sishu提案モデル（リングスターネットワークなどの構成）
class Model_SishuESN(Model):
    """
    Sishu提案モデル
    ニューロン間の結合形態は指定、結合強度は乱数で与える
    スペクトル半径を組み込む
    """
    #コンストラクタ
    def __init__(self, param: dict, evaluation: any):
        super().__init__(param, evaluation)

        #パラメータ取得
        self.D_u = self.Param["Model_SishuESN_D_u"]            #入力信号次元
        self.D_x = self.Param["Model_SishuESN_D_x"]            #ニューロン数（リスト型）
        self.D_y = self.Param["Model_SishuESN_D_y"]            #出力信号次元

        self.Ring = self.Param["Model_SishuESN_Ring"]          #リングネットワーク
        self.Star = self.Param["Model_SishuESN_Star"]          #スターネットワーク

        self.ImputScale = self.Param["Model_SishuESN_InputScale"]   #入力スケーリング
        self.Rho = self.Param["Model_SishuESN_Rho"]                 #スペクトル半径

        self.sigma = self.Param["Model_SishuESN_sigma"]         #リングネットワークの有無
        self.mu = self.Param["Model_SishuESN_mu"]               #スターネットワークの有無

        self.a = self.Param["Model_SishuESN_a"]
        self.b = self.Param["Model_SishuESN_b"]
        self.c = self.Param["Model_SishuESN_c"]
        self.k0 = self.Param["Model_SishuESN_k0"]

        self.k1 = self.Param["Model_SishuESN_k1"]
        self.k2 = self.Param["Model_SishuESN_k2"]
        self.alpha = self.Param["Model_SishuESN_alpha"]
        self.beta = self.Param["Model_SishuESN_beta"]

        self.k = self.Param["Model_SishuESN_k"]                #Chialvoの変数：k
        
        self.D_z = self.D_x + self.D_u                           #特徴ベクトル次元
        
        #リザバーインスタンス
        param = self.Param.copy()
        param.update({
            "SishuReservoir_D_u" : self.D_u,
            "SishuReservoir_D_x" : self.D_x,

            "SishuReservoir_Ring" : self.Ring,
            "SishuReservoir_Star" : self.Star,

            "SishuReservoir_InputScale" : self.ImputScale,
            "SishuReservoir_Rho" : self.Rho,

            "SishuReservoir_sigma" : self.sigma,
            "SishuReservoir_mu" : self.mu,

            "SishuReservoir_a" : self.a,
            "SishuReservoir_b" : self.b,
            "SishuReservoir_c" : self.c,
            "SishuReservoir_k0" : self.k0,

            "SishuReservoir_k1" : self.k1,
            "SishuReservoir_k2" : self.k2,
            "SishuReservoir_alpha" : self.alpha,
            "SishuReservoir_beta" : self.beta,

            "SishuReservoir_k" : self.k
            })
        self.SishuReservoir = Reservoir.Module_SishuReservoir(param, self)

        #リードアウトインスタンス
        param = self.Param.copy()
        param.update({
            "LinerTransformer_D_z" : self.D_z,
            "LinerTransformer_D_y" : self.D_y
            })
        self.Readout_LinerTransformer = Readout.Module_Readout_LinerTransformer(param, self)

    #順伝播（リザバーのみ）
    def forwardReservoir(self, u: np.ndarray) -> np.ndarray:
        s = self.SishuReservoir.forward(u)
        z = np.concatenate([s, u])
        self.SishuReservoir.update()
        return z , s[10:21]
    
    #順伝播（リードアウトのみ）
    def forwardReadout(self, z: np.ndarray) -> np.ndarray:
        return self.Readout_LinerTransformer.forward(z)
    
    #順伝播
    def forward(self, u: np.ndarray) -> np.ndarray: 
        s = self.SishuReservoir.forward(u)
        z = np.concatenate([s, u])
        y = self.Readout_LinerTransformer.forward(z)
        self.SishuReservoir.update()
        return y
    
    #順伝播（エラー出力付き）
    def forwardWithRMSE(self, u: np.ndarray, y_d: np.ndarray) -> tuple: 
        s = self.SishuReservoir.forward(u)
        z = np.concatenate([s, u])
        y = self.Readout_LinerTransformer.forward(z)
        e = self.Readout_LinerTransformer.RMSE(z, y_d)
        self.SishuReservoir.update()
        return y, e, s[10:21]
    
    #学習
    def fit(self, Z: np.ndarray, Y_d: np.ndarray):
        self.Readout_LinerTransformer.fit(Z, Y_d)
        
    #初期化
    def reset(self): 
        self.SishuReservoir.reset()
        self.Readout_LinerTransformer.reset()

    #ディープコピー
    def clone(self):
        new = Model_SishuESN(self.Param.copy(), self.Evaluation)
        new.SishuReservoir = self.SishuReservoir.clone()
        new.Readout_LinerTransformer = self.Readout_LinerTransformer.clone()
