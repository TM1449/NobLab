#####################################################################
#信川研ESNプロジェクト用
#制作者：中村亮介 吉原悠斗 吉田聡太 飯沼貴大
#作成日：2023/05/24
#####################################################################

#====================================================================
#＜このプログラムのTodoメモ＞
#・オンライン学習未移植＠Ver1
#・多層リードアウトは構造をパラメータとして指定できるようにする
#・多層リードアウトをクローンとリセットに対応させる

#====================================================================
import numpy as np
import torch

import Module

#====================================================================
#モジュール

#********************************************************************
#リードアウト

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#継承元
class Module_Readout(Module.Module):
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
class Module_Readout_LinearTransformer(Module_Readout):
    """
    線形変換器の読み出し層
    """
    def __init__(self, param: dict, parent = None) -> None:
        super().__init__(param, parent)

        #パラメータ取得
        self.D_z = self.Param["LinearTransformer_D_z"]       #特徴ベクトル次元
        self.D_y = self.Param["LinearTransformer_D_y"]       #出力信号次元
        self.Beta = self.Param["LinearTransformer_Beta"]     #正規化係数
        self.Bias = np.ones([1])                            #バイアス

        #重み初期化
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
                                 * np.eye(self.D_z + 1)).dot(X.T).dot(Y_d)
        
    #初期化
    def reset(self): pass

    #ディープコピー
    def clone(self):
        new = Module_Readout_LinearTransformer(self.Param.copy(), self.Parent)
        new.W_out = self.W_out

        return new

class Module_Torch_Readout_LinearTransformer(Module_Readout_LinearTransformer):
    """
    Torch版線形変換器の読み出し層
    """
    def __init__(self, param: dict, parent = None) -> None:
        super(Module_Readout, self).__init__(param, parent)

        #パラメータ取得
        self.Network = Network_Readout_LinearTransformer(param, self)
        
    #順伝播
    def forward(self, z: np.ndarray) -> np.ndarray:
        return self.Network.forward(z)
    
    #現在ステップの誤差計算
    def RMSE(self, z: np.ndarray, y_d: np.ndarray) -> np.ndarray:
        return self.Network.RMSE(z, y_d)
    
    #学習
    def fit(self, Z: np.ndarray, Y_d: np.ndarray):
        self.Network.fit(Z, Y_d)
        
    #初期化
    def reset(self):
        self.Network.reset()

    #ディープコピー
    def clone(self):
        return self.Network.clone()
    
class Network_Readout_LinearTransformer(Module.Network):
    """
    Torchで実装した線形変換器の読み出し層ネットワーク
    """
    def __init__(self, param: dict, parent = None) -> None:
        super().__init__(param, parent)

        #パラメータ取得
        self.DeviceCode = self.Param["Project_DeviceCode"]  #CPU/GPUを使うか（CPU -> cpu, GPU -> gpu:n（nはデバイス番号，無くてもいい））
        self.DataType = self.Param["Project_DataType"]      #Pytorchのデータ型

        self.D_z = self.Param["LinearTransformer_D_z"]       #特徴ベクトル次元
        self.D_y = self.Param["LinearTransformer_D_y"]       #出力信号次元
        self.Beta = self.Param["LinearTransformer_Beta"]     #正規化係数
        self.Bias = np.ones([1])                            #バイアス

        #ハイパーパラメータのtorchヘの変換
        self.Torch_D_y = torch.tensor(self.D_y, device = self.DeviceCode, dtype = self.DataType)
        self.Torch_Beta = torch.tensor(self.Beta, device = self.DeviceCode, dtype = self.DataType)
        self.Torch_Bias = torch.tensor(self.Bias, device = self.DeviceCode, dtype = self.DataType)

        #重み初期化
        self.W_out = None
        
    #順伝播
    def forward(self, z: torch.tensor) -> torch.tensor:
        return torch.matmul(torch.cat([z, self.Torch_Bias]), self.W_out)
    
    #現在ステップの誤差計算
    def RMSE(self, z: torch.tensor, y_d: torch.tensor) -> torch.tensor:
        return torch.sqrt(torch.sum((y_d - torch.matmul(torch.cat([z, self.Torch_Bias]), self.W_out))**2) / self.Torch_D_y)
    
    #学習
    def fit(self, Z: torch.tensor, Y_d: torch.tensor) -> torch.tensor:
        X = torch.cat([Z, torch.ones([Z.shape[0], 1], device = self.DeviceCode, dtype = self.DataType) * self.Torch_Bias], 1)
        #inverseは精度的に不安定なのでsolveを使用
        #double型にキャスト
        X = X.to(torch.double)
        Y_d = Y_d.to(torch.double)
        #self.W_out = torch.matmul(
        #                torch.matmul(
        #                    torch.inverse(
        #                        torch.matmul(torch.t(X), X) + self.Torch_Beta * torch.eye(self.D_z + 1, device = self.DeviceCode, dtype = torch.double))
        #                    , torch.t(X))
        #                , Y_d).to(self.DataType)
        self.W_out = torch.linalg.solve(
                        torch.matmul(torch.t(X), X) + self.Torch_Beta * torch.eye(self.D_z + 1, device = self.DeviceCode, dtype = torch.double),
                        torch.matmul(torch.t(X), Y_d)).to(self.DataType)
        
    #初期化
    #未実装#################################
    def reset(self):
        print("ERROR")

    #ディープコピー
    #未実装#################################
    def clone(self) -> any:
        print("ERROR")
    
#--------------------------------------------------------------------
#多層リードアウト
class Module_Readout_DNN(Module_Readout):
    """
    多層NNの読み出し層（Torchを利用しているが，Numpy版）
    ！！！後で構造をパラメータとして指定できるようにすること！！！
    """
    def __init__(self, param: dict, parent = None) -> None:
        super().__init__(param, parent)

        #パラメータ取得
        self.DeviceCode = self.Param["Project_DeviceCode"]      #CPU/GPUを使うか（CPU -> cpu, GPU -> gpu:n（nはデバイス番号，無くてもいい））
        self.DataType = self.Param["Project_DataType"]          #Pytorchのデータ型

        self.D_z = self.Param["DNN_D_z"]                        #特徴ベクトル次元
        self.D_y = self.Param["DNN_D_y"]                        #出力信号次元
        
        #ネットワーク生成
        self.Network = Network_Readout_DNN(param, self)
        
    #順伝播
    def forward(self, z: np.ndarray) -> np.ndarray:
        z = torch.tensor(z, device = self.DeviceCode, dtype = self.DataType).clone() if type(z) is np.ndarray else z.clone()
        return self.Network.forward(z).to(device = 'cpu').detach().numpy().copy()
    
    #現在ステップの誤差計算
    def RMSE(self, z: np.ndarray, y_d: np.ndarray) -> np.ndarray:
        z = torch.tensor(z, device = self.DeviceCode, dtype = self.DataType).clone() if type(z) is np.ndarray else z.clone()
        y_d = torch.tensor(y_d, device = self.DeviceCode, dtype = self.DataType).clone() if type(y_d) is np.ndarray else y_d.clone()
        return np.array(self.Network.RMSE(z, y_d))
    
    #学習
    def fit(self, Z: np.ndarray, Y_d: np.ndarray) -> np.ndarray:
        Z = torch.tensor(Z, device = self.DeviceCode, dtype = self.DataType).clone() if type(Z) is np.ndarray else Z.clone()
        Y_d = torch.tensor(Y_d, device = self.DeviceCode, dtype = self.DataType).clone() if type(Y_d) is np.ndarray else Y_d.clone()
        
        return np.array([self.Network.fit(Z, Y_d)])
        
    #初期化
    #未実装#################################
    def reset(self): 
        print("ERROR")

    #ディープコピー
    def clone(self) -> any:
        new = Module_Readout_DNN(self.Param.copy(), self.Parent)
        new.Network = self.Network.clone()

        return new

class Network_Readout_DNN(Module.Network):
    """
    多層NNの読み出し層用ネットワーク，多層NN
    ！！！現在のところ構造はここにハードコーディング！！！
    ！！！後で構造をパラメータとして指定できるようにすること！！！
    """
    def __init__(self, param: dict, parent = None) -> None:
        super().__init__(param, parent)

        #パラメータ取得
        self.D_z = self.Param["DNN_D_z"]                        #特徴ベクトル次元
        self.D_y = self.Param["DNN_D_y"]                        #出力信号次元
        self.LearningRate = self.Param["DNN_LearningRate"]      #学習率
        self.MaxLearningLoop = self.Param["DNN_MaxLearningLoop"]#最大学習ループ数（使わない場合は0にする）
        self.AimingError = self.Param["DNN_AimingError"]        #目標（最小）誤差（使わない場合は0にする）
        
        #ネットワーク生成
        self._makeNetwork()
        
    #順伝播
    def forward(self, z: torch.tensor) -> torch.tensor:
        signal = z
        signal = self.Relu_1(self.Dense_1(signal))
        signal = self.Relu_2(self.Dense_2(signal))
        signal = self.Relu_3(self.Dense_3(signal))
        signal = self.Relu_4(self.Dense_4(signal))
        return self.Dense_5(signal)
    
    #現在ステップの誤差計算
    def RMSE(self, z: torch.tensor, y_d: torch.tensor) -> torch.tensor:
        return self.Criterion(self(z), y_d).item()
    
    #学習
    def fit(self, Z: torch.tensor, Y_d: torch.tensor) -> torch.tensor:
        self.train()#学習モード
        #学習ループ
        error = 1           #誤差
        loop_count = 0      #ループ回数
        #一定のループ回数経過か一定未満の誤差で脱出
        while (loop_count < self.MaxLearningLoop or self.MaxLearningLoop == 0) and error > self.AimingError:
            loss = self.Criterion(self(Z), Y_d)
            self.Optimizer.zero_grad()
            loss.backward()
            self.Optimizer.step()
            print("Loop Count : " + str(loop_count) + ", Error : " + str(loss.item()))
            error = loss.item()
            loop_count += 1

        self.eval()#推論モード
        #順伝播
        loss = self.Criterion(self(Z), Y_d)
        #誤差を返す
        return loss.item()
        
    #ネットワーク生成
    def _makeNetwork(self):
        self.D_Dense_1 = 32
        self.D_Dense_2 = 32
        self.D_Dense_3 = 32
        self.D_Dense_4 = 32
        
        self.Dense_1 = torch.nn.Linear(self.D_z, self.D_Dense_1)
        self.Relu_1 = torch.nn.ReLU()
        self.Dense_2 = torch.nn.Linear(self.D_Dense_1, self.D_Dense_2)
        self.Relu_2 = torch.nn.ReLU()
        self.Dense_3 = torch.nn.Linear(self.D_Dense_2, self.D_Dense_3)
        self.Relu_3 = torch.nn.ReLU()
        self.Dense_4 = torch.nn.Linear(self.D_Dense_3, self.D_Dense_4)
        self.Relu_4 = torch.nn.ReLU()
        self.Dense_5 = torch.nn.Linear(self.D_Dense_4, self.D_y)

        self.Criterion = torch.nn.functional.mse_loss
        self.Optimizer = torch.optim.Adam(self.parameters(), lr = self.LearningRate)

    #初期化
    #未実装#################################
    def reset(self):
        print("ERROR")

    #ディープコピー
    #未実装#################################
    def clone(self) -> any:
        print("ERROR")
    