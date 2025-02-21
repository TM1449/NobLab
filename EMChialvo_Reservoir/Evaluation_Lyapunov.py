#####################################################################
#信川研ESNプロジェクト用
#制作者：中村亮介 吉原悠斗 吉田聡太 飯沼貴大
#作成日：2023/05/24
#####################################################################

#====================================================================
#＜このプログラムのTodoメモ＞
#・リアプノフ次元未移植＠Ver1
#・
#・

#====================================================================
import time

import numpy as np
import torch

#====================================================================
#評価

#********************************************************************
#継承元
class Evaluation:
    """
    評価クラス
    全ての評価はこれを継承
    親オブジェクトは今のところ無し．
    実験パラメータをparamで受け，複製したparamに結果を入れて返す．
    """
    #コンストラクタ
    def __init__(self, param: dict, parent: any = None):
        self.Param = param
        
        self.Parent = parent                            #親オブジェクト
        
        #パラメータ取得
        self.F_UsePytorch = self.Param["Project_F_UsePytorch"]  #Pytorchを使うか（多層リードアウトでは強制的に使用）
        self.DeviceCode = self.Param["Project_DeviceCode"]      #CPU/GPUを使うか（CPU -> cpu, GPU -> gpu:n（nはデバイス番号，無くてもいい））
        self.DataType = self.Param["Project_DataType"]          #Pytorchのデータ型

    #本体
    def __call__(self) -> dict: pass

    #収集行列作成補助（テンソルが入れ子のリストを２次元テンソルに変換）
    def ListTensorToTensor(self, data: list) -> torch.tensor:
        out = torch.zeros([len(data), len(data[0])], device = data[0].device, dtype = data[0].dtype)
        for i, d in enumerate(data): 
            out[i] = d
        return out
    
#********************************************************************
#利用可能評価指標
class Evaluation_MLE(Evaluation):
    """
    Max Lyapunov Exponent評価クラス（摂動ベース）
    """
    #コンストラクタ
    def __init__(self, param: dict, parent: any = None):
        super().__init__(param, parent)

        #パラメータ取得
        self.F_OutputLog = self.Param["MLE_F_OutputLog"]            #経過の出力を行うか

        if self.F_OutputLog : print("*** Evaluation Max-Lyapunov-Exponent ***")
        if self.F_OutputLog : print("+++ Initializing +++")

        self.D_u = self.Param["MLE_D_u"]                            #入力信号次元
        self.D_y = self.Param["MLE_D_y"]                            #出力信号次元
        self.D_x = self.Param["MLE_D_x"]

        self.Length_Burnin = self.Param["MLE_Length_Burnin"]        #空走用データ時間長
        self.Length_Test = self.Param["MLE_Length_Test"]            #評価用データ時間長
        self.Length_Total = self.Length_Burnin + self.Length_Test   #全体データ時間長
        self.Epsilon = self.Param["MLE_Epsilon"]                    #摂動ε

        #評価用タスク（Type型）
        self.T_Task = self.Param["MLE_T_Task"]
        #タスクのインスタンス
        param = self.Param.copy()
        param.update({
            "Task_D_u" : self.D_u,
            "Task_D_y" : self.D_y,
            "Task_Length" : self.Length_Total,
            })
        self.Task = self.T_Task(param, self)
        
        #モデル（Type型）
        self.T_Model = self.Param["MLE_T_Model"]
        #モデルのインスタンス
        param = self.Param.copy()
        param.update({
            })
        self.Model = self.T_Model(param, self)

        #作図出力（Type型）
        self.T_Output = self.Param["MLE_T_Output"]
        #作図出力のインスタンス
        param = self.Param.copy()
        param.update({
            })
        self.Output = self.T_Output(param, self)
        
    #本体
    def __call__(self) -> dict:
        if self.F_OutputLog : print("*** Started Evaluation Max-Lyapunov-Exponent ***")

        #エコー収集
        if self.F_OutputLog : print("+++ Collecting Echo +++")
        
        #バッチ
        T = list(range(self.Length_Total))
        U = [None for _ in range(self.Length_Total)]
        MMLE = [None for _ in range(self.Length_Total)]#瞬時リアプノフ指数

        #基準軌道の配列
        Standard_0 = np.zeros((3 * self.D_x, self.Length_Total))
        Standard_1 = np.zeros((3 * self.D_x, self.Length_Total))

        #摂動の配列
        Pert_0 = np.zeros((3 * self.D_x, self.Length_Total))
        Pert_1 = np.zeros((3 * self.D_x, self.Length_Total))

        #リザバー層の配列
        RS_xO_MLE = np.array([None for _ in range(self.D_x)])      #リザバー層のX
        RS_yO_MLE = np.array([None for _ in range(self.D_x)])      #リザバー層のY
        RS_phiO_MLE = np.array([None for _ in range(self.D_x)])    #リザバー層のPhi

        RS_x_MLE = np.array([None for _ in range(self.D_x)])      #リザバー層のX
        RS_y_MLE = np.array([None for _ in range(self.D_x)])      #リザバー層のY
        RS_phi_MLE = np.array([None for _ in range(self.D_x)])    #リザバー層のPhi

        #摂動軌道用のリザバー層の配列
        RS_xPO_MLE = np.array([None for _ in range(self.D_x)])      #リザバー層のX
        RS_yPO_MLE = np.array([None for _ in range(self.D_x)])      #リザバー層のY
        RS_phiPO_MLE = np.array([None for _ in range(self.D_x)])    #リザバー層のPhi

        RS_xP_MLE = np.array([None for _ in range(self.D_x)])      #リザバー層のX
        RS_yP_MLE = np.array([None for _ in range(self.D_x)])      #リザバー層のY
        RS_phiP_MLE = np.array([None for _ in range(self.D_x)])    #リザバー層のPhi
        
        #モデルリセット
        self.Model.reset()
        
        #Burn-in OOOOOOOO
        if self.F_OutputLog : print("--- Burn-in Process---")
        for i, t in enumerate(T[0 : self.Length_Burnin]):
            if self.F_OutputLog : print("\r%d / %d"%(i, self.Length_Burnin), end = "")
            U[t], _ = self.Task.getData(t)
            self.Model.forwardReservoir(U[t])
        if self.F_OutputLog : print("\n", end = "")
        
        if self.F_OutputLog : print("--- Add perturbation Process---")
        
        #リザバー層の全ニューロンの誤差
        e = np.ones(3 * self.D_x)
        e = (e / np.linalg.norm(e)) * self.Epsilon
        

        if self.F_OutputLog : print("+++ Evaluating Model +++")
        #時間発展させ評価
        for i, t in enumerate(T[self.Length_Burnin : self.Length_Burnin + self.Length_Test]):
            if self.F_OutputLog : print("\r%d / %d"%(i, self.Length_Test), end = "")
            U[t], _ = self.Task.getData(t)

            #基準軌道の現時刻のニューロンの値、基準軌道の1時刻先のニューロンの値
            #摂動軌道の現時刻のニューロンの値、摂動軌道の1時刻先のニューロンの値
            RS_xO_MLE, RS_yO_MLE, RS_phiO_MLE, \
                RS_x_MLE, RS_y_MLE, RS_phi_MLE, \
                    RS_xPO_MLE, RS_yPO_MLE, RS_phiPO_MLE, \
                        RS_xP_MLE, RS_yP_MLE, RS_phiP_MLE = self.Model.forwardReservoir_MLE(U[t], e[0:self.D_x], e[self.D_x:2*self.D_x], e[2*self.D_x:3*self.D_x])
            
            #基準軌道：現時刻のニューロンの値
            Standard_0[:,t] = np.concatenate([RS_xO_MLE, RS_yO_MLE, RS_phiO_MLE])
            #基準軌道：1時刻先のニューロンの値
            Standard_1[:,t] = np.concatenate([RS_x_MLE, RS_y_MLE, RS_phi_MLE])
            #摂動軌道：現時刻のニューロンの値
            Pert_0[:,t] = np.concatenate([RS_xPO_MLE, RS_yPO_MLE, RS_phiPO_MLE])
            #摂動軌道：1時刻先のニューロンの値
            Pert_1[:,t] = np.concatenate([RS_xP_MLE, RS_yP_MLE, RS_phiP_MLE])

            #1時刻前と現時刻の差ベクトルを算出
            Vector_0 = (Pert_0[:, t] - Standard_0[:, t])
            Vector_1 = (Pert_1[:, t] - Standard_1[:, t])

            #1時刻前と現時刻の差ベクトルの大きさを算出
            Norm_0 = np.linalg.norm(Vector_0)
            Norm_1 = np.linalg.norm(Vector_1)

            #最大リアプノフ指数の計算
            MMLE[t] = np.log(Norm_1 / Norm_0 + 1e-20)

            #次の摂動の計算
            e = (Vector_1 / Norm_1) * self.Epsilon

        if self.F_OutputLog : print("\n", end = "")
        
        #指標計算
        if self.F_OutputLog : print("--- Calculating ---")
        start = self.Length_Burnin
        end = self.Length_Burnin + self.Length_Test
        array_MMLE = np.array(MMLE[start : end])
        MLE = np.mean(array_MMLE, axis = 0)#リアプノフ指数
        
        #終了処理
        if self.F_OutputLog : print("+++ Storing Results +++")
        results = self.Param.copy()
        results.update({
            "MLE_R_MLE" : MLE,
            "MLE_R_T" : T,
            "MLE_R_U" : U,
            "MLE_R_MMLE" : MMLE,
            })
        
        #作図出力
        self.Output(results)
        
        if self.F_OutputLog : print("*** Finished Evaluation Max-Lyapunov-Exponent ***")

        return results
    
#--------------------------------------------------------------------
