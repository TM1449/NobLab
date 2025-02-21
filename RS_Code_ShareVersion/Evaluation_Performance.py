#####################################################################
#信川研ESNプロジェクト用
#制作者：中村亮介 吉原悠斗 吉田聡太 飯沼貴大
#作成日：2023/05/24
#####################################################################

#====================================================================
#＜このプログラムのTodoメモ＞
#・波形出力未移植＠Ver1
#・Delay Capacity未実装
#・マルチスケールエントロピー未移植＠Ver1
#・コンシステンシー未移植＠Ver1
#・共分散ランク未実装
#・固有値分布未移植＠Ver1
#・入力信号分布未実装

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
class Evaluation_NRMSE(Evaluation):
    """
    NRMSE評価クラス
    """
    #コンストラクタ
    def __init__(self, param: dict, parent: any = None):
        super().__init__(param, parent)

        #パラメータ取得
        self.F_OutputLog = self.Param["NRMSE_F_OutputLog"]      #経過の出力を行うか

        if self.F_OutputLog : print("*** Evaluation NRMSE ***")
        if self.F_OutputLog : print("+++ Initializing +++")

        self.D_u = self.Param["NRMSE_D_u"]                      #入力信号次元
        self.D_y = self.Param["NRMSE_D_y"]                      #出力信号次元
        self.Length_Burnin = self.Param["NRMSE_Length_Burnin"]  #空走用データ時間長
        self.Length_Train = self.Param["NRMSE_Length_Train"]    #学習用データ時間長
        self.Length_Test = self.Param["NRMSE_Length_Test"]      #評価用データ時間長
        self.Length_Total = self.Length_Burnin + self.Length_Train + self.Length_Test#全体データ時間長

        #評価用タスク（Type型）
        self.T_Task = self.Param["NRMSE_T_Task"]
        #タスクのインスタンス
        param = self.Param.copy()
        param.update({
            "Task_D_u" : self.D_u,
            "Task_D_y" : self.D_y,
            "Task_Length" : self.Length_Total,
            })
        self.Task = self.T_Task(param, self)
        
        #モデル（Type型）
        self.T_Model = self.Param["NRMSE_T_Model"]
        #モデルのインスタンス
        param = self.Param.copy()
        param.update({
            })
        self.Model = self.T_Model(param, self)

        #作図出力（Type型）
        self.T_Output = self.Param["NRMSE_T_Output"]
        #作図出力のインスタンス
        param = self.Param.copy()
        param.update({
            })
        self.Output = self.T_Output(param, self)
        
    #本体
    def __call__(self) -> dict:
        if self.F_OutputLog : print("*** Started Evaluation NRMSE ***")

        #エコー収集
        if self.F_OutputLog : print("+++ Collecting Echo +++")
        
        #バッチ
        T = list(range(self.Length_Total))
        U = [None for _ in range(self.Length_Total)]
        Z = [None for _ in range(self.Length_Total)]
        Y = [None for _ in range(self.Length_Total)]
        Y_d = [None for _ in range(self.Length_Total)]
        E = [None for _ in range(self.Length_Total)]

        #モデルリセット
        self.Model.reset()

        #学習時間計測開始
        TimeForTraining_start = time.perf_counter()
        
        #Burn-in
        if self.F_OutputLog : print("--- Burn-in Process---")
        for i, t in enumerate(T[0 : self.Length_Burnin]):
            if self.F_OutputLog : print("\r%d / %d"%(i, self.Length_Burnin), end = "")
            U[t], Y_d[t] = self.Task.getData(t)
            self.Model.forwardReservoir(U[t])
        if self.F_OutputLog : print("\n", end = "")

        #データ収集
        if self.F_OutputLog : print("--- Collecting Data ---")
        for i, t in enumerate(T[self.Length_Burnin : self.Length_Burnin + self.Length_Train]):
            if self.F_OutputLog : print("\r%d / %d"%(i, self.Length_Train), end = "")
            U[t], Y_d[t] = self.Task.getData(t)
            Z[t] = self.Model.forwardReservoir(U[t])
        if self.F_OutputLog : print("\n", end = "")
    
        #学習
        if self.F_OutputLog : print("+++ Training Model +++")
        
        start = self.Length_Burnin
        end = self.Length_Burnin + self.Length_Train
        #Torch分岐
        array_Z = self.ListTensorToTensor(Z[start : end]) if self.F_UsePytorch else np.array(Z[start : end])
        array_Y_d = self.ListTensorToTensor(Y_d[start : end]) if self.F_UsePytorch else np.array(Y_d[start : end])
        self.Model.fit(array_Z, array_Y_d)
        
        #学習時間計測終了(オーバーフローしてた場合はNaN)
        TimeForTraining_end = time.perf_counter()
        if TimeForTraining_end < TimeForTraining_start:
            TimeForTraining = np.nan
        else : TimeForTraining = TimeForTraining_end - TimeForTraining_start

        #評価
        if self.F_OutputLog : print("+++ Evaluating Model +++")
        if self.F_OutputLog : print("--- Testing ---")
        #評価時間計測開始
        TimeForTesting_start = time.perf_counter()
        
        for i, t in enumerate(T[self.Length_Burnin + self.Length_Train : self.Length_Burnin + self.Length_Train + self.Length_Test]):
            if self.F_OutputLog : print("\r%d / %d"%(i, self.Length_Test), end = "")
            U[t], Y_d[t] = self.Task.getData(t)
            Y[t], E[t] = self.Model.forwardWithRMSE(U[t], Y_d[t])
        if self.F_OutputLog : print("\n", end = "")
        
        #評価時間計測終了(オーバーフローしてた場合はNaN)
        TimeForTesting_end = time.perf_counter()
        if TimeForTesting_end < TimeForTesting_start:
            TimeForTesting = np.nan
        else : TimeForTesting = TimeForTesting_end - TimeForTesting_start

        #指標計算
        if self.F_OutputLog : print("--- Calculating ---")
        start = self.Length_Burnin + self.Length_Train
        end = self.Length_Burnin + self.Length_Train + self.Length_Test
        #Torch分岐
        array_Y = self.ListTensorToTensor(Y[start : end]).to(device = 'cpu').detach().numpy().copy() if self.F_UsePytorch else np.array(Y[start : end])
        array_Y_d = self.ListTensorToTensor(Y_d[start : end]).to(device = 'cpu').detach().numpy().copy() if self.F_UsePytorch else np.array(Y_d[start : end])
        NRMSE = np.mean(np.sqrt(np.mean((array_Y - array_Y_d)**2, axis = 0) / (np.var(array_Y_d, axis = 0) + 10**-14)))#（発散防止）
        LogNRMSE = np.log(NRMSE + 10**-14)#（発散防止）
        
        #終了処理
        if self.F_OutputLog : print("+++ Storing Results +++")
        if self.F_UsePytorch:#Torch分岐
            U = [d.to(device = 'cpu').detach().numpy().copy() if d is not None else None for d in U]
            Y = [d.to(device = 'cpu').detach().numpy().copy() if d is not None else None for d in Y]
            Y_d = [d.to(device = 'cpu').detach().numpy().copy() if d is not None else None for d in Y_d]
            E = [d.to(device = 'cpu').detach().numpy().copy() if d is not None else None for d in E]
        results = self.Param.copy()
        results.update({
            "NRMSE_R_NRMSE" : NRMSE,
            "NRMSE_R_LogNRMSE" : LogNRMSE,
            "NRMSE_R_TimeForTraining" : TimeForTraining,
            "NRMSE_R_TimeForTesting" : TimeForTesting,
            "NRMSE_R_T" : T,
            "NRMSE_R_U" : U,
            "NRMSE_R_Y" : Y,
            "NRMSE_R_Y_d" : Y_d,
            "NRMSE_R_E" : E,
            })

        #作図出力
        self.Output(results)
        
        if self.F_OutputLog : print("*** Finished Evaluation NRMSE ***")

        return results
    
#--------------------------------------------------------------------
class Evaluation_MC(Evaluation):
    """
    Memory Capacity評価クラス
    """
    #コンストラクタ
    def __init__(self, param: dict, parent: any = None):
        super().__init__(param, parent)

        #パラメータ取得
        self.F_OutputLog = self.Param["MemoryCapacity_F_OutputLog"]     #経過の出力を行うか

        if self.F_OutputLog : print("*** Evaluation Memory-Capacity ***")
        if self.F_OutputLog : print("+++ Initializing +++")

        self.D_u = self.Param["MemoryCapacity_D_u"]                     #入力信号次元
        self.D_y = self.Param["MemoryCapacity_D_y"]                     #出力信号次元
        self.Length_Burnin = self.Param["MemoryCapacity_Length_Burnin"] #空走用データ時間長
        self.Length_Train = self.Param["MemoryCapacity_Length_Train"]   #学習用データ時間長
        self.Length_Test = self.Param["MemoryCapacity_Length_Test"]     #評価用データ時間長
        self.Length_Total = self.Length_Burnin + self.Length_Train + self.Length_Test#全体データ時間長
        self.MaxTau = self.Param["MemoryCapacity_MaxTau"]               #評価する最大遅延

        #評価用タスク（Type型）
        self.T_Task = self.Param["MemoryCapacity_T_Task"]
        #タスクのインスタンス
        param = self.Param.copy()
        param.update({
            "Task_D_u" : self.D_u,
            "Task_D_y" : self.D_y,
            "Task_Length" : self.Length_Total,
            "Task_MaxTau" : self.MaxTau
            })
        self.Task = self.T_Task(param, self)
        
        #モデル（Type型）
        self.T_Model = self.Param["MemoryCapacity_T_Model"]
        #モデルのインスタンス
        param = self.Param.copy()
        param.update({
            })
        self.Model = self.T_Model(param, self)

        #作図出力（Type型）
        self.T_Output = self.Param["MemoryCapacity_T_Output"]
        #作図出力のインスタンス
        param = self.Param.copy()
        param.update({
            })
        self.Output = self.T_Output(param, self)
        
    #本体
    def __call__(self) -> dict:
        if self.F_OutputLog : print("*** Started Evaluation Memory-Capacity ***")

        #エコー収集
        if self.F_OutputLog : print("+++ Collecting Echo +++")
        
        #バッチ
        T = list(range(self.Length_Total))
        U = [None for _ in range(self.Length_Total)]
        Z = [None for _ in range(self.Length_Total)]

        #モデルリセット
        self.Model.reset()
        
        #Burn-in
        if self.F_OutputLog : print("--- Burn-in Process---")
        for i, t in enumerate(T[0 : self.Length_Burnin]):
            if self.F_OutputLog : print("\r%d / %d"%(i, self.Length_Burnin), end = "")
            U[t], _ = self.Task.getData(t)
            self.Model.forwardReservoir(U[t])
        if self.F_OutputLog : print("\n", end = "")

        #データ収集
        if self.F_OutputLog : print("--- Collecting Data for Training ---")
        for i, t in enumerate(T[self.Length_Burnin : self.Length_Burnin + self.Length_Train]):
            if self.F_OutputLog : print("\r%d / %d"%(i, self.Length_Train), end = "")
            U[t], _ = self.Task.getData(t)
            Z[t] = self.Model.forwardReservoir(U[t])
        if self.F_OutputLog : print("\n", end = "")
    
        if self.F_OutputLog : print("--- Collecting Data for Test ---")
        for i, t in enumerate(T[self.Length_Burnin + self.Length_Train : self.Length_Burnin + self.Length_Train + self.Length_Test]):
            if self.F_OutputLog : print("\r%d / %d"%(i, self.Length_Test), end = "")
            U[t], _ = self.Task.getData(t)
            Z[t] = self.Model.forwardReservoir(U[t])
        if self.F_OutputLog : print("\n", end = "")

        #テスト
        if self.F_OutputLog : print("+++ Evaluating Model +++")
        Tau = np.array(list(range(1, self.MaxTau + 1)))
        MC_Tau = np.zeros([len(Tau)])
        for i, tau in enumerate(Tau):
            if self.F_OutputLog : print("\r--- Tau : %d / %d ---"%(tau, self.MaxTau), end = "")
                
            #学習データ取得
            start = self.Length_Burnin
            end = self.Length_Burnin + self.Length_Train
            array_Z_train = np.array(Z[start : end])
            array_Y_d_tau_train = np.array([
                self.Task.getDataTau(t, tau)[1] for t in range(start, end)])

            #学習
            self.Model.fit(array_Z_train, array_Y_d_tau_train)
        
            #テスト
            start = self.Length_Burnin + self.Length_Train
            end = self.Length_Burnin + self.Length_Train + self.Length_Test
            array_Y_tau_test = np.array([self.Model.forwardReadout(Z[t]) for t in range(start, end)])
            array_Y_d_tau_test = np.array([self.Task.getDataTau(t, tau)[1] for t in range(start, end)])
            
            #評価処理
            Cov = np.cov([array_Y_d_tau_test.reshape([-1]), array_Y_tau_test.reshape([-1])], bias = True)
            MC_Tau[i] = (Cov[0,1]**2) / (Cov[0,0] * Cov[1,1])
        if self.F_OutputLog : print("\n", end = "")
            
        #指標計算
        if self.F_OutputLog : print("--- Calculating ---")
        MC = np.sum(MC_Tau)

        #終了処理
        if self.F_OutputLog : print("+++ Storing Results +++")
        results = self.Param.copy()
        results.update({
            "MemoryCapacity_R_MC" : MC,
            "MemoryCapacity_R_Tau" : Tau,
            "MemoryCapacity_R_MC_Tau" : MC_Tau
            })
        
        #作図出力
        self.Output(results)
        
        if self.F_OutputLog : print("*** Finished Evaluation Memory-Capacity ***")

        return results
    