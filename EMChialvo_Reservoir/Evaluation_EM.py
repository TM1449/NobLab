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
#・波形出力未移植＠Ver1
#・Delay Capacity未実装
#・最大リアプノフ指数未移植＠Ver1
#・マルチスケールエントロピー未移植＠Ver1
#・コンシステンシー未移植＠Ver1
#・リアプノフ次元未移植＠Ver1
#・共分散ランク未実装
#・固有値分布未移植＠Ver1
#・入力信号分布未実装

#====================================================================
import time

import numpy as np

#NRMSE_full_list : list = []

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
        
    #本体
    def __call__(self) -> dict: pass
    
#********************************************************************
#利用可能モデル
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
        self.D_x = self.Param["NRMSE_D_x"]

        self.Length_Burnin = self.Param["NRMSE_Length_Burnin"]  #空走用データ時間長
        self.Length_Train = self.Param["NRMSE_Length_Train"]    #学習用データ時間長
        self.Length_Test = self.Param["NRMSE_Length_Test"]      #評価用データ時間長
        self.Length_Total = self.Length_Burnin + self.Length_Train + self.Length_Test#全体データ時間長
        self.RS_neuron = self.Param["Model_Reservoir_Neurons"]  #リザバー層のニューロン数

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
        #global NRMSE_full_list
        if self.F_OutputLog : print("*** Started Evaluation NRMSE ***")

        #エコー収集
        if self.F_OutputLog : print("+++ Collecting Echo +++")
        
        #バッチ
        T = list(range(self.Length_Total))                      #全体時間
        U = [None for _ in range(self.Length_Total)]            #入力信号
        Z = [None for _ in range(self.Length_Total)]            #リザバー層からの信号
        Y = [None for _ in range(self.Length_Total)]            #出力層の信号
        Y_d = [None for _ in range(self.Length_Total)]          #教師信号（のはず？）
        E = [None for _ in range(self.Length_Total)]            #誤差


        RS_X = np.array([None for _ in range(self.Length_Total * self.RS_neuron)]).reshape(-1,self.Length_Total)      #リザバー層のX
        RS_Y = np.array([None for _ in range(self.Length_Total * self.RS_neuron)]).reshape(-1,self.Length_Total)      #リザバー層のY
        RS_Phi = np.array([None for _ in range(self.Length_Total * self.RS_neuron)]).reshape(-1,self.Length_Total)      #リザバー層のPhi

        RS_X_A = np.array([None for _ in range(self.Length_Total * self.D_x)]).reshape(-1,self.Length_Total)      #リザバー層のX
        RS_Y_A = np.array([None for _ in range(self.Length_Total * self.D_x)]).reshape(-1,self.Length_Total)      #リザバー層のY
        RS_Phi_A = np.array([None for _ in range(self.Length_Total * self.D_x)]).reshape(-1,self.Length_Total)      #リザバー層のPhi

        RS_HeatMap = np.array([None for _ in range(self.Length_Total * self.D_x)]).reshape(-1,self.Length_Total)    #リザバー層のXのヒートマップ

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
            Z[t], RS_X[:, t], RS_Y[:, t], RS_Phi[:, t], RS_X_A[:, t], RS_Y_A[:, t], RS_Phi_A[:, t] = self.Model.forwardReservoir(U[t])
        if self.F_OutputLog : print("\n", end = "")
    
        #学習
        if self.F_OutputLog : print("+++ Training Model +++")
        start = self.Length_Burnin
        end = self.Length_Burnin + self.Length_Train
        array_Z = np.array(Z[start : end])
        array_Y_d = np.array(Y_d[start : end])
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
            Y[t], E[t], RS_HeatMap[:, t], RS_X[:, t], RS_Y[:, t], RS_Phi[:, t], RS_X_A[:, t], RS_Y_A[:, t], RS_Phi_A[:, t] = self.Model.forwardWithRMSE(U[t], Y_d[t])
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
        array_Y = np.array(Y[start : end])
        array_Y_d = np.array(Y_d[start : end])
        NRMSE = np.mean(np.sqrt(np.mean((array_Y - array_Y_d)**2, axis = 0) / (np.var(array_Y_d, axis = 0) + 0.0000001)))#発散抑制
        LogNRMSE = np.log(NRMSE)
        #NRMSE_full_list.append(NRMSE)
        
        #終了処理
        if self.F_OutputLog : print("+++ Storing Results +++")
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

            "Reservoir_X" : RS_X,
            "Reservoir_Y" : RS_Y,
            "Reservoir_Phi" : RS_Phi,
            
            "Reservoir_X_All" : RS_X_A,
            "Reservoir_Y_All" : RS_Y_A,
            "Reservoir_Phi_All" : RS_Phi_A,

            "Reservoir_HeatMap" : RS_HeatMap,
            })
        
        outputs = self.Param.copy()
        outputs.update({
            "NRMSE_R_NRMSE" : NRMSE,
            "NRMSE_R_LogNRMSE" : LogNRMSE,
            "NRMSE_R_TimeForTraining" : TimeForTraining,
            "NRMSE_R_TimeForTesting" : TimeForTesting
            })

        #作図出力
        self.Output(results)
        
        if self.F_OutputLog : print("*** Finished Evaluation NRMSE ***")

        return outputs
    
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

        if self.F_OutputLog : print("*** Evaluation Memory Capacity ***")
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
            Z[t], _, _, _, _, _, _ = self.Model.forwardReservoir(U[t])
        if self.F_OutputLog : print("\n", end = "")
    
        if self.F_OutputLog : print("--- Collecting Data for Test ---")
        for i, t in enumerate(T[self.Length_Burnin + self.Length_Train : self.Length_Burnin + self.Length_Train + self.Length_Test]):
            if self.F_OutputLog : print("\r%d / %d"%(i, self.Length_Test), end = "")
            U[t], _ = self.Task.getData(t)
            Z[t], _, _, _, _, _, _ = self.Model.forwardReservoir(U[t])
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
        outputs = self.Param.copy()
        outputs.update({
            "MemoryCapacity_R_MC" : MC,
            "MemoryCapacity_R_Tau" : Tau,
            "MemoryCapacity_R_MC_Tau" : MC_Tau
            })

        #作図出力
        self.Output(results)
        
        if self.F_OutputLog : print("*** Finished Evaluation Memory-Capacity ***")

        return outputs

#********************************************************************
#利用可能モデル
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
        MLE_TS = [None for _ in range(self.Length_Total)]#リアプノフ指数時系列
        MLE_Sum = 0 #リアプノフ指数の合計

        #基準軌道の配列
        Standard_0 = np.zeros((3 * self.D_x, self.Length_Total))
        Standard_1 = np.zeros((3 * self.D_x, self.Length_Total))

        #摂動の配列
        Pert_0 = np.zeros((3 * self.D_x, self.Length_Total))
        Pert_1 = np.zeros((3 * self.D_x, self.Length_Total))

        #リザバー層の配列
        RS_xO_MLE = np.array([None for _ in range(self.D_x)])      #リザバー層のX: 基準軌道用
        RS_yO_MLE = np.array([None for _ in range(self.D_x)])      #リザバー層のY: 基準軌道用
        RS_phiO_MLE = np.array([None for _ in range(self.D_x)])    #リザバー層のPhi: 基準軌道用

        RS_x_MLE = np.array([None for _ in range(self.D_x)])      #リザバー層のX: 基準軌道用
        RS_y_MLE = np.array([None for _ in range(self.D_x)])      #リザバー層のY: 基準軌道用
        RS_phi_MLE = np.array([None for _ in range(self.D_x)])    #リザバー層のPhi: 基準軌道用

        #摂動軌道用のリザバー層の配列
        RS_xPO_MLE = np.array([None for _ in range(self.D_x)])      #リザバー層のX: 摂動軌道用
        RS_yPO_MLE = np.array([None for _ in range(self.D_x)])      #リザバー層のY: 摂動軌道用
        RS_phiPO_MLE = np.array([None for _ in range(self.D_x)])    #リザバー層のPhi: 摂動軌道用

        RS_xP_MLE = np.array([None for _ in range(self.D_x)])      #リザバー層のX: 摂動軌道用
        RS_yP_MLE = np.array([None for _ in range(self.D_x)])      #リザバー層のY: 摂動軌道用
        RS_phiP_MLE = np.array([None for _ in range(self.D_x)])    #リザバー層のPhi: 摂動軌道用
        
        #モデルリセット
        self.Model.reset()
        
        #Burn-in
        if self.F_OutputLog : print("--- Burn-in Process---")
        for i, t in enumerate(T[0 : self.Length_Burnin]):
            if self.F_OutputLog : print("\r%d / %d"%(i, self.Length_Burnin), end = "")
            U[t], _ = self.Task.getData(t)
            self.Model.forwardReservoir(U[t])
        if self.F_OutputLog : print("\n", end = "")
        
        if self.F_OutputLog : print("--- Add perturbation Process---")
        
        #リザバー層の全ニューロンの誤差
        e0 = np.ones(3 * self.D_x)
        e = (e0 / np.linalg.norm(e0)) * self.Epsilon
        
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

            #瞬間最大リアプノフ指数の計算
            MMLE[t] = np.log(Norm_1 / Norm_0)

            #各時間ステップごとのリアプノフ指数を格納
            MLE_Sum += MMLE[t]
            MLE_TS[t] = MLE_Sum / (i + 1)

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
            "MLE_R_MLE_TS" : MLE_TS,
            })
        
        outputs = self.Param.copy()
        outputs.update({
            "MLE_R_MLE" : MLE,
            "MLE_R_MMLE" : MMLE,
            "MLE_R_MLE_TS" : MLE_TS,
            })
        
        #作図出力
        self.Output(results)
        
        if self.F_OutputLog : print("*** Finished Evaluation Max-Lyapunov-Exponent ***")

        return outputs
    

#********************************************************************
#利用可能モデル
class Evaluation_CovMatrixRank(Evaluation):
    """
    Covariance Matrix Rank評価クラス
    """
    #コンストラクタ
    def __init__(self, param: dict, parent: any = None):
        super().__init__(param, parent)

        #パラメータ取得
        self.F_OutputLog = self.Param["CovMatrixRank_F_OutputLog"]      #経過の出力を行うか

        if self.F_OutputLog : print("*** Evaluation Covariance Matrix Rank ***")
        if self.F_OutputLog : print("+++ Initializing +++")

        self.D_u = self.Param["CovMatrixRank_D_u"]                      #入力信号次元
        self.D_x = self.Param["CovMatrixRank_D_x"]
        self.D_y = self.Param["CovMatrixRank_D_y"]                      #出力信号次元
        
        self.Length_Burnin = self.Param["CovMatrixRank_Length_Burnin"]  #空走用データ時間長
        self.Length_Test = self.Param["CovMatrixRank_Length_Test"]      #評価用データ時間長
    
        self.Length_Total = self.Length_Burnin + self.Length_Test       #全体データ時間長
    
        #評価用タスク（Type型）
        self.T_Task = self.Param["CovMatrixRank_T_Task"]
        #タスクのインスタンス
        param = self.Param.copy()
        param.update({
            "Task_D_u" : self.D_u,
            "Task_D_y" : self.D_y,
            "Task_Length" : self.Length_Total,
            })
        self.Task = self.T_Task(param, self)
        
        #モデル（Type型）
        self.T_Model = self.Param["CovMatrixRank_T_Model"]
        #モデルのインスタンス
        param = self.Param.copy()
        param.update({
            })
        self.Model = self.T_Model(param, self)

        #作図出力（Type型）
        self.T_Output = self.Param["CovMatrixRank_T_Output"]
        #作図出力のインスタンス
        param = self.Param.copy()
        param.update({
            })
        self.Output = self.T_Output(param, self)
        
    #本体
    def __call__(self) -> dict:
        if self.F_OutputLog : print("*** Started Evaluation Covariance Matrix Rank ***")

        #エコー収集
        if self.F_OutputLog : print("+++ Collecting Echo +++")
        
        #バッチ
        T = list(range(self.Length_Total))                      #全体時間
        U = [None for _ in range(self.Length_Total)]            #入力信号

        X = np.array([None for _ in range(self.Length_Total * self.D_x)], dtype=float).reshape(-1, self.Length_Total)      #リザバー層のX
        
        #モデルリセット
        self.Model.reset()
        
        #Burn-in
        if self.F_OutputLog : print("--- Burn-in Process---")
        for i, t in enumerate(T[0 : self.Length_Burnin]):
            if self.F_OutputLog : print("\r%d / %d"%(i, self.Length_Burnin), end = "")
            U[t], _ = self.Task.getData(t)
            self.Model.forwardReservoir(U[t])
        if self.F_OutputLog : print("\n", end = "")

        #評価
        if self.F_OutputLog : print("+++ Evaluating Model +++")
        if self.F_OutputLog : print("--- Testing ---")
                
        for i, t in enumerate(T[self.Length_Burnin : self.Length_Burnin + self.Length_Test]):
            if self.F_OutputLog : print("\r%d / %d"%(i, self.Length_Test), end = "")
            U[t], _ = self.Task.getData(t)
            _, _, _, _, X[:, t], _, _ = self.Model.forwardReservoir(U[t])
        if self.F_OutputLog : print("\n", end = "")
        
        #指標計算
        if self.F_OutputLog : print("--- Calculating ---")
        start = self.Length_Burnin
        end = self.Length_Burnin + self.Length_Test

        #リザバー層のXの共分散行列のランクを計算
        Omega = X[:, start : end].T
        Bias = np.ones((self.Length_Test,1))         #バイアス項
        Omega = np.concatenate([Omega, Bias], 1)            #バイアス項を追加
        
        
        #論文の定義通りに共分散行列を計算
        CovOmega = np.dot(Omega.T, Omega)
        Gamma = np.linalg.matrix_rank(CovOmega)
        

        #終了処理
        if self.F_OutputLog : print("+++ Storing Results +++")
        results = self.Param.copy()
        results.update({
            "CovMatrixRank_R_CovarianceMatrix" : CovOmega,
            "CovMatrixRank_R_CovMatrixRank" : Gamma,
            "CovMatrixRank_R_T" : T,
            "CovMatrixRank_R_U" : U,
            
            "Reservoir_X" : X,
            })
        
        outputs = self.Param.copy()
        outputs.update({
            "CovMatrixRank_R_CovarianceMatrix" : CovOmega,
            "CovMatrixRank_R_CovMatrixRank" : Gamma,
            })

        #作図出力
        self.Output(results)
        
        if self.F_OutputLog : print("*** Finished Evaluation NRMSE ***")

        return outputs


class Evaluation_DelayCapacity(Evaluation):
    """
    Dalay Capacity評価クラス
    """
    #コンストラクタ
    def __init__(self, param: dict, parent: any = None):
        super().__init__(param, parent)

        #パラメータ取得
        self.F_OutputLog = self.Param["DelayCapacity_F_OutputLog"]                  #経過の出力を行うか

        if self.F_OutputLog : print("*** Evaluation Delay Capacity ***")
        if self.F_OutputLog : print("+++ Initializing +++")

        self.D_u = self.Param["DelayCapacity_D_u"]                                  #入力信号次元
        self.D_x = self.Param["DelayCapacity_D_x"]                                  #リザバー層次元
        self.D_y = self.Param["DelayCapacity_D_y"]                                  #出力信号次元
        
        self.Length_Burnin = self.Param["DelayCapacity_Length_Burnin"]              #空走用データ時間長
        self.Length_Tdc = self.Param["DelayCapacity_Length_Tdc"]                  #評価用データ時間長
        self.Length_Taumax = self.Param["DelayCapacity_Length_Taumax"]                            #最大遅延量

        self.Length_Total = self.Length_Burnin + self.Length_Taumax + self.Length_Tdc    #全体データ長

        #評価用タスク（Type型）
        self.T_Task = self.Param["DelayCapacity_T_Task"]
        #タスクのインスタンス
        param = self.Param.copy()
        param.update({
            "Task_D_u" : self.D_u,
            "Task_D_y" : self.D_y,
            "Task_Length" : self.Length_Total,
            })
        self.Task = self.T_Task(param, self)
        
        #モデル（Type型）
        self.T_Model = self.Param["DelayCapacity_T_Model"]
        #モデルのインスタンス
        param = self.Param.copy()
        param.update({
            })
        self.Model = self.T_Model(param, self)

        #作図出力（Type型）
        self.T_Output = self.Param["DelayCapacity_T_Output"]
        #作図出力のインスタンス
        param = self.Param.copy()
        param.update({
            })
        self.Output = self.T_Output(param, self)
        
    #本体
    def __call__(self) -> dict:
        if self.F_OutputLog : print("*** Started Evaluation Delay Capacity ***")

        #エコー収集
        if self.F_OutputLog : print("+++ Collecting Echo +++")
        
        #バッチ
        T = list(range(self.Length_Total))                      #全体時間
        U = [None for _ in range(self.Length_Total)]            #入力信号
        Y = [None for _ in range(self.Length_Total)]            #出力層の信号

        TimeDC = list(range(self.Length_Taumax))        
        
        RS_X = np.array([None for _ in range(self.Length_Total * self.D_x)], dtype=float).reshape(-1, self.Length_Total)      #リザバー層のX
        CovTrace = np.zeros(self.Length_Taumax)

        #モデルリセット
        self.Model.reset()
        
        #Burn-in
        if self.F_OutputLog : print("--- Burn-in Process---")
        for i, t in enumerate(T[0 : self.Length_Burnin]):
            if self.F_OutputLog : print("\r%d / %d"%(i, self.Length_Burnin), end = "")
            U[t], _ = self.Task.getData(t)
            Y[t], _, _, _, RS_X[:, t], _, _ = self.Model.forwardReservoir(U[t])
        if self.F_OutputLog : print("\n", end = "")

        #データ収集
        if self.F_OutputLog : print("+++ Collecting Data for Tau_max +++")        
        for i, t in enumerate(T[self.Length_Burnin : self.Length_Burnin + self.Length_Taumax]):
            if self.F_OutputLog : print("\r%d / %d"%(i, self.Length_Taumax), end = "")
            U[t], _ = self.Task.getData(t)
            Y[t], _, _, _, RS_X[:, t], _, _ = self.Model.forwardReservoir(U[t])
        if self.F_OutputLog : print("\n", end = "")
        
        if self.F_OutputLog : print("+++ Collecting Data for T_dc +++")        
        for i, t in enumerate(T[self.Length_Burnin + self.Length_Taumax : self.Length_Burnin + self.Length_Taumax + self.Length_Tdc]):
            if self.F_OutputLog : print("\r%d / %d"%(i, self.Length_Tdc), end = "")
            U[t], _ = self.Task.getData(t)
            Y[t], _, _, _, RS_X[:, t], _, _ = self.Model.forwardReservoir(U[t])
        if self.F_OutputLog : print("\n", end = "")

        #指標計算
        if self.F_OutputLog : print("--- Calculating for Tau ---")

        #空走時間とTau_max
        start = self.Length_Burnin 
        #T_dcの時間だけ行列収集
        end = self.Length_Burnin + self.Length_Tdc - self.Length_Taumax

        #行列の白色化関数
        def Whitening_Matrix(X):
            #行列サイズ
            Ti, Ne = X.shape

            #各ニューロンの時間平均ベクトル
            #X_ave: [Time * Neurons]

            #平均化したリザバー状態ベクトル
            X_Cent = X - np.mean(X, axis=0)

            #共分散行列
            CovX = ((np.dot(X_Cent.T, X_Cent)) / (Ti)) + np.eye(Ne) * 1e-10
            
            #特異値分解
            Uw, Sw, Vw = np.linalg.svd(CovX)
            Sigma = np.diag(1 / np.sqrt(Sw))

            #白色化行列１
            #多分こっちが合ってる、はず
            White_X0L = np.dot(X_Cent, Vw.T)
            White_X0 = np.dot(White_X0L, Sigma)
            
            #白色化行列２
            """
            White_X0 = np.dot(np.dot(Sigma, Vw), X_Cent.T)
            """
            return White_X0

        
        for tau in range(1, self.Length_Taumax + 1):
            if self.F_OutputLog :print("\r%d / %d"%(tau, self.Length_Taumax), end = "")

            #リザバー層のXの状態ベクトル（(T_dc - Taumax)×ニューロン数）の転置
            RS_X_Standard = RS_X[:, start : end].T

            #Tauステップだけ遅らせたリザバー層のXの状態ベクトル（(T_dc - Taumax)×ニューロン数）の転置
            RS_X_Delay = RS_X[:, start - tau : end - tau].T
            
            #基準時間の白色化行列
            X0_Standard = Whitening_Matrix(RS_X_Standard)
            #遅延ありの白色化行列
            X0_Delay = Whitening_Matrix(RS_X_Delay)

            #共分散行列の計算１
            C_tau = np.dot(X0_Standard.T, X0_Delay) / self.Length_Tdc
            
            #共分散行列の計算２
            #C_tau = np.dot(X0_Standard, X0_Delay.T) / self.Length_Tdc
            
            #print(f"共分散行列のサイズ：{C_tau.shape}")
            CovTrace[tau-1] = np.trace(np.abs(C_tau))
            #print(f"DC: {CovTrace[tau-1]}")
        if self.F_OutputLog : print("\n", end = "")

        DelayCapacity = np.mean(CovTrace)

        #終了処理
        if self.F_OutputLog : print("+++ Storing Results +++")
        results = self.Param.copy()
        results.update({
            "DelayCapacity_R_DelayCapacity" : DelayCapacity,

            "DelayCapacity_R_DelayCapacity_Taumax" : TimeDC,
            "DelayCapacity_R_DelayCapacity_Time" : CovTrace,
            "DelayCapacity_R_T" : T,
            "DelayCapacity_R_U" : U,
            "DelayCapacity_R_Y" : Y,            
            })
        
        outputs = self.Param.copy()
        outputs.update({
            "DelayCapacity_R_DelayCapacity" : DelayCapacity,
            "DelayCapacity_R_DelayCapacity_Time" : CovTrace,
            })

        #作図出力
        self.Output(results)
        
        if self.F_OutputLog : print("*** Finished Evaluation NRMSE ***")

        return outputs
