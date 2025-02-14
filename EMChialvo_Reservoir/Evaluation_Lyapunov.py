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
        #modelの全サブリザバーを取得
        L_sub = self.Model.getSubReservoirs()
        L_PathName = [sub.PathName for sub in L_sub]
        L_PathShowName = [sub.PathShowName for sub in L_sub]
        
        #copy生成
        L_Model_e = [self.Model.clone() for _ in L_sub]
        L_sub_e = [model_e.getSubReservoirs()[i] for i, model_e in enumerate(L_Model_e)]
            
        #摂動追加
        Gamma_e_0 = self.Epsilon
        for sub_e in L_sub_e:
            e = np.ones(sub_e.s.shape)
            e = e / np.linalg.norm(e, ord = 2) * self.Epsilon
            sub_e.s_old += e
        
        if self.F_OutputLog : print("+++ Evaluating Model +++")
        #時間発展させ評価
        for i, t in enumerate(T[self.Length_Burnin : self.Length_Burnin + self.Length_Test]):
            if self.F_OutputLog : print("\r%d / %d"%(i, self.Length_Test), end = "")
            U[t], _ = self.Task.getData(t)
            
            #時間発展
            self.Model.forwardReservoir(U[t])
            for model_e in L_Model_e:
                model_e.forwardReservoir(U[t])
            
            #瞬時最大リアプノフと次のGamma計算
            MMLt = np.zeros([len(L_sub)])#瞬時リアプノフ
            for i, sub_e in enumerate(L_sub_e):
                Gamma_e_t = np.linalg.norm(sub_e.s_old - L_sub[i].s_old, ord = 2) 
                sub_e.s_old = L_sub[i].s_old + (Gamma_e_0 / (Gamma_e_t + 1e-14)) * (sub_e.s_old - L_sub[i].s_old)#（発散防止）
                MMLt[i] = Gamma_e_t / Gamma_e_0#瞬時リアプノフ
            MMLE[t] = np.log(MMLt + 1e-14)#瞬時リアプノフ指数（発散防止）
                
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
            "MLE_R_PathName" : L_PathName,
            "MLE_R_PathShowName" : L_PathShowName
            })
        
        #作図出力
        self.Output(results)
        
        if self.F_OutputLog : print("*** Finished Evaluation Max-Lyapunov-Exponent ***")

        return results
    
#--------------------------------------------------------------------
class Evaluation_LS(Evaluation):
    """
    Lyapunov Spectrum評価クラス
    """
    #コンストラクタ
    def __init__(self, param: dict, parent: any = None):
        super().__init__(param, parent)

        #パラメータ取得
        self.F_OutputLog = self.Param["LS_F_OutputLog"]            #経過の出力を行うか

        if self.F_OutputLog : print("*** Evaluation Lyapunov-Spectrum ***")
        if self.F_OutputLog : print("+++ Initializing +++")

        self.D_u = self.Param["LS_D_u"]                            #入力信号次元
        self.D_y = self.Param["LS_D_y"]                            #出力信号次元
        self.Length_Burnin = self.Param["LS_Length_Burnin"]        #空走用データ時間長
        self.Length_Test = self.Param["LS_Length_Test"]            #評価用データ時間長
        self.Length_Total = self.Length_Burnin + self.Length_Test   #全体データ時間長
        self.Epsilon = self.Param["LS_Epsilon"]                    #摂動ε

        #評価用タスク（Type型）
        self.T_Task = self.Param["LS_T_Task"]
        #タスクのインスタンス
        param = self.Param.copy()
        param.update({
            "Task_D_u" : self.D_u,
            "Task_D_y" : self.D_y,
            "Task_Length" : self.Length_Total,
            })
        self.Task = self.T_Task(param, self)
        
        #モデル（Type型）
        self.T_Model = self.Param["LS_T_Model"]
        #モデルのインスタンス
        param = self.Param.copy()
        param.update({
            })
        self.Model = self.T_Model(param, self)

        #作図出力（Type型）
        self.T_Output = self.Param["LS_T_Output"]
        #作図出力のインスタンス
        param = self.Param.copy()
        param.update({
            })
        self.Output = self.T_Output(param, self)
        
    #本体
    def __call__(self) -> dict:
        if self.F_OutputLog : print("*** Started Evaluation Lyapunov-Spectrum ***")

        #エコー収集
        if self.F_OutputLog : print("+++ Collecting Echo +++")
        
        #バッチ
        T = list(range(self.Length_Total))
        U = [None for _ in range(self.Length_Total)]

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
        #modelの全サブリザバーを取得
        L_sub = self.Model.getSubReservoirs()
        L_PathName = [sub.PathName for sub in L_sub]
        L_PathShowName = [sub.PathShowName for sub in L_sub]
        
        #copy生成（モデルを各サブリザバー各ニューロンごとに複製）
        L_Model_e_Pert_Gram = [[self.Model.clone() for _ in range(sub.D_x)] for sub in L_sub]
        L_sub_e_Pert_Gram = [[model_e[j].getSubReservoirs()[i] for j in range(len(model_e))] for i, model_e in enumerate(L_Model_e_Pert_Gram)]
        MLS_Pert_Gram = [None for _ in range(self.Length_Total)]#瞬時リアプノフスペクトル
        
        #Debug
        L_Model_e_Pert_QR = [[self.Model.clone() for _ in range(sub.D_x)] for sub in L_sub]
        L_sub_e_Pert_QR = [[model_e[j].getSubReservoirs()[i] for j in range(len(model_e))] for i, model_e in enumerate(L_Model_e_Pert_QR)]
        MLS_Pert_QR = [None for _ in range(self.Length_Total)]#瞬時リアプノフスペクトル
        
        #Debug
        Q = [np.eye(sub.D_x) for sub in L_sub]
        MLS_Math_QR = [None for _ in range(self.Length_Total)]#瞬時リアプノフスペクトル
            
        #Debug
        L_Model_e_Pert_MLE = [self.Model.clone() for _ in L_sub]
        L_sub_e_Pert_MLE = [model_e.getSubReservoirs()[i] for i, model_e in enumerate(L_Model_e_Pert_MLE)]
        MLS_Pert_MLE = [None for _ in range(self.Length_Total)]#瞬時最大リアプノフ指数
            
        #摂動追加
        for sub_e in L_sub_e_Pert_Gram:
            e = np.eye(sub_e[0].D_x)
            e = e / np.linalg.norm(e, axis = 1, ord = 2) * self.Epsilon
            for i in range(sub_e[0].D_x):
                sub_e[i].s_old += e[i]
                
        #Debug
        for sub_e in L_sub_e_Pert_QR:
            e = np.eye(sub_e[0].D_x)
            e = e / np.linalg.norm(e, axis = 1, ord = 2) * self.Epsilon
            for i in range(sub_e[0].D_x):
                sub_e[i].s_old += e[i]
        
        #Debug
        for sub_e in L_sub_e_Pert_MLE:
            e = np.ones(sub_e.s.shape)
            e = e / np.linalg.norm(e, ord = 2) * self.Epsilon
            sub_e.s_old += e
        
        if self.F_OutputLog : print("+++ Evaluating Model +++")
        #時間発展させ評価
        for i, t in enumerate(T[self.Length_Burnin : self.Length_Burnin + self.Length_Test]):
            if self.F_OutputLog : print("\r%d / %d"%(i, self.Length_Test), end = "")
            U[t], _ = self.Task.getData(t)
            
            #時間発展
            #Debug
            Js = [sub.Jacobian(U[t]) for sub in L_sub]

            self.Model.forwardReservoir(U[t])

            for model_e in L_Model_e_Pert_Gram:
                for j in range(model_e[0].D_x):
                    model_e[j].forwardReservoir(U[t])
                    
            #Debug
            for model_e in L_Model_e_Pert_QR:
                for j in range(model_e[0].D_x):
                    model_e[j].forwardReservoir(U[t])
            
            #Debug
            for model_e in L_Model_e_Pert_MLE:
                model_e.forwardReservoir(U[t])
            
            #瞬時リアプノフスペクトルと次のGamma計算
            MLS_Pert_Gram[t] = [None for _ in range(len(L_sub))]#瞬時リアプノフ指数
            MLS_Pert_QR[t] = [None for _ in range(len(L_sub))]#瞬時リアプノフ指数
            MLS_Math_QR[t] = [None for _ in range(len(L_sub))]#瞬時リアプノフ指数
            MLS_Pert_MLE[t] = [None for _ in range(len(L_sub))]#瞬時リアプノフ指数
            for i in range(len(L_sub)):

                # 摂動ベース（グラムシュミット）
                d_tau = np.array([L_sub_e_Pert_Gram[i][j].s_old - L_sub[i].s_old for j in range(L_sub[i].D_x)])
                d_Orthogonal = d_tau.copy()
                norm_d_Orthogonal = np.zeros((L_sub[i].D_x))
                for k in range(L_sub[i].D_x):
                    for l in range(k):
                        d_Orthogonal[k] -= np.dot(d_tau[k], d_Orthogonal[l]) / (norm_d_Orthogonal[l]**2 + 1e-20) * d_Orthogonal[l]
                    norm_d_Orthogonal[k] = np.linalg.norm(d_Orthogonal[k], ord = 2)
                    L_sub_e_Pert_Gram[i][k].s_old = L_sub[i].s_old + self.Epsilon / norm_d_Orthogonal[k] * d_Orthogonal[k]
                
                MLS_Pert_Gram[t][i] = np.log(norm_d_Orthogonal / self.Epsilon + 10e-20)#瞬時リアプノフスペクトル（発散防止）

                
                #Debug
                # 摂動ベース（QR分解）
                d_tau = np.array([L_sub_e_Pert_QR[i][j].s_old - L_sub[i].s_old for j in range(L_sub[i].D_x)])
                pQ, pR = np.linalg.qr(d_tau.T)
                for k in range(L_sub[i].D_x):
                    L_sub_e_Pert_QR[i][k].s_old = L_sub[i].s_old + self.Epsilon * pQ.T[k]
                MLS_Pert_QR[t][i] = np.log(np.abs(np.diag(pR)) / self.Epsilon + 1e-20)

                
                #Debug
                # 数学ベース（QR分解）
                Q[i] = Js[i] @ Q[i]
                Q[i], R = np.linalg.qr(Q[i])
                MLS_Math_QR[t][i] = np.log(np.abs(np.diag(R)) + 1e-20)
                
                
                #Debug
                # 摂動ベース（最大リアプノフ指数）
                Gamma_e_t = np.linalg.norm(L_sub_e_Pert_MLE[i].s_old - L_sub[i].s_old, ord = 2) 
                sub_e.s_old = L_sub[i].s_old + (self.Epsilon / (Gamma_e_t + 1e-20)) * (L_sub_e_Pert_MLE[i].s_old - L_sub[i].s_old)#（発散防止）
                MLS_Pert_MLE[t][i] = np.log(Gamma_e_t / self.Epsilon + 1e-20)#瞬時リアプノフ
                

        if self.F_OutputLog : print("\n", end = "")
        
        #指標計算
        if self.F_OutputLog : print("--- Calculating ---")
        start = self.Length_Burnin
        end = self.Length_Burnin + self.Length_Test

        #結果整理
        array_MLS_Pert_Gram = [None for _ in range(len(L_sub))]
        LS_Pert_Gram = [None for _ in range(len(L_sub))]
        for i in range(len(L_sub)):
            array_MLS_Pert_Gram[i] = np.array([MLS_t[i] for MLS_t in MLS_Pert_Gram[start : end]])
            LS_Pert_Gram[i] = np.mean(array_MLS_Pert_Gram[i], axis = 0)#リアプノフスペクトル
            
        #Debug
        array_MLS_Pert_QR = [None for _ in range(len(L_sub))]
        LS_Pert_QR = [None for _ in range(len(L_sub))]
        for i in range(len(L_sub)):
            array_MLS_Pert_QR[i] = np.array([MLS_t[i] for MLS_t in MLS_Pert_QR[start : end]])
            LS_Pert_QR[i] = np.mean(array_MLS_Pert_QR[i], axis = 0)#リアプノフスペクトル
            
        #Debug
        array_MLS_Math_QR = [None for _ in range(len(L_sub))]
        LS_Math_QR = [None for _ in range(len(L_sub))]
        for i in range(len(L_sub)):
            array_MLS_Math_QR[i] = np.array([MLS_t[i] for MLS_t in MLS_Math_QR[start : end]])
            LS_Math_QR[i] = np.mean(array_MLS_Math_QR[i], axis = 0)#リアプノフスペクトル
        
        #Debug
        array_MLS_Pert_MLE = [None for _ in range(len(L_sub))]
        LS_Pert_MLE = [None for _ in range(len(L_sub))]
        for i in range(len(L_sub)):
            array_MLS_Pert_MLE[i] = np.array([MLS_t[i] for MLS_t in MLS_Pert_MLE[start : end]])
            LS_Pert_MLE[i] = np.mean(array_MLS_Pert_MLE[i], axis = 0)#リアプノフスペクトル
        
        #Debug
        print("< MLE Pert Gram >")
        for i in range(len(LS_Pert_Gram)):
            print(L_PathShowName[i] + 
                    "(" + L_PathName[i] + ") : " + 
                    str(LS_Pert_Gram[i]))
                
        #Debug
        print("< MLE Pert QR >")
        for i in range(len(LS_Pert_QR)):
            print(L_PathShowName[i] + 
                    "(" + L_PathName[i] + ") : " + 
                    str(LS_Pert_QR[i]))
            
        #Debug
        print("< MLE Math QR >")
        for i in range(len(LS_Math_QR)):
            print(L_PathShowName[i] + 
                    "(" + L_PathName[i] + ") : " + 
                    str(LS_Math_QR[i]))
            
        #Debug
        print("< MLE Pert MLE >")
        for i in range(len(LS_Pert_MLE)):
            print(L_PathShowName[i] + 
                    "(" + L_PathName[i] + ") : " + 
                    str(LS_Pert_MLE[i]))

        #終了処理
        if self.F_OutputLog : print("+++ Storing Results +++")
        results = self.Param.copy()
        results.update({
            "LS_R_LS" : LS_Pert_Gram,
            "LS_R_T" : T,
            "LS_R_U" : U,
            "LS_R_MLS" : array_MLS_Pert_Gram,
            "LS_R_PathName" : L_PathName,
            "LS_R_PathShowName" : L_PathShowName
            })
        
        #作図出力
        self.Output(results)
        
        if self.F_OutputLog : print("*** Finished Evaluation Lyapunov-Spectrum ***")

        return results
