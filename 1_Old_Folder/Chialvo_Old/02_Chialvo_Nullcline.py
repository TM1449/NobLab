import datetime

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.integrate as integrate
from mpl_toolkits.mplot3d import Axes3D


Param = {
    "Chialvo_a"                 : 0.89,
    "Chialvo_b"                 : 0.6,
    "Chialvo_c"                 : 0.28,
    "Chialvo_k0"                : 0.04,

    "Chialvo_k"                 : 3.2,
    "Chialvo_k1"                : 0.1,
    "Chialvo_k2"                : 0.2,
    "Chialvo_alpha"             : 0.1,
    "Chialvo_beta"              : 0.2,

    "Chialvo_RunTime"           : 1500,
    "Chialvo_PltStart"          : 0,
    "Chialvo_PltEnd"            : 1500,
    #===============================================================================================================
    
    "Plot_X"                    : False,
    "Plot_Y"                    : False,
    "Plot_Phi"                  : False,
    
    #===============================================================================================================

    "Plot_XandY"                : False,
    "Plot_old_Nullcline"        : False,
    
    #===============================================================================================================
    "Plot_new_X_Nullcline"      : True,
    "Plot_new_Y_Nullcline"      : False,
    "Plot_new_Phi_Nullcline"    : False,
}

class Chialvo:
    def __init__(self, param: dict):
        self.Param = param

    def __call__(self) -> dict: pass

class Chialvo_Map(Chialvo):
    def __init__(self, param: dict):
        super().__init__(param)

        #従来Chialvo変数
        self.a = self.Param["Chialvo_a"]
        self.b = self.Param["Chialvo_b"]
        self.c = self.Param["Chialvo_c"]
        self.k0 = self.Param["Chialvo_k0"]

        #電磁束下におけるChialvo変数
        self.k = self.Param["Chialvo_k"]
        self.k1 = self.Param["Chialvo_k1"]
        self.k2 = self.Param["Chialvo_k2"]
        self.alpha = self.Param["Chialvo_alpha"]
        self.beta = self.Param["Chialvo_beta"]

        #実行時間
        self.Run_Time = self.Param["Chialvo_RunTime"]
        self.plt_startTime = self.Param["Chialvo_PltStart"]
        self.plt_EndTime = self.Param["Chialvo_PltEnd"]

        self.Plot_X = self.Param["Plot_X"]
        self.Plot_XandY = self.Param["Plot_XandY"]
        self.Plot_old_Nullcline = self.Param["Plot_old_Nullcline"]
        
        self.Plot_new_X_Nullcline = self.Param["Plot_new_X_Nullcline"]
        self.Plot_new_Y_Nullcline = self.Param["Plot_new_Y_Nullcline"]
        self.Plot_new_Phi_Nullcline = self.Param["Plot_new_Phi_Nullcline"]
        
    def __call__(self):

        FigSize = (16,9)
        FontSize_Label = 48                 #ラベルのフォントサイズ
        FontSize_Title = 0                 #タイトルのフォントサイズ
        LineWidth = 2                       #線の太さ
        FileFormat = ".png"#".pdf"          #ファイルフォーマット

        if self.Plot_X:
            self.x = np.random.uniform(-1,1,self.Run_Time)
            self.y = np.random.uniform(-1,1,self.Run_Time)
            self.phi = np.random.uniform(-1,1,self.Run_Time)

            for n in range(self.Run_Time-1):
                self.x[n+1] = self.x[n] ** 2 * np.exp(self.y[n] - self.x[n]) + self.k0 \
                    + self.k * self.x[n] * (self.alpha + 3 * self.beta * self.phi[n] ** 2)
                self.y[n+1] = self.a * self.y[n] - self.b * self.x[n] + self.c
                self.phi[n+1] = self.k1 * self.x[n] - self.k2 * self.phi[n]

            print("Display of x values by time")
            fig = plt.figure(figsize = FigSize)
            ax = fig.add_subplot(1,1,1)

            Title = None
            plt.tick_params(labelsize=48)
            ax.set_title(Title, fontsize = FontSize_Title)
            ax.set_xlabel("Time Step", fontsize = FontSize_Label)
            ax.set_ylabel("x", fontsize = FontSize_Label)

            ax.plot(self.x[self.plt_startTime:self.plt_EndTime],'-',lw = LineWidth)
            fig.tight_layout()
            plt.show()

        if self.Plot_XandY:
            self.x = np.random.uniform(-0.01,0.01,self.Run_Time)
            self.y = np.random.uniform(-0.01,0.01,self.Run_Time)
            self.phi = np.random.uniform(-1,1,self.Run_Time)

            for n in range(self.Run_Time-1):
                self.x[n+1] = self.x[n] ** 2 * np.exp(self.y[n] - self.x[n]) + self.k0 \
                    + self.k * self.x[n] * (self.alpha + 3 * self.beta * self.phi[n] ** 2)
                self.y[n+1] = self.a * self.y[n] - self.b * self.x[n] + self.c
                self.phi[n+1] = self.k1 * self.x[n] - self.k2 * self.phi[n]
            
            print("Display of x and y phase space at different times")
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)

            plt.plot(self.x[self.plt_startTime:self.plt_EndTime],self.y[self.plt_startTime:self.plt_EndTime])
            plt.show()

        if self.Plot_old_Nullcline:
            print("old_Nullcline")
            """
            psはプロット範囲の開始地点
            peはプロット範囲の終了地点
            """
            self.ps = -3
            self.pe = 3

            """
            Et, Ex, Eyは描写用の変数.
            """
            self.Et = 0.1
            
            self.Ex = np.arange(self.ps, self.pe, self.Et)
            self.Ey = np.arange(self.ps, self.pe, self.Et)

            self.X, self.Y = np.meshgrid(self.Ex, self.Ey)

            self.Ex = self.X ** 2 * np.exp(self.Y - self.X) + self.k0 - self.X
            self.Ey = self.a * self.Y - self.b * self.X + self.c - self.Y
            
            """
            dt, dx, dyは計算用の変数.
            """
            self.dt = 0.0001
            self.dx = np.arange(self.ps, self.pe, self.dt)
            self.dy = np.arange(self.ps, self.pe, self.dt)

            self.fx = np.log(self.dx - self.k0) - 2 * np.log(self.dx) + self.dx
            self.fy = (self.a * self.dy + self.c - self.dy) / self.b

            self.x = np.random.uniform(0.045,0.055,self.Run_Time)
            self.y = np.random.uniform(2.25,2.35,self.Run_Time)

            for n in range(self.Run_Time-1):
                self.x[n+1] = self.x[n] ** 2 * np.exp(self.y[n] - self.x[n]) + self.k0
                self.y[n+1] = self.a * self.y[n] - self.b * self.x[n] + self.c

            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)

            ax.plot(self.dx, self.fx)
            ax.plot(self.fy, self.dx)
            ax.plot(self.x[self.plt_startTime:self.plt_EndTime], self.y[self.plt_startTime:self.plt_EndTime])
            ax.set_xlim(-3,3)
            ax.set_ylim(-3,3)

            ax.quiver(self.X, self.Y, self.Ex / np.sqrt(pow(self.Ex, 2) + pow(self.Ey, 2)), self.Ey / np.sqrt(pow(self.Ex, 2) + pow(self.Ey, 2)))
            #ax.streamplot(self.X, self.Y, self.Ey, self.Ex)
            plt.grid()
            plt.show()

        if self.Plot_new_X_Nullcline:
            print("new_X_Nullcline")
            #===============================================================================================================
            """
            psはプロット範囲の開始地点
            peはプロット範囲の終了地点
            """
            self.ps = -0.6
            self.pe = 1.6

            #===============================================================================================================
            """
            Et, Ex, Eyは描写用の変数.
            """
            self.Edt = 0.1
            
            self.Edx = np.arange(self.ps, self.pe, self.Edt)
            self.Edy = np.arange(self.ps, self.pe, self.Edt)

            self.X, self.Y = np.meshgrid(self.Edx, self.Edy)
            
            self.Ex = pow(self.X, 2) * np.exp(((self.b - self.a + 1) * self.X - self.c) / (self.a - 1)) + self.k0 \
                + ((3 * self.k * self.beta * pow(self.k1, 2)) / pow(1 + self.k2, 2)) * pow(self.X, 3)\
                    + self.X * self.k * self.alpha - self.X
            
            self.Ey = self.X - self.Y

            #===============================================================================================================
            """
            dt, dx, dyは計算用の変数.
            """
            self.dt = 0.00001

            self.dx = np.arange(self.ps, self.pe, self.dt)
            self.dy = np.arange(self.ps, self.pe, self.dt)

            self.fx = self.dx ** 2 * np.exp(((self.b - self.a + 1) * self.dx - self.c) / (self.a - 1)) + self.k0 \
                + ((3 * self.k * self.beta * self.k1 ** 2) / (1 + self.k2) ** 2) * self.dx ** 3 \
                    + self.dx * self.k * self.alpha
            
            self.fy = self.dx

            #===============================================================================================================

            self.x = np.random.uniform(-1,1,self.Run_Time)
            self.y = np.random.uniform(-1,1,self.Run_Time)
            self.phi = np.random.uniform(-1,1,self.Run_Time)

            for n in range(self.Run_Time-1):
                self.x[n+1] = self.x[n] ** 2 * np.exp(self.y[n] - self.x[n]) + self.k0 \
                    + self.k * self.x[n] * (self.alpha + 3 * self.beta * self.phi[n] ** 2)
                self.y[n+1] = self.a * self.y[n] - self.b * self.x[n] + self.c
                self.phi[n+1] = self.k1 * self.x[n] - self.k2 * self.phi[n]

            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)

            ax.plot(self.dx, self.fx)
            ax.plot(self.dx, self.fy)
            ax.set_xlim(self.ps,self.pe)
            ax.set_ylim(self.ps,self.pe)

            ax.quiver(self.X, self.Y, self.Ex / np.sqrt(pow(self.Ex, 2) + pow(self.Ey, 2)), self.Ey / np.sqrt(pow(self.Ex, 2) + pow(self.Ey, 2)))
            #ax.plot(self.x[self.plt_startTime:self.plt_EndTime],self.y[self.plt_startTime:self.plt_EndTime])
            #ax.streamplot(self.X, self.Y, self.Ey, self.Ex)
            plt.grid()
            plt.show()
        
        if self.Plot_new_Y_Nullcline:
            print("new_Y_Nullcline")
            #===============================================================================================================
            """
            psはプロット範囲の開始地点
            peはプロット範囲の終了地点
            """
            self.ps = -10
            self.pe = 30

            #===============================================================================================================
            """
            Et, Ex, Eyは描写用の変数.
            """
            self.Edt = 0.5
            
            self.Edx = np.arange(self.ps, self.pe, self.Edt)
            self.Edy = np.arange(self.ps, self.pe, self.Edt)

            self.X, self.Y = np.meshgrid(self.Edx, self.Edy)
            
            self.Ex = self.X ** 2 * np.log(self.Y - self.X) + self.k0 + self.k * self.X * (self.alpha + 3 * self.beta * pow((self.k1 * self.X) / (self.k2 + 1), 2)) - self.X
            self.Ey = self.a * self.Y - self.b * self.X + self.c - self.Y

            #===============================================================================================================
            """
            dt, dx, dyは計算用の変数.
            """
            self.dt = 0.001

            self.dx = np.arange(self.ps, self.pe, self.dt)
            self.dy = np.arange(self.ps, self.pe, self.dt)

            self.fx = self.dx ** 2 * np.log(self.dy - self.dx) + self.k0 + self.k * self.dx * (self.alpha + 3 * self.beta * pow((self.k1 * self.dx) / (self.k2 + 1), 2))
            self.fy = self.a * self.dy - self.b * self.dx + self.c
            #===============================================================================================================
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)

            ax.plot(self.dx, self.fy)
            ax.plot(self.fx, self.dy)
            ax.set_xlim(-3,20)
            ax.set_ylim(-3,20)

            plt.quiver(self.X, self.Y, self.Ex / np.sqrt(pow(self.Ex, 2) + pow(self.Ey, 2)), self.Ey / np.sqrt(pow(self.Ex, 2) + pow(self.Ey, 2)), cmap = 'Blues')
            #ax.streamplot(self.X, self.Y, self.Ey, self.Ex)
            plt.grid()
            plt.show()

        if self.Plot_new_Phi_Nullcline:
            print("new_Phi_Nullcline")
            #===============================================================================================================
            """
            psはプロット範囲の開始地点
            peはプロット範囲の終了地点
            """
            self.ps = -10
            self.pe = 30

            #===============================================================================================================
            """
            Et, Ex, Eyは描写用の変数.
            """
            self.Edt = 0.5
            
            self.Edx = np.arange(self.ps, self.pe, self.Edt)
            self.Edphi = np.arange(self.ps, self.pe, self.Edt)

            self.X, self.Phi = np.meshgrid(self.Edx, self.Edphi)

            self.Ex = (self.Phi * (1 + self.k2)) / self.k1
            self.Ephi = np.sqrt((1 / (3 * self.beta)) * ((1 / (self.k * self.X)) * \
                (self.X - self.X ** 2 * np.exp(((self.b - self.a + 1) * self.X - self.c) / (self.a - 1)) - self.k0) - self.alpha))
            #===============================================================================================================
            """
            dt, dx, dyは計算用の変数.
            """
            self.dt = 0.001

            self.dx = np.arange(self.ps, self.pe, self.dt)
            self.dphi = np.arange(self.ps, self.pe, self.dt)
            
            self.fx = (self.dphi * (1 + self.k2)) / self.k1
            self.fphi = np.sqrt((1 / (3 * self.beta)) * ((1 / (self.k * self.dx)) * \
                (self.dx - self.dx ** 2 * np.exp(((self.b - self.a + 1) * self.dx - self.c) / (self.a - 1)) - self.k0) - self.alpha))
            
            self.ffphi = -1*np.sqrt((1 / (3 * self.beta)) * ((1 / (self.k * self.dx)) * \
                (self.dx - self.dx ** 2 * np.exp(((self.b - self.a + 1) * self.dx - self.c) / (self.a - 1)) - self.k0) - self.alpha))
            #===============================================================================================================
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)

            
            #ax.plot(self.dx, self.ffphi)
            ax.plot(self.dx, self.fphi)
            ax.plot(self.fx, self.dphi)
            ax.set_xlim(-3,20)
            ax.set_ylim(-3,20)

            plt.quiver(self.X, self.Phi, self.Ex / np.sqrt(pow(self.Ex, 2) + pow(self.Ephi, 2)), self.Ephi / np.sqrt(pow(self.Ex, 2) + pow(self.Ephi, 2)))
            #plt.streamplot(self.X, self.Phi, self.Ephi, self.Ex)
            plt.grid()
            plt.show()

print("###Main Program has started###")
Chialvo_Map(Param)()
print("###All Processes have finished###")