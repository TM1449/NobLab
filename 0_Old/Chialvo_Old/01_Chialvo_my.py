import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D


#使用する変数，配列の定義
a = 0.89
b = 0.6
c = 0.28
k0 = 0.04
k1 = 0.1
k2 = 0.2
alph = 0.1
beta = 0.2

k = 0
N = 100
mu = 0.005
R = 10
sigma = 0.00724
t = 20000  #このtは、論文でいうn

Rho = 1

class ChialvoMap:
    """
    Chiavlo写像の作成
    """
    def __init__(self, a, b, c, k0, k, k1 ,k2 ,alph ,beta ,N ,mu ,R ,sigma ,t):
        """
        モデルの初期化を行う
        """
        self.a = a #回復の時定数
        self.b = b #回復過程の活性化依存数
        self.c = c #オフセット
        self.k0 = k0 #時間に依存しないパラメータ
        self.k = k #電磁束の結合強度

        #以下のパラメータは電磁束パラメータ
        self.k1 = k1
        self.k2 = k2
        self.alph = alph
        self.beta = beta

        self.N = N #ネットワークのサイズ
        self.mu = mu #ニューロンとスターネットワークの中心ニューロン間の結合強度
        self.R = R #結合数
        self.sigma = sigma #リングスターネットワークのニューロンの結合強度
        self.t = t #時間ステップ

    def run(self):
        #self.t = t は、7行目あたりで指定した変数を持ってくる
        self.t = t

        #self.x,y,phiは、それぞれ、ニューロン数の行数、時間ステップ分の列数を用意した
        self.x = np.zeros((N,t))
        self.y = np.zeros((N,t))
        self.phi = np.zeros((N,t))

        #初期値の変更，シード値を設定
        np.random.seed(None)  # シード値は任意の整数
    
        #乱数を生成（0列目の全要素に代入）
        for m in range(0,N):
            self.x[m, 0] = np.random.uniform(-1,1)
            self.y[m, 0] = 1
            self.phi[m, 0] = 1
            
        #初期値確認
        """print(self.x[:,0])
        print(self.y[:,0])
        print(self.phi[:,0])"""

        """-----------------------------------------------------"""
        def M(x):
            return self.alph + 3 * self.beta * x ** 2
        
        """-----------------------------------------------------"""
        #self.ring_matrixは、リングネットワークの隣接行列を生成する
        self.ring_matrix = np.identity(N)
        self.ring_matrix[0,0] = 0
        
        for i in range(1,N):
            #これが対角成分周辺の1を生成する
            self.ring_matrix[i:i+R+1,i:i+R+1] = 1
            #これらが範囲外でも対応させる部分（左下や右上の1を生成する）
            self.ring_matrix[N-R+i-1:, i] = 1
            self.ring_matrix[i,N-R+i-1:] = 1
        for i in range(1,N):
            self.ring_matrix[i,i] = -2 * R

        self.ring_matrix *= (self.sigma / (2 * R))
        """-----------------------------------------------------"""    
        #self.star_matrixは、スターネットワークの隣接行列を生成する
        self.star_matrix = np.zeros((N,N))

        self.star_matrix[0,0] = -self.mu * (N-1)
        self.star_matrix[0,1:] = self.mu
        self.star_matrix[1:,0] = -self.mu

        for i in range(1,N):
            self.star_matrix[i,i] += self.mu
        """-----------------------------------------------------"""
        #self.Ring_Star_matrixは、リングスターネットワークの隣接行列を生成する（ただ足し合わせた）
        self.Ring_Star_matrix = Rho *(self.ring_matrix + self.star_matrix)

        """W = np.ones((N,N)) * np.random.uniform(-1,1,(N,N))

        w , v = np.linalg.eig(W)

        self.Ring_Star_matrix = Rho * W / np.amax(w.real)"""
        """-----------------------------------------------------"""

        """print("リングネットワークは")
        print(self.ring_matrix)
        print("スターネットワークは")
        print(self.star_matrix)
        print("リングスターネットワークは")
        print(self.Ring_Star_matrix)"""

        """-----------------------------------------------------"""

        #for文中にある「 \ 」は改行の意図。除法の意図ではない
            
        #これは改良版
        #最初に全てのニューロンの計算を行う形にし、その後、中心ニューロンを改めて計算し、上書きする形にした。
        #その場合、エラーは現れなかった。
        for n in range(0, t-1):
            self.x[:, n+1] = self.x[:, n] ** 2 * np.exp(self.y[:, n] - self.x[:, n]) + self.k0 + self.k * self.x[:, n] * M(self.phi[:,n]) + self.Ring_Star_matrix @ self.x[:,n]
            self.y[:, n+1] = self.a * self.y[:, n] - self.b * self.x[:, n] + self.c
            self.phi[:, n+1] = self.k1 * self.x[:, n] - self.k2 * self.phi[:, n]

        """-----------------------------------------------------"""
        """-----------------------------------------------------"""    
        

#下以降はまだ未着手
#Chialvo写像のインスタンスを作成する。
chialvo_map = ChialvoMap(a=a, b=b, c=c, k0=k0, k=k, k1=k1, k2=k2, alph=alph, beta=beta, N=N, mu=mu, R=R, sigma=sigma, t=t)

#実行する
chialvo_map.run()

#ヒートマップの出力用に転置した
#過渡期を除外する
plot_start = 15000
x_data = chialvo_map.x[:, plot_start:]
x_data_T = x_data.T  # ヒートマップ用に転置

#軌跡を位相平面にプロットする。
fig = plt.figure()
ax = fig.add_subplot(111)


plt.plot(chialvo_map.x[10,:],'.')
plt.xlabel('Times')
plt.ylabel('x')
plt.show()


#ヒートマップを表示する。
sns.heatmap(x_data_T, cmap='hsv',vmin=-1,vmax=1)
plt.title(f"σ={sigma}, μ={mu}, R={R}, k={k}")
plt.xlabel('Nodes')
plt.ylabel('Time')
plt.gca().invert_yaxis()  # y軸の順序を反転
#これをオンにすると名前付きで保存される
#plt.savefig(f"σ={sigma},μ={mu},R={R},k={k},heatmap.png")
plt.show()

plt.plot(chialvo_map.x[:,t-1],'.')
plt.title(f"σ={sigma}, μ={mu}, R={R}, k={k}")
plt.xlabel('Nodes')
plt.ylabel('x')
plt.ylim(0) #y軸範囲指定
#これをオンにすると名前付きで保存される
#plt.savefig(f"σ={sigma},μ={mu},R={R},k={k},plot.png")
plt.show()

plt.plot(chialvo_map.x[1,:],'.')
plt.xlabel('Times')
plt.ylabel('x')
plt.show()