import numpy as np

def build_ring_star_matrix(N, R, mu, sigma):
    """
    リングスター結合行列 (N×N) を返す。
    - リング結合: 各ノードは左右 R 個の近隣ノードと sigma 強度で結合
    - スター結合: 中心ノード (0 番) と他ノードが mu 強度で相互結合
    """
    # リング結合：まず隣接行列 A_ring（一次元ラップアラウンド）を作る
    ring = np.identity(N)  # N×N の単位行列
    ring[0, 0] = 0  # 中心ノードは除外

    for i in range(1, N):
        # 対角成分周辺に 1 を配置
        ring[i: i+R+1, i: i+R+1] = 1

        # 左下と右上のラップアラウンド（範囲外：左下や右上）を考慮
        ring[N-R+i-1:, i] = 1
        ring[i, N-R+i-1:] = 1
    # 対角成分は -2R に設定
    for j in range(1, N):
        ring[j, j] = -2 * R
    
    Ring_M = (sigma / (2 * R)) * ring

    # スター結合：ノード 0 が中心
    Star_M = np.zeros((N, N))
    # 中心ノード 0 は他ノードと結合する：行 0 の off-diagonal を +mu、対角には −mu*(N-1)
    Star_M[0, 0] = -mu * (N - 1)
    Star_M[0, 1:] = mu
    Star_M[1:, 0] = -mu

    for z in range(1, N):
        Star_M[z, z] += mu

    # 合成
    return print(Ring_M + Star_M)

build_ring_star_matrix(10, 1, 1, 0)