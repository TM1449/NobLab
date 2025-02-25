import numpy as np

def create_ring_adjacency_matrix(N, weight_range=(-1, 1)):
    """
    N個のノードを持つリング型の隣接行列を作成する
    
    :param N: ノード数（隠れニューロン数）
    :param weight_range: (最小値, 最大値) のタプルで、ランダム重みの範囲
    :return: N x N の隣接行列（NumPy配列）
    """
    W = np.zeros((N, N))  # 初期化
    
    for i in range(N):
        forward_index = (i + 1) % N  # 次のノード
        backward_index = (i - 1) % N  # 前のノード

        Rand = np.random.uniform(-1,1)
        
        # ランダムな重みを設定（もしくは固定値）
        W[i, forward_index] = np.random.uniform(*weight_range)
        W[i, backward_index] = np.random.uniform(*weight_range)
    return W

# ノード数（隠れ層のニューロン数）
N = 5
W_ring = create_ring_adjacency_matrix(N)

print(W_ring)

A = [0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001, 0.0009, 0.0008, 0.0007, 0.0006, 0.0005, 0.0004, 0.0003, 0.0002, 0.0001]
B = [-5.0, -4.75, -4.5, -4.25, -4.0, -3.75, -3.5, -3.25, -3.0, -2.75, -2.5, -2.25, -2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0]
print(len(A))
print(len(B))