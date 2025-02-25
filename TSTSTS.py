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
