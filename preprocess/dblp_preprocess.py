import numpy as np
def com_npy2comms(comm_path,dataset):

    # 加载 .npy 文件中的社区信息
    comms = np.load(f'{comm_path}/comms.npy', allow_pickle=True)
    # 将其保存为文本格式：.comms 文件
    with open(f'{comm_path}/{dataset}.comms', 'w') as f:
        # 写第一行：社区编号
        f.write(' '.join(str(i) for i in range(len(comms))) + '\n')
        # 写每一行的节点ID
        for community in comms:
            f.write(' '.join(str(node) for node in community) + '\n')
    print(f'存到了{comm_path}/{dataset}.comms中')

def edge_npy2edges(edge_path,dataset):
    # 加载边信息，通常为 N×2 的数组，每行一条边
    edges = np.load(f'{edge_path}/edges.npy')
    # 保存为 .edges 文本文件
    with open(f'{edge_path}/{dataset}.edges', 'w') as f:
        for u, v in edges:
            f.write(f"{u} {v}\n")

import networkx as nx
import numpy as np

def compute_coreness_feats(edge_file, feat_file, normalize=True):
    '''
    构建初始的特征
    :param edge_file:
    :param feat_file:
    :param normalize:
    :return:
    '''
    # 1. 读取边，构建图
    G = nx.read_edgelist(edge_file, nodetype=int)
    nodes = sorted(G.nodes())
    node2idx = {node: idx for idx, node in enumerate(nodes)}  # 可确保顺序一致

    # 2. 计算 coreness（k-core 值）
    core_dict = nx.core_number(G)  # {node: coreness}
    coreness = np.array([core_dict[node] for node in nodes], dtype=np.float32)

    # 3. 可选归一化（min-max 归一化到 [0,1]）
    if normalize:
        coreness = (coreness - coreness.min()) / (coreness.max() - coreness.min() + 1e-8)

    # 4. 写入 .feats 文件
    with open(feat_file, 'w') as f:
        num_nodes = len(nodes)
        feat_dim = 1  # 只有 coreness 一个特征
        f.write(f"{num_nodes} {feat_dim}\n")
        for node, feat in zip(nodes, coreness):
            f.write(f"{node} {feat:.6f}\n")

if __name__ == '__main__':
    data = 'amazon'

    comm_path = f'../data/{data}/'
    com_npy2comms(comm_path, data)
    edge_path = f'../data/{data}/'
    edge_npy2edges(edge_path,data)
    compute_coreness_feats(f'../data/{data}/{data}.edges', f'../data/{data}/{data}.feats')



