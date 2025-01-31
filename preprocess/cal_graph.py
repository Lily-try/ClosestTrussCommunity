import numpy as np
import torch
import networkx as nx
import os

from citation_loader import citation_graph_reader, citation_feature_reader

#统计图的信息
root = '../data'
dataset = 'cocs' # ['photo','dblp','physics','reddit','texas','wisconsin']:

#统计特征数据的唯独
if dataset in ['cora', 'pubmed', 'citeseer']:
    nodes_feats = citation_feature_reader(root, dataset)  # numpy.ndaaray:(2708,1433)
    nodes_feats = torch.from_numpy(nodes_feats)  # 转换成tensor
    node_in_dim = nodes_feats.shape[1]
    # print(f'{args.dataset}的feats dtype: {nodes_feats.dtype}')
elif dataset in ['cocs','photo','dblp','physics','reddit','texas','wisconsin']:
    with open(f'{root}/{dataset}/{dataset}.feats', "r") as f:
        # 每行特征转换为列表，然后堆叠为 ndarray
        nodes_feats = np.array([list(map(float, line.strip().split())) for line in f])
        nodes_feats = torch.from_numpy(nodes_feats)  # 转换成tensor
        node_in_dim = nodes_feats.shape[1]
print(f'{dataset}的节点特征维度：',nodes_feats.shape)

#统计邻接矩阵的维度
if dataset in ['cora', 'pubmed', 'citeseer']:
    graphx = citation_graph_reader(root, dataset)  # 读取图 nx格式的
    n_nodes = graphx.number_of_nodes()
elif dataset in ['cocs','photo','dblp','physics','reddit','texas','wisconsin']:
    graphx = nx.Graph()
    with open(f'{root}/{dataset}/{dataset}.edges', "r") as f:
        for line in f:
            node1, node2 = map(int, line.strip().split())
            graphx.add_edge(node1, node2)
print(f'{dataset}:', graphx)
# 计算平均度数
degrees = [degree for _, degree in graphx.degree()]  # 获取所有节点的度数
average_degree = sum(degrees) / len(degrees) if degrees else 0  # 避免零节点时的除零错误
print(f"平均度数: {average_degree}")

#统计社区数量和平均大小
communities = {}
file_path = os.path.join(root, dataset, f'{dataset}.comms')
with open(file_path, 'r', encoding='utf-8') as f:
    # 跳过第一行的标签列表
    next(f)
    label = 0
    for line in f:
        # 假设每行是由空格分隔的节点ID
        node_ids = line.strip().split()
        communities[label] = [int(node_id) for node_id in node_ids]
        label += 1

num_communities = len(communities) # 统计社区数量
community_sizes = [len(nodes) for nodes in communities.values()]# 计算社区大小

average_size = sum(community_sizes) / num_communities if num_communities > 0 else 0# 计算平均大小
print(f"{dataset}的社区数量: {num_communities}")
print(f"社区的平均大小: {average_size}")