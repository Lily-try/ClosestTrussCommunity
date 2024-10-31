import argparse
import networkx as nx
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import numpy as np
import os
# from utils import txt_utils, citation_utils, snap_utils
from preprocess import txt_utils
from utils import citation_loader

def add_noise_edges(adj, node_labels, noise_level, random_seed=0):
    np.random.seed(random_seed)
    num_nodes = adj.shape[0]
    num_edges = adj.sum() / 2
    if noise_level == 1:
        num_added_edges = int(num_edges * 0.3)
    elif noise_level == 2:
        num_added_edges = int(num_edges * 0.6)
    elif noise_level == 3:
        num_added_edges = int(num_edges * 0.9)
    else:
        return None

    # node_labels = np.argmax(node_labels, axis=-1)

    rows = []  # 所有的源节点
    cols = []  # 所有的目标节点
    edge_dict = dict()  # 存储现有的边，以保证不会重复加边
    for src, dst in zip(adj.tocoo().row, adj.tocoo().col):  # 便利每条边
        if src <= dst: continue
        if src not in edge_dict: edge_dict[src] = set()
        edge_dict[src].add(dst)
        rows.append(src)
        cols.append(dst)
        rows.append(dst)
        cols.append(src)

    num_sampled = 0  # 已添加的噪声边数量
    noise_edge_list = []  # 添加的噪声边
    while num_sampled < num_added_edges:
        # 随机采样2个节点，添加噪声边
        sampled_src, sampled_dst = np.random.choice(num_nodes, 2, replace=False)
        if sampled_src <= sampled_dst:
            tmp = sampled_src
            sampled_src = sampled_dst
            sampled_dst = tmp
        if sampled_src in edge_dict:  # 检查随机采样的边是否已存在
            if sampled_dst in edge_dict[sampled_src]:
                continue
            if node_labels[sampled_dst] == node_labels[sampled_src]:  # 确保采样的边的标签不同
                continue
        if [sampled_src, sampled_dst] not in noise_edge_list:
            noise_edge_list.append([sampled_src, sampled_dst])  # 将满足条件的噪声边记录
            num_sampled += 1

    for src, dst in noise_edge_list:  # 将生成的噪声边2个断电都加入，确保邻接矩阵的对成型
        rows.append(src)
        cols.append(dst)
        rows.append(dst)
        cols.append(src)
    noisy_adj = csr_matrix(([1] * len(rows), (rows, cols)), shape=(num_nodes, num_nodes))
    return noisy_adj


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--root', type=str, default='../data', help='data store root')
    parser.add_argument('--dataset', type=str, default='citeseer',
                        choices=['football', 'facebook_all', 'cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'],
                        help='dataset')
    parser.add_argument('--noise_level', type=int, default=3, choices=[1, 2, 3], help='noisy level')
    parser.add_argument('--ptb_rate', type=float, default=0.40, help='pertubation rate')
    parser.add_argument('--type', type=str, default='add', help='attack type', choices=['add', 'remove', 'flip'])


    args = parser.parse_args()

    # 读取原始邻接矩阵
    dataset = args.dataset
    if dataset in ['football', 'facebook_all']:
        adj = txt_utils.load_txt_adj(args.root, dataset)
        # 读取features
        # 读取labels
        labels = txt_utils.read_comms_to_labels(args.root, dataset)

    if dataset in ['cora', 'citeseer', 'pubmed']:  # 引文网络，deeprobust本身就有的
        # 读取临界矩阵
        graph = citation_loader.citation_graph_reader(args.root, args.dataset)  # 读取图 nx格式的
        adj = nx.adjacency_matrix(graph)  # 转换为CSR格式的稀疏矩阵
        # 读取标签
        labels = citation_loader.citation_target_reader(args.root, dataset)  # 读取标签,ndarray:(2708,1)
    if dataset in ['dblp', 'amazon']:  # sanp数据集上的
        # edge, labels = snap_utils.load_snap(args.root, data_set='com_' + dataset, com_size=3)  # edge是list:1049866
        # 将edge转换成csr_matrix
        pass

    # 注入随机噪声
    modified_adj = add_noise_edges(adj, labels, args.noise_level, random_seed=0)

    # 存储修改后的邻接矩阵
    # 存储成npz格式.
    path = os.path.join(args.root, args.dataset, 'add')
    name = f'{args.dataset}_add_{args.noise_level}'
    sp.save_npz(os.path.join(path, name), modified_adj)


