import os
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import scipy.sparse as sp
import argparse

import citation_loader
from preprocess import txt_utils


def del_sample_edge(ori_edge_index, sample_ratio, seed=123):
    '''
    从对称邻接矩阵的下三角部分采样，也就是随机删除(1-ratio)的比例删除边
    :param ori_edge_index:边列表
    :param sample_ratio :表示采样边的比例
    :param seed：随机数种子，用来确保结果可重复
    '''
    # sample from half side of the symmetric adjacency matrix
    #提取对称邻接矩阵的一半边
    half_edge_index = []
    for i in range(ori_edge_index.shape[1]): #迭代每一列（即每条边）
        if ori_edge_index[0, i] < ori_edge_index[1, i]: #仅保留下三角部分(u<v)
            half_edge_index.append(ori_edge_index[:, i].view(2, -1)) #调整张量形状为[2,1]，便于后续拼接；并将符合条件的边加入列表
    #将边按列拼接为[2,num_half_edges]
    half_edge_index = torch.cat(half_edge_index, dim=1)
    np.random.seed(seed) #设置随机数种子确保结果可重复
    #计算边数
    num_edge = half_edge_index.shape[1]
    #随机选择ratio*num_edge条边；参数replace=False确保不重复采样
    samples = np.random.choice(num_edge, size=int(sample_ratio * num_edge), replace=False)
    #利用采样的索引samples从haf_edge_index中提取对应的边
    sampled_edge_index = half_edge_index[:, samples]
    #恢复对称性
    sampled_edge_index = torch.cat([sampled_edge_index, sampled_edge_index[torch.LongTensor([1,0])]], dim=1)
    #返回采样后的边
    return sampled_edge_index


def edge_delete(prob_del, adj, enforce_connected=False):
    '''
    邻接矩阵adj中的边以prob_del的几率被删除可选择强制确保图的连通性
    :param prob_del:边被删除的概率
    :param adj :图的邻接矩阵，scipy.sparse
    :param enforce_connected :布尔值，是否强制删除边后图仍然保持连通性
    '''
    rnd = np.random.RandomState(1234) #固定随机数种子
    adj= adj.toarray() #转为numpy数组
    del_adj = np.array(adj, dtype=np.float32) #复制，用于存储删除后的结果
    #创建随机矩阵，每个元素分别以prob_del和（1-prob_del）的几率为是=0或1
    #*np.triu(np.ones_like(adj)),1)仅保留上三角（不包括对角线），用于表示无向图的边删除掩码
    smpl = rnd.choice([0., 1.], p=[prob_del, 1. - prob_del], size=adj.shape) * np.triu(np.ones_like(adj), 1)
    #将上三角矩阵对称化，得到完整的无向图边删除掩码
    smpl += smpl.transpose()
    #应用删除掩码元素逐点相乘，删除的边对应矩阵元素为0
    del_adj *= smpl

    #强制保持图的连通性（可选）
    if enforce_connected:
        add_edges = 0
        for k, a in enumerate(del_adj): #逐行检查del_adj中的每个节点的边
            if not list(np.nonzero(a)[0]):#如果节点k没有任何边(孤立节点）
                #找到该节点k在原始图中连接的节点列表
                prev_connected = list(np.nonzero(adj[k, :])[0])
                #随机选择一个节点添加边，并进行对称处理
                other_node = rnd.choice(prev_connected)
                del_adj[k, other_node] = 1
                del_adj[other_node, k] = 1
                add_edges += 1
    #转为稀疏矩阵并返回。
    del_adj= sp.csr_matrix(del_adj)

    return del_adj



def compute_basic_stats(adj):
    '''
    获取邻接矩阵的节点数边数和边密度
    :param adj:
    :return:
    '''
    num_nodes = adj.shape[0]
    num_edges = adj.nnz // 2  # 无向图，边数是非零元素的一半
    density = num_edges / (num_nodes * (num_nodes - 1) / 2)
    return num_nodes, num_edges, density
def compute_connectivity_stats(adj):
    '''
    测试邻接矩阵的连通分量个数以及最大的连通分量大小
    :param adj:
    :return:
    '''
    graph = nx.from_scipy_sparse_array(adj)
    num_components = nx.number_connected_components(graph)
    largest_cc_size = len(max(nx.connected_components(graph), key=len))
    return num_components, largest_cc_size

def compute_degree_distribution(adj):
    '''
    测试邻接矩阵的度分布，和平均度
    :param adj:
    :return:
    '''
    degrees = adj.sum(axis=1).ravel()  # 转为一维数组
    avg_degree = np.mean(degrees)
    return degrees, avg_degree

def compute_graph_properties(adj):
    '''
    最短路径长度和平均聚类系数
    :param adj:
    :return:
    '''
    graph = nx.from_scipy_sparse_array(adj)
    avg_shortest_path = nx.average_shortest_path_length(graph) if nx.is_connected(graph) else float('inf')
    avg_clustering = nx.average_clustering(graph)
    return avg_shortest_path, avg_clustering

def visualize_graph(adj, title):
    '''
    将前后的图可视化
    :param adj:
    :param title:
    :return:
    '''
    graph = nx.from_scipy_sparse_array(adj)
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, node_size=20, edge_color='gray')
    plt.title(title)
    plt.show()



if __name__ == '__main__':
    '''测试生成incomplete graph的方法是否正确执行'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=15, help='Random seed.')
    parser.add_argument('--root', type=str, default='../data', help='data store root')
    parser.add_argument('--dataset', type=str, default='citeseer',
                        choices=['football', 'facebook_all', 'cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'],
                        help='dataset')
    parser.add_argument('--ptb_rate', type=float, default=0.30, help='pertubation rate')

    args = parser.parse_args()

    #1.从文件中加载ori_edge_index
    # 读取原始邻接矩阵
    dataset = args.dataset
    if dataset in ['football', 'facebook_all']:
        adj = txt_utils.load_txt_adj(args.root, dataset)
        # labels = txt_utils.read_comms_to_labels(args.root, dataset)
        pass
    if dataset in ['cora', 'citeseer', 'pubmed']:  # 引文网络，deeprobust本身就有的
        graph = citation_loader.citation_graph_reader(args.root, args.dataset)  # 读取图 nx格式的
        num_nodes = graph.number_of_nodes()
        ori_edge_index = np.array(list(graph.edges)).T  # 转置
        ori_edge_index = torch.tensor(ori_edge_index, dtype=torch.long)
        print(type(ori_edge_index))
        # adj = nx.adjacency_matrix(graph)  # 转换为CSR格式的稀疏矩阵
        # labels = citation_loader.citation_target_reader(args.root, dataset)  # 读取标签,ndarray:(2708,1)
    if dataset in ['dblp', 'amazon']:  # sanp数据集上的
        # edge, labels = snap_utils.load_snap(args.root, data_set='com_' + dataset, com_size=3)  # edge是list:1049866
        # 将edge转换成csr_matrix
        pass


    #2.进行加噪
    sample_ratio = 1-args.ptb_rate
    sampled_edge_index = del_sample_edge(ori_edge_index,sample_ratio,args.seed) #返回的类型是edge_index

    #3.将加噪后的邻接矩阵存入文件
    modified_adj = sampled_edge_index.T.numpy() #转置并转换为Numpy   {ndarrayL(6332,2)}
    row = modified_adj[:,0] #获取起点
    col = modified_adj[:,1] #获取终点  col:{ndarray:(2,)},row:{ndarray:(2,)}
    modified_adj_sparse = sp.csr_matrix((torch.ones(len(row)).numpy(), (row, col)), shape=(num_nodes, num_nodes))

    save_path = os.path.join(args.root,args.dataset,'del')
    name = f'{args.dataset}_del_{args.ptb_rate}.npz'
    print(f'Saving to:{os.path.join(save_path,name)}')
    sp.save_npz(os.path.join(save_path,name),modified_adj_sparse)

    #4.测试修改后的邻接矩阵的变化
    #①基本属性
    ori_stats = compute_basic_stats(nx.adjacency_matrix(graph))
    del_stats = compute_basic_stats(modified_adj_sparse)
    print(f"Original Graph: Nodes={ori_stats[0]}, Edges={ori_stats[1]}, Density={ori_stats[2]:.4f}")
    print(f"Modified Graph: Nodes={del_stats[0]}, Edges={del_stats[1]}, Density={del_stats[2]:.4f}")
    #②连通性
    ori_connectivity = compute_connectivity_stats(nx.adjacency_matrix(graph))
    del_connectivity = compute_connectivity_stats(modified_adj_sparse)

    print(f"Original Graph: Components={ori_connectivity[0]}, Largest CC={ori_connectivity[1]}")
    print(f"Modified Graph: Components={del_connectivity[0]}, Largest CC={del_connectivity[1]}")

    #③度分布变化
    ori_degrees, ori_avg_degree = compute_degree_distribution(nx.adjacency_matrix(graph))
    del_degrees, del_avg_degree = compute_degree_distribution(modified_adj_sparse)

    print(f"Original Graph: Avg Degree={ori_avg_degree:.2f}")
    print(f"Modified Graph: Avg Degree={del_avg_degree:.2f}")

    # plt.hist(ori_degrees, bins=30, alpha=0.5, label='Original')
    # plt.hist(del_degrees, bins=30, alpha=0.5, label='Modified')
    # plt.xlabel('Degree')
    # plt.ylabel('Count')
    # plt.legend()
    # plt.title('Degree Distribution')
    # plt.show()

    #④图谱特性
    ori_properties = compute_graph_properties(nx.adjacency_matrix(graph))
    del_properties = compute_graph_properties(modified_adj_sparse)

    print(f"Original Graph: Avg Path Length={ori_properties[0]:.4f}, Avg Clustering={ori_properties[1]:.4f}")
    print(f"Modified Graph: Avg Path Length={del_properties[0]:.4f}, Avg Clustering={del_properties[1]:.4f}")
    #
    # #⑤将前后的图可视化
    # visualize_graph(nx.adjacency_matrix(graph), "Original Graph")
    # visualize_graph(modified_adj_sparse, "Modified Graph")




