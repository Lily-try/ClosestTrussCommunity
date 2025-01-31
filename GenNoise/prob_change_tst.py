import argparse
import os
import sys
import pickle as pkl
import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csgraph
from scipy.linalg import eigh


'''
统计攻击前后图的变化
用于测试攻击是否确实使得图中的某些性质发生了变化。

'''

def get_graph_properties(graphx):
    """
    获取图的性质，包括度分布、平均度、聚类系数、连通分量数量和图直径。

    参数：
    adj -- networkx 图

    返回：
    properties -- 一个包含图性质的字典
    """

    # 度分布
    degrees = [d for n, d in graphx.degree()]
    degree_distribution = np.bincount(degrees) / len(graphx.nodes())

    # 平均度
    avg_degree = np.mean(degrees)

    # 全局聚类系数
    clustering_coefficient = nx.average_clustering(graphx)

    #局部聚类系数均值
    local_clustering = nx.clustering(graphx)
    local_clustering_avg = sum(local_clustering.values()) / len(local_clustering)    # 将局部聚类系数转换为列表以便计算平均值

    # 连通分量数量
    connected_components = nx.number_connected_components(graphx)

    # 图直径
    if nx.is_connected(graphx):
        diameter = nx.diameter(graphx)
    else:
        # 计算每个连通分量的直径，取最大值
        diameters = [nx.diameter(graphx.subgraph(component)) for component in nx.connected_components(graphx)]
        diameter = max(diameters)

    properties = {
        'degree_distribution': degree_distribution,
        'avg_degree': avg_degree,
        'clustering_coefficient': clustering_coefficient,
        'local_clustering_avg': local_clustering_avg,
        'connected_components': connected_components,
        'diameter': diameter
    }

    return properties


def compare_graphs_properties(original_G, modified_G):
    """
    比较攻击前后的图的性质变化。

    参数：
    original_adj -- 原始图的邻接矩阵（scipy 稀疏矩阵）
    modified_adj -- 攻击后的图的邻接矩阵（scipy 稀疏矩阵）
    """
    print(f'攻击前图有{original_G.number_of_nodes()}个节点，{original_G.number_of_edges()}条边')
    print(f'攻击后图有{modified_G.number_of_nodes()}个节点，{modified_G.number_of_edges()}条边')

    # 获取攻击前后的图性质
    original_properties = get_graph_properties(original_G)
    modified_properties = get_graph_properties(modified_G)
    # 比较性质
    print("攻击前后的图结构性质对比：")

    # 度分布比较
    print("\n1. 度分布变化:")
    print(f"攻击前: {original_properties['degree_distribution']}")
    print(f"攻击后: {modified_properties['degree_distribution']}")

    # 平均度比较
    print("\n2. 平均度变化:")
    print(f"攻击前: {original_properties['avg_degree']}")
    print(f"攻击后: {modified_properties['avg_degree']}")

    # 聚类系数比较
    print("\n3. 全局聚类系数变化:")
    print(f"攻击前: {original_properties['clustering_coefficient']:.4f}")
    print(f"攻击后: {modified_properties['clustering_coefficient']:.4f}")

    #平均局部聚类系数
    print("\n3. 平均局部聚类系数变化:")
    print(f"攻击前: {original_properties['local_clustering_avg']:.4f}")
    print(f"攻击后: {modified_properties['local_clustering_avg']:.4f}")

    # 连通分量数量比较
    print("\n4. 连通分量数量变化:")
    print(f"攻击前: {original_properties['connected_components']}")
    print(f"攻击后: {modified_properties['connected_components']}")

    # 图直径比较
    print("\n5. 图直径变化:")
    print(f"攻击前: {original_properties['diameter']}")
    print(f"攻击后: {modified_properties['diameter']}")

def compare_edges(original_graph, modified_graph):
    # 获取攻击前后的边集
    original_edges = set(original_graph.edges())
    modified_edges = set(modified_graph.edges())

    # 计算新增和删除的边
    added_edges = modified_edges - original_edges
    removed_edges = original_edges - modified_edges

    print("\n边的变化:")
    print(f"新增边数量: {len(added_edges)}")
    print(f"删除边数量: {len(removed_edges)}")
    # print(f"新增的边: {added_edges}")
    # print(f"删除的边: {removed_edges}")

def compare_laplacian_eigenvalues(original_graph, modified_graph, k=5):
    # 计算拉普拉斯矩阵
    original_laplacian = nx.normalized_laplacian_matrix(original_graph)
    modified_laplacian = nx.normalized_laplacian_matrix(modified_graph)

    # 计算前 k 个最小特征值
    original_eigenvalues = eigh(original_laplacian.toarray(), subset_by_index=[0, k-1])[0]
    modified_eigenvalues = eigh(modified_laplacian.toarray(), subset_by_index=[0, k-1])[0]

    print("\n拉普拉斯特征值的变化:")
    print(f"攻击前前 {k} 个特征值: {original_eigenvalues}")
    print(f"攻击后前 {k} 个特征值: {modified_eigenvalues}")

# 示例用法
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=str,default='../data')
    parser.add_argument('--dataset',type=str,default='cora')
    parser.add_argument('--attack', type=str, default='meta',choices=['none','random','meta'])
    parser.add_argument('--type', type=str, default='add',help='random attack type',choices=['add','remove','flip'])
    parser.add_argument('--ptb_rate',type=float,default=0.1)

    args = parser.parse_args()
    #读取攻击前的图
    if args.dataset in ['cora','pubmed','citeseer']:
        names = ['graph']
        objects = []  # 用于存储从文件中读取的对象
        for i in range(len(names)):  # 这里只有1个元素，因此只会执行一次循环
            # ../data/cora/raw/ind.cora.graph
            path = os.path.join(args.root, args.dataset, 'raw', 'ind.{0}.{1}'.format(args.dataset, names[i]))
            with open(path, 'rb') as f:
                if sys.version_info > (3, 0):  # 检查python版本信息是否大于3.9.
                    objects.append(pkl.load(f, encoding='latin1'))  # 使用latin1编码从打开的文件'f'中加载对象并添加到Objects列表中
                else:
                    objects.append(pkl.load(f))  # python2.x不支持编码参数
        original_G = nx.from_dict_of_lists(objects[0])  # 从objcts中提取第一个对象（字典）

    #读取攻击后的图
    path = os.path.join(args.root,args.dataset,args.attack,f'{args.dataset}_{args.attack}_{args.ptb_rate}.npz')
    modified_adj=sp.load_npz(path)
    modified_G=nx.from_scipy_sparse_array(modified_adj)


    compare_edges(original_G, modified_G)
    compare_laplacian_eigenvalues(original_G, modified_G)

    #对比图性质
    compare_graphs_properties(original_G, modified_G)


    # #测试去除孤立点
    # origin_isolated_nodes = [node for node,degree in original_G.degree() if degree==0]
    # original_G.remove_nodes_from(origin_isolated_nodes)
    # print('原始图中去除孤立点后的节点数量',original_G.number_of_nodes())
    # #测试去除孤立点
    # modified_isolated_nodes = [node for node,degree in modified_G.degree() if degree==0]
    # modified_G.remove_nodes_from(modified_isolated_nodes)
    # print('攻击后图中去除孤立点后的节点数量',modified_G.number_of_nodes())

