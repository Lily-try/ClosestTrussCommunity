# coding = utf-8
# usr/bin/env python

'''
Author: Jantory
Email: zhang_ze_yu@outlook.com

Time: 2021/9/8 12:24 上午
Software: PyCharm
File: noise.py
desc:
'''
import argparse
import heapq
import copy
import networkx as nx
import numpy as np
import random
import math
from networkx.convert_matrix import from_numpy_array


import os
import scipy.sparse as sp
from collections import defaultdict

from scipy.sparse import issparse, lil_matrix, csr_matrix

from citation_loader import citation_graph_reader

#设置随机种子确保随机性可重复
np.random.seed(1)
def no_operation(p, mx):
    #不修改图的邻接矩阵，直接返回原图。
    "Do not modify the original graph"
    return mx


'''
We manipulate the graph using according to the local information
'''
def del_edge_degree_avg(p, mx):
    """
    基于度的边操作：
    功能：根据节点的度信息删除边，使高于平均度的节点的度降至平均值。
    p：概率，决定被操作的节点比例。
    mx：图的非稀疏邻接矩阵。
      For nodes have k-highest degree (k is determined by a probability): If their
      degrees are higher than the average, we delete existing edges so that the
      degrees after modification is at the average degree of the original graph;
      If their degrees are smaller than the average, we add new edges so that the
      degrees after modification is also at the average. In case of digital decimal,
      the whole procedure chooses to round floor.
      Input: p - Probability
             mx - Non-sparse adjacency matrix for the graph (require to be symmetric).
    """
    adj = copy.deepcopy(mx) #复制邻接矩阵以保持原始图不变。
    G = from_numpy_array(adj)
    ## Degree of each node
    degree = dict(G.degree) #使用 NetworkX 构造图并计算节点度。

    avg = sum(degree.values()) / len(adj) #计算平均度。
    k = int(p * len(adj))
    #选择度最高的前 k 个节点。
    top_k_vertex = heapq.nlargest(k, degree, degree.__getitem__)

    for i in top_k_vertex: #对于这些节点：如果度数高于平均值，则随机删除一些边。
        idx = list(adj[i].nonzero()[0])  # Find the idx of non-zero values
        if len(idx) > avg:
            idx_del = random.sample(idx, int(len(idx) - avg))
            for j in idx_del:
                adj[i][j] = 0
                adj[j][i] = 0
        else:
            continue

    #返回修改后的邻接矩阵。
    return adj


def del_edge_degree_mode(p, mx):
    '''
    基于度的众数删除边。  delm?
    :param p:
    :param mx:
    :return:
    '''
    is_sparse = issparse(mx)
    adj = copy.deepcopy(mx)
    # 构建图
    if is_sparse:
        G = nx.from_scipy_sparse_array(adj)
    else:
        G = nx.from_numpy_array(adj)
    ## Degree of each node
    degree = dict(G.degree)
    deg_values = np.array(list(degree.values()))
    counts = np.bincount(deg_values)
    mode = np.argmax(counts)
    k = int(p * adj.shape[0]) if is_sparse else int(p * len(adj))
    top_k_vertex = heapq.nlargest(k, degree, degree.__getitem__)

    # 修改邻接矩阵
    if is_sparse:
        adj = adj.tolil()
        for i in top_k_vertex:
            idx = adj.rows[i]
            if len(idx) > mode:
                idx_del = random.sample(idx, len(idx) - mode)
                for j in idx_del:
                    adj[i, j] = 0
                    adj[j, i] = 0
            else:
                continue
        return adj.tocsr()
    else:
        for i in top_k_vertex:
            idx = list(np.nonzero(adj[i])[0])
            if len(idx) > mode:
                idx_del = random.sample(idx, len(idx) - mode)
                for j in idx_del:
                    adj[i][j] = 0
                    adj[j][i] = 0
            else:
                continue
        return adj


# def flip_edge_degree_avg(p, mx):
#     '''
#     翻转边（删除部分现有边，同时添加新边）以达到平均度。
#     :param p:
#     :param mx:
#     :return:
#     '''
#     is_sparse = issparse(mx)
#     adj = copy.deepcopy(mx)
#     # 构建图
#     if is_sparse:
#         G = nx.from_scipy_sparse_array(adj)
#     else:
#         G = nx.from_numpy_array(adj)
#
#     ## Degree of each node
#     degree = dict(G.degree)
#     avg = sum(degree.values()) / len(adj)
#     k = int(p * adj.shape[0]) if is_sparse else int(p * len(adj))
#     top_k_vertex = heapq.nlargest(k, degree, degree.__getitem__)
#
#     if is_sparse:
#         adj = adj.tolil()
#         for i in top_k_vertex:
#             idx_ones = adj.rows[i]  # Find the idx of non-zero values
#             if len(idx_ones) > avg:
#                 all_columns = set(range(adj.shape[1]))  # 所有列的索引，假设矩阵是adj.shape[1]列
#                 idx_zeros = list(all_columns - idx_ones)  # 计算差集，得到零值的列索引
#                 idx_del = random.sample(idx_ones, int(len(idx_ones) - avg - 1))
#                 idx_add = random.sample(idx_zeros, int(len(idx_ones) - avg - 1))
#                 for j in idx_del:
#                     adj[i, j] = 0
#                     adj[j, i] = 0
#                 for j in idx_add:
#                     adj[i, j] = 1
#                     adj[j, i] = 1
#         return adj.tocsr()
#     else:
#         for i in top_k_vertex:
#             idx_ones = list(mx[i].nonzero()[0])  # Find the idx of non-zero values
#             if len(idx_ones) > avg:
#                 idx_zeros = list(np.where(mx[i] == 0)[0])
#                 idx_del = random.sample(idx_ones, int(len(idx_ones) - avg - 1))
#                 idx_add = random.sample(idx_zeros, int(len(idx_ones) - avg - 1))
#                 for j in idx_del:
#                     adj[i][j] = 0
#                     adj[j][i] = 0
#                 for j in idx_add:
#                     adj[i][j] = 1
#                     adj[j][i] = 1
#     return adj
def flip_edge_degree_avg(p, mx):
    adj = copy.deepcopy(mx)
    G = from_numpy_array(adj)
    ## Degree of each node
    degree = dict(G.degree)

    avg = sum(degree.values()) / len(adj)
    k = int(p * len(adj))
    top_k_vertex = heapq.nlargest(k, degree, degree.__getitem__)

    for i in top_k_vertex:
        idx_ones = list(mx[i].nonzero()[0])  # Find the idx of non-zero values
        if len(idx_ones) > avg:
            idx_zeros = list(np.where(mx[i] == 0)[0])
            idx_del = random.sample(idx_ones, int(len(idx_ones) - avg - 1))
            idx_add = random.sample(idx_zeros, int(len(idx_ones) - avg - 1))
            for j in idx_del:
                adj[i][j] = 0
                adj[j][i] = 0
            for j in idx_add:
                adj[i][j] = 1
                adj[j][i] = 1

    return adj





def flip_edge_degree_mode(p, mx):
    '''
    翻转边（删除部分现有边，同时添加新边）以达到众数度。
    :param p:
    :param mx:
    :return:
    '''
    print('in flip_edge_degree_mode method ')
    is_sparse = issparse(mx)
    adj = copy.deepcopy(mx)
    # 构建图
    if is_sparse:
        G = nx.from_scipy_sparse_array(adj)
    else:
        G = nx.from_numpy_array(adj)

    ## Degree of each node
    degree = dict(G.degree)

    deg_values = np.array(list(degree.values()))
    counts = np.bincount(deg_values)
    mode = np.argmax(counts)

    k = int(p * adj.shape[0]) if is_sparse else int(p * len(adj))
    top_k_vertex = heapq.nlargest(k, degree, degree.__getitem__)

    total_add = 0
    total_del = 0
    if is_sparse:
        adj = adj.tolil()
        for i in top_k_vertex:
            idx_ones = adj.rows[i]
            print(f'{i}点的非零列索引为{len(idx_ones)}')
            if len(idx_ones) > mode:  # 如果边数超过众数
                all_columns = set(range(adj.shape[1]))  # 所有列的索引，假设矩阵是adj.shape[1]列
                idx_ones_set = set(idx_ones)  # 确保idx_ones是
                idx_zeros = list(all_columns - idx_ones_set)  # 计算差集，得到零值的列索引
                print(f'{i}点的零列索引为{len(idx_zeros)}')
                num_change = int(len(idx_ones) - mode - 1)
                if num_change <= 0:
                    continue
                idx_del = random.sample(idx_ones, int(len(idx_ones) - mode - 1))
                # print(f'需要删除{len(idx_del)}条边')
                for j in idx_del:
                    adj[i,j] = 0
                    adj[j,i] = 0
                total_del += len(idx_del)
                # 先判断能不能采样
                if num_change <= len(idx_zeros):
                    idx_add = random.sample(idx_zeros, int(len(idx_ones) - mode - 1))
                    # print(f'需要添加{len(idx_add)}条边')
                    for j in idx_add:
                        adj[i,j] = 1
                        adj[j,i] = 1
                    total_add += len(idx_add)
                else:
                    print(f"[WARNING] 节点 {i} 可加边不足：需要 {num_change}，可加 {len(idx_zeros)}，跳过")
        return adj.tocsr()
    else:
        for i in top_k_vertex:
            idx_ones = list(mx[i].nonzero()[0]) # Find the idx of non-zero values
            if len(idx_ones) > mode:  # 如果超过众数
                idx_zeros = list(np.where(mx[i] == 0)[0])
                num_change = int(len(idx_ones) - mode - 1)
                if num_change <= 0:
                    continue
                idx_del = random.sample(idx_ones, int(len(idx_ones) - mode - 1))
                for j in idx_del:
                    adj[i][j] = 0
                    adj[j][i] = 0
                total_del += len(idx_del)
                # 先判断能不能采样
                if num_change <= len(idx_zeros):
                    idx_add = random.sample(idx_zeros, int(len(idx_ones) - mode - 1))
                    for j in idx_add:
                        adj[i][j] = 1
                        adj[j][i] = 1
                    total_add += len(idx_add)
                else:
                    print(f"[WARNING] 节点 {i} 可加边不足：需要 {num_change}，可加 {len(idx_zeros)}，跳过")
        print(f"[RESULT] 共删边 {total_del} 条，加边 {total_add} 条")
        return adj




def add_edge_degree_avg(p, mx):
    '''
    添加边以达到平均度
    :param p:
    :param mx:
    :return:
    '''
    adj = copy.deepcopy(mx)
    G = from_numpy_array(adj)
    ## Degree of each node
    degree = dict(G.degree)

    avg = sum(degree.values()) / len(adj)
    k = int(p * len(adj))
    top_k_vertex = heapq.nlargest(k, degree, degree.__getitem__)

    s = 0
    for i in top_k_vertex:
        idx_ones = list(mx[i].nonzero()[0])  # Find the idx of non-zero values
        if len(idx_ones) > avg:
            idx_zeros = list(np.where(mx[i] == 0)[0])
            idx_add = random.sample(idx_zeros, int((len((idx_ones)) - avg) / 1.11))
            s += len(idx_add)

            for j in idx_add:
                adj[i][j] = 1
                adj[j][i] = 1

    return adj


def add_edge_degree_mode(p, mx):
    '''
    添加边以达到众数度。
    :param p:
    :param mx:
    :return:
    '''
    adj = copy.deepcopy(mx)
    G = from_numpy_array(adj)
    ## Degree of each node
    degree = dict(G.degree)

    deg_values = np.array(list(degree.values()))
    counts = np.bincount(deg_values)
    mode = np.argmax(counts)

    k = int(p * len(adj))
    top_k_vertex = heapq.nlargest(k, degree, degree.__getitem__)

    s = 0
    for i in top_k_vertex:
        idx_ones = list(mx[i].nonzero()[0])  # Find the idx of non-zero values
        if len(idx_ones) > mode:
            idx_zeros = list(np.where(mx[i] == 0)[0])
            idx_add = random.sample(idx_zeros, int((len(idx_ones) - mode) / 1.2))

            for j in idx_add:
                adj[i][j] = 1
                adj[j][i] = 1

    return adj

'''
We manipulate the graph using according to the community information
'''
def graph_partition(G):
    '''
    This function is used for get the best graph partition. We use cached file to avoid
    unnecessary overhead. The dataset here is Cora.
    Input G - The networkx type graph we want to know the partiton
    '''
    # import community as community_louvain
    import community.community_louvain as community_louvain
    import json

    file_path = 'community.txt'

    if not os.path.isfile(file_path):
        partition = community_louvain.best_partition(G)
        json.dump(partition, open(file_path, 'w'))
        return partition
    else:
        partition = json.load(open(file_path))
        new_partition = dict()
        for key in partition.keys():
            new_partition[eval(key)] = partition[key]
        return new_partition


def graph_edge_partition(G):
    '''
    This function is used for getting the edge partitions. Edges whose endpoints are in
    the same partition are grouped, and edges connect two partitions are grouped
    together.
    Input: partition - the Leuvan partition of graph G
           G - The networkx type graph we want to know the partiton
    '''
    edge_partition = defaultdict(set)
    partition = graph_partition(G)

    for edge in G.edges():
        if partition[edge[0]] == partition[edge[1]]:
            edge_partition[partition[edge[0]]].add((edge[0], edge[1]))
        else:
            edge_partition[-1].add((edge[0], edge[1]))
    return edge_partition, partition


def del_edge_community_avg(p, mx):
    '''
    删除边以达到平均度。
    :param p:
    :param mx:
    :return:
    '''
    adj = copy.deepcopy(mx)
    G = from_numpy_array(adj)
    num_del = sum(sum(mx)) - sum(sum(del_edge_degree_avg(p, mx)))
    fixed_ratio = num_del / sum(sum(adj))  ##fix the original ratio
    edge_partition = graph_edge_partition(G)[0]
    for key in edge_partition.keys():
        for iter in range(int(fixed_ratio * len(edge_partition[key]))):
            edge_partition[key].pop()
    new_adj = np.zeros((len(adj), len(adj)))
    for key in edge_partition.keys():
        for value in edge_partition[key]:
            new_adj[value[0]][value[1]] = 1
            new_adj[value[1]][value[0]] = 1
    return new_adj


def del_edge_community_mode(p, mx):
    '''
    从社区的边列表中删除一部分边以达到众数度。
    :param p:
    :param mx:
    :return:
    '''
    print('按照众数删除社区内部边')
    is_sparse = issparse(mx)
    adj = copy.deepcopy(mx)
    # 构建图
    if is_sparse:
        G = nx.from_scipy_sparse_array(adj)
        total_edges = adj.sum()
    else:
        G = nx.from_numpy_array(adj)
        total_edges = np.sum(adj)

    #调用了按照度的众数删除边的方法（已经改为兼容版）
    cleaned_adj = del_edge_degree_mode(p, mx)
    # 计算需要删的边数
    edges_after = cleaned_adj.sum() if is_sparse else np.sum(cleaned_adj)
    num_del = total_edges - edges_after
    # num_del = sum(sum(mx)) - sum(sum(del_edge_degree_mode(p, mx)))
    # fixed_ratio = num_del / sum(sum(adj))  ##fix the original ratio
    fixed_ratio = num_del / total_edges
    # 获取社区边划分
    edge_partition = graph_edge_partition(G)[0]

    # 删除部分社区内边
    for key in edge_partition.keys():
        for iter in range(int(fixed_ratio * len(edge_partition[key]))):
            edge_partition[key].pop()
    # new_adj = np.zeros((len(adj), len(adj)))
    # 重建邻接矩阵（dense）
    # 6. 构建新邻接矩阵（与原输入类型一致）
    n = adj.shape[0]
    if is_sparse:
        new_adj = lil_matrix((n, n), dtype=int)
    else:
        new_adj = np.zeros((n, n), dtype=int)

    for key in edge_partition.keys():
        for u, v in edge_partition[key]:
            new_adj[u, v] = 1
            new_adj[v, u] = 1

    return new_adj.tocsr() if is_sparse else new_adj


def flip_edge_community_avg(p, mx):
    '''
    翻转边以达到平均度。
    :param p:
    :param mx:
    :return:
    '''
    adj = copy.deepcopy(mx)
    G = from_numpy_array(adj)
    num_del = sum(sum(mx)) - sum(sum(del_edge_degree_avg(p, mx)))
    fixed_ratio = num_del / sum(sum(adj))  ##fix the original ratio
    edge_partition, partition = graph_edge_partition(G)
    partition_trans = defaultdict(set)

    for key in partition.keys():
        partition_trans[partition[key]].add(key)

    for key in edge_partition.keys():
        num_operate = int(fixed_ratio * len(edge_partition[key]))
        add_edges = set()
        if key == -1:
            while num_operate > 0:
                sample_population = list(edge_partition.keys())
                sample_population.remove(-1)
                c1, c2 = random.sample(sample_population, 2)
                source = random.sample(partition_trans[c1], 1)[0]
                target = random.sample(partition_trans[c2], 1)[0]
                (source_, target_) = (source, target) if source < target else (target, source)
                if (source_, target_) not in edge_partition[key] \
                        and (source_, target_) not in add_edges:
                    add_edges.add((source_, target_))
                    num_operate = num_operate - 1
        else:
            while num_operate > 0:
                if len(partition_trans[key]) > 6:  ## to avoid dead loop, because some partitions are too small
                    source, target = random.sample(partition_trans[key], 2)
                    (source_, target_) = (source, target) if source < target else (target, source)
                    if source_ != target_ and (source_, target_) not in edge_partition[key] and \
                            (source_, target_) not in add_edges:
                        add_edges.add((source_, target_))
                        num_operate = num_operate - 1
                else:
                    source, target = random.sample(partition_trans[key], 2)
                    (source_, target_) = (source, target) if source < target else (target, source)
                    if source_ != target_:
                        add_edges.add((source_, target_))
                        num_operate = num_operate - 1
        for iter in range(int(fixed_ratio * len(edge_partition[key]))):
            edge_partition[key].pop()
        edge_partition[key] = edge_partition[key].union(add_edges)

    new_adj = np.zeros((len(adj), len(adj)))
    for key in edge_partition.keys():
        for value in edge_partition[key]:
            new_adj[value[0]][value[1]] = 1
            new_adj[value[1]][value[0]] = 1
    return new_adj


def flip_edge_community_mode(p, mx):
    '''翻转边以达到众数度。'''
    adj = copy.deepcopy(mx)
    G = from_numpy_array(adj)
    num_del = sum(sum(mx)) - sum(sum(del_edge_degree_mode(p, mx)))
    fixed_ratio = num_del / sum(sum(adj))  ##fix the original ratio
    edge_partition, partition = graph_edge_partition(G)
    partition_trans = defaultdict(set)

    for key in partition.keys():
        partition_trans[partition[key]].add(key)

    for key in edge_partition.keys():
        num_operate = int(fixed_ratio * len(edge_partition[key]))
        add_edges = set()
        if key == -1:
            while num_operate > 0:
                sample_population = list(edge_partition.keys())
                sample_population.remove(-1)
                c1, c2 = random.sample(sample_population, 2)
                source = random.sample(partition_trans[c1], 1)[0]
                target = random.sample(partition_trans[c2], 1)[0]
                (source_, target_) = (source, target) if source < target else (target, source)
                if (source_, target_) not in edge_partition[key] \
                        and (source_, target_) not in add_edges:
                    add_edges.add((source_, target_))
                    num_operate = num_operate - 1
        else:
            while num_operate > 0:
                if len(partition_trans[key]) > 0.01 * len(adj):  ## to avoid dead loop, because some partitions are too small
                    source, target = random.sample(partition_trans[key], 2)
                    (source_, target_) = (source, target) if source < target else (target, source)
                    if source_ != target_ and (source_, target_) not in edge_partition[key] and \
                            (source_, target_) not in add_edges:
                        add_edges.add((source_, target_))
                        num_operate = num_operate - 1
                else:
                    source, target = random.sample(partition_trans[key], 2)
                    (source_, target_) = (source, target) if source < target else (target, source)
                    if source_ != target_:
                        add_edges.add((source_, target_))
                        num_operate = num_operate - 1
        for iter in range(int(fixed_ratio * len(edge_partition[key]))):
            edge_partition[key].pop()
        edge_partition[key] = edge_partition[key].union(add_edges)

    new_adj = np.zeros((len(adj), len(adj)))
    for key in edge_partition.keys():
        for value in edge_partition[key]:
            new_adj[value[0]][value[1]] = 1
            new_adj[value[1]][value[0]] = 1
    return new_adj


def add_edge_community_avg(p, mx):
    '''添加边以达到平均度。'''
    adj = copy.deepcopy(mx)
    G = from_numpy_array(adj)
    num_del = sum(sum(mx)) - sum(sum(del_edge_degree_avg(p, mx)))
    fixed_ratio = num_del / sum(sum(adj))  ##fix the original ratio
    edge_partition, partition = graph_edge_partition(G)
    partition_trans = defaultdict(set)

    for key in partition.keys():
        partition_trans[partition[key]].add(key)

    for key in edge_partition.keys():
        num_operate = int(fixed_ratio * len(edge_partition[key]))
        add_edges = set()
        if key == -1:
            while num_operate > 0:
                sample_population = list(edge_partition.keys())
                sample_population.remove(-1)
                c1, c2 = random.sample(sample_population, 2)
                source = random.sample(partition_trans[c1], 1)[0]
                target = random.sample(partition_trans[c2], 1)[0]
                (source_, target_) = (source, target) if source < target else (target, source)
                if (source_, target_) not in edge_partition[key] \
                        and (source_, target_) not in add_edges:
                    add_edges.add((source_, target_))
                    num_operate = num_operate - 1
        else:
            while num_operate > 0:
                if len(partition_trans[key]) > 6:  ## to avoid dead loop
                    source, target = random.sample(partition_trans[key], 2)
                    (source_, target_) = (source, target) if source < target else (target, source)
                    if source_ != target_ and (source_, target_) not in edge_partition[key] and \
                            (source_, target_) not in add_edges:
                        add_edges.add((source_, target_))
                        num_operate = num_operate - 1
                else:
                    source, target = random.sample(partition_trans[key], 2)
                    (source_, target_) = (source, target) if source < target else (target, source)
                    if source_ != target_:
                        add_edges.add((source_, target_))
                        num_operate = num_operate - 1
        edge_partition[key] = edge_partition[key].union(add_edges)

    new_adj = np.zeros((len(adj), len(adj)))
    for key in edge_partition.keys():
        for value in edge_partition[key]:
            new_adj[value[0]][value[1]] = 1
            new_adj[value[1]][value[0]] = 1
    return new_adj


def add_edge_community_mode(p, mx):
    '''添加边以达到众数度。'''
    adj = copy.deepcopy(mx)
    G = from_numpy_array(adj)
    num_del = sum(sum(mx)) - sum(sum(del_edge_degree_mode(p, mx)))
    fixed_ratio = num_del / sum(sum(adj))  ##fix the original ratio
    edge_partition, partition = graph_edge_partition(G)
    partition_trans = defaultdict(set)

    for key in partition.keys():
        partition_trans[partition[key]].add(key)

    for key in edge_partition.keys():
        num_operate = int(fixed_ratio * len(edge_partition[key]))
        add_edges = set()
        if key == -1:
            while num_operate > 0:
                sample_population = list(edge_partition.keys())
                sample_population.remove(-1)
                c1, c2 = random.sample(sample_population, 2)
                source = random.sample(partition_trans[c1], 1)[0]
                target = random.sample(partition_trans[c2], 1)[0]
                (source_, target_) = (source, target) if source < target else (target, source)
                if (source_, target_) not in edge_partition[key] \
                        and (source_, target_) not in add_edges:
                    add_edges.add((source_, target_))
                    num_operate = num_operate - 1
        else:
            while num_operate > 0:
                if len(partition_trans[key]) > 0.01 * len(adj):  ## to avoid dead loop
                    source, target = random.sample(partition_trans[key], 2)
                    (source_, target_) = (source, target) if source < target else (target, source)
                    if source_ != target_ and (source_, target_) not in edge_partition[key] and \
                            (source_, target_) not in add_edges:
                        add_edges.add((source_, target_))
                        num_operate = num_operate - 1
                else:
                    source, target = random.sample(partition_trans[key], 2)
                    (source_, target_) = (source, target) if source < target else (target, source)
                    if source_ != target_:
                        add_edges.add((source_, target_))
                        num_operate = num_operate - 1
        edge_partition[key] = edge_partition[key].union(add_edges)

    new_adj = np.zeros((len(adj), len(adj)))
    for key in edge_partition.keys():
        for value in edge_partition[key]:
            new_adj[value[0]][value[1]] = 1
            new_adj[value[1]][value[0]] = 1
    return new_adj


'''
We manipulate the graph using according to the global information
'''
def graph_roles(G):
    '''
    This function is used for get the roles in the graph. We set the number of roles
    6. The result is cached to avoid unnecessary overhead. The dataset here is Cora.
    Input G - The networkx type graph we want to know the partiton
    根据图的结构特征，使用 graphrole 库提取节点的角色信息，并将结果缓存到文件以节省计算时间
    '''
    from graphrole import RecursiveFeatureExtractor, RoleExtractor
    import json

    file_path = 'global.txt'

    if not os.path.isfile(file_path): #检查缓存文件是否存在。如果不存在：
        # extract features
        feature_extractor = RecursiveFeatureExtractor(G)
        features = feature_extractor.extract_features()

        role_extractor = RoleExtractor(n_roles=6) #角色数目固定为6
        role_extractor.extract_role_factors(features)
        node_roles = role_extractor.roles
        json.dump(node_roles, open(file_path, 'w')) #提取角色并保存结果。
        return node_roles #返回角色数据。
    else:
        node_roles = json.load(open(file_path))
        new_node_roles = dict()
        for key in node_roles.keys():
            new_node_roles[eval(key)] = node_roles[key]
        return new_node_roles


def role_edge_partition(G):
    '''
    根据节点角色对边进行分组。
    同一角色的节点之间的边归为一组。
    不同角色之间的边归为另一组。
    This function is used for getting the edge partitions according to graph roles.
    Edges whose endpoints are the same role are grouped, and edges connect different
    roles are grouped
    together.
    Input: G - The networkx type graph we want to know the partiton
    '''
    edge_partition = defaultdict(set)
    node_roles = graph_roles(G)
    for edge in G.edges(): #遍历图中的每条边。
        #获取边两端节点的角色。
        if edge[1] not in node_roles:
            print(f"Edge: {edge}, Node Roles: {node_roles.get(edge[1], 'Missing')}")
        source = eval(node_roles[edge[0]][-1])
        target = eval(node_roles[edge[1]][-1])

        if source == target:   #如果两端节点角色相同，边归入 (source, target) 组。
            edge_partition[(source, target)].add((edge[0], edge[1]))
        else: #如果两端节点角色不同，确保 (source, target) 按升序排列后归入组。
            edge_partition[(source, target) if source < target else (target, source)].add((edge[0], edge[1]))
    #返回边的分组结果和节点角色。
    return edge_partition, node_roles


def del_edge_global_avg(p, mx):
    '''
    基于角色信息按平均度删除边。
    :param p:概率，用于决定被操作的边数量
    :param mx:邻接矩阵。
    :return:
    '''
    adj = copy.deepcopy(mx)
    G = from_numpy_array(adj)
    #计算需要删除的边数量。
    num_del = sum(sum(mx)) - sum(sum(del_edge_degree_avg(p, mx)))
    #确定删除边的比例。
    fixed_ratio = num_del / sum(sum(adj))  ##fix the original ratio
    edge_partition = role_edge_partition(G)[0]
    #按角色分组删除边。
    for key in edge_partition.keys(): #每组删除的边数量为该组边数乘以 fixed_ratio。
        for iter in range(int(fixed_ratio * len(edge_partition[key]))):
            edge_partition[key].pop()
    #生成新的邻接矩阵，保留剩余边。
    new_adj = np.zeros((len(adj), len(adj)))
    for key in edge_partition.keys():
        for value in edge_partition[key]:
            new_adj[value[0]][value[1]] = 1
            new_adj[value[1]][value[0]] = 1
    #返回修改后的邻接矩阵。
    return new_adj


def del_edge_global_mode(p, mx):
    '''基于角色信息按众数度删除边'''
    adj = copy.deepcopy(mx)
    G = from_numpy_array(adj)
    num_del = sum(sum(mx)) - sum(sum(del_edge_degree_mode(p, mx)))
    fixed_ratio = num_del / sum(sum(adj))  ##fix the original ratio
    edge_partition = role_edge_partition(G)[0]
    for key in edge_partition.keys():
        for iter in range(int(fixed_ratio * len(edge_partition[key]))):
            edge_partition[key].pop()
    new_adj = np.zeros((len(adj), len(adj)))
    for key in edge_partition.keys():
        for value in edge_partition[key]:
            new_adj[value[0]][value[1]] = 1
            new_adj[value[1]][value[0]] = 1
    return new_adj


def flip_edge_global_avg(p, mx):
    adj = copy.deepcopy(mx)
    G = from_numpy_array(adj)
    num_del = sum(sum(mx)) - sum(sum(del_edge_degree_avg(p, mx)))
    fixed_ratio = num_del / sum(sum(adj))  ##fix the original ratio
    edge_partition, node_roles = role_edge_partition(G)
    node_roles_trans = defaultdict(set)

    for key in node_roles.keys():
        node_roles_trans[eval(node_roles[key][-1])].add(key)

    for key in edge_partition.keys():
        num_operate = int(fixed_ratio * len(edge_partition[key]))
        add_edges = set()
        while num_operate > 0:
            source = random.sample(node_roles_trans[key[0]], 1)[0]
            target = random.sample(node_roles_trans[key[1]], 1)[0]
            (source_, target_) = (source, target) if source < target else (target, source)
            if source_ != target_ and (source_, target_) not in edge_partition[key] and (
            source_, target_) not in add_edges:
                add_edges.add((source_, target_))
                num_operate = num_operate - 1
        for iter in range(int(fixed_ratio * len(edge_partition[key]))):
            edge_partition[key].pop()
        edge_partition[key] = edge_partition[key].union(add_edges)

    new_adj = np.zeros((len(adj), len(adj)))
    for key in edge_partition.keys():
        for value in edge_partition[key]:
            new_adj[value[0]][value[1]] = 1
            new_adj[value[1]][value[0]] = 1
    return new_adj


def flip_edge_global_mode(p, mx):
    '''
    按照论文这应该是最强的
    :param p:
    :param mx:
    :return:
    '''
    adj = copy.deepcopy(mx)
    G = from_numpy_array(adj)
    num_del = sum(sum(mx)) - sum(sum(del_edge_degree_mode(p, mx)))
    fixed_ratio = num_del / sum(sum(adj))  ##fix the original ratio
    edge_partition, node_roles = role_edge_partition(G)
    node_roles_trans = defaultdict(set)

    for key in node_roles.keys():
        node_roles_trans[eval(node_roles[key][-1])].add(key)

    for key in edge_partition.keys():
        num_operate = int(fixed_ratio * len(edge_partition[key]))
        add_edges = set()
        while num_operate > 0:
            source = random.sample(node_roles_trans[key[0]], 1)[0]
            target = random.sample(node_roles_trans[key[1]], 1)[0]
            (source_, target_) = (source, target) if source < target else (target, source)
            if source_ != target_ and (source_, target_) not in edge_partition[key] and (
            source_, target_) not in add_edges:
                add_edges.add((source_, target_))
                num_operate = num_operate - 1
        for iter in range(int(fixed_ratio * len(edge_partition[key]))):
            edge_partition[key].pop()
        edge_partition[key] = edge_partition[key].union(add_edges)

    new_adj = np.zeros((len(adj), len(adj)))
    for key in edge_partition.keys():
        for value in edge_partition[key]:
            new_adj[value[0]][value[1]] = 1
            new_adj[value[1]][value[0]] = 1
    return new_adj


def add_edge_global_avg(p, mx):
    adj = copy.deepcopy(mx)
    G = from_numpy_array(adj)
    num_del = sum(sum(mx)) - sum(sum(del_edge_degree_avg(p, mx)))
    fixed_ratio = num_del / sum(sum(adj))  ##fix the original ratio
    edge_partition, node_roles = role_edge_partition(G)
    node_roles_trans = defaultdict(set)

    for key in node_roles.keys():
        node_roles_trans[eval(node_roles[key][-1])].add(key)

    for key in edge_partition.keys():
        num_operate = int(fixed_ratio * len(edge_partition[key]))
        add_edges = set()
        while num_operate > 0:
            source = random.sample(node_roles_trans[key[0]], 1)[0]
            target = random.sample(node_roles_trans[key[1]], 1)[0]
            (source_, target_) = (source, target) if source < target else (target, source)
            if source_ != target_ and (source_, target_) not in edge_partition[key] and (
            source_, target_) not in add_edges:
                add_edges.add((source_, target_))
                num_operate = num_operate - 1
        edge_partition[key] = edge_partition[key].union(add_edges)

    new_adj = np.zeros((len(adj), len(adj)))
    for key in edge_partition.keys():
        for value in edge_partition[key]:
            new_adj[value[0]][value[1]] = 1
            new_adj[value[1]][value[0]] = 1
    return new_adj


def add_edge_global_mode(p, mx):
    adj = copy.deepcopy(mx)
    G = from_numpy_array(adj)
    num_del = sum(sum(mx)) - sum(sum(del_edge_degree_mode(p, mx)))
    fixed_ratio = num_del / sum(sum(adj))  ##fix the original ratio
    edge_partition, node_roles = role_edge_partition(G)
    node_roles_trans = defaultdict(set)

    for key in node_roles.keys():
        node_roles_trans[eval(node_roles[key][-1])].add(key)

    for key in edge_partition.keys():
        num_operate = int(fixed_ratio * len(edge_partition[key]))
        add_edges = set()
        while num_operate > 0:
            source = random.sample(node_roles_trans[key[0]], 1)[0]
            target = random.sample(node_roles_trans[key[1]], 1)[0]
            (source_, target_) = (source, target) if source < target else (target, source)
            if source_ != target_ and (source_, target_) not in edge_partition[key] and (
            source_, target_) not in add_edges:
                add_edges.add((source_, target_))
                num_operate = num_operate - 1
        edge_partition[key] = edge_partition[key].union(add_edges)

    new_adj = np.zeros((len(adj), len(adj)))
    for key in edge_partition.keys():
        for value in edge_partition[key]:
            new_adj[value[0]][value[1]] = 1
            new_adj[value[1]][value[0]] = 1
    return new_adj

def load_graph(root,dataset):
    if dataset in ['cora', 'pubmed', 'citeseer']:
        graphx = citation_graph_reader(root, dataset)  # 读取图 nx格式的
        print(graphx)
        n_nodes = graphx.number_of_nodes()
        adj_matrix = nx.to_numpy_array(graphx,dtype=int)
    elif dataset in ['cocs','photo']:
        graphx = nx.Graph()
        with open(f'{args.root}/{args.dataset}/{args.dataset}.edges', "r") as f:
            for line in f:
                node1, node2 = map(int, line.strip().split())
                graphx.add_edge(node1, node2)
        print(f'{args.dataset}:', graphx)
        n_nodes = graphx.number_of_nodes()
        adj_matrix = nx.to_numpy_array(graphx, dtype=int)
        # adj_matrix = nx.to_scipy_sparse_array(graphx, format='csr', dtype=int)
    elif dataset in ['dblp','amazon']:
        graphx = nx.Graph()
        with open(f'{args.root}/{args.dataset}/{args.dataset}.edges', "r") as f:
            for line in f:
                node1, node2 = map(int, line.strip().split())
                graphx.add_edge(node1, node2)
        print(f'{args.dataset}:', graphx)
        n_nodes = graphx.number_of_nodes()
        # adj_matrix = nx.to_numpy_array(graphx, dtype=int) #内存会溢出？
        adj_matrix = nx.to_scipy_sparse_array(graphx, format='csr', dtype=int)
    elif dataset in ['facebook','fb107']:
        graphx = nx.read_edgelist(f'{root}/{dataset}/{dataset}.edges', nodetype=int, data=False)
        print(graphx)
        n_nodes = graphx.number_of_nodes()
        adj_matrix = nx.to_numpy_array(graphx, dtype=int)
    # 判断是否存在孤立节点
    row_sums = adj_matrix.sum(axis=1)
    isolated_nodes = np.where(row_sums == 0)[0]
    if len(isolated_nodes) > 0:
        print(f'发现孤立节点，节点索引:{isolated_nodes}')
    else:
        print('没有孤立节点')
        return adj_matrix
def save_graph(matrix,save_path):
    sp.save_npz(save_path, sp.csr_matrix(matrix))  # 转换为稀疏矩阵格式保存
    print(f"修改后的邻接矩阵已保存至: {save_path}")


if __name__ == '__main__':
    func_dict = {0: no_operation,
                 1: del_edge_degree_avg, 2: del_edge_degree_mode,
                 3: del_edge_community_avg, 4: del_edge_community_mode,
                 5: del_edge_global_avg, 6: del_edge_global_mode,
                 7: flip_edge_degree_avg, 8: flip_edge_degree_mode,
                 9: flip_edge_community_avg, 10: flip_edge_community_mode,
                 11: flip_edge_global_avg, 12: flip_edge_global_mode,
                 13: add_edge_degree_avg, 14: add_edge_degree_mode,
                 15: add_edge_community_avg, 16: add_edge_community_mode,
                 17: add_edge_global_avg, 18: add_edge_global_mode}
    #一个缩写形式，便于文件命名
    name_dict =  {0: no_operation,
                 1: del_edge_degree_avg, 2: 'delm',
                 3: del_edge_community_avg, 4: 'cdelm',
                 5: del_edge_global_avg, 6: 'gdelm',
                 7: flip_edge_degree_avg, 8: 'flipm',
                 9: flip_edge_community_avg, 10: 'cflipm',
                 11: flip_edge_global_avg, 12: 'gflipm',
                 13: add_edge_degree_avg, 14: add_edge_degree_mode,
                 15: add_edge_community_avg, 16: add_edge_community_mode,
                 17: add_edge_global_avg, 18: 'gaddm'}
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--root', type=str, default='../data', help='data store root')
    #choices=['fb107','cora','citeseer','cocs','facebook_all', 'cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'],
    parser.add_argument('--dataset', type=str, default='amazon',help='dataset')
    parser.add_argument('--method', type=int, default=4,choices=[6,12,18,4,10,2,8],help='noise type')
    parser.add_argument('--ptb_rate', type=float, default=0.40, help='pertubation rate')

    args = parser.parse_args()
    print(f'读取{args.dataset}数据集')
    #① 读取原始邻接矩阵
    adj_matrix = load_graph(args.root,args.dataset)

    #② 进行加噪
    # modified_adj= del_edge_global_mode(args.ptb_rate,adj_matrix)   #
    modified_adj= func_dict[args.method](args.ptb_rate,adj_matrix)


    #③ 保存加噪后的邻接矩阵
    save_path = os.path.join(args.root, args.dataset, name_dict[args.method])
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_name = f'{args.dataset}_{name_dict[args.method]}_{args.ptb_rate}.npz'

    save_graph(modified_adj,f'{save_path}/{save_name}')

    #4读取加噪后的邻接矩阵
    adj_csr_matrix = sp.load_npz(f'{save_path}/{save_name}')
    graphx = nx.from_scipy_sparse_array(adj_csr_matrix)
    print('加入噪声后的邻接矩阵：',graphx)
    n_nodes = graphx.number_of_nodes()
