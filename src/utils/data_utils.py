#!/usr/bin/env python
# encoding: utf-8
# File Name: data_util.py
# Author: Jiezhong Qiu
# Create Time: 2019/12/30 14:20
import os

import scipy
from ogb.nodeproppred import DglNodePropPredDataset

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import torch as th
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

import utils.util_funcs as uf
from utils.proj_settings import *

from tqdm import tqdm
from heapq import heapify, heappushpop, merge as heap_merge, nlargest, nsmallest
import pickle
from utils.citation_loader import citation_graph_reader,citation_feature_reader,citation_target_reader
import dgl
def graph_normalization(g, cuda):
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    if cuda:
        norm = norm.cuda()
    g.ndata['norm'] = norm.unsqueeze(1)
    return g


def stratified_train_test_split(label_idx, labels, num_nodes, train_rate, seed=2021):
    num_train_nodes = int(train_rate / 100 * num_nodes) #划分比例是5%，则训练集有5/100*2708=135个节点
    test_rate_in_labeled_nodes = (len(labels) - num_train_nodes) / len(labels) #验证+测试集的节点比例
    train_idx, test_and_valid_idx = train_test_split(
        label_idx, test_size=test_rate_in_labeled_nodes, random_state=seed, shuffle=True, stratify=labels)
    valid_idx, test_idx = train_test_split(
        test_and_valid_idx, test_size=.5, random_state=seed, shuffle=True, stratify=labels[test_and_valid_idx])
    return train_idx, valid_idx, test_idx

def save_split_indices(dataset, root, train_idx, val_idx, test_idx):
    save_dir = os.path.join(root, dataset,'gsr')
    os.makedirs(save_dir, exist_ok=True)

    np.savetxt(os.path.join(save_dir, f'{dataset}.train'), train_idx, fmt='%d')
    np.savetxt(os.path.join(save_dir, f'{dataset}.val'), val_idx, fmt='%d')
    np.savetxt(os.path.join(save_dir, f'{dataset}.test'), test_idx, fmt='%d')

    print(f"Saved split indices to {save_dir}")

# def preprocess_data(dataset, train_percentage):
#     import dgl
#
#     # Modified from AAAI21 FA-GCN
#     if dataset in ['cora', 'citeseer', 'pubmed']:
#         load_default_split = train_percentage <= 0
#         edge = np.loadtxt(f'{DATA_PATH}/{dataset}/{dataset}.edge', dtype=int).tolist()
#         features = np.loadtxt(f'{DATA_PATH}/{dataset}/{dataset}.feature')
#         labels = np.loadtxt(f'{DATA_PATH}/{dataset}/{dataset}.label', dtype=int)
#         if load_default_split:
#             train = np.loadtxt(f'{DATA_PATH}/{dataset}/{dataset}.train', dtype=int)
#             val = np.loadtxt(f'{DATA_PATH}/{dataset}/{dataset}.val', dtype=int)
#             test = np.loadtxt(f'{DATA_PATH}/{dataset}/{dataset}.test', dtype=int)
#         else:
#             train, val, test = stratified_train_test_split(np.arange(len(labels)), labels, len(labels), train_percentage)
#         nclass = len(set(labels.tolist()))
#         print(dataset, nclass)
#
#         U = [e[0] for e in edge]
#         V = [e[1] for e in edge]
#         g = dgl.graph((U, V))
#         g = dgl.to_simple(g)
#         g = dgl.remove_self_loop(g)
#         g = dgl.to_bidirected(g)
#
#         features = normalize_features(features)
#         features = th.FloatTensor(features)
#         labels = th.LongTensor(labels)
#         train = th.LongTensor(train)
#         val = th.LongTensor(val)
#         test = th.LongTensor(test)
#
#     elif dataset in ['airport', 'blogcatalog', 'flickr']:
#         load_default_split = train_percentage <= 0
#         adj_orig = pickle.load(open(f'{DATA_PATH}/{dataset}/{dataset}_adj.pkl', 'rb'))  # sparse
#         features = pickle.load(open(f'{DATA_PATH}/{dataset}/{dataset}_features.pkl', 'rb'))  # sparase
#         labels = pickle.load(open(f'{DATA_PATH}/{dataset}/{dataset}_labels.pkl', 'rb'))  # tensor
#         if th.is_tensor(labels):
#             labels = labels.numpy()
#
#         if load_default_split:
#             tvt_nids = pickle.load(open(f'{DATA_PATH}/{dataset}/{dataset}_tvt_nids.pkl', 'rb'))  # 3 array
#             train = tvt_nids[0]
#             val = tvt_nids[1]
#             test = tvt_nids[2]
#         else:
#             train, val, test = stratified_train_test_split(np.arange(len(labels)), labels, len(labels),
#                                                            train_percentage)
#         nclass = len(set(labels.tolist()))
#         print(dataset, nclass)
#
#         adj_orig = adj_orig.tocoo()
#         U = adj_orig.row.tolist()
#         V = adj_orig.col.tolist()
#         g = dgl.graph((U, V))
#         g = dgl.to_simple(g)
#         g = dgl.remove_self_loop(g)
#         g = dgl.to_bidirected(g)
#
#         if dataset in ['airport']:
#             features = normalize_features(features)
#
#         if sp.issparse(features):
#             features = torch.FloatTensor(features.toarray())
#         else:
#             features = th.FloatTensor(features)
#
#         labels = th.LongTensor(labels)
#         train = th.LongTensor(train)
#         val = th.LongTensor(val)
#         test = th.LongTensor(test)
#
#     elif dataset in ['arxiv']:
#         dataset = DglNodePropPredDataset(name='ogbn-arxiv', root='data/ogb_arxiv')
#         split_idx = dataset.get_idx_split()
#         train, val, test = split_idx["train"], split_idx["valid"], split_idx["test"]
#         g, labels = dataset[0]
#         features = g.ndata['feat']
#         nclass = 40
#         labels = labels.squeeze()
#         g = dgl.to_bidirected(g)
#         g = dgl.to_bidirected(g)
#     if dataset in ['citeseer']:
#         g = dgl.add_self_loop(g)
#     return g, features, features.shape[1], nclass, labels, train, val, test


def load_features(root,dataset):
    if dataset in ['cora', 'pubmed', 'citeseer']:
        nodes_feats = citation_feature_reader(root, dataset)  # numpy.ndaaray:(2708,1433)
        nodes_feats = scipy.sparse.csr_matrix(nodes_feats)
    elif dataset in ['cocs', 'photo', 'dblp', 'physics', 'reddit', 'texas', 'wisconsin']:
        with open(f'{root}/{dataset}/{dataset}.feats', "r") as f:
            # 每行特征转换为列表，然后堆叠为 ndarray
            nodes_feats = np.array([list(map(float, line.strip().split())) for line in f])
            nodes_feats = scipy.sparse.csr_matrix(nodes_feats)
    else: #这个是源码自己使用的
        nodes_feats = scipy.sparse.load_npz('./ptb_graphs/%s_features.npz' % (dataset))
    print(f'{dataset} nodes_feats type:{type(nodes_feats)}, nodes_feats shape:{nodes_feats.shape}')

    # 如果是稀疏矩阵，转为 dense 后再转 tensor
    if scipy.sparse.issparse(nodes_feats):
        nodes_feats = nodes_feats.toarray()
    #将特征标准化
    features = normalize_features(nodes_feats) #ndarray
    features = torch.FloatTensor(features)
    return features  # shape: [N, F]

def load_adj(root,dataset,attack,ptb_rate): #torch.Tenso
    if attack == 'none':  # 使用原始数据
        if dataset in ['cora', 'pubmed', 'citeseer']:
            graphx = citation_graph_reader(root, dataset)  # 读取图 nx格式的
            print(graphx)
            n_nodes = graphx.number_of_nodes()
        elif dataset in ['cocs']:
            graphx = nx.Graph()
            with open(f'{root}/{dataset}/{dataset}.edges', "r") as f:
                for line in f:
                    node1, node2 = map(int, line.strip().split())
                    graphx.add_edge(node1, node2)
            print(f'{dataset}:', graphx)
            n_nodes = graphx.number_of_nodes()
    elif attack == 'random':
        path = os.path.join(root, dataset, attack,
                            f'{dataset}_{attack}_{type}_{ptb_rate}.npz')
        adj_csr_matrix = sp.load_npz(path)
        graphx = nx.from_scipy_sparse_array(adj_csr_matrix)
        print(graphx)
        n_nodes = graphx.number_of_nodes()
    elif attack in ['del', 'gflipm', 'gdelm', 'add','gaddm']:
        path = os.path.join(root, dataset, attack,
                            f'{dataset}_{attack}_{ptb_rate}.npz')
        adj_csr_matrix = sp.load_npz(path)
        graphx = nx.from_scipy_sparse_array(adj_csr_matrix)
        print(graphx)
        n_nodes = graphx.number_of_nodes()
    else:
        raise ValueError(f'Unsupported attack type:{attack}')

    #需要节点是从0开始的连续编号？
    #将其转换成DGLGraph
    g = dgl.from_networkx(graphx)
    g = dgl.to_simple(g)
    g = dgl.remove_self_loop(g)
    g = dgl.to_bidirected(g)
    return g

def read_community_data(root,dataset):
    """
    从文件读取community。
    :param file_path: 包含社区数据的文件路径。
    :return: 一个字典，键是社区标签，值是该社区的节点列表。
    """
    communities = {}
    # print(os.path.join(root,dataset,f'{dataset}_comms.txt'))
    file_path = os.path.join(root,dataset,f'{dataset}.comms')
    with open(file_path, 'r',encoding='utf-8') as f:
        # 跳过第一行的标签列表
        next(f)
        label = 0
        for line in f:
            # 假设每行是由空格分隔的节点ID
            node_ids = line.strip().split()
            communities[label] = [int(node_id) for node_id in node_ids]
            label += 1
    return communities

def load_labels(root,dataset,n_nodes):
    communities = read_community_data(root,dataset)
    if n_nodes is None:
        # 如果没提供节点总数，就自动获取最大节点ID
        all_nodes = [node for nodes in communities.values() for node in nodes]
        n_nodes = max(all_nodes) + 1
    labels = [-1] * n_nodes  # 先填 -1 表示未分配的节点（可选）
    for label, node_list in communities.items():
        for node in node_list:
            labels[node] = label
    # 检查是否还有未分配的节点
    if -1 in labels:
        print("Warning: some nodes were not assigned to any community.")
    return torch.LongTensor(labels)  # shape: [N]

def preprocess_data(root,dataset,attack,ptb_rate,train_percentage=0):
    #1.加载邻接矩阵并创建dgl图
    g = load_adj(root,dataset,attack,ptb_rate)
    #2.加载节点特征
    features = load_features(root,dataset)
    #3.加载节点标签数据
    labels = load_labels(root,dataset,g.num_nodes())
    nclass = len(set(labels.tolist()))

    #4.加载训练集、验证集和测试集
    load_default_split = train_percentage<=0
    load_dir = f'{root}/{dataset}/gsr/'
    if os.path.isfile(f'{load_dir}/{dataset}.train') and load_default_split: #要求读取默认的，并且缺失已经生成了默认的
        train = np.loadtxt(f'{load_dir}/{dataset}.train', dtype=int)
        val = np.loadtxt(f'{load_dir}/{dataset}/{dataset}.val', dtype=int)
        test = np.loadtxt(f'{load_dir}/{dataset}/{dataset}.test', dtype=int)
    else:#重新生成训练集、、、
        if th.is_tensor(labels):
            labels = labels.numpy()
        train,val,test =  stratified_train_test_split(np.arange(len(labels)), labels, len(labels),train_percentage)
        save_split_indices(dataset=dataset, root=root, train_idx=train, val_idx=val, test_idx=test)

    #将读取的数据都转换成tensor
    labels = th.LongTensor(labels)
    train = th.LongTensor(train)
    val = th.LongTensor(val)
    test = th.LongTensor(test)

    print('labels的类别数量',nclass)
    print('标签节点个数',len(labels))
    print('训练集的形状依次是：',train.shape, val.shape, test.shape) #是：torch.Size([140]）torch.Size([500]）torch.Size([1000])

    return g, features, features.shape[1], nclass, labels, train, val, test

# * ============================= Torch =============================
def topk_sim_edges(sim_mat, k, row_start_id, largest):
    v, i = th.topk(sim_mat.flatten(), k, largest=largest)
    inds = np.array(np.unravel_index(i.cpu().numpy(), sim_mat.shape)).T
    inds[:, 0] = inds[:, 0] + row_start_id
    ind_tensor = th.tensor(inds).to(sim_mat.device)
    # ret = th.cat((th.tensor(inds).to(sim_mat.device), v.view((-1, 1))), dim=1)
    return ind_tensor, v  # v.view((-1, 1))


def global_topk(input, k, largest):
    # https://stackoverflow.com/questions/64241325/top-k-indices-of-a-multi-dimensional-tensor
    v, i = th.topk(input.flatten(), k, largest=largest)
    return np.array(np.unravel_index(i.cpu().numpy(), input.shape)).T.tolist()


def exists_zero_lines(h):
    zero_lines = th.where(th.sum(h, 1) == 0)[0]
    if len(zero_lines) > 0:
        # raise ValueError('{} zero lines in {}s!\nZero lines:{}'.format(len(zero_lines), 'emb', zero_lines))
        print(f'{len(zero_lines)} zero lines !\nZero lines:{zero_lines}')
        return True
    return False


def batch_pairwise_cos_sim(mat, batch_size):
    # Normalization
    print()
    return


def cosine_sim_torch(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)
    # return 1 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)


def edge_lists_to_set(_):
    return set(list(map(tuple, _)))


def graph_edge_to_lot(g):
    # graph_edge_to list of (row_id, col_id) tuple
    return list(map(tuple, np.column_stack([_.cpu().numpy() for _ in g.edges()]).tolist()))

def scalable_graph_refine(g, emb, rm_num, add_num, batch_size, fsim_weight, device, norm=False):
    def _update_topk(sim, start, mask, k, prev_inds, prev_sim, largest):
        # Update TopK similarity and inds
        top_inds, top_sims = topk_sim_edges(sim + mask, k, start, largest)
        temp_inds = th.cat((prev_inds, top_inds))
        temp_sim = th.cat((prev_sim, top_sims))
        current_best = temp_sim.topk(k, largest=largest).indices
        return temp_sim[current_best], temp_inds[current_best]

    edges = set(graph_edge_to_lot(g))
    num_batches = int(g.num_nodes() / batch_size) + 1
    if add_num + rm_num == 0:
        return g.edges()

    if norm:
        # Since maximum value of a similarity matrix is fixed as 1, we only have to calculate the minimum value
        fsim_min, ssim_min = 99, 99
        for row_i in tqdm(range(num_batches), desc='Calculating minimum similarity'):
            # ! Initialize batch inds
            start = row_i * batch_size
            end = min((row_i + 1) * batch_size, g.num_nodes())
            if end <= start:
                break

            # ! Calculate similarity matrix
            fsim_min = min(fsim_min, cosine_sim_torch(emb['F'][start:end], emb['F']).min())
            ssim_min = min(ssim_min, cosine_sim_torch(emb['S'][start:end], emb['S']).min())
    # ! Init index and similairty tensor
    # Edge indexes should not be saved as floats in triples, since the number of nodes may well exceeds the maximum of float16 (65504)
    rm_inds, add_inds = [th.tensor([(0, 0) for i in range(_)]).type(th.int32).to(device)
                         for _ in [1, 1]]  # Init with one random point (0, 0)
    add_sim = th.ones(1).type(th.float16).to(device) * -99
    rm_sim = th.ones(1).type(th.float16).to(device) * 99

    for row_i in tqdm(range(num_batches), desc='Batch filtering edges'):
        # ! Initialize batch inds
        start = row_i * batch_size
        end = min((row_i + 1) * batch_size, g.num_nodes())
        if end <= start:
            break

        # ! Calculate similarity matrix
        f_sim = cosine_sim_torch(emb['F'][start:end], emb['F'])
        s_sim = cosine_sim_torch(emb['S'][start:end], emb['S'])
        if norm:
            f_sim = (f_sim - fsim_min) / (1 - fsim_min)
            s_sim = (s_sim - ssim_min) / (1 - ssim_min)
        sim = fsim_weight * f_sim + (1 - fsim_weight) * s_sim

        # ! Get masks
        # Edge mask
        edge_mask, diag_mask = [th.zeros_like(sim).type(th.int8) for _ in range(2)]
        row_gids, col_ids = g.out_edges(g.nodes()[start: end])
        edge_mask[row_gids - start, col_ids] = 1
        # Diag mask
        diag_r, diag_c = zip(*[(_ - start, _) for _ in range(start, end)])
        diag_mask[diag_r, diag_c] = 1
        # Add masks: Existing edges and diag edges should be masked
        add_mask = (edge_mask + diag_mask) * -99
        # Remove masks: Non-Existing edges should be masked (diag edges have 1 which is maximum value)
        rm_mask = (1 - edge_mask) * 99

        # ! Update edges to remove and add
        if rm_num > 0:
            k = max(len(rm_sim), rm_num)
            rm_sim, rm_inds = _update_topk(sim, start, rm_mask, k, rm_inds, rm_sim, largest=False)
        if add_num > 0:
            k = max(len(add_sim), add_num)
            add_sim, add_inds = _update_topk(sim, start, add_mask, k, add_inds, add_sim, largest=True)

    # ! Graph refinement
    if rm_num > 0:
        rm_edges = [tuple(_) for _ in rm_inds.cpu().numpy().astype(int).tolist()]
        edges -= set(rm_edges)
    if add_num > 0:
        add_edges = [tuple(_) for _ in add_inds.cpu().numpy().astype(int).tolist()]
        edges |= set(add_edges)
    # assert uf.load_pickle('EdgesGeneratedByOriImplementation') == sorted(edges)
    return edges


@uf.time_logger
def cosine_similarity_n_space(m1=None, m2=None, dist_batch_size=100):
    NoneType = type(None)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if type(m1) != torch.Tensor:  # only numpy conversion supported
        m1 = torch.from_numpy(m1).float()
    if type(m2) != torch.Tensor and type(m2) != NoneType:
        m2 = torch.from_numpy(m2).float()  # m2 could be None

    m2 = m1 if m2 is None else m2
    assert m1.shape[1] == m2.shape[1]

    result = torch.zeros([1, m2.shape[0]])

    for row_i in tqdm(range(0, int(m1.shape[0] / dist_batch_size) + 1), desc='Calculating pairwise similarity'):
        start = row_i * dist_batch_size
        end = min([(row_i + 1) * dist_batch_size, m1.shape[0]])
        if end <= start:
            break
        rows = m1[start: end]
        # sim = cosine_similarity(rows, m2) # rows is O(1) size
        sim = cosine_sim_torch(rows.to(device), m2.to(device))

        result = torch.cat((result, sim.cpu()), 0)

    result = result[1:, :]  # deleting the first row, as it was used for setting the size only
    del sim
    return result  # return 1 - ret # should be used with sklearn cosine_similarity


@uf.time_logger
def matrix_rowwise_cosine_sim(a, b, eps=1e-8):
    """
    calculate cosine similarity between matrix a and b
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / th.max(a_n, eps * th.ones_like(a_n))
    b_norm = b / th.max(b_n, eps * th.ones_like(b_n))
    sim_mt = th.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def mp_to_relations(mp):
    return [f"{mp[t_id]}{mp[t_id + 1]}" for t_id in range(len(mp) - 1)]


# ! Torch Scaling Functions

def standarize(input):
    return (input - input.mean(0, keepdims=True)) / input.std(0, keepdims=True)


def row_norm(input):
    return F.normalize(input, p=1, dim=1)


def col_norm(input):
    return F.normalize(input, p=1, dim=0)


def min_max_scaling(input, type='col'):
    '''
    min-max scaling modified from https://discuss.pytorch.org/t/how-to-efficiently-normalize-a-batch-of-tensor-to-0-1/65122/5

    Parameters
    ----------
    input (2 dimensional torch tensor): input data to scale
    type (str): type of scaling, row, col, or global.

    Returns (2 dimensional torch tensor): min-max scaled torch tensor
    -------
    Example input tensor (list format):
        [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
    Scaled tensor (list format):
        [[0.0, 0.0], [0.25, 0.25], [0.5, 0.5], [1.0, 1.0]]

    '''
    if type in ['row', 'col']:
        dim = 0 if type == 'col' else 1
        input -= input.min(dim).values
        input /= input.max(dim).values
        # corner case: the row/col's minimum value equals the maximum value.
        input[input.isnan()] = 0
        return input
    elif type == 'global':
        return (input - input.min()) / (input.max() - input.min())
    else:
        ValueError('Invalid type of min-max scaling.')
