import argparse
import copy
import datetime
import os
# import random
import random
import torch.nn.functional as F
import pymetis as metis

import torch
import torch as th
from community import community_louvain
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, jaccard_score
import numpy as np
from torch_geometric.data import Data
from torch_sparse import spspmm, coalesce
from torch_geometric.utils import remove_self_loops, add_remaining_self_loops

from utils.query_utils import gen_all_queries, read_community_data, read_queries_from_file, write_queries_to_file, \
    write_onlynodes_to_file
from utils.preprocess import preprocess_graph_dataset, relabel_graph, reorder_features, remap_query_data, \
    remap_cluster_membership
# from model_GCN import ConRC
from utils.citation_loader import citation_feature_reader
from utils.cocle_val_utils import get_model_path, get_res_path, get_comm_path
# from coclep_citation import validation_part_pre
from utils.log_utils import get_log_path, get_logger
from models.COCLE import COCLE
from utils.load_utils import load_graph, loadQuerys


def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False  # 与上面一条代码配套使用，True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现
    torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.enabled = False  # 禁用cudnn使用非确定性算法

    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # 在cuda 10.2及以上的版本中，需要设置以下环境变量来保证cuda的结果可复现
    os.environ['PYTHONHASHSEED'] = str(seed)  # 禁止hash随机化
    # torch.use_deterministic_algorithms(True) # 一些操作使用了原子操作，不是确定性算法，不能保证可复现，设置这个禁用原子操作，保证使用确定性算法


def f1_score_(comm_find, comm):
    lists = [x for x in comm_find if x in comm]
    if len(lists) == 0:
        # print("f1, pre, rec", 0.0, 0.0, 0.0)
        return 0.0, 0.0, 0.0
    pre = len(lists) * 1.0 / len(comm_find)
    rec = len(lists) * 1.0 / len(comm)
    f1 = 2 * pre * rec / (pre + rec)
    # print("f1, pre, rec", f1, pre, rec)
    return f1, pre, rec


def NMI_score(comm_find, comm, n_nodes):
    truthlabel = np.zeros((n_nodes), dtype=int)
    truthlabel[comm] = 1
    prelabel = np.zeros((n_nodes), dtype=int)
    prelabel[comm_find] = 1
    score = normalized_mutual_info_score(truthlabel, prelabel)
    # print("q, nmi:", score)
    return score


def ARI_score(comm_find, comm, n_nodes):
    truthlabel = np.zeros((n_nodes), dtype=int)
    truthlabel[comm] = 1
    prelabel = np.zeros((n_nodes), dtype=int)
    prelabel[comm_find] = 1
    score = adjusted_rand_score(truthlabel, prelabel)
    # print("q, ari:", score)

    return score


def JAC_score(comm_find, comm, n_nodes):
    truthlabel = np.zeros((n_nodes), dtype=int)
    truthlabel[comm] = 1
    prelabel = np.zeros((n_nodes), dtype=int)
    prelabel[comm_find] = 1
    score = jaccard_score(truthlabel, prelabel)
    # print("q, jac:", score)
    return score


def validation__(val, cluster_membership, sg_nodes, n_nodes, nodes_feats, device, g, model, nodes_adj, k):
    scorelists = []
    for q, comm in val:
        nodeslists = sg_nodes[cluster_membership[q]]
        nodelistb = []
        if q in nodes_adj:
            neighbor = nodes_adj[q]
            nodelistb = [x for x in neighbor if x not in nodeslists]
        nodeslists = nodeslists + nodelistb
        mask = [False] * n_nodes
        for u in nodeslists:
            mask[u] = True
        feats = nodes_feats[mask].to(device)
        nodeslists = sorted(nodeslists)
        nodes_ = {}
        for i, u in enumerate(nodeslists):
            nodes_[u] = i
        sub = g.subgraph(nodeslists)
        src = []
        dst = []
        for id1, id2 in sub.edges:
            id1_ = nodes_[id1]
            id2_ = nodes_[id2]
            src.append(id1_)
            dst.append(id2_)
            src.append(id2_)
            dst.append(id1_)
        edge_index = torch.tensor([src, dst]).to(device)
        edge_index_aug, egde_attr = hypergraph_construction(edge_index, len(nodeslists), k=k)
        edge_index = add_remaining_self_loops(edge_index, num_nodes=len(nodeslists))[0]

        h = model((nodes_[q], None, edge_index, edge_index_aug, feats))

        # numerator = torch.mm(h[nodes_[q]].unsqueeze(0), h.t())
        # norm = torch.norm(h, dim=-1, keepdim=True)
        # denominator = torch.mm(norm[nodes_[q]].unsqueeze(0), norm.t())
        # sim = numerator / denominator

        # 为什么到了photo数据集这里就报错了
        # sim = F.cosine_similarity(h[nodes_[q]].unsqueeze(0), h, dim=1)  # (115,)
        # simlists = torch.sigmoid(sim.squeeze(0)).to(torch.device('cpu')).numpy().tolist()  # torch.sigmoid(simlists).numpy().tolist()

        sim = F.cosine_similarity(h[nodes_[q]].unsqueeze(0), h, dim=1)  # (num_nodes_in_subgraph,)
        simlists = torch.sigmoid(sim).cpu().numpy().tolist()

        comm_ = [nodes_[x] for x in comm if x in nodeslists]
        if len(comm_) != len(comm):
            size = len(comm) - len(comm_)
            for i in range(size):
                comm_.append(i + len(nodeslists))
        scorelists.append([nodes_[q], comm_, simlists])
    s_ = 0.1
    f1_m = 0.0
    s_m = s_
    while (s_ <= 0.9):
        f1_x = 0.0
        # print("------------------------------", s_)
        for q, comm, simlists in scorelists:
            comm_find = []
            for i, score in enumerate(simlists):
                if score >= s_ and i not in comm_find:
                    comm_find.append(i)

            comm_find = set(comm_find)
            comm_find = list(comm_find)
            comm = set(comm)
            comm = list(comm)
            f1, pre, rec = f1_score_(comm_find, comm)
            f1_x = f1_x + f1  # pre
        f1_x = f1_x / len(val)
        if f1_m < f1_x:
            f1_m = f1_x
            s_m = s_
        s_ = s_ + 0.05
    # print("------------------------", s_m, f1_m)
    logger.info(f'best threshold: {s_m}, validation_set Avg F1: {f1_m}')
    return s_m, f1_m


def validation_part_pre(val, cluster_membership, sg_nodes, n_nodes, nodes_feats, device, g, model, nodes_adj, k):
    '''
    改为选择precision最优的结果
    :param val:
    :param nodes_feats:
    :param model:
    :param edge_index:
    :param edge_index_aug:
    :return:
    '''
    scorelists = []
    for q, comm in val:
        nodeslists = sg_nodes[cluster_membership[q]]
        nodelistb = []
        if q in nodes_adj:
            neighbor = nodes_adj[q]
            nodelistb = [x for x in neighbor if x not in nodeslists]
        nodeslists = nodeslists + nodelistb
        mask = [False] * n_nodes
        for u in nodeslists:
            mask[u] = True
        feats = nodes_feats[mask].to(device)
        nodeslists = sorted(nodeslists)
        nodes_ = {}
        for i, u in enumerate(nodeslists):
            nodes_[u] = i
        sub = g.subgraph(nodeslists)
        src = []
        dst = []
        for id1, id2 in sub.edges:
            id1_ = nodes_[id1]
            id2_ = nodes_[id2]
            src.append(id1_)
            dst.append(id2_)
            src.append(id2_)
            dst.append(id1_)
        edge_index = torch.tensor([src, dst]).to(device)
        edge_index_aug, egde_attr = hypergraph_construction(edge_index, len(nodeslists), k=k)
        edge_index = add_remaining_self_loops(edge_index, num_nodes=len(nodeslists))[0]
        h = model((nodes_[q], None, edge_index, edge_index_aug, feats))

        # 计算余弦相似度
        sim = F.cosine_similarity(h[nodes_[q]].unsqueeze(0), h, dim=1)  # (115,)
        # 使用 torch.sigmoid 将相似度值转换为概率，然后使用 squeeze(0) 移除多余的维度，
        # 并将结果转移到 CPU，最后转换为 NumPy 数组并转换为 Python 列表。
        simlists = torch.sigmoid(sim).cpu().numpy().tolist()
        # simlists = torch.sigmoid(sim.squeeze(0)).to(torch.device('cpu')).numpy().tolist()  # torch.sigmoid(simlists).numpy().tolist()

        # print("simlists type:", type(simlists))  #<class 'list'>
        # print("simlists content:", simlists)
        comm_ = [nodes_[x] for x in comm if x in nodeslists]
        if len(comm_) != len(comm):
            size = len(comm) - len(comm_)
            for i in range(size):
                comm_.append(i + len(nodeslists))
        scorelists.append([nodes_[q], comm_, simlists])
    s_ = 0.1  # 阈值？？
    pre_m = 0.0
    s_m = s_  # 记录可以取的最大的社区阈值
    while (s_ <= 0.9):  # 结束循环后得到的是从0.1按照0.05的步长不断增加社区阈值可以得到的最大的平均f1值f1_m和最优的s_取值s_m。
        pre_x = 0.0
        # print("------------------------------", s_) #s_是什么？？
        for q, comm, simlists in scorelists:
            comm_find = []
            for i, score in enumerate(simlists):  # i是每个节点的编号；score是q与每个节点的相似得分。
                if score >= s_ and i not in comm_find:
                    comm_find.append(i)

            comm_find = set(comm_find)
            comm_find = list(comm_find)
            comm = set(comm)
            comm = list(comm)
            f1, pre, rec = f1_score_(comm_find, comm)
            pre_x = pre_x + pre  # 累加此样本的f1得分
        pre_x = pre_x / len(val)  # 总的f1得分除以验证集样本数量
        if pre_m < pre_x:  # 如果此社区阈值下得到的平均f1得分更高
            pre_m = pre_x
            s_m = s_
        s_ = s_ + 0.05  # 将s_进行增大。
    logger.info(f'best threshold: {s_m}, validation_set Avg Pre: {pre_m}')
    return s_m, pre_m


# def loadQuerys(dataset, root, train_n, val_n, test_n, train_path, test_path, val_path):
#     # path_train = root + dataset + '/' + dataset + train_path
#     path_train = os.path.join(root, dataset, f'{dataset}_{train_path}_{train_n}.txt')
#     if not os.path.isfile(path_train):
#         raise Exception("No such file: %s" % path_train)
#     train_lists = []
#     for line in open(path_train, encoding='utf-8'):
#         q, pos, comm = line.split(",")
#         q = int(q)
#         pos = pos.split(" ")
#         pos_ = [int(x) for x in pos if int(x)!=q]
#         comm = comm.split(" ")
#         comm_ = [int(x) for x in comm]
#         if len(train_lists)>=train_n:
#             break
#         train_lists.append((q, pos_, comm_))
#     path_test = root + dataset + '/' + dataset + test_path
#     if not os.path.isfile(path_test):
#         raise Exception("No such file: %s" % path_test)
#     test_lists = []
#     for line in open(path_test, encoding='utf-8'):
#         q, comm = line.split(",")
#         q = int(q)
#         comm = comm.split(" ")
#         comm_ = [int(x) for x in comm]
#         if len(test_lists)>=test_n:
#             break
#         test_lists.append((q, comm_))
#     '''
#     val_lists_ = test_lists[test_n:]
#     test_lists = test_lists[:test_n]
#     val_lists = []
#     for q, comm in val_lists_:
#         val_lists.append((q, comm))
#
#     '''
#     path_val = root + dataset + '/' + dataset + val_path
#     if not os.path.isfile(path_val):
#         raise Exception("No such file: %s" % path_val)
#     val_lists = []
#     for line in open(path_val, encoding='utf-8'):
#         q, pos, comm = line.split(",")
#         q = int(q)
#         pos = pos.split(" ")
#         pos_ = [int(x) for x in pos if int(x)!=q]
#         comm = comm.split(" ")
#         comm_ = [int(x) for x in comm]
#         if len(val_lists)>=val_n:
#             break
#         val_lists.append((q, comm_))
#     #'''
#
#     return train_lists, val_lists, test_lists

'''
进行图划分
graph：是图，
cluster_number：是要划分的簇的数目
'''


def metis_clustering(cluster_number, graph):
    # 使用metis的方法进行图划分，
    adjacency_list = []
    for node in graph.nodes():
        neighbors = list(graph.neighbors(node))
        # adjacency_list.append([node_to_index[neighbor] for neighbor in neighbors])
        adjacency_list.append(neighbors)
    (st, parts) = metis.part_graph(cluster_number, adjacency=adjacency_list)
    clusters = list(set(parts))  # 创建包含所有簇的列表，并通过set()进行簇标识去重。
    cluster_membership = {}  # 字典，用于存储每个节点所属的簇
    for i, node in enumerate(graph.nodes):  # 遍历图中的节点，同时获得节点索引i和节点对象node
        cluster = parts[i]  # 获取第i个节点所属簇的标识
        cluster_membership[node] = cluster  # 将节点`node`和其所属的簇的标识`cluster`添加到字典中，建立了节点到簇的映射关系
    return clusters, cluster_membership  # 返回簇的列表和节点到簇的映射


def louvain_clustering(graph):
    # 生成每个节点所属的社区（返回的是dict：node -> community_id）
    partition = community_louvain.best_partition(graph)
    clusters = list(set(partition.values()))
    return clusters, partition


def gen_new_queries(args, valid_nodes):
    print('从gt社区中生成查询任务并划分')
    # 3.从comms文件中读取所有社区列表
    if args.dataset in ['photo_stb', 'photo_gsr', 'cocs_stb', 'cocs_gsr']:
        communities = read_community_data(args.root, args.dataset[:-4])
    else:
        communities = read_community_data(args.root, args.dataset)
    all_query_nums = args.train_size + args.val_size + args.test_size
    gen_all_queries(args.root, args.dataset, communities, all_query_nums, args.pos_size, args.pos_size, threshold=.8,
                    constrain='True', valid_nodes=valid_nodes)

    split_ratios = (args.train_size, args.val_size, args.test_size)
    queries = read_queries_from_file(f'{args.root}/{args.dataset}/{args.dataset}_all_queries.txt')  # 读取查询任务
    dataset_root = os.path.join(args.root, args.dataset)
    train_path = os.path.join(dataset_root, f'{args.dataset}_{args.pos_size}_pos_train_{split_ratios[0]}.txt')
    val_path = os.path.join(dataset_root, f'{args.dataset}_{args.pos_size}_val_{split_ratios[1]}.txt')
    test_path = os.path.join(dataset_root, f'{args.dataset}_{args.pos_size}_test_{split_ratios[2]}.txt')
    train_size, val_size, test_size = split_ratios
    assert sum(split_ratios) == len(queries), "Sum of split ratios must equal the total number of queries"
    # 划分任务
    train_queries = queries[:train_size]
    validation_queries = queries[train_size:train_size + val_size]
    test_queries = queries[train_size + val_size:train_size + val_size + test_size]

    # 写入训练集文件
    write_queries_to_file(train_queries, train_path)
    # 写入验证集文件
    write_queries_to_file(validation_queries, val_path)
    # 写入测试集文件
    write_queries_to_file(test_queries, test_path)
    # 只写入测试集的查询节点（用于传统方法）
    query_path = os.path.join(args.root, args.dataset, f'{args.dataset}_querynode.txt')
    print(f'File will be saved at: {query_path}')  # 打印路径
    write_onlynodes_to_file(test_queries, query_path)


def gen_new_queries_(root, dataset, train_size, val_size, test_size, pos_size, valid_nodes):
    print('从gt社区中生成查询任务并划分')
    # 3.从comms文件中读取所有社区列表
    if dataset in ['photo_stb', 'photo_gsr', 'cocs_stb', 'cocs_gsr']:
        communities = read_community_data(root, dataset[:-4])
    else:
        communities = read_community_data(root, dataset)
    all_query_nums = train_size + val_size + test_size
    gen_all_queries(root, dataset, communities, all_query_nums, pos_size, pos_size, threshold=.8,
                    constrain='True', valid_nodes=valid_nodes)

    split_ratios = (train_size, val_size, test_size)
    queries = read_queries_from_file(f'{root}/{dataset}/{dataset}_all_queries.txt')  # 读取查询任务
    dataset_root = os.path.join(root, dataset)
    train_path = os.path.join(dataset_root, f'{dataset}_{pos_size}_pos_train_{split_ratios[0]}.txt')
    val_path = os.path.join(dataset_root, f'{dataset}_{pos_size}_val_{split_ratios[1]}.txt')
    test_path = os.path.join(dataset_root, f'{dataset}_{pos_size}_test_{split_ratios[2]}.txt')
    train_size, val_size, test_size = split_ratios
    assert sum(split_ratios) == len(queries), "Sum of split ratios must equal the total number of queries"
    # 划分任务
    train_queries = queries[:train_size]
    validation_queries = queries[train_size:train_size + val_size]
    test_queries = queries[train_size + val_size:train_size + val_size + test_size]

    # 写入训练集文件
    write_queries_to_file(train_queries, train_path)
    # 写入验证集文件
    write_queries_to_file(validation_queries, val_path)
    # 写入测试集文件
    write_queries_to_file(test_queries, test_path)
    # 只写入测试集的查询节点（用于传统方法）
    query_path = os.path.join(root, dataset, f'{dataset}_querynode.txt')
    print(f'File will be saved at: {query_path}')  # 打印路径
    write_onlynodes_to_file(test_queries, query_path)


'''
从文件夹中加载数据集
cluster_number：分区个数
返回值：
node_feats：节点特征
train_ ：训练集
val：验证集
test：测试集
node_in_dim ：
n_nodes ：图中节点数量
g(graphx)：图
sg_nodes：字典，key是分区，value是分区中的节点编号
clusters：分区标识列表
cluster_membership：字典，标记了每个节点所处的分区标识
node_adj：字典，用于存储节点的邻接信息。无向图
'''


def load_data(
        args):  # dataset, root, train_n, val_n, test_n, feats_path, cluster_number, train_path,test_path, val_path):
    # 加载图。存在一个问题是重新编号后，最终保存的.comms文件是找不到原始的id的了。
    graphx, n_nodes = load_graph(args.root, args.dataset, args.attack, args.ptb_rate)
    graphx, mapping = relabel_graph(graphx)

    # 生成查询
    valid_nodes = set(graphx.nodes())
    gen_new_queries(args, valid_nodes)
    # 下面这行则是不同于COLCE的地方，进行了图分区。
    # clusters是分区簇列表，cluster_membership是字典
    # cluster是{list:100}从0一直到99；cluster_membership是{dict:2708}例如key是节点id，value是节点所属的分区id。0:80;
    print("---------------------cluster开始-------------------------------")
    clusters, cluster_membership = metis_clustering(args.cluster_number, graphx)
    # clusters,cluster_membership = louvain_clustering(graphx)
    cluster_membership = remap_cluster_membership(cluster_membership, mapping)
    print("---------------------cluster结束-------------------------------")

    nodes_adj = {}  # 字典，用于存储节点的邻接信息。无向图
    for id1, id2 in graphx.edges:
        if id1 not in nodes_adj:
            nodes_adj[id1] = [id2]
        else:
            nodes_adj[id1].append(id2)
        if id2 not in nodes_adj:
            nodes_adj[id2] = [id1]
        else:
            nodes_adj[id2].append(id1)

    print("---------------------graph-------------------------------")
    sg_nodes = {}  # 字典，用于存储分区后的节点信息。key是分区标识，value是分区内的节点

    for u, c in cluster_membership.items():
        if c not in sg_nodes:
            sg_nodes[c] = []
        sg_nodes[c].append(u)  # {c1:1,2,3},{c2:4,5,6}

    # 调用loadQuery加载训练、验证和测试数据
    logger.info('正在加载训练数据')
    train, val, test = loadQuerys(args.dataset, args.root, args.train_size, args.val_size, args.test_size,
                                  args.train_path, args.test_path, args.val_path)
    train, val, test = remap_query_data(train, val, test, mapping)
    logger.info('加载训练数据完成')

    print("====================3.加载特征数据featd==================================")
    logger.info('正在加载特征数据')
    if args.dataset in ['cora', 'citeseer_stb', 'pubmed', 'citeseer']:
        nodes_feats = citation_feature_reader(args.root, args.dataset)  # numpy.ndaaray:(2708,1433)
        nodes_feats = torch.from_numpy(nodes_feats)  # 转换成tensor
        node_in_dim = nodes_feats.shape[1]
        print(f'{args.dataset}的feats dtype: {nodes_feats.dtype}')
    elif args.dataset in ['cora_stb', 'cora_gsr']:
        nodes_feats = citation_feature_reader(args.root, args.dataset[:-4])  # numpy.ndaaray:(2708,1433)
        nodes_feats = torch.from_numpy(nodes_feats)  # 转换成tensor
        node_in_dim = nodes_feats.shape[1]
    elif args.dataset in ['fb107_gsr', 'fb107_stb']:
        feats_array = np.loadtxt(f'{args.root}/{args.dataset[:-4]}/{args.dataset[:-4]}.feat', delimiter=' ',
                                 dtype=np.float32)
        print(type(feats_array))
        # nodes_feats = fnormalize(feats_array)  # 将特征进行归一化
        nodes_feats = torch.from_numpy(feats_array)
        node_in_dim = nodes_feats.shape[1]
    elif args.dataset in ['cocs', 'photo']:
        with open(f'{args.root}/{args.dataset}/{args.dataset}.feats', "r") as f:
            # 每行特征转换为列表，然后堆叠为 ndarray,注意要是float32
            nodes_feats = np.array([list(map(float, line.strip().split())) for line in f], dtype=np.float32)
            print(f'{args.dataset}的nodes_feats.dtype = {nodes_feats.dtype}')
            print(f'{args.dataset}的节点特征shape:', nodes_feats.shape)
            nodes_feats = torch.from_numpy(nodes_feats)  # 转换成tensor
            node_in_dim = nodes_feats.shape[1]
    elif args.dataset in ['dblp', 'amazon']:
        with open(f'{args.root}/{args.dataset}/{args.dataset}.feats', "r") as f:
            first_line = f.readline().strip().split()
            num_nodes, feat_dim = int(first_line[0]), int(first_line[1])
            feat_dict = {}
            for line in f:
                parts = line.strip().split()
                node_id = int(parts[0])
                feats = list(map(float, parts[1:]))
                feat_dict[node_id] = feats
            # 根据 node id 顺序填充
            nodes_feats = np.zeros((num_nodes, feat_dim), dtype=np.float32)
            for node_id, feats in feat_dict.items():
                nodes_feats[node_id] = feats
            nodes_feats = torch.from_numpy(nodes_feats)
            node_in_dim = nodes_feats.shape[1]
    elif args.dataset in ['dblp_gsr', 'dblp_stb', 'amazon_gsr', 'amazon_stb']:
        with open(f'{args.root}/{args.dataset[:-4]}/{args.dataset[:-4]}.feats', "r") as f:
            first_line = f.readline().strip().split()
            num_nodes, feat_dim = int(first_line[0]), int(first_line[1])
            feat_dict = {}
            for line in f:
                parts = line.strip().split()
                node_id = int(parts[0])
                feats = list(map(float, parts[1:]))
                feat_dict[node_id] = feats
            # 根据 node id 顺序填充
            nodes_feats = np.zeros((num_nodes, feat_dim), dtype=np.float32)
            for node_id, feats in feat_dict.items():
                nodes_feats[node_id] = feats
            nodes_feats = torch.from_numpy(nodes_feats)
            node_in_dim = nodes_feats.shape[1]
    elif args.dataset in ['photo_gsr', 'photo_stb', 'cocs_stb', 'cocs_gsr']:
        feats_array = np.loadtxt(f'{args.root}/{args.dataset[:-4]}/{args.dataset[:-4]}.feats', delimiter=' ',
                                 dtype=np.float32)
        print(type(feats_array))
        # nodes_feats = fnormalize(feats_array)  # 将特征进行归一化
        nodes_feats = torch.from_numpy(feats_array)
        node_in_dim = nodes_feats.shape[1]
    # elif args.dataset.startswith(('fb', 'wfb', 'fa')):  # 不加入中心节点
    elif args.dataset in ['fb107', 'wfb107']:  # 不加入中心节点
        feats_array = np.loadtxt(f'{args.root}/{args.dataset}/{args.dataset}.feat', delimiter=' ', dtype=np.float32)
        print(type(feats_array))
        # nodes_feats = fnormalize(feats_array)  # 将特征进行归一化
        nodes_feats = torch.from_numpy(feats_array)
        node_in_dim = nodes_feats.shape[1]
    elif args.dataset in ['facebook']:  # 读取pyg中的特征数据
        feats_array = np.loadtxt(f'{args.root}/{args.dataset}/{args.dataset}.feat', dtype=float, delimiter=' ')
        nodes_feats = torch.tensor(feats_array, dtype=torch.float32)
        node_in_dim = nodes_feats.shape[1]
    elif args.dataset in ['facebook_gsr']:  # 读取pyg中的特征数据
        feats_array = np.loadtxt(f'{args.root}/{args.dataset[:-4]}/{args.dataset[:-4]}.feat', dtype=float,
                                 delimiter=' ')
        nodes_feats = torch.tensor(feats_array, dtype=torch.float32)
        node_in_dim = nodes_feats.shape[1]
    else:
        logger.info('加载节点特征失败，数据集不匹配')
    nodes_feats = reorder_features(nodes_feats, mapping)
    logger.info('加载节点特征完成完成')
    # path_feat = root + dataset + '/' + feats_path #特征文件的路径
    # if not os.path.isfile(path_feat):
    #     raise Exception("No such file: %s" % path_feat)
    # feats_node = {}
    # count = 1
    # for line in open(path_feat, encoding='utf-8'):
    #     if count == 1:
    #         node_n_, node_in_dim = line.split()
    #         node_in_dim = int(node_in_dim)
    #         count = count + 1
    #     else:
    #         emb = [float(x) for x in line.split()]
    #         id = int(emb[0])
    #         emb = emb[1:]
    #         feats_node[id] = emb
    # nodes_feats = []
    #
    # for i in range(0, n_nodes):
    #     if i not in feats_node:
    #         nodes_feats.append([0.0] * node_in_dim)
    #     else:
    #         nodes_feats.append(feats_node[i])
    # nodes_feats = th.tensor(nodes_feats)
    '''  
    nodes_feats = nodes_feats.transpose(0, 1)
    rowsum = nodes_feats.sum(1)
    rowsum[rowsum == 0] = 1
    print(rowsum)
    nodes_feats = nodes_feats / rowsum[:, np.newaxis]
    nodes_feats = nodes_feats.transpose(0, 1)
    #'''

    # 进行统一编号处理
    # graphx, nodes_feats, train, val, test, cluster_membership = preprocess_graph_dataset(
    #     graphx, nodes_feats, train, val, test, cluster_membership
    # )
    return nodes_feats, train, val, test, node_in_dim, n_nodes, graphx, \
        sg_nodes, clusters, cluster_membership, nodes_adj


class TwoHopNeighbor(object):
    def __call__(self, data):
        edge_index, edge_attr = data.edge_index, data.edge_attr
        N = data.num_nodes
        # 强制转换为 long 类型 #为了防止spspmm报错
        if edge_index.dtype != torch.long:
            edge_index = edge_index.long()
        data.edge_index = edge_index  # 确保同步回 data

        # edge_index = edge_index.long()
        # assert edge_index.dtype == torch.long, f"edge_index.dtype is {edge_index.dtype}, expected torch.long"
        value = edge_index.new_ones((edge_index.size(1),), dtype=torch.float)
        # print("edge_index dtype:", edge_index.dtype)
        # print("edge_index shape:", edge_index.shape)
        # print("edge_index fixed dtype:", edge_index.dtype)
        index, value = spspmm(edge_index, value, edge_index, value, N, N, N, True)
        value.fill_(0)
        index, value = remove_self_loops(index, value)

        edge_index = torch.cat([edge_index, index], dim=1)
        if edge_attr is None:
            data.edge_index, _ = coalesce(edge_index, None, N, N)
        else:
            value = value.view(-1, *[1 for _ in range(edge_attr.dim() - 1)])
            value = value.expand(-1, *list(edge_attr.size())[1:])
            edge_attr = torch.cat([edge_attr, value], dim=0)
            data.edge_index, edge_attr = coalesce(edge_index, edge_attr, N, N)
            data.edge_attr = edge_attr

        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


'''
创建超图
输入：
原始子图中的edge_index
num_nodes：节点数量
k：应该是文章中的r，表示1-hop邻居
'''


def hypergraph_construction(edge_index, num_nodes, k=1):
    # print(f'hypergraph_construction中的edgeindex的{edge_index.dtype}')
    if k == 1:
        edge_index, edge_attr = add_remaining_self_loops(edge_index, num_nodes=num_nodes)
    else:
        neighbor_augment = TwoHopNeighbor()
        hop_data = Data(edge_index=edge_index, edge_attr=None)
        hop_data.num_nodes = num_nodes
        for _ in range(k - 1):
            hop_data = neighbor_augment(hop_data)
        hop_edge_index = hop_data.edge_index
        hop_edge_attr = hop_data.edge_attr
        edge_index, edge_attr = add_remaining_self_loops(hop_edge_index, hop_edge_attr, num_nodes=num_nodes)
    return edge_index, edge_attr


'''
将训练集进行重构
g：是原始图
sg_nodes：字典，key是分区，value是分区中的节点编号
cluster_membership：字典，key是节点编号，value是该节点所在的分区编号
返回：
train_lists 可以在中pytorch中使用的格式
train_lists.append((nodes_[q]-q的索引, pos_-pos节点是索引, edge_index-边, edge_index_aug-增强图边, feats-特征))
'''


def decompose_train(g, train, val, nodes_feats, n_nodes, cluster_membership,
                    sg_nodes, k, nodes_adj):
    train_lists = []
    # q是查询节点，pos是正例，train是社区节点列表。从每个社区中随机抽取一个节点作为查询节点
    for q, pos, comm in train:
        nodeslists = sg_nodes[cluster_membership[q]]  # q所在分区的节点列表

        nodelistb = []  # 存储此分区边界节点
        if q in nodes_adj:  # 检查查询节点q是否在节点邻接列表中
            neighbor = nodes_adj[q]  # 如果在，则获得查询节点q的邻接节点列表
            nodelistb = [x for x in neighbor if x not in nodeslists]  # 不在此分区中的邻居
        nodelistb = set(nodelistb)  # 去重，确保只出现1次
        nodelistb = list(nodelistb)  # 转换为列表
        nodeslists = nodeslists + nodelistb  # 将nodeListb中的边界节点加入到Nodelists中
        lists = [x for x in comm if x not in nodeslists]  # 不在此分区（分区节点+边界节点）但在社区中的顶点。
        # 打印此分区的节点数，此分区边界结点数，被分到其他分区的节点数，社区节点数
        print(len(nodeslists), len(nodelistb), len(lists), len(comm))
        mask = [False] * n_nodes  # 创建初始为false的长度为节点数的布尔列表mask
        for u in nodeslists:  # 将在此分区中的节点设置为True
            mask[u] = True
        feats = nodes_feats[mask]  # 使用mask获得此分区（分区+边界）的节点特征
        nodeslists = sorted(nodeslists)  # 排序，确保节点的顺序
        nodes_ = {}  # 字典，key是节点编号，value是在nodelist中的索引
        for i, u in enumerate(nodeslists):
            nodes_[u] = i
        sub = g.subgraph(nodeslists)  # 创建分区（分区+边界）的导出子图  。怎么知道子图的边是哪些的？？？？
        src = []  # 存储子图中边的源节点
        dst = []  # 存储子图中边的目标节点
        # 用于构建超图边，然后将它们与原始图的边合并，结果存储在edge)index中。
        for id1, id2 in sub.edges:
            id1_ = nodes_[id1]  # 获得源节点id1在nodes中的索引
            id2_ = nodes_[id2]  # 获得目标节点id2在nodes中的索引
            src.append(id1_)
            dst.append(id2_)
            src.append(id2_)
            dst.append(id1_)
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        # print("在decompose_train中edge_index fixed dtype:", edge_index.dtype)
        edge_index_aug, egde_attr = hypergraph_construction(edge_index, len(nodeslists), k=k)
        edge_index = add_remaining_self_loops(edge_index, num_nodes=len(nodeslists))[0]  # 添加自环？
        pos_ = [nodes_[x] for x in pos if x in nodeslists]  # 在此分区（分区+边界）中的标记节点的索引。
        if edge_index.dtype != torch.long:
            edge_index = edge_index.long()
            edge_index_aug = edge_index_aug.long()
            # print('出现了不是long的')
        # pos_ = [nodes_[x] for x in comm if x in nodeslists and x!=q]
        train_lists.append((nodes_[q], pos_, edge_index, edge_index_aug, feats))

    return train_lists


'''
进行图划分的社区搜索方法
'''


def Val_CommunitySearch(args,logger):
    # ── 放到 import 之后，进入 Val_CommunitySearch() 之初 ──
    intra_sum_H = inter_sum_H = 0.0
    intra_cnt_H = inter_cnt_H = 0
    sig_intra_sum_H = sig_inter_sum_H = 0.0
    sig_intra_cnt_H = sig_inter_cnt_H = 0
    intra_sum_X = inter_sum_X = 0.0
    intra_cnt_X = inter_cnt_X = 0

    sig_intra_sum_X = sig_inter_sum_X = 0.0
    sig_intra_cnt_X = sig_inter_cnt_X = 0

    #一、加载数据
    preprocess_start = datetime.datetime.now()  # 进行社区搜索前的时间
    # load_data加载数据集
    nodes_feats, train_, val, test, node_in_dim, n_nodes, g, \
        sg_nodes, clusters, cluster_membership, nodes_adj = load_data(args)

    print(f'feats的shape：{node_in_dim} ,{nodes_feats.shape}')
    print(f'节点数：{g.number_of_nodes()},边数：{g.number_of_edges()}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 提前把所有节点特征做 L2 归一化，方便后续复用
    X_norm_global = F.normalize(nodes_feats, p=2, dim=1).to(device)  # (N, d_x)
    logger.info(f'device:{device}')  # 打印设备

    model = COCLE(node_in_dim, args.hidden_dim, args.num_layers, args.drop_out, args.tau, device, args.alpha,
                  args.lam, args.k)  # COCLEP中的模型，目前和EmbLearner是一样的
    logger.info(f'embLearner: {args.method}')
    optimizer = th.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.to(device)
    pre_process_time = (datetime.datetime.now() - preprocess_start).seconds  # 到现在结束了预处理过程？

    bst_model_path = get_model_path('./results/coclep/res_model/', args)
    logger.info(f'#################### Starting evaluation######################')
    # 目前是加载具有最优f1的模型
    if args.val_type == 'pre':
        model.load_state_dict(torch.load(f'{bst_model_path}_pre.pkl'))  # 加载模型
    else:
        model.load_state_dict(torch.load(f'{bst_model_path}_f1.pkl'))  # 加载模型
    model.eval()
    model.eval()  # 进行模型验证

    eval_start = datetime.datetime.now()
    with th.no_grad():
        # 使用验证集数据找打最佳阈值s_
        if args.val_type == 'f1':
            s_, f1_ = validation__(val, cluster_membership, sg_nodes,
                                   n_nodes, nodes_feats, device, g, model, nodes_adj, args.k)
            logger.info(f'evaluation time = {datetime.datetime.now() - eval_start}, best s_={s_}, best val f1_={f1_}')
        elif args.val_type == 'pre':
            s_, pre_ = validation_part_pre(val, cluster_membership, sg_nodes,
                                          n_nodes, nodes_feats, device, g, model, nodes_adj, args.k)
            logger.info(f'evaluation time = {datetime.datetime.now() - eval_start}, best s_={s_}, best val pre_={pre_}')

        # 开始测试
        logger.info(f'#################### starting test  ####################')
        test_start = datetime.datetime.now()
        for q, comm in test:  # 进行测试。。
            nodeslists = sg_nodes[cluster_membership[q]]
            nodelistb = []
            if q in nodes_adj:
                neighbor = nodes_adj[q]
                nodelistb = [x for x in neighbor if x not in nodeslists]
            nodelistb = set(nodelistb)
            nodelistb = list(nodelistb)
            nodeslists = nodeslists + nodelistb
            mask = [False] * n_nodes
            for u in nodeslists:
                mask[u] = True
            feats = nodes_feats[mask].to(device)
            nodeslists = sorted(nodeslists)
            nodes_ = {}
            for i, u in enumerate(nodeslists):
                nodes_[u] = i
            sub = g.subgraph(nodeslists)
            src = []
            dst = []
            for id1, id2 in sub.edges:
                id1_ = nodes_[id1]
                id2_ = nodes_[id2]
                src.append(id1_)
                dst.append(id2_)
                src.append(id2_)
                dst.append(id1_)
            edge_index = torch.tensor([src, dst]).to(device)
            edge_index_aug, egde_attr = hypergraph_construction(edge_index, len(nodeslists), k=args.k)
            edge_index = add_remaining_self_loops(edge_index, num_nodes=len(nodeslists))[0]  # 得到可以输入模型的数据

            h = model((nodes_[q], None, edge_index, edge_index_aug, feats))  # 运行模型得到
            h = F.normalize(h, p=2, dim=1)

            # ---------- A) 取社区 ground-truth 索引 ----------
            comm_set = set(comm)  # comm 来自数据集给的真社区
            comm_idx = torch.tensor([nodes_[u] for u in comm_set if u in nodes_],
                                    device=device)  # 映射到子图局部索引

            if len(comm_idx) < 2:
                continue  # 单点社区跳过

            out_idx = torch.tensor([i for i in range(len(nodeslists))
                                    if i not in comm_idx],
                                   device=device)

            # ---------- B) 嵌入 H 的相似度 ----------
            h_c = h[comm_idx]  # (m, d_h)
            h_out = h[out_idx]  # (n, d_h)

            # 社区内两两相似总和 ½(||Σ h_c||² − m)
            sum_h_c = h_c.sum(dim=0)  # (d_h,)
            intra_sum_H += 0.5 * (sum_h_c @ sum_h_c - len(comm_idx))
            intra_cnt_H += len(comm_idx) * (len(comm_idx) - 1) // 2

            # 社区 ↔ 外部
            sum_h_out = h_out.sum(dim=0)
            inter_sum_H += (sum_h_c @ sum_h_out)
            inter_cnt_H += len(comm_idx) * len(out_idx)

            # ---------- C) 原始特征 X 的相似度 ----------
            X_c = X_norm_global[nodeslists][:, :][comm_idx]  # (m, d_x)
            X_out = X_norm_global[nodeslists][:, :][out_idx]  # (n, d_x)

            sum_x_c = X_c.sum(dim=0)
            intra_sum_X += 0.5 * (sum_x_c @ sum_x_c - len(comm_idx))
            intra_cnt_X += len(comm_idx) * (len(comm_idx) - 1) // 2

            sum_x_out = X_out.sum(dim=0)
            inter_sum_X += (sum_x_c @ sum_x_out)
            inter_cnt_X += len(comm_idx) * len(out_idx)

            # ② 循环内 —— 在你算完 X_c / X_out 后，加下面几行
            # —— X 的 Sigmoid 版相似度 -----------------------------
            sig_sims_intra_x = torch.mm(X_c, X_c.T).triu(diagonal=1)
            sig_sims_intra_x = torch.sigmoid(sig_sims_intra_x)
            sig_intra_sum_X += sig_sims_intra_x.sum().item()
            sig_intra_cnt_X += sig_sims_intra_x.nonzero(as_tuple=False).size(0)

            sig_sims_inter_x = torch.mm(X_c, X_out.T)
            sig_sims_inter_x = torch.sigmoid(sig_sims_inter_x)
            sig_inter_sum_X += sig_sims_inter_x.sum().item()
            sig_inter_cnt_X += sig_sims_inter_x.numel()


            #----测试归一化后的结果
            sig_sims_intra = torch.mm(h_c, h_c.T)  # (m,m)
            sig_sims_intra = sig_sims_intra.triu(diagonal=1)  # 上三角去对角
            sig_sims_intra = torch.sigmoid(sig_sims_intra)  # ★ 加激活
            sig_intra_sum_H += sig_sims_intra.sum().item()
            sig_intra_cnt_H += sig_sims_intra.nonzero(as_tuple=False).size(0)

            sig_sims_inter = torch.mm(h_c, h_out.T)  # (m,n)
            sig_sims_inter = torch.sigmoid(sig_sims_inter)  # ★ 加激活
            sig_inter_sum_H += sig_sims_inter.sum().item()
            sig_inter_cnt_H += sig_sims_inter.numel()
    # ----------------------- 计算平均值 -----------------------
    μ_intra_H = intra_sum_H / intra_cnt_H
    μ_inter_H = inter_sum_H / inter_cnt_H

    sig_μ_intra_H =  sig_intra_sum_H /  sig_intra_cnt_H
    sig_μ_inter_H =  sig_inter_sum_H /  sig_inter_cnt_H

    μ_intra_X = intra_sum_X / intra_cnt_X
    μ_inter_X = inter_sum_X / inter_cnt_X

    sig_μ_intra_X = sig_intra_sum_X / sig_intra_cnt_X
    sig_μ_inter_X = sig_inter_sum_X / sig_inter_cnt_X

    logger.info(f"H:  μ_intra={μ_intra_H:.4f}, μ_inter={μ_inter_H:.4f}")
    logger.info(f"归一化后的H:  sig_μ_intra={sig_μ_intra_H:.4f}, sig_μ_inter={sig_μ_inter_H:.4f}")
    logger.info(f"X:  μ_intra={μ_intra_X:.4f}, μ_inter={μ_inter_X:.4f}")
    logger.info(f"sigmoid-X: μ_intra={sig_μ_intra_X:.4f}, μ_inter={sig_μ_inter_X:.4f}")

    return μ_intra_H, μ_inter_H, μ_intra_X, μ_inter_X


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--count', type=int, default=1)  # 计数？？？
    parser.add_argument('--root', type=str, default='./data')  # 默认的数据集根节点
    parser.add_argument("--log", type=bool, default=True, help='run prepare_data or not')

    parser.add_argument('--method', type=str, default='COCLEP',
                        choices=['EmbLearner', 'COCLE', 'EmbLearnerWithoutHyper', 'EmbLearnerwithWeights'])
    parser.add_argument('--model_path', type=str, default='CS')  # 模型路径
    parser.add_argument('--m_model_path', type=str, default='META')

    parser.add_argument('--dataset', type=str, default='cocs')  # 默认的数据集
    parser.add_argument('--pos_size', type=int, default=3)  # 每个训练查询顶点给出的标签数
    parser.add_argument('--train_size', type=int, default=300)  # 默认的训练集大小
    parser.add_argument('--val_size', type=int, default=100)  # 默认的验证集大小
    parser.add_argument('--test_size', type=int, default=500)  # 默认的测试集大小
    parser.add_argument('--train_path', type=str, default='3_pos_train')  # 训练积路径
    parser.add_argument('--test_path', type=str, default='3_test')  # 测试集路径
    parser.add_argument('--val_path', type=str, default='3_val')  # 验证集路径
    parser.add_argument('--feats_path', type=str, default='feats.txt')  # 默认的特征数据
    parser.add_argument('--val_type', type=str, default='f1', help='pre or f1 to val')

    # 控制攻击方法、攻击类型和攻击率
    # choices=['none','meta', 'random_remove', 'random_flip', 'random_add', 'meta_attack', 'add', 'del','gflipm', 'gdelm', 'gaddm', 'cdelm', 'cflipm', 'delm', 'flipm']
    parser.add_argument('--attack', type=str, default='none')
    parser.add_argument('--type', type=str, default='add', help='random attack type', choices=['add', 'remove', 'flip'])
    parser.add_argument('--noise_level', type=int, default=3, choices=[1, 2, 3], help='noisy level')
    parser.add_argument('--ptb_rate', type=float, default=0.0, help='pertubation rate')

    # 模型batch大小，隐藏层维度，训练epoch数，drop_out，学须率lr，权重衰减weight_decay
    parser.add_argument('--batch_size', type=int, default=64)  # 默认的batch大小
    parser.add_argument('--hidden_dim', type=int, default=256)  # 隐藏层维度
    parser.add_argument('--num_layers', type=int, default=3)  # 模型层数
    parser.add_argument('--epoch_n', type=int, default=100)  # epoch数
    parser.add_argument('--drop_out', type=float, default=0.1)  # drop_out率
    parser.add_argument('--lr', type=float, default=0.001)  # 学习率
    parser.add_argument('--weight_decay', type=float, default=0.0005)  # 权重衰减率

    # 注意力系数tau，不同损失函的比率，超图跳数k
    parser.add_argument('--tau', type=float, default=0.2)  # 超参数tau
    parser.add_argument('--alpha', type=float, default=0.2)  # 超参数alpha
    parser.add_argument('--lam', type=float, default=0.2)  # 超参数lambda
    parser.add_argument('--k', type=int, default=2)  # 超参数？用1还是2？

    parser.add_argument('--cluster_number', type=int, default=10)  # 聚类的数量（我猜是分区的数量）

    parser.add_argument('--test_every', type=int, default=200)  # ？？？？？？
    parser.add_argument('--result', type=str, default='_Cluster_CLCS_result.txt')  # ？？？？
    # 权重计算模型及其学习率
    parser.add_argument('--mw_net', type=str, default='MLP', choices=['MLP', 'GCN'], help='type of meta-weighted model')
    parser.add_argument('--m_lr', type=float, default=0.005, help='learning rate of meta model')
    # 更新图的阈值。pa加边阈值，pd删边阈值。
    parser.add_argument('--pa', type=float, default=0.7)
    parser.add_argument('--pd', type=float, default=0.3)
    parser.add_argument('--n_p', type=int, default=5, help='number of positive pairs per node')
    parser.add_argument("--n_n", type=int, default=5, help='number of negitive pairs per node')
    parser.add_argument('--sigma', type=float, default=100,
                        help='the parameter to control the variance of sample weights in rec loss')
    parser.add_argument('--t_delete', type=float, default=0.1,
                        help='threshold of eliminating the edges')
    parser.add_argument('--gamma', type=float, default=0.01,
                        help='weight of rec loss')

    args = parser.parse_args()

    if args.log:
        log_path = get_log_path('./log/coclep/', args)
        logger = get_logger(log_path)
        print(f'save logger to {log_path}')
    else:
        logger = get_logger()

    μ_intra_H, μ_inter_H, μ_intra_X, μ_inter_X = Val_CommunitySearch(args, logger)
    print(f"{args.dataset}_{args.attack}_{args.ptb_rate}：μ_intra_H = {μ_intra_H:.4f},  μ_inter = {μ_inter_H:.4f}")
    print(f"{args.dataset}_{args.attack}_{args.ptb_rate}：μ_intra_x = {μ_intra_X:.4f},  μ_inter = {μ_inter_X:.4f}")
















