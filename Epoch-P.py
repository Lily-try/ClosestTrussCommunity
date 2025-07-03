import argparse
import copy
import datetime
import glob
import os
#import random
import random
import re

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

from Epoch import extract_epoch
from utils.query_utils import gen_all_queries, read_community_data, read_queries_from_file, write_queries_to_file, \
    write_onlynodes_to_file
from utils.preprocess import preprocess_graph_dataset, relabel_graph, reorder_features, remap_query_data, \
    remap_cluster_membership
# from model_GCN import ConRC
from utils.citation_loader import citation_feature_reader
from utils.cocle_val_utils import get_model_path, get_res_path, get_comm_path, get_epoch_res_path
# from coclep_citation import validation_part_pre
from utils.log_utils import get_log_path, get_logger
from models.COCLE import COCLE
from utils.load_utils import load_graph, loadQuerys

'''
对于大数据集测试epoch的影响
'''

def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False# 与上面一条代码配套使用，True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现
    torch.backends.cudnn.deterministic = True # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.enabled = False  # 禁用cudnn使用非确定性算法

    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # 在cuda 10.2及以上的版本中，需要设置以下环境变量来保证cuda的结果可复现
    os.environ['PYTHONHASHSEED'] = str(seed)  # 禁止hash随机化
    #torch.use_deterministic_algorithms(True) # 一些操作使用了原子操作，不是确定性算法，不能保证可复现，设置这个禁用原子操作，保证使用确定性算法

def f1_score_(comm_find, comm):

    lists = [x for x in comm_find if x in comm]
    if len(lists) == 0:
        #print("f1, pre, rec", 0.0, 0.0, 0.0)
        return 0.0, 0.0, 0.0
    pre = len(lists) * 1.0 / len(comm_find)
    rec = len(lists) * 1.0 / len(comm)
    f1 = 2 * pre * rec / (pre + rec)
    #print("f1, pre, rec", f1, pre, rec)
    return f1, pre, rec

def NMI_score(comm_find, comm, n_nodes):

    truthlabel = np.zeros((n_nodes), dtype=int)
    truthlabel[comm] = 1
    prelabel = np.zeros((n_nodes), dtype=int)
    prelabel[comm_find] = 1
    score = normalized_mutual_info_score(truthlabel, prelabel)
    #print("q, nmi:", score)
    return score

def ARI_score(comm_find, comm, n_nodes):

    truthlabel = np.zeros((n_nodes), dtype=int)
    truthlabel[comm] = 1
    prelabel = np.zeros((n_nodes), dtype=int)
    prelabel[comm_find] = 1
    score = adjusted_rand_score(truthlabel, prelabel)
    #print("q, ari:", score)

    return score

def JAC_score(comm_find, comm, n_nodes):
    truthlabel = np.zeros((n_nodes), dtype=int)
    truthlabel[comm] = 1
    prelabel = np.zeros((n_nodes), dtype=int)
    prelabel[comm_find] = 1
    score = jaccard_score(truthlabel, prelabel)
    #print("q, jac:", score)
    return score

def validation__(val, cluster_membership, sg_nodes, n_nodes, nodes_feats, device, g, model, nodes_adj,k):
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

        #为什么到了photo数据集这里就报错了
        # sim = F.cosine_similarity(h[nodes_[q]].unsqueeze(0), h, dim=1)  # (115,)
        # simlists = torch.sigmoid(sim.squeeze(0)).to(torch.device('cpu')).numpy().tolist()  # torch.sigmoid(simlists).numpy().tolist()

        sim = F.cosine_similarity(h[nodes_[q]].unsqueeze(0), h, dim=1)  # (num_nodes_in_subgraph,)
        simlists = torch.sigmoid(sim).cpu().numpy().tolist()


        comm_ = [nodes_[x] for x in comm if x in nodeslists]
        if len(comm_)!=len(comm):
            size = len(comm)-len(comm_)
            for i in range(size):
                comm_.append(i+len(nodeslists))
        scorelists.append([nodes_[q], comm_, simlists])
    s_ = 0.1
    f1_m = 0.0
    s_m = s_
    while(s_<=0.9):
        f1_x = 0.0
        # print("------------------------------", s_)
        for q, comm, simlists in scorelists:
            comm_find = []
            for i, score in enumerate(simlists):
                if score >=s_ and i not in comm_find:
                    comm_find.append(i)

            comm_find = set(comm_find)
            comm_find = list(comm_find)
            comm = set(comm)
            comm = list(comm)
            f1, pre, rec = f1_score_(comm_find, comm)
            f1_x= f1_x+f1#pre
        f1_x = f1_x/len(val)
        if f1_m<f1_x:
            f1_m = f1_x
            s_m = s_
        s_ = s_+0.05
    # print("------------------------", s_m, f1_m)
    logger.info(f'best threshold: {s_m}, validation_set Avg F1: {f1_m}')
    return s_m, f1_m

def validation_part_pre(val, cluster_membership, sg_nodes, n_nodes, nodes_feats, device, g, model, nodes_adj,k):
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

'''
进行图划分
graph：是图，
cluster_number：是要划分的簇的数目
'''
def metis_clustering(cluster_number,graph):
    # 使用metis的方法进行图划分，
    adjacency_list=[]
    for node in graph.nodes():
        neighbors = list(graph.neighbors(node))
        # adjacency_list.append([node_to_index[neighbor] for neighbor in neighbors])
        adjacency_list.append(neighbors)
    (st, parts) = metis.part_graph(cluster_number,adjacency=adjacency_list)
    clusters = list(set(parts)) #创建包含所有簇的列表，并通过set()进行簇标识去重。
    cluster_membership = {} #字典，用于存储每个节点所属的簇
    for i, node in enumerate(graph.nodes): #遍历图中的节点，同时获得节点索引i和节点对象node
        cluster = parts[i] #获取第i个节点所属簇的标识
        cluster_membership[node] = cluster #将节点`node`和其所属的簇的标识`cluster`添加到字典中，建立了节点到簇的映射关系
    return clusters, cluster_membership     #返回簇的列表和节点到簇的映射

def louvain_clustering(graph):
    # 生成每个节点所属的社区（返回的是dict：node -> community_id）
    partition = community_louvain.best_partition(graph)
    clusters = list(set(partition.values()))
    return clusters, partition

def gen_new_queries(args,valid_nodes):
    print('从gt社区中生成查询任务并划分')
    # 3.从comms文件中读取所有社区列表
    if args.dataset in ['photo_stb','photo_gsr','cocs_stb','cocs_gsr']:
        communities = read_community_data(args.root,  args.dataset[:-4])
    else:
        communities = read_community_data(args.root, args.dataset)
    all_query_nums = args.train_size + args.val_size + args.test_size
    gen_all_queries(args.root, args.dataset, communities, all_query_nums, args.pos_size, args.pos_size, threshold=.8,
                    constrain='True', valid_nodes=valid_nodes)

    split_ratios = (args.train_size, args.val_size, args.test_size)
    queries = read_queries_from_file(f'{args.root}/{args.dataset}/{args.dataset}_all_queries.txt')  # 读取查询任务
    dataset_root = os.path.join(args.root, args.dataset)
    train_path = os.path.join(dataset_root, f'{args.dataset}_{args.attack}_{args.ptb_rate}_{args.pos_size}_pos_train_{split_ratios[0]}.txt')
    val_path = os.path.join(dataset_root, f'{args.dataset}_{args.attack}_{args.ptb_rate}_{args.pos_size}_val_{split_ratios[1]}.txt')
    test_path = os.path.join(dataset_root, f'{args.dataset}_{args.attack}_{args.ptb_rate}_{args.pos_size}_test_{split_ratios[2]}.txt')
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


def gen_new_queries_(root,dataset,train_size,val_size,test_size,pos_size,valid_nodes):
    print('从gt社区中生成查询任务并划分')
    # 3.从comms文件中读取所有社区列表
    if dataset in ['photo_stb','photo_gsr','cocs_stb','cocs_gsr']:
        communities = read_community_data(root,  dataset[:-4])
    else:
        communities = read_community_data(root, dataset)
    all_query_nums = train_size +val_size + test_size
    gen_all_queries(root, dataset, communities, all_query_nums,pos_size,pos_size, threshold=.8,
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
    query_path = os.path.join(root,dataset, f'{dataset}_querynode.txt')
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
def load_data(args):#dataset, root, train_n, val_n, test_n, feats_path, cluster_number, train_path,test_path, val_path):
    #加载图。存在一个问题是重新编号后，最终保存的.comms文件是找不到原始的id的了。
    graphx, n_nodes = load_graph(args.root, args.dataset, args.attack, args.ptb_rate)
    graphx, mapping = relabel_graph(graphx)

    #生成查询
    valid_nodes = set(graphx.nodes())
    gen_new_queries(args,valid_nodes)
    #下面这行则是不同于COLCE的地方，进行了图分区。
    #clusters是分区簇列表，cluster_membership是字典
    #cluster是{list:100}从0一直到99；cluster_membership是{dict:2708}例如key是节点id，value是节点所属的分区id。0:80;
    print("---------------------cluster开始-------------------------------")
    clusters, cluster_membership = metis_clustering(args.cluster_number,graphx)
    # clusters,cluster_membership = louvain_clustering(graphx)
    cluster_membership = remap_cluster_membership(cluster_membership, mapping)
    print("---------------------cluster结束-------------------------------")

    nodes_adj = {} #字典，用于存储节点的邻接信息。无向图
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
    sg_nodes = {} #字典，用于存储分区后的节点信息。key是分区标识，value是分区内的节点

    for u, c in cluster_membership.items():
        if c not in sg_nodes:
            sg_nodes[c] = []
        sg_nodes[c].append(u) #{c1:1,2,3},{c2:4,5,6}

    #调用loadQuery加载训练、验证和测试数据
    logger.info('正在加载训练数据')
    train, val, test = loadQuerys(args.dataset, args.root, args.train_size, args.val_size, args.test_size, args.train_path, args.test_path, args.val_path)
    train, val, test = remap_query_data(train, val, test, mapping)
    logger.info('加载训练数据完成')

    print("====================3.加载特征数据featd==================================")
    logger.info('正在加载特征数据')
    if args.dataset in ['cora','citeseer_stb','pubmed','citeseer']:
        nodes_feats = citation_feature_reader(args.root, args.dataset)  # numpy.ndaaray:(2708,1433)
        nodes_feats = torch.from_numpy(nodes_feats)  # 转换成tensor
        node_in_dim = nodes_feats.shape[1]
        print(f'{args.dataset}的feats dtype: {nodes_feats.dtype}')
    elif args.dataset in ['cora_stb','cora_gsr']:
        nodes_feats = citation_feature_reader(args.root, args.dataset[:-4])  # numpy.ndaaray:(2708,1433)
        nodes_feats = torch.from_numpy(nodes_feats)  # 转换成tensor
        node_in_dim = nodes_feats.shape[1]
    elif args.dataset in ['fb107_gsr','fb107_stb']:
        feats_array = np.loadtxt(f'{args.root}/{args.dataset[:-4]}/{args.dataset[:-4]}.feat', delimiter=' ', dtype=np.float32)
        print(type(feats_array))
        # nodes_feats = fnormalize(feats_array)  # 将特征进行归一化
        nodes_feats = torch.from_numpy(feats_array)
        node_in_dim = nodes_feats.shape[1]
    elif args.dataset in ['cocs','photo']:
        with open(f'{args.root}/{args.dataset}/{args.dataset}.feats', "r") as f:
            # 每行特征转换为列表，然后堆叠为 ndarray,注意要是float32
            nodes_feats = np.array([list(map(float, line.strip().split())) for line in f],dtype=np.float32)
            print(f'{args.dataset}的nodes_feats.dtype = {nodes_feats.dtype}')
            print(f'{args.dataset}的节点特征shape:', nodes_feats.shape)
            nodes_feats = torch.from_numpy(nodes_feats)  # 转换成tensor
            node_in_dim = nodes_feats.shape[1]
    elif args.dataset in ['dblp','amazon']:
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
    elif args.dataset in ['dblp_gsr','dblp_stb','amazon_gsr','amazon_stb']:
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
    elif args.dataset in ['photo_gsr','photo_stb','cocs_stb','cocs_gsr']:
        feats_array = np.loadtxt(f'{args.root}/{args.dataset[:-4]}/{args.dataset[:-4]}.feats', delimiter=' ', dtype=np.float32)
        print(type(feats_array))
        # nodes_feats = fnormalize(feats_array)  # 将特征进行归一化
        nodes_feats = torch.from_numpy(feats_array)
        node_in_dim = nodes_feats.shape[1]
    # elif args.dataset.startswith(('fb', 'wfb', 'fa')):  # 不加入中心节点
    elif args.dataset in ['fb107','wfb107']:  # 不加入中心节点
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
        feats_array = np.loadtxt(f'{args.root}/{args.dataset[:-4]}/{args.dataset[:-4]}.feat', dtype=float, delimiter=' ')
        nodes_feats = torch.tensor(feats_array, dtype=torch.float32)
        node_in_dim = nodes_feats.shape[1]
    else:
        logger.info('加载节点特征失败，数据集不匹配')
    nodes_feats = reorder_features(nodes_feats, mapping)
    logger.info('加载节点特征完成完成')
    '''  
    nodes_feats = nodes_feats.transpose(0, 1)
    rowsum = nodes_feats.sum(1)
    rowsum[rowsum == 0] = 1
    print(rowsum)
    nodes_feats = nodes_feats / rowsum[:, np.newaxis]
    nodes_feats = nodes_feats.transpose(0, 1)
    #'''

    #进行统一编号处理
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
        value = edge_index.new_ones((edge_index.size(1), ), dtype=torch.float)
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
    #q是查询节点，pos是正例，train是社区节点列表。从每个社区中随机抽取一个节点作为查询节点
    for q, pos, comm in train:
        nodeslists = sg_nodes[cluster_membership[q]] #q所在分区的节点列表

        nodelistb = [] #存储此分区边界节点
        if q in nodes_adj: #检查查询节点q是否在节点邻接列表中
            neighbor = nodes_adj[q] #如果在，则获得查询节点q的邻接节点列表
            nodelistb = [x for x in neighbor if x not in nodeslists] #不在此分区中的邻居
        nodelistb = set(nodelistb) # 去重，确保只出现1次
        nodelistb = list(nodelistb)# 转换为列表
        nodeslists = nodeslists+nodelistb # 将nodeListb中的边界节点加入到Nodelists中
        lists = [x for x in comm if x not in nodeslists] #不在此分区（分区节点+边界节点）但在社区中的顶点。
        #打印此分区的节点数，此分区边界结点数，被分到其他分区的节点数，社区节点数
        print(len(nodeslists), len(nodelistb), len(lists), len(comm))
        mask = [False]*n_nodes #创建初始为false的长度为节点数的布尔列表mask
        for u in nodeslists: #将在此分区中的节点设置为True
            mask[u] = True
        feats = nodes_feats[mask] #使用mask获得此分区（分区+边界）的节点特征
        nodeslists = sorted(nodeslists) #排序，确保节点的顺序
        nodes_ = {} #字典，key是节点编号，value是在nodelist中的索引
        for i, u in enumerate(nodeslists):
            nodes_[u]=i
        sub = g.subgraph(nodeslists) #创建分区（分区+边界）的导出子图  。怎么知道子图的边是哪些的？？？？
        src = [] #存储子图中边的源节点
        dst = []#存储子图中边的目标节点
        # 用于构建超图边，然后将它们与原始图的边合并，结果存储在edge)index中。
        for id1, id2 in sub.edges:
            id1_ = nodes_[id1]   #获得源节点id1在nodes中的索引
            id2_ = nodes_[id2]   #获得目标节点id2在nodes中的索引
            src.append(id1_)
            dst.append(id2_)
            src.append(id2_)
            dst.append(id1_)
        edge_index = torch.tensor([src, dst],dtype=torch.long)
        # print("在decompose_train中edge_index fixed dtype:", edge_index.dtype)
        edge_index_aug, egde_attr = hypergraph_construction(edge_index, len(nodeslists), k = k)
        edge_index = add_remaining_self_loops(edge_index, num_nodes=len(nodeslists))[0] #添加自环？
        pos_ = [nodes_[x] for x in pos if x in nodeslists]  #在此分区（分区+边界）中的标记节点的索引。
        if edge_index.dtype != torch.long:
            edge_index = edge_index.long()
            edge_index_aug = edge_index_aug.long()
            # print('出现了不是long的')
        #pos_ = [nodes_[x] for x in comm if x in nodeslists and x!=q]
        train_lists.append((nodes_[q], pos_, edge_index, edge_index_aug, feats))

    return train_lists

'''
进行图划分的社区搜索方法
'''
def CommunitySearch(args):
    preprocess_start = datetime.datetime.now() #进行社区搜索前的时间
    #load_data加载数据集
    nodes_feats, train_, val, test, node_in_dim, n_nodes, g, \
    sg_nodes, clusters, cluster_membership, nodes_adj = load_data(args)
    print(f'feats的shape：{node_in_dim} ,{nodes_feats.shape}')
    print(f'节点数：{g.number_of_nodes()},边数：{g.number_of_edges()}')
    print('开始重构训练集')
    #对训练集进行重构，得到了可以直接输入到pytorch的训练集 train_lists.append((nodes_[q], pos_, edge_index, edge_index_aug, feats))
    trainlists  = decompose_train(g, train_, val, nodes_feats, n_nodes,cluster_membership,
                                            sg_nodes, args.k, nodes_adj)
    print(len(trainlists))  #打印数量
    print('训练集重构成功')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    logger.info(f'device:{device}') #打印设备

    train = []
    for q, pos, edge_index, edge_index_aug, feats in trainlists:
        if edge_index.dtype != torch.long:
            edge_index = edge_index.long()
        if edge_index_aug.dtype != torch.long:
            edge_index_aug = edge_index_aug.long()

        edge_index = edge_index.to(device)
        edge_index_aug = edge_index_aug.to(device)
        feats = feats.to(device)
        train.append((q, pos, edge_index, edge_index_aug, feats))


    model = COCLE(node_in_dim, args.hidden_dim, args.num_layers, args.drop_out, args.tau, device, args.alpha,
                       args.lam, args.k)  # COCLEP中的模型，目前和EmbLearner是一样的
    logger.info(f'embLearner: {args.method}')
    optimizer = th.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.to(device)
    # model.reset_parameters()
    pre_process_time = (datetime.datetime.now() - preprocess_start).seconds #到现在结束了预处理过程？


    logger.info('start trainning')
    train_start = datetime.datetime.now()
    optimizer.zero_grad() #梯度清零
    val_epochs_time = 0.0  # 记录整个epoch下的时间
    bst_model_path = get_model_path('./results/coclep/res_model/', args)
    # 这部分是训练的代码
    ckpt_epochs = []
    for epoch in range(120): #训练模型
        model.train()
        start = datetime.datetime.now()
        loss_b = 0.0
        i = 0
        for q, pos, edge_index, edge_index_aug, feats in train:
            if len(pos) == 0:
                i = i + 1
                continue
            loss,h = model((q, pos, edge_index, edge_index_aug, feats))
            loss_b = loss_b + loss
            loss.backward() #反向传播
            if (i + 1) % args.batch_size == 0:
                optimizer.step() #更新模型参数
                optimizer.zero_grad()
            i = i + 1
        if epoch % 10 == 0:
            ckpt_file = f'{bst_model_path}_{epoch}.pkl'
            ckpt_epochs.append(epoch)
            torch.save(model.state_dict(), ckpt_file)  #
            logger.info(f'[Checkpoint] saved to {ckpt_file}')
        epoch_time = (datetime.datetime.now() - start).seconds #当前的epoch，每个epoch的时间
        logger.info(f'epoch_loss = {loss_b}, epoch = {epoch}, epoch_time = {epoch_time}')
    training_time = (datetime.datetime.now() - train_start).seconds - val_epochs_time  # 将验证的时间减去
    logger.info(f'trainning time = {training_time},validate time ={val_epochs_time}')

    logger.info(f'#################### Starting evaluation######################')

    all_ckpts = glob.glob(f'{bst_model_path}_*.pkl')
    # 过滤掉名字不合法的文件
    ckpt_files = [p for p in all_ckpts if extract_epoch(p) is not None]
    # 如果目录里啥都没有，打个日志提醒一下（可选）
    if not ckpt_files:
        logger.warning(f'No valid checkpoints found under pattern {bst_model_path}_*.pkl')
    # 对合法检查点按 epoch 升序排序
    ckpt_files.sort(key=extract_epoch)

    eval_start = datetime.datetime.now()

    # 结果文件准备
    output = get_epoch_res_path('./results/coclep/', args)
    os.makedirs(os.path.dirname(output), exist_ok=True)

    with open(output, 'a+', encoding='utf-8') as fh:
        for ckpt in ckpt_files:
            epoch_id = int(re.search(r'_(\d+)\.pkl', ckpt).group(1))
            logger.info(f'------ Evaluating checkpoint @ epoch {epoch_id} ------')
            model.load_state_dict(torch.load(ckpt, map_location=device))
            model.eval()
            eval_start = datetime.datetime.now()
            F1 = Pre = Rec = nmi_score = ari_score = jac_score = count = 0.0
            with th.no_grad():
                # 使用验证集数据找打最佳阈值s_
                if args.val_type == 'f1':
                    # s_, f1_ = validation(val, nodes_feats, embLearner, edge_index, edge_index_aug)
                    s_, best_metric = validation__(val, cluster_membership, sg_nodes,
                                           n_nodes, nodes_feats, device, g, model, nodes_adj, args.k)
                    logger.info(
                        f'evaluation time = {datetime.datetime.now() - eval_start}, best s_={s_}, best val f1_={best_metric}')
                elif args.val_type == 'pre':
                    # s_, pre_ = validation_pre(val, nodes_feats, embLearner, edge_index, edge_index_aug)
                    s_, best_metric = validation_part_pre(val, cluster_membership, sg_nodes,
                                                  n_nodes, nodes_feats, device, g, model, nodes_adj, args.k)
                    logger.info(
                        f'evaluation time = {datetime.datetime.now() - eval_start}, best s_={s_}, best val pre_={best_metric}')
                val_running_time = (datetime.datetime.now() - eval_start).seconds  # 结束了测试运行的时间
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

                    count = count + 1

                    numerator = torch.mm(h[nodes_[q]].unsqueeze(0), h.t())
                    norm = torch.norm(h, dim=-1, keepdim=True)
                    denominator = torch.mm(norm[nodes_[q]].unsqueeze(0), norm.t())
                    sim = numerator / denominator
                    # 得到每个节点的相似度得分
                    simlists = torch.sigmoid(sim.squeeze(0)).to(
                        torch.device('cpu')).numpy().tolist()  # torch.sigmoid(simlists).numpy().tolist()

                    comm_find = []
                    for i, score in enumerate(simlists):
                        if score >= s_ and nodeslists[i] not in comm_find:  # 将此查询节点子图中满足得分阈值的节点加入到comm_find
                            comm_find.append(nodeslists[i])
                    lists = []  # 用于存储边界节点所在的分区编号
                    for qb in nodelistb:  # 处理边界节点
                        if cluster_membership[qb] in lists:  # 如果qb所在的分区编号在list中，则继续
                            continue
                        lists.append(cluster_membership[qb])  # 将qb所在的分区编号加入到list
                        qb_ = nodes_[qb]  # 获得qb的索引
                        if simlists[qb_] < s_:  # 判断qb的相似度得分不满足阈值，则不管
                            continue
                        nodeslists_ = sg_nodes[cluster_membership[qb]]  # 否则，将获得qb所在的分区编号中的所有节点nodelists

                        nodelistb_ = []
                        if qb in nodes_adj:
                            neighbor_ = nodes_adj[qb]
                            nodelistb_ = [x for x in neighbor_ if x not in nodeslists_]
                        nodelistb_ = set(nodelistb_)
                        nodelistb_ = list(nodelistb_)
                        nodeslists_ = nodeslists_ + nodelistb_  # 继续构造这个分区+边界。

                        qb = q  # 并将查询节点继续在新的子图上进行查询

                        mask_ = [False] * n_nodes
                        for u in nodeslists_:
                            mask_[u] = True
                        feats = nodes_feats[mask_].to(device)
                        nodeslists_ = sorted(nodeslists_)
                        nodes__ = {}
                        for i, u in enumerate(nodeslists_):
                            nodes__[u] = i
                        sub_ = g.subgraph(nodeslists_)
                        src_ = []
                        dst_ = []
                        for id1_, id2_ in sub_.edges:
                            id1__ = nodes__[id1_]
                            id2__ = nodes__[id2_]
                            src_.append(id1__)
                            dst_.append(id2__)
                            src_.append(id2__)
                            dst_.append(id1__)
                        edge_index_ = torch.tensor([src_, dst_])
                        edge_index_aug_, egde_attr_ = hypergraph_construction(edge_index_, len(nodeslists_), k=args.k)
                        edge_index_ = add_remaining_self_loops(edge_index_)[0].to(device)
                        h_ = model((nodes__[qb], None, edge_index_, edge_index_aug_, feats))  # 运行模型得到相应的表示
                        h_[nodes__[qb]] = h_[nodes__[qb]] + h[nodes_[q]]

                        numerator_ = torch.mm(h_[nodes__[qb]].unsqueeze(0), h_.t())
                        norm_ = torch.norm(h_, dim=-1, keepdim=True)
                        denominator_ = torch.mm(norm_[nodes__[qb]].unsqueeze(0), norm_.t())
                        sim_ = numerator_ / denominator_
                        simlists_ = torch.sigmoid(sim_.squeeze(0)).to(
                            torch.device('cpu')).numpy().tolist()

                        for i, score in enumerate(simlists_):
                            if score >= s_ and nodeslists_[i] not in comm_find:
                                comm_find.append(nodeslists_[i])

                    comm_find = set(comm_find)
                    comm_find = list(comm_find)
                    # 将找到的社区结果存入文件
                    comm_path = get_comm_path('./results/coclep/', args)
                    logger.info(f'找到的社区被存入了{comm_path}')
                    with open(comm_path, 'a', encoding='utf-8') as f:
                        line = str(q) + "," + " ".join(str(u) for u in comm_find)
                        f.write(line + "\n")

                    comm = set(comm)
                    comm = list(comm)
                    f1, pre, rec = f1_score_(comm_find, comm)
                    F1 = F1 + f1
                    Pre = Pre + pre
                    Rec = Rec + rec

                    nmi = NMI_score(comm_find, comm, n_nodes)
                    nmi_score = nmi_score + nmi

                    ari = ARI_score(comm_find, comm, n_nodes)
                    ari_score = ari_score + ari

                    jac = JAC_score(comm_find, comm, n_nodes)
                    jac_score = jac_score + jac
            # 结束了测试阶段，计算测试集上的平均F1,Pre和Rec并打印
        test_running_time = (datetime.datetime.now() - test_start).seconds  # 结束了测试运行的时间

        F1 = F1 / len((test))
        Pre = Pre / len((test))
        Rec = Rec / len((test))
        nmi_score = nmi_score / len(test)
        ari_score = ari_score / len(test)
        jac_score = jac_score / len(test)
        logger.info(f'Test time = {test_running_time}')
        logger.info(f'Test_set Avg：F1 = {F1}, Pre = {Pre}, Rec = {Rec}, s = {s_}')
        logger.info(f'Test_set Avg NMI = {nmi_score}, ARI = {ari_score}, JAC = {jac_score}')
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fh.write(
            f"epoch: {epoch_id}\n"
            f"args: {args}\n"
            f"val_type: {args.val_type}\n"
            f"best_comm_threshold: {s_}\n"
            f"best_validation_metric: {best_metric}\n"
            f"pre_process_time: {pre_process_time}\n"
            f"training_time: {training_time}\n"
            f"val_time: {val_running_time}\n"
            f"test_time: {test_running_time}\n"
            f"F1: {F1}\nPre: {Pre}\nRec: {Rec}\n"
            f"NMI: {nmi_score}\nARI: {ari_score}\nJAC: {jac_score}\n"
            f"timestamp: {now}\n"
            "----------------------------------------\n"
        )
        fh.flush()  # 立即写盘，防止中途退出丢数据

        # ★★★ 这里清理 GPU 缓存，让下一轮更干净
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    return F1, Pre, Rec, nmi_score, ari_score, jac_score, pre_process_time, training_time,val_running_time, test_running_time

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--count', type=int, default=1) #计数？？？
    parser.add_argument('--root', type=str, default='./data') #默认的数据集根节点
    parser.add_argument("--log", type=bool, default=True, help='run prepare_data or not')

    parser.add_argument('--method', type=str, default='COCLEP',
                        choices=['EmbLearner', 'COCLE', 'EmbLearnerWithoutHyper', 'EmbLearnerwithWeights'])
    parser.add_argument('--model_path', type=str, default='CS') #模型路径
    parser.add_argument('--m_model_path', type=str, default='META')


    parser.add_argument('--dataset', type=str, default='photo_stb') #默认的数据集
    parser.add_argument('--pos_size', type=int, default=3) #每个训练查询顶点给出的标签数
    parser.add_argument('--train_size', type=int, default=300)  # 默认的训练集大小
    parser.add_argument('--val_size', type=int, default=100)  # 默认的验证集大小
    parser.add_argument('--test_size', type=int, default=500)  # 默认的测试集大小
    parser.add_argument('--train_path', type=str, default='3_pos_train')  # 训练积路径
    parser.add_argument('--test_path', type=str, default='3_test')  # 测试集路径
    parser.add_argument('--val_path', type=str, default='3_val')  # 验证集路径
    parser.add_argument('--feats_path', type=str, default='feats.txt') #默认的特征数据
    parser.add_argument('--val_type', type=str, default='f1', help='pre or f1 to val')

    # 控制攻击方法、攻击类型和攻击率
    #choices=['none','meta', 'random_remove', 'random_flip', 'random_add', 'meta_attack', 'add', 'del','gflipm', 'gdelm', 'gaddm', 'cdelm', 'cflipm', 'delm', 'flipm']
    parser.add_argument('--attack', type=str, default='none')
    parser.add_argument('--type', type=str, default='add', help='random attack type', choices=['add', 'remove', 'flip'])
    parser.add_argument('--noise_level', type=int, default=3, choices=[1, 2, 3], help='noisy level')
    parser.add_argument('--ptb_rate', type=float, default=0.30, help='pertubation rate')

    # 模型batch大小，隐藏层维度，训练epoch数，drop_out，学须率lr，权重衰减weight_decay
    parser.add_argument('--batch_size', type=int, default=64)#默认的batch大小
    parser.add_argument('--hidden_dim', type=int, default=256) #隐藏层维度
    parser.add_argument('--num_layers', type=int, default=3) #模型层数
    parser.add_argument('--epoch_n', type=int, default=100) #epoch数
    parser.add_argument('--drop_out', type=float, default=0.1) #drop_out率
    parser.add_argument('--lr', type=float, default=0.001) # 学习率
    parser.add_argument('--weight_decay', type=float, default=0.0005)# 权重衰减率

    # 注意力系数tau，不同损失函的比率，超图跳数k
    parser.add_argument('--tau', type=float, default=0.2) #超参数tau
    parser.add_argument('--alpha', type=float, default=0.2) #超参数alpha
    parser.add_argument('--lam', type=float, default=0.2) #超参数lambda
    parser.add_argument('--k', type=int, default=2) #超参数？用1还是2？

    parser.add_argument('--cluster_number', type=int, default=100)  # 聚类的数量（我猜是分区的数量）


    parser.add_argument('--test_every', type=int, default=200) #？？？？？？
    parser.add_argument('--result', type=str, default='_Cluster_CLCS_result.txt') #？？？？
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

    pre_process_time_A, train_model_running_time_A,val_time_A, test_running_time_A = 0.0, 0.0,0.0,0.0
    count = 0 #计数，模型更新次数？？
    F1lists = [] #F1值
    Prelists = [] #精度
    Reclists = [] #Recall召回率
    nmi_scorelists = []
    ari_scorelists = []
    jac_scorelists = [] #Jaccard相似性

    
    for i in range(args.count): #count的作用我执行多少次，默认是1；之前不是实验中有说以5次运行的平均值巴拉巴拉。。。
        seed_all(0) #种子
        # args.model_path='Cluster-CLCS-6'+"-"+str(count)
        count = count+1
        print('= ' * 20) #哦，输出20个等号分隔
        print(count) #打印当前的count
        now = datetime.datetime.now()
        logger.info(f'##第 {count} 次执行, Starting Time: {now.strftime("%Y-%m-%d %H:%M:%S")}')
        #args.dataset='SimpleEx' #测试数据集
        F1, Pre, Rec, nmi_score, ari_score, jac_score, pre_process_time, train_model_running_time,val_time,test_running_time = \
            CommunitySearch(args) #记录各个所需的值
        # 打印结束时间
        logger.info(f'##第{count}次Finishing Time: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        running_time = (datetime.datetime.now() - now).seconds
        # 打印总的运行时间
        logger.info(f'##第{count}次Running Time(s): {running_time}')
        print('= ' * 20)

        F1lists.append(F1)
        Prelists.append(Pre)
        Reclists.append(Rec)
        nmi_scorelists.append(nmi_score)
        ari_scorelists.append(ari_score)
        jac_scorelists.append(jac_score)
        # 累计预处理时间、训练时间和测试时间
        pre_process_time_A = pre_process_time_A + pre_process_time #ALL总共的预处理时间
        train_model_running_time_A = train_model_running_time_A + train_model_running_time #ALL总共的模型运行时间
        test_running_time_A = test_running_time_A + test_running_time #总共的测试运行的时间


    F1_std = np.std(F1lists) #标准差
    F1_mean = np.mean(F1lists) #均值
    Pre_std = np.std(Prelists)
    Pre_mean = np.mean(Prelists)
    Rec_std = np.std(Reclists)
    Rec_mean = np.mean(Reclists)
    nmi_std = np.std(nmi_scorelists)
    nmi_mean = np.mean(nmi_scorelists)
    ari_std = np.std(ari_scorelists)
    ari_mean = np.mean(ari_scorelists)
    jac_std = np.std(jac_scorelists)
    jac_mean = np.mean(jac_scorelists)

    pre_process_time_A = pre_process_time_A/float(args.count) #平均的预处理时间
    train_model_running_time_A = train_model_running_time_A/float(args.count) #平均的模型训练时间
    val_time_A = val_time_A / float(args.count)
    test_running_time_A = test_running_time_A/float(args.count) #平均的运行时间
    single_query_time = test_running_time_A / float(args.test_size)  # 除以测试集大小得到测试时间

    output = get_epoch_res_path('s./results/coclep/', args)
    with open(output, 'a+',encoding='utf-8') as fh:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = (
            f"average {args}\n"
            f"pre_process_time: {pre_process_time_A}\n"
            f"train_model_running_time: {train_model_running_time_A}\n"
            f"val_time_A: {val_time_A}\n"
            f"test_running_time: {test_running_time_A}\n"
            f"single_query_time: {single_query_time}\n"
            f"F1 mean: {F1_mean}\n"
            f"F1 std: {F1_std}\n"
            f"Pre mean: {Pre_mean}\n"
            f"Pre std: {Pre_std}\n"
            f"Rec mean: {Rec_mean}\n"
            f"Rec std: {Rec_std}\n"
            f"nmi_score mean: {nmi_mean}\n"
            f"nmi std: {nmi_std}\n"
            f"ari_score mean: {ari_mean}\n"
            f"ari std: {ari_std}\n"
            f"jac mean: {jac_mean}\n"
            f"jac std: {jac_std}\n"
            f"current_time: {current_time}\n"
            "----------------------------------------\n"
        )
        fh.write(line + "\n")
        fh.close()














