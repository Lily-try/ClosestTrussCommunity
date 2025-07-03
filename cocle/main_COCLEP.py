import argparse
import datetime
import os
#import random
import random

import pymetis as metis
import networkx as nx
import torch
import torch as th
from community import community_louvain
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, jaccard_score
import numpy as np
from torch_geometric.data import Data
from torch_sparse import spspmm, coalesce
from torch_geometric.utils import remove_self_loops, add_remaining_self_loops

from citation_loader import citation_feature_reader
from model_GCN import ConRC
from load_utils import load_graph, loadQuerys
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

def validation__(val, cluster_membership, sg_nodes, n_nodes, nodes_feats, device, g, model, nodes_adj):
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
        edge_index_aug, egde_attr = hypergraph_construction(edge_index, len(nodeslists), k=args.k)
        edge_index = add_remaining_self_loops(edge_index, num_nodes=len(nodeslists))[0]

        h = model((nodes_[q], None, edge_index, edge_index_aug, feats))

        numerator = torch.mm(h[nodes_[q]].unsqueeze(0), h.t())
        norm = torch.norm(h, dim=-1, keepdim=True)
        denominator = torch.mm(norm[nodes_[q]].unsqueeze(0), norm.t())
        sim = numerator / denominator
        simlists = torch.sigmoid(sim.squeeze(0)).to(
            torch.device('cpu')).numpy().tolist()  # torch.sigmoid(simlists).numpy().tolist()
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
        print("------------------------------", s_)
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
    print("------------------------", s_m, f1_m)
    return s_m, f1_m

def loadQuerys(dataset, root, train_n, val_n, test_n, train_path, test_path, val_path):
    path_train = root + dataset + '/' + dataset + train_path
    if not os.path.isfile(path_train):
        raise Exception("No such file: %s" % path_train)
    train_lists = []
    for line in open(path_train, encoding='utf-8'):
        q, pos, comm = line.split(",")
        q = int(q)
        pos = pos.split(" ")
        pos_ = [int(x) for x in pos if int(x)!=q]
        comm = comm.split(" ")
        comm_ = [int(x) for x in comm]
        if len(train_lists)>=train_n:
            break
        train_lists.append((q, pos_, comm_))
    path_test = root + dataset + '/' + dataset + test_path
    if not os.path.isfile(path_test):
        raise Exception("No such file: %s" % path_test)
    test_lists = []
    for line in open(path_test, encoding='utf-8'):
        q, comm = line.split(",")
        q = int(q)
        comm = comm.split(" ")
        comm_ = [int(x) for x in comm]
        if len(test_lists)>=test_n:
            break
        test_lists.append((q, comm_))
    '''
    val_lists_ = test_lists[test_n:]
    test_lists = test_lists[:test_n]
    val_lists = []
    for q, comm in val_lists_:
        val_lists.append((q, comm))

    '''
    path_val = root + dataset + '/' + dataset + val_path
    if not os.path.isfile(path_val):
        raise Exception("No such file: %s" % path_val)
    val_lists = []
    for line in open(path_val, encoding='utf-8'):
        # q, pos, comm = line.split(",")
        q,comm = line.split(",")
        q = int(q)
        # pos = pos.split(" ")
        # pos_ = [int(x) for x in pos if int(x)!=q]
        comm = comm.split(" ")
        comm_ = [int(x) for x in comm]
        if len(val_lists)>=val_n:
            break
        val_lists.append((q, comm_))
    #'''

    return train_lists, val_lists, test_lists

'''
进行图划分
graph：是图，
cluster_number：是要划分的簇的数目
'''
def metis_clustering(cluster_number,graph):
    # 使用metis的方法进行图划分，
    # adjacency_list=[]
    # for node in graph.nodes():
    #     neighbors = list(graph.neighbors(node))
    #     # adjacency_list.append([node_to_index[neighbor] for neighbor in neighbors])
    #     adjacency_list.append(neighbors)
    #
    # assert isinstance(adjacency_list, list)
    # for neighbors in adjacency_list:
    #     assert isinstance(neighbors, list)
    #     for n in neighbors:
    #         assert isinstance(n, int)
    # #删除自环/重复邻居
    # adjacency_list = [list(set(neigh)) for neigh in adjacency_list]
    # adjacency_list = [
    #     [n for n in neigh if n != i]  # remove self-loops
    #     for i, neigh in enumerate(adjacency_list)
    # ]
    # total_nodes = len(adjacency_list)
    # for i, neigh in enumerate(adjacency_list):
    #     if len(neigh) == 0:
    #         # 随机选择一个不同于自己的节点
    #         candidates = list(range(total_nodes))
    #         candidates.remove(i)
    #         rand_neighbor = random.choice(candidates)
    #         print(f"Node {i} is isolated. Connecting it to {rand_neighbor}")
    #         adjacency_list[i].append(rand_neighbor)
    #         adjacency_list[rand_neighbor].append(i)  # 保持无向图一致性
    #
    # max_index = len(adjacency_list) - 1
    # for i, neigh in enumerate(adjacency_list):
    #     for n in neigh:
    #         if n < 0 or n > max_index:
    #             raise ValueError(f"Invalid neighbor index: {n} at node {i}")
    #
    # assert len(adjacency_list) == len(graph.nodes), "Mismatch between adjacency_list and graph nodes"
    #
    # # #空节点（无邻居）会导致 Metis 崩溃：
    # # for i, neigh in enumerate(adjacency_list):
    # #     if len(neigh) == 0:
    # #         print(f"Node {i} is isolated.")
    # #         # 可选择：删除该节点或加入一个虚拟邻居（如随机连接一个节点）
    # (st, parts) = metis.part_graph(cluster_number,adjacency=adjacency_list)
    # print('metis.part_graph指定成功了')

    #使用替代方法
    nodes = list(graph.nodes())  # 原始索引列表
    partition = community_louvain.best_partition(graph)
    parts = [partition[node] for node in graph.nodes()]
    clusters = list(set(parts)) #创建包含所有簇的列表，并通过set()进行簇标识去重。
    cluster_membership = {} #字典，用于存储每个节点所属的簇
    for i, node in enumerate(graph.nodes): #遍历图中的节点，同时获得节点索引i和节点对象node
        cluster = parts[i] #获取第i个节点所属簇的标识
        cluster_membership[node] = cluster #将节点`node`和其所属的簇的标识`cluster`添加到字典中，建立了节点到簇的映射关系
    print('Louvain clustering 成功完成')
    return clusters, cluster_membership,nodes     #返回簇的列表和节点到簇的映射

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
def load_data(dataset, root, train_n, val_n, test_n, feats_path, cluster_number, train_path,
              test_path, val_path):
    # 加载图
    graphx, n_nodes = load_graph(args.root, args.dataset, args.attack, args.ptb_rate)

    #下面这行则是不同于COLCE的地方，进行了图分区。clusters是分区簇列表cluster_membership是字典
    print("---------------------cluster-开始------------------------------")
    clusters, cluster_membership,nodes_in_partition  = metis_clustering(cluster_number,graphx)
    print("---------------------cluster完毕------------------------------")

    # 原索引 -> 新索引 映射
    old_to_new = {old: new for new, old in enumerate(nodes_in_partition)}
    # 替换 nodes_adj 构建逻辑（仅使用参与聚类的边）
    nodes_adj = {}
    for id1, id2 in graphx.edges:
        if id1 not in old_to_new or id2 not in old_to_new:
            continue  # 只保留参与聚类的边
        new_id1 = old_to_new[id1]
        new_id2 = old_to_new[id2]
        nodes_adj.setdefault(new_id1, []).append(new_id2)
        nodes_adj.setdefault(new_id2, []).append(new_id1)

    # nodes_adj = {} #字典，用于存储节点的邻接信息。无向图
    # for id1, id2 in graphx.edges:
    #     if id1 not in nodes_adj:
    #         nodes_adj[id1] = [id2]
    #     else:
    #         nodes_adj[id1].append(id2)
    #     if id2 not in nodes_adj:
    #         nodes_adj[id2] = [id1]
    #     else:
    #         nodes_adj[id2].append(id1)

    print("---------------------graph-------------------------------")
    # sg_nodes = {} #字典，用于存储分区后的节点信息。key是分区标识，value是分区内的节点
    # for u, c in cluster_membership.items():
    #     if c not in sg_nodes:
    #         sg_nodes[c] = []
    #     sg_nodes[c].append(u) #{c1:1,2,3},{c2:4,5,6}

    sg_nodes = {}
    for u, c in cluster_membership.items():  # u 是新索引（0～N-1）
        sg_nodes.setdefault(c, []).append(u)

    #调用loadQuery加载训练、验证和测试数据
    train, val, test = loadQuerys(dataset, root, train_n, val_n, test_n, train_path, test_path, val_path)

    print("======================featd==================================")
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
    elif args.dataset.startswith(('fb', 'wfb', 'fa')):  # 不加入中心节点
        feats_array = np.loadtxt(f'{args.root}/{args.dataset}/{args.dataset}.feat', delimiter=' ', dtype=np.float32)
        print(type(feats_array))
        # nodes_feats = fnormalize(feats_array)  # 将特征进行归一化
        nodes_feats = torch.from_numpy(feats_array)
        node_in_dim = nodes_feats.shape[1]
    elif args.dataset in ['facebook']:  # 读取pyg中的特征数据
        feats_array = np.loadtxt(f'{args.root}/{args.dataset}/{args.dataset}.feat', dtype=float, delimiter=' ')
        nodes_feats = torch.tensor(feats_array, dtype=torch.float32)
        node_in_dim = nodes_feats.shape[1]
    else:
        print('加载节点特征失败，数据集不匹配')
    print('加载节点特征完成完成')

    nodes_feats = nodes_feats[torch.tensor(nodes_in_partition, dtype=torch.long)]
    n_nodes = len(nodes_in_partition)
    print('nodes_in_partition个数:{n_nodes}')
    '''  
    nodes_feats = nodes_feats.transpose(0, 1)
    rowsum = nodes_feats.sum(1)
    rowsum[rowsum == 0] = 1
    print(rowsum)
    nodes_feats = nodes_feats / rowsum[:, np.newaxis]
    nodes_feats = nodes_feats.transpose(0, 1)
    #'''
    return nodes_feats, train, val, test, node_in_dim, n_nodes, graphx, \
           sg_nodes, clusters, cluster_membership, nodes_adj

class TwoHopNeighbor(object):
    def __call__(self, data):
        edge_index, edge_attr = data.edge_index, data.edge_attr
        N = data.num_nodes

        value = edge_index.new_ones((edge_index.size(1), ), dtype=torch.float)

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
        edge_index = torch.tensor([src, dst])
        edge_index_aug, egde_attr = hypergraph_construction(edge_index, len(nodeslists), k = k)
        edge_index = add_remaining_self_loops(edge_index, num_nodes=len(nodeslists))[0] #添加自环？
        pos_ = [nodes_[x] for x in pos if x in nodeslists]  #在此分区（分区+边界）中的标记节点的索引。
        #pos_ = [nodes_[x] for x in comm if x in nodeslists and x!=q]
        train_lists.append((nodes_[q], pos_, edge_index, edge_index_aug, feats))

    return train_lists

'''
进行图划分的社区搜索方法
'''
def CommunitySearch(args):
    pretime = datetime.datetime.now() #进行社区搜索前的时间
    #load_data加载数据集
    nodes_feats, train_, val, test, node_in_dim, n_nodes, g, \
    sg_nodes, clusters, cluster_membership, nodes_adj = load_data(args.dataset, args.root,
              args.train_size, args.val_size, args.test_size,args.feats_path, args.cluster_number,
              args.train_path, args.test_path, args.val_path)

    print('数据集加载完毕')
    #对训练集进行重构，得到了可以直接输入到pytorch的训练集 train_lists.append((nodes_[q], pos_, edge_index, edge_index_aug, feats))
    trainlists  = decompose_train(g, train_, val, nodes_feats, n_nodes,cluster_membership,
                                            sg_nodes, args.k, nodes_adj)
    #打印数量
    print(len(trainlists))

    print(torch.cuda.device_count())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    print(device) #打印设备

    train = []
    for q, pos, edge_index, edge_index_aug, feats in trainlists:
        edge_index = edge_index.to(device)
        edge_index_aug = edge_index_aug.to(device)
        feats = feats.to(device)
        train.append((q, pos, edge_index, edge_index_aug, feats))

    model = ConRC(node_in_dim, args.hidden_dim, args.num_layers, args.drop_out, args.tau,
                  device, args.alpha, args.lam, args.k) #创建模型对象
    optimizer = th.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.to(device)
    model.reset_parameters()
    pre_process_time = (datetime.datetime.now() - pretime).seconds #到现在结束了预处理过程？

    model_path = './model/' + args.dataset + '_' + args.model_path + '.pkl' #存储模型

    now = datetime.datetime.now()
    optimizer.zero_grad() #梯度清零
    
    for epoch in range(args.epoch_n): #训练模型
        model.train()
        start = datetime.datetime.now()
        loss_b = 0.0
        i = 0
        for q, pos, edge_index, edge_index_aug, feats in train:
            if len(pos) == 0:
                i = i + 1
                continue
            loss = model((q, pos, edge_index, edge_index_aug, feats))
            loss_b = loss_b + loss
            loss.backward() #反向传播
            if (i + 1) % args.batch_size == 0:
                optimizer.step() #更新模型参数
                optimizer.zero_grad()
            i = i + 1

        epoch_time = (datetime.datetime.now() - start).seconds #当前的epoch，每个epoch的时间
        print("loss", loss_b, epoch, epoch_time)

    torch.save(model.state_dict(), model_path)  # 存储模型参数
    print("------------------------evalue-------------------------")
    #if args.early_stop==1:

    model.load_state_dict(torch.load(model_path)) #从文件中加载模型
    model.eval() #进行模型验证


    F1 = 0.0
    Pre = 0.0
    Rec = 0.0
    nmi_score = 0.0
    ari_score = 0.0
    jac_score = 0.0
    count = 0.0

    f1_score, precision, recall = 0.0, 0.0, 0.0
    train_model_running_time = (datetime.datetime.now() - now).seconds  # '''
    now = datetime.datetime.now()
    with th.no_grad():
        s_, f1_ = validation__(val, cluster_membership, sg_nodes,
                               n_nodes, nodes_feats, device, g, model, nodes_adj)

        print(len(test))

        for q, comm in test: #进行测试。。

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
            edge_index = add_remaining_self_loops(edge_index, num_nodes=len(nodeslists))[0]  #得到可以输入模型的数据

            h = model((nodes_[q], None, edge_index, edge_index_aug, feats)) #运行模型得到

            count = count + 1

            numerator = torch.mm(h[nodes_[q]].unsqueeze(0), h.t())
            norm = torch.norm(h, dim=-1, keepdim=True)
            denominator = torch.mm(norm[nodes_[q]].unsqueeze(0), norm.t())
            sim = numerator/denominator
            #得到每个节点的相似度得分
            simlists = torch.sigmoid(sim.squeeze(0)).to(torch.device('cpu')).numpy().tolist()#torch.sigmoid(simlists).numpy().tolist()

            comm_find = []
            for i, score in enumerate(simlists):
                if score >=s_ and nodeslists[i] not in comm_find:  #将此查询节点子图中满足得分阈值的节点加入到comm_find
                    comm_find.append(nodeslists[i])
            lists = [] #用于存储边界节点所在的分区编号
            for qb in nodelistb:  #处理边界节点
                if cluster_membership[qb] in lists: #如果qb所在的分区编号在list中，则继续
                    continue
                lists.append(cluster_membership[qb])#将qb所在的分区编号加入到list
                qb_ = nodes_[qb] #获得qb的索引
                if simlists[qb_] <s_: #判断qb的相似度得分不满足阈值，则不管
                    continue
                nodeslists_ = sg_nodes[cluster_membership[qb]] #否则，将获得qb所在的分区编号中的所有节点nodelists

                nodelistb_ = []
                if qb in nodes_adj:
                    neighbor_ = nodes_adj[qb]
                    nodelistb_ = [x for x in neighbor_ if x not in nodeslists_]
                nodelistb_ = set(nodelistb_)
                nodelistb_ = list(nodelistb_)
                nodeslists_ = nodeslists_ + nodelistb_ #继续构造这个分区+边界。

                qb = q #并将查询节点继续在新的子图上进行查询

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
                h_= model((nodes__[qb], None, edge_index_, edge_index_aug_, feats)) #运行模型得到相应的表示
                h_[nodes__[qb]] = h_[nodes__[qb]]+h[nodes_[q]]

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
            comm = set(comm)
            comm = list(comm)
            f1, pre, rec = f1_score_(comm_find, comm)
            F1 = F1 + f1
            Pre = Pre + pre
            Rec = Rec + rec
            print("count", count)
            print("f1, pre, rec", q, f1, pre, rec)
            print("f1, pre, rec", q, F1 / count, Pre / count, Rec / count)

            nmi = NMI_score(comm_find, comm, n_nodes)
            nmi_score = nmi_score + nmi

            ari = ARI_score(comm_find, comm, n_nodes)
            ari_score = ari_score + ari

            jac = JAC_score(comm_find, comm, n_nodes)
            jac_score = jac_score + jac


    F1 = F1 / len((test))
    Pre = Pre / len((test))
    Rec = Rec / len((test))
    print("F1, Pre, Rec, s", F1, Pre, Rec, s_)


    nmi_score = nmi_score/len(test)
    print("NMI: ", nmi_score)

    ari_score = ari_score/len(test)
    print("ARI: ", ari_score)

    jac_score = jac_score/len(test)
    print("JAC: ", jac_score)

    test_running_time = (datetime.datetime.now() - now).seconds
    output = args.root+'/result/'+args.dataset+args.result
    with open(output, 'a+') as fh:
        line = str(args)+" pre_process_time "+str(pre_process_time)+" train_model_running_time "+\
               str(train_model_running_time)+" test_running_time "+str(test_running_time)+" F1 "+str(F1)\
               +" Pre "+str(Pre)+" Rec "+str(Rec)+" nmi_score "+str(nmi_score)+" ari_score "+str(ari_score)\
               +" jac_score " + str(jac_score)+" F1_ "+str(f1_score)\
               +" Pre_ "+str(precision)+" Rec_ "+str(recall)
        fh.write(line + "\n")
        fh.close()
    return F1, Pre, Rec, nmi_score, ari_score, jac_score, pre_process_time, train_model_running_time, test_running_time

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='photo') #默认的数据集
    parser.add_argument('--root', type=str, default='../data/') #默认的数据集根节点
    parser.add_argument('--feats_path', type=str, default='/football_core_emb_.txt') #默认的特征数据
    parser.add_argument('--train_size', type=int, default=300) #默认的训练集大小
    parser.add_argument('--val_size', type=int, default=100)#默认的验证集大小
    parser.add_argument('--test_size', type=int, default=500)#默认的测试集大小
    parser.add_argument('--batch_size', type=int, default=64)#默认的batch大小
    parser.add_argument('--hidden_dim', type=int, default=256) #隐藏层维度
    parser.add_argument('--num_layers', type=int, default=3) #模型层数
    parser.add_argument('--epoch_n', type=int, default=100) #epoch数
    parser.add_argument('--drop_out', type=float, default=0.1) #drop_out率
    parser.add_argument('--lr', type=float, default=0.001) # 学习率
    parser.add_argument('--weight_decay', type=float, default=0.0005)# 权重衰减率
    parser.add_argument('--tau', type=float, default=0.2) #超参数tau
    parser.add_argument('--cluster_number', type=int, default=100) #聚类的数量（我猜是分区的数量）


    # 控制攻击方法、攻击类型和攻击率
    #choices=['none','meta', 'random_remove', 'random_flip', 'random_add', 'meta_attack', 'add', 'del','gflipm', 'gdelm', 'gaddm', 'cdelm', 'cflipm', 'delm', 'flipm']
    parser.add_argument('--attack', type=str, default='none')
    parser.add_argument('--type', type=str, default='add', help='random attack type', choices=['add', 'remove', 'flip'])
    parser.add_argument('--noise_level', type=int, default=3, choices=[1, 2, 3], help='noisy level')
    parser.add_argument('--ptb_rate', type=float, default=0.30, help='pertubation rate')

    parser.add_argument('--alpha', type=float, default=0.2) #超参数alpha
    parser.add_argument('--lam', type=float, default=0.2) #超参数lambda
    parser.add_argument('--train_path', type=str, default='_3_pos_train_300.txt') #训练积路径
    parser.add_argument('--test_path', type=str, default='_3_test_500.txt')#测试集路径
    parser.add_argument('--val_path', type=str, default='_3_val_100.txt') #验证集路径
    parser.add_argument('--model_path', type=str, default='Cluster-CLCS-') #模型路径
    parser.add_argument('--count', type=int, default=1) #计数？？？
    parser.add_argument('--pos_size', type=int, default=3) #每个训练查询顶点给出的标签数
    parser.add_argument('--k', type=int, default=2) #超参数？
    parser.add_argument('--test_every', type=int, default=200) #？？？？？？
    parser.add_argument('--result', type=str, default='_Cluster_CLCS_result.txt') #？？？？


    args = parser.parse_args()

    pre_process_time_A, train_model_running_time_A, test_running_time_A = 0.0, 0.0, 0.0

    count = 0 #计数，模型更新次数？？
    F1lists = [] #F1值
    Prelists = [] #精度
    Reclists = [] #Recall召回率
    nmi_scorelists = []
    ari_scorelists = []
    jac_scorelists = [] #Jaccard相似性

    
    for i in range(args.count): #count的作用我执行多少次，默认是1；之前不是实验中有说以5次运行的平均值巴拉巴拉。。。
        seed_all(0) #种子
        args.model_path='Cluster-CLCS-6'+"-"+str(count)
        count = count+1
        print('= ' * 20) #哦，输出20个等号分隔
        print(count) #打印当前的count
        now = datetime.datetime.now()
        print('##  Starting Time:', now.strftime("%Y-%m-%d %H:%M:%S"), flush=True) #打印程序开始运行的时间
        #args.dataset='SimpleEx' #测试数据集
        F1, Pre, Rec, nmi_score, ari_score, jac_score, pre_process_time, train_model_running_time, test_running_time = \
            CommunitySearch(args) #记录各个所需的值
        F1lists.append(F1)
        Prelists.append(Pre)
        Reclists.append(Rec)
        nmi_scorelists.append(nmi_score)
        ari_scorelists.append(ari_score)
        jac_scorelists.append(jac_score)
        pre_process_time_A = pre_process_time_A + pre_process_time #ALL总共的预处理时间
        train_model_running_time_A = train_model_running_time_A + train_model_running_time #ALL总共的模型运行时间
        test_running_time_A = test_running_time_A + test_running_time #总共的测试运行的时间
        print('## Finishing Time:', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True) #打印测试结束时的时间
        running_time = (datetime.datetime.now() - now).seconds
        print('## Running Time(s):', running_time)# 打印结束-开始=程序运行的总秒数。
        print('= ' * 20) #打印20个*号来分隔


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
    test_running_time_A = test_running_time_A/float(args.count) #平均的运行时间
    output = args.root+'/result/'+args.dataset+args.result #记录模型最终的结果
    #向文件中写入平均预处理时间，平均模型训练时间，平均测试运行时间。
    #写入F1均值，F1标准差，Pre均值，Pre标准差，Rec均值，Rec标准差
    #写入NMi均值，NMI标准差，Ari均值，Ari标准差，Jac均值，Jac标准差
    with open(output, 'a+') as fh:
        line = "average "+str(args)+" pre_process_time "+str(pre_process_time_A)+" train_model_running_time "+\
               str(train_model_running_time_A)+" test_running_time "+str(test_running_time_A)+\
               " F1 mean "+str(F1_mean)+" F1 std "+ str(F1_std)+" Pre mean "+str(Pre_mean)+" Pre std "+\
               str(Pre_std)+" Rec mean "+str(Rec_mean)+"Rec std "+str(Rec_std)+" nmi_score mean "+\
               str(nmi_mean)+" nmi std "+str(nmi_std)+" ari_score mean "+str(ari_mean)+" ari std "+\
               str(ari_std)+" jac mean "+str(jac_mean)+" jac std "+str(jac_std)
        fh.write(line + "\n")
        fh.close()














