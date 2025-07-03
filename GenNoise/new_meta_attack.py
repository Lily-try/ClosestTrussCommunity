import os
import pathlib
import pickle

import networkx as nx
import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import time
from deeprobust.graph.defense import GCN
from deeprobust.graph.global_attack import MetaApprox, Metattack
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
import argparse
import citation_loader
from citation_loader import citation_target_reader
# from preprocess import txt_utils
# 开始计时
start_time = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
#设置数据集
#choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed','footbal'],
parser.add_argument('--dataset', type=str, default='facebook', help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.20,  help='pertubation rate')
parser.add_argument('--model', type=str, default='Meta-Self',
        choices=['Meta-Self', 'A-Meta-Self', 'Meta-Train', 'A-Meta-Train'], help='model variant')
parser.add_argument('--root', type=str, default='../data',  help='data store root')
parser.add_argument('--require_lcc',type = bool, default=False, help='Whether use largest connected components or not',)
#加载配置并固定随机数种子
args = parser.parse_args()
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if device != 'cpu':
    torch.cuda.manual_seed(args.seed)

def load_community_labels(filepath, n_nodes=None, default_label=-1):
    """
    读取社区文件并返回形状为 (n_nodes, 1) 的 labels 数组。

    :param filepath: 社区文件路径，第一行为社区标签，后续每行对应一个社区包含的节点 ID。
    :param n_nodes: 节点总数，若为 None，则从最大节点 ID 推断。
    :param default_label: 默认标签，用于未出现在任何社区的节点。
    :return: labels: np.ndarray of shape (n_nodes, 1)
    """
    with open(filepath, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    # 解析社区标签
    community_labels = list(map(int, lines[0].split()))

    # 构造 node -> label 映射
    node_to_label = {}
    for label_idx, line in enumerate(lines[1:]):
        node_ids = map(int, line.split())
        for node_id in node_ids:
            node_to_label[node_id] = community_labels[label_idx]

    # 推断节点总数（若未指定）
    if n_nodes is None:
        n_nodes = max(node_to_label.keys()) + 1

    # 构造 label 数组
    labels = np.full((n_nodes, 1), default_label, dtype=int)
    for node_id, label in node_to_label.items():
        labels[node_id] = label

    return labels
def count_nodes_in_comms(filepath):
    """
    统计 .comms 文件中出现的所有节点总数（去重后）
    """
    with open(filepath, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    node_set = set()
    for line in lines[1:]:  # 跳过第一行标签
        node_ids = map(int, line.split())
        node_set.update(node_ids)

    return len(node_set), node_set
def get_train_val_test(nnodes, val_size=0.1, test_size=0.8, stratify=None, seed=None):
    '''
    参照deeprobust中的实现，来对labels进行分割。区别在于这里没有将random_state始终赋值为None
    :param nnodes:
    :param val_size:
    :param test_size:
    :param stratify:
    :param seed:
    :return:
    '''
    # assert stratify is not None, 'Stratify parameter cannot be None!'
    if seed is not None:#固定随机数种子
        np.random.seed(seed)

    idx = np.arange(nnodes)
    train_size = 1 - val_size - test_size
    #Deeprobust的实现与此的区别是，deeprobust将random_state始终赋值为None
    idx_train_and_val, idx_test = train_test_split(idx, train_size=train_size + val_size, test_size=test_size,
                                                   stratify=stratify, random_state=None)
    if stratify is not None:
        stratify = stratify[idx_train_and_val]

    idx_train, idx_val = train_test_split(idx_train_and_val, train_size=(train_size / (train_size + val_size)),
                                          test_size=(val_size / (train_size + val_size)), stratify=stratify,
                                          random_state=None)
    return idx_train, idx_val, idx_test
def load_node_features(filepath):
    """
    加载节点特征文件 cora_feats.txt

    :param filepath: 特征文件路径
    :return: features: ndarray of shape (n_nodes, feat_dim)
    """
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    max_node_id, feat_dim = map(int, lines[0].split())
    n_nodes = max_node_id + 1  # 节点编号从 0 开始

    features = np.zeros((n_nodes, feat_dim), dtype=np.float32)

    for line in lines[1:]:
        parts = line.split()
        node_id = int(parts[0])
        feats = list(map(float, parts[1:]))
        features[node_id] = feats

    return features

#加载数据集
dataset = args.dataset
'''
graph:scipy.sparse.csr_matrix
feature:scipy.sparse.csr_matrix or numpy.array
label:numpy.array
'''
# if dataset in ['cora', 'citeseer', 'pubmed']:#使用存储库中提供的。
#     # 需要adj,features,划分训练集得到不同的label
#     graph = citation_utils.citation_graph_reader(args.root, dataset)  # 读取图 nx格式的
#     adj = nx.adjacency_matrix(graph)  # 转换为CSR格式的稀疏矩阵
#     # 读取特征和标签，并根据最大连通分量更新他们
#     features = citation_utils.citation_feature_reader(args.root, dataset)  # numpy.ndaaray:(2708,1433)
#     labels = citation_utils.citation_target_reader(args.root, dataset)  # 读取标签,ndarray:(2708,1)
#     if args.require_lcc:#过滤只保留最大连通分量
#         print('攻击时只保留了最大连通分量')
#         lcc = citation_utils.largest_connected_components(adj, n_components=1)
#         adj = adj[lcc][:, lcc]
#         features = features[lcc]
#         labels = labels[lcc]
#     idx_train,idx_val,idx_test = get_train_val_test(adj.shape[0], val_size=0.1, test_size=0.8, stratify=labels)
#     idx_unlabeled = np.union1d(idx_val,idx_test)
#
# if dataset in ['dblp','amazon']:#这几个是有gt社区但没有节点特征的。
#     edge, labels = snap_utils.load_snap(args.root, data_set='com_' + dataset, com_size=3)  # edge是list:1049866
#     #将edge转换成csr_matrix

#第一步：加载图数据和特征数据
if dataset in ['football','facebook_all']:
    graphx = nx.Graph()
    with open(f'{args.root}/{args.dataset}/{args.dataset}.txt', "r") as f:
        for line in f:
            node1, node2 = map(int, line.strip().split())
            graphx.add_edge(node1, node2)
    adj = nx.adjacency_matrix(graphx)  # 转换为CSR格式的稀疏矩阵
        # 读取features
    features = load_node_features(f'{args.root}/{args.dataset}/{args.dataset}_feats.txt')

    len_comm_nodes, node_set = count_nodes_in_comms(f"{args.root}/{args.dataset}/{args.dataset}.comms")
    print(len_comm_nodes)
    labels = load_community_labels(f"{args.root}/{args.dataset}/{args.dataset}.comms", n_nodes=graphx.number_of_nodes())
    idx_train, idx_val, idx_test = get_train_val_test(adj.shape[0], val_size=0.1, test_size=0.8, stratify=labels)  #
    idx_unlabeled = np.union1d(idx_val, idx_test)

if dataset in ['cora', 'citeseer', 'pubmed']:#引文网络，deeprobust本身就有的
    #从pyg中的原始数据集中读取数据
    graph = citation_loader.citation_graph_reader(args.root, args.dataset)  # 读取图 nx格式的
    print(graph)
    adj = nx.adjacency_matrix(graph)  # 转换为CSR格式的稀疏矩阵
    #节点特征
    features = citation_loader.citation_feature_reader(args.root, args.dataset) # numpy.ndaaray:(2708,1433)
    #标签 。ndarray(n_nodes,1)
    labels = citation_target_reader(args.root, dataset)  # 读取标签,ndarray:(2708,1)
    idx_train, idx_val, idx_test = get_train_val_test(adj.shape[0], val_size=0.1, test_size=0.8, stratify=labels)
    idx_unlabeled = np.union1d(idx_val, idx_test)
elif dataset in ['cocs','photo']:
    print(f'给{dataset}数据集加入{args.ptb_rate}的噪声')
    # #读取图
    # graphx = nx.Graph()
    # with open(f'{args.root}/{args.dataset}/{args.dataset}.edges', "r") as f:
    #     for line in f:
    #         node1, node2 = map(int, line.strip().split())
    #         graphx.add_edge(node1, node2)
    # print(f'{args.dataset}:', graphx)
    # adj = nx.adjacency_matrix(graphx)  # 转换为CSR格式的稀疏矩阵
    # #读取节点特征
    data_list = torch.load(f'{args.root}/{args.dataset}/{dataset}_dgl.pt')
    adj_tensor = data_list[0]
    features_tensor =data_list[1]
    labels_tensor  = data_list[2]
    # --- 1. 转换邻接矩阵为 scipy.sparse.csr_matrix ---
    if adj_tensor.is_sparse:
        # PyTorch sparse tensor 转为 COO 格式，然后转为 SciPy
        coo = adj_tensor.coalesce()
        indices = coo.indices().cpu().numpy()
        values = coo.values().cpu().numpy()
        shape = coo.shape
        adj = sp.coo_matrix((values, (indices[0], indices[1])), shape=shape).tocsr()
    else:
        # 若为 dense tensor，先转为 numpy，再转为 sparse
        adj = sp.csr_matrix(adj_tensor.cpu().numpy())

    # --- 2. 转换特征为 numpy 或 sparse matrix ---
    features = features_tensor.cpu().numpy()
    # 如果需要稀疏格式，可以添加判断：
    # features = sp.csr_matrix(features)  # 如果你希望特征为稀疏格式
    # --- 3. 转换标签为 numpy array ---
    labels = labels_tensor.cpu().numpy()
    idx_train, idx_val, idx_test = get_train_val_test(adj.shape[0], val_size=0.1, test_size=0.8, stratify=labels)
    idx_unlabeled = np.union1d(idx_val, idx_test)

elif dataset in ['fb107','facebook']:
    #加载图
    graphx = nx.read_edgelist(f'{args.root}/{args.dataset}/{args.dataset}.edges', nodetype=int, data=False)
    print(graphx)
    adj = nx.adjacency_matrix(graphx)  # 转换为CSR格式的稀疏矩阵
    #加载feats
    features = np.loadtxt(f'{args.root}/{args.dataset}/{args.dataset}.feat', delimiter=' ', dtype=np.float32)
    print(type(features))
    len_comm_nodes, node_set = count_nodes_in_comms(f"{args.root}/{args.dataset}/{args.dataset}.comms")
    print(len_comm_nodes)
    #加载labels
    labels = load_community_labels(f"{args.root}/{args.dataset}/{args.dataset}.comms", n_nodes=graphx.number_of_nodes())
    idx_train, idx_val, idx_test = get_train_val_test(adj.shape[0], val_size=0.1, test_size=0.8, stratify=None) #
    idx_unlabeled = np.union1d(idx_val, idx_test)


#计算扰动数量
perturbations = int(args.ptb_rate * (adj.sum()//2))
#将adj, features, labels处理成tensor格式
adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False)

if args.dataset in ['fb107','facebook']:
    labels = labels.squeeze()

#设置代理模型
surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1, nhid=16,
        dropout=0.5, with_relu=False, with_bias=True, weight_decay=5e-4, device=device)

surrogate = surrogate.to(device)
surrogate.fit(features, adj, labels, idx_train)

#设置攻击模型
if 'Self' in args.model:
    lambda_ = 0
if 'Train' in args.model:
    lambda_ = 1
if 'Both' in args.model:
    lambda_ = 0.5
if 'A' in args.model:
    # model = MetaApprox(model=surrogate, nnodes=adj.shape[0], feature_shape=features.shape, attack_structure=True, attack_features=False, device=device, lambda_=lambda_)
    model = MetaApprox(model=surrogate, nnodes=adj.shape[0], feature_shape=features.shape, attack_structure=True, attack_features=True, device=device, lambda_=lambda_)

else:
    model = Metattack(model=surrogate, nnodes=adj.shape[0], feature_shape=features.shape,  attack_structure=True, attack_features=True, device=device, lambda_=lambda_)

model = model.to(device)

def test(adj):
    ''' test on CSGCN '''
    # adj = normalize_adj_tensor(adj)
    gcn = GCN(nfeat=features.shape[1],
              nhid=args.hidden,
              nclass=labels.max().item() + 1,
              dropout=args.dropout, device=device)
    gcn = gcn.to(device)
    gcn.fit(features, adj, labels, idx_train) # train without model picking
    # gcn.fit(features, adj, labels, idx_train, idx_val) # train with validation model picking
    output = gcn.output.cpu()
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

    return acc_test.item()

def test_path():
    modified_adj = adj
    name = pathlib.Path(f'{args.dataset}_meta_{args.ptb_rate*100}.npz')
    if type(modified_adj) is torch.Tensor:
        sparse_adj = to_scipy(modified_adj)
        sp.save_npz(os.path.join(args.root,args.dataset,'meta',name), sparse_adj)
    else:
        sp.save_npz(os.path.join(args.root, args.dataset, 'meta', name), modified_adj)


def to_scipy(tensor):
    """将dense/sparse tensor转换成scipy matrix"""
    if is_sparse_tensor(tensor):
        values = tensor._values()
        indices = tensor._indices()
        return sp.csr_matrix((values.cpu().numpy(), indices.cpu().numpy()), shape=tensor.shape)
    else:
        indices = tensor.nonzero().t()
        values = tensor[indices[0], indices[1]]
        return sp.csr_matrix((values.cpu().numpy(), indices.cpu().numpy()), shape=tensor.shape)


def main():
    #进行攻击
    torch.cuda.empty_cache()
    model.attack(features, adj, labels, idx_train, idx_unlabeled, perturbations, ll_constraint=False)
    modified_adj = model.modified_adj
    path = os.path.join(args.root,args.dataset,'meta')
    if not os.path.exists(path):
        os.makedirs(path)
    name = f'{args.dataset}_meta_{args.ptb_rate}'
    try:
        if isinstance(modified_adj,torch.Tensor):
            sparse_adj = to_scipy(modified_adj) #转换为scipy稀疏矩阵
            sp.save_npz(os.path.join(path, name), sparse_adj)
        else:
            sp.save_npz(os.path.join(path, name), modified_adj)
        print('文件保存成功！')
    except Exception as e:
        print('保存文件时发生错误：',e)
    #存储成npz
    # model.save_adj(root=os.path.join(args.root,args.dataset,'meta'), name=f'{args.dataset}_meta_{args.ptb_rate}')
    # model.save_features(root=os.path.join(args.root,args.dataset,'meta'), name=f'{args.dataset}_meta_{args.ptb_rate}_feat')

if __name__ == '__main__':
    main()
    # 结束计时
    end_time = time.time()
    execution_time = end_time - start_time
    # 打印总用时
    print(f'Total execution time: {execution_time:.4f} seconds')

    # 日志文件路径
    log_file_path = '../data/log/execution_log.txt'
    # 创建或追加到日志文件
    with open(log_file_path, 'a') as log_file:
        log_file.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"Script: {__file__}\n")
        log_file.write(f"Arguments: Seed={args.seed}, Dataset={args.dataset}, Perturbation Rate={args.ptb_rate}\n")
        log_file.write(f"Execution Time: {execution_time:.4f} seconds\n")
        log_file.write("--------------------------------------------------------\n")
