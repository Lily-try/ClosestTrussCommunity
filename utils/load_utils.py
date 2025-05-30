import os
import networkx as nx
import scipy.sparse as sp
import numpy as np
import torch
from scipy.sparse import csr_matrix
from torch_geometric.datasets import AttributedGraphDataset
from torch_geometric.utils import remove_self_loops, add_remaining_self_loops
from torch_sparse import spspmm, coalesce
from torch_geometric.data import Data

from utils.citation_loader import citation_graph_reader,citation_feature_reader
from utils import citation_loader
# from utils import efacebook_utils
from utils import ego_utils
'''
加载社区搜索所需的图、特征矩阵、训练、验证、测试集
'''

class TwoHopNeighbor(object):
    '''
    计算图中节点的两跳邻居
    '''

    def __call__(self, data):
        edge_index, edge_attr = data.edge_index, data.edge_attr
        N = data.num_nodes

        value = edge_index.new_ones((edge_index.size(1),), dtype=torch.float)

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


def hypergraph_construction(edge_index, num_nodes, k=1):
    '''
    构建超图
    :param edge_index: 原始图
    :param num_nodes: 节点数目
    :param k: 以k跳邻居构建超边
    :return:
    '''
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


def loadQuerys(dataset, root, train_n, val_n, test_n, train_path, test_path, val_path):
    '''
    加载数据的训练集、验证集和测试集
    :param dataset:
    :param root:
    :param train_n:
    :param val_n:
    :param test_n:
    :param train_path:
    :param test_path:
    :param val_path:
    :return:
    '''
    path_train = os.path.join(root,dataset,f'{dataset}_{train_path}_{train_n}.txt')
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

    path_test = os.path.join(root,dataset,f'{dataset}_{test_path}_{test_n}.txt')
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
    path_val = os.path.join(root,dataset,f'{dataset}_{val_path}_{val_n}.txt')
    if not os.path.isfile(path_val):
        raise Exception("No such file: %s" % path_val)
    val_lists = []
    for line in open(path_val, encoding='utf-8'):
        q, comm = line.split(",")
        q = int(q)
        comm = comm.split(" ")
        comm_ = [int(x) for x in comm]
        if len(val_lists)>=val_n:
            break
        val_lists.append((q, comm_))

    return train_lists, val_lists, test_lists


def load_data(args):
    '''********************1. 加载图数据******************************'''
    max = 0
    edges = []
    if args.attack == 'none':  # 使用原始数据
        if args.dataset_name in ['football', 'facebook_all']:
            path = os.path.join(args.root_name, args.dataset_name, f'{args.dataset_name}.txt')
            for line in open(path, encoding='utf-8'):
                node1, node2 = line.split(" ")
                node1_ = int(node1)
                node2_ = int(node2)
                if node1_ == node2_:
                    continue
                if max < node1_:
                    max = node1_
                if max < node2_:
                    max = node2_
                edges.append([node1_, node2_])
            n_nodes = max + 1
            nodeslists = [x for x in range(n_nodes)]
            graphx = nx.Graph()  # 学一下怎么用的。
            graphx.add_nodes_from(nodeslists)
            graphx.add_edges_from(edges)
            print(graphx)
            del edges
        elif args.dataset_name in ['cora', 'pubmed', 'citeseer']:
            graphx = citation_graph_reader(args.root_name, args.dataset_name)  # 读取图 nx格式的
            print(graphx)
            n_nodes = graphx.number_of_nodes()
        elif args.dataset_name.startswith('fb'): #读取ego-facebook的邻接矩阵
            graphx = ego_utils.ego_graph_reader(args.root_name, args.dataset_name)
            # graph_path = f'{args.root}/{args.dataset}/{args.dataset[2:]}.edges'
            # graphx = efacebook_utils.read_edge_file(graph_path)
            print(graphx)
            n_nodes = graphx.number_of_nodes()
        elif args.dataset_name in ['facebook']: #从pyg上读取的facebook数据
            dataset = AttributedGraphDataset(root='../data', name='facebook')
            data = dataset[0]
            #存在错误呢
        elif args.dataset_name.startswith('wfb'):
            edges_path = "{}/{}/{}.edges".format(args.root_name, args.dataset_name, args.dataset_name)
            graphx = nx.read_edgelist(edges_path, nodetype=int)
            n_nodes = graphx.number_of_nodes()
    elif args.attack == 'random':
        # 读取npz版本的。
        path = os.path.join(args.root_name, args.dataset_name, args.attack,
                            f'{args.dataset_name}_{args.attack}_{args.type}_{args.ptb_rate}.npz')
        adj_csr_matrix = sp.load_npz(path)
        graphx = nx.from_scipy_sparse_array(adj_csr_matrix)
        print(graphx)
        n_nodes = graphx.number_of_nodes()
    elif args.attack =='add': #metaGC中自己注入随机噪声
        # 读取npz版本的。
        path = os.path.join(args.root_name, args.dataset_name, args.attack,
                            f'{args.dataset_name}_{args.attack}_{args.noise_level}.npz')
        adj_csr_matrix = sp.load_npz(path)
        graphx = nx.from_scipy_sparse_array(adj_csr_matrix)
        print(graphx)
        n_nodes = graphx.number_of_nodes()

    # 计算aa指标
    aa_indices = nx.adamic_adar_index(graphx)
    # 初始化 Adamic-Adar 矩阵
    aa_matrix = np.zeros((n_nodes, n_nodes))
    # 计算 Adamic-Adar 指数
    for u, v, p in aa_indices:
        aa_matrix[u, v] = p
        aa_matrix[v, u] = p  # 因为是无向图，所以也需要填充对称位置
    # 转换为张量
    aa_tensor = torch.tensor(aa_matrix, dtype=torch.float32)

    src = []
    dst = []
    for id1, id2 in graphx.edges:
        src.append(id1)
        dst.append(id2)
        src.append(id2)
        dst.append(id1)
    # 这两行是获得存储成稀疏矩阵的格式，加权模型中使用
    num_nodes = graphx.number_of_nodes()
    adj_matrix = csr_matrix(([1] * len(src), (src, dst)), shape=(num_nodes, num_nodes))
    # 构建超图
    edge_index = torch.tensor([src, dst])
    edge_index_aug, egde_attr = hypergraph_construction(edge_index, n_nodes, k=args.k)  # 构建超图
    edge_index = add_remaining_self_loops(edge_index, num_nodes=n_nodes)[0]

    '''2:************************加载训练数据**************************'''
    train, val, test = loadQuerys(args.dataset_name, args.root_name, args.train_size, args.val_size, args.test_size,
                                  args.train_path, args.test_path, args.val_path)

    '3.*************加载特征数据************'
    if args.dataset_name in ['football', 'facebook_all']:
        path_feat = os.path.join(args.root_name, args.dataset_name, f'{args.dataset_name}_feats.txt')
        if not os.path.isfile(path_feat):
            raise Exception("No such file: %s" % path_feat)
        feats_node = {}
        count = 1
        for line in open(path_feat, encoding='utf-8'):
            if count == 1:
                node_n_, node_in_dim = line.split()
                node_in_dim = int(node_in_dim)
                count = count + 1
            else:
                emb = [float(x) for x in line.split()]
                id = int(emb[0])
                emb = emb[1:]
                feats_node[id] = emb
        nodes_feats = []
        for i in range(0, n_nodes):
            if i not in feats_node:
                nodes_feats.append([0.0] * node_in_dim)
            else:
                nodes_feats.append(feats_node[i])
        nodes_feats = torch.tensor(nodes_feats)
        print(f"{args.dataset_name} feats dtype:", nodes_feats.dtype)

    elif args.dataset_name in ['cora', 'pubmed']:
        nodes_feats = citation_feature_reader(args.root_name, args.dataset_name)  # numpy.ndaaray:(2708,1433)
        nodes_feats = torch.from_numpy(nodes_feats)  # 转换成tensor
        node_in_dim = nodes_feats.shape[1]
    elif args.dataset_name.startswith('fb'): #读取ego facebook的特征数据
        nodes_feats = ego_utils.ego_feature_reader(args.root_name, args.dataset_name)
        nodes_feats = ego_utils.fnormalize(nodes_feats) #将特征进行归一化
        nodes_feats = torch.from_numpy(nodes_feats)  # 转换成tensor
        nodes_feats = nodes_feats.to(torch.float32) # 转换成float32,确保和模型一致
        node_in_dim = nodes_feats.shape[1]
    elif args.dataset_name.startswith('wfb'):
        feature_file = "{}/{}/{}.feat".format(args.root_name, args.dataset_name, args.dataset_name)
        feats_array = ego_utils.load_features(feature_file)
        nodes_feats = torch.tensor(feats_array, dtype=torch.float32)
        node_in_dim = nodes_feats.shape[1]

    return nodes_feats, train, val, test, node_in_dim, n_nodes, edge_index, edge_index_aug, adj_matrix, aa_tensor



def load_graph(root,dataset,attack,ptb_rate=None,type=None):
    '''
    太多地方用到了，写一个公共的避免多个地方进行修改
    目前缺少对facebook数据集的读取
    :param args:
    :return:
    '''
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
        elif dataset.startswith('fb'):  #fbxxx的邻接矩阵
            graphx = ego_utils.ego_graph_reader(root, dataset)
            # graph_path = f'{args.root}/{args.dataset}/{args.dataset[2:]}.edges'
            # graphx = efacebook_utils.read_edge_file(graph_path)
            print(graphx)
            n_nodes = graphx.number_of_nodes()
        elif dataset in ['cora_stb','cora_gsr']:
            path = os.path.join(root, dataset, attack,f'{dataset}_raw.npz')
            print(f'加载图路径：{path}')
            adj_csr_matrix = sp.load_npz(path)
            graphx = nx.from_scipy_sparse_array(adj_csr_matrix)
            print(graphx)
            n_nodes = graphx.number_of_nodes()
    elif attack in ['add','random_remove','gaddm','del','random_add','gdelm','random_flip','gflipm','cdelm','cflipm','delm','flipm']:
        path = os.path.join(root, dataset, attack,
                            f'{dataset}_{attack}_{ptb_rate}.npz')
        print(f'加载图路径：{path}')
        adj_csr_matrix = sp.load_npz(path)
        graphx = nx.from_scipy_sparse_array(adj_csr_matrix)
        print(graphx)
        n_nodes = graphx.number_of_nodes()
    else:
        print('攻击类型不合法')

        path = os.path.join(root, dataset, attack,
                            f'{dataset}_{attack}_{ptb_rate}.npz')
        print(f'加载图路径：{path}')
        adj_csr_matrix = sp.load_npz(path)
        graphx = nx.from_scipy_sparse_array(adj_csr_matrix)
        print(graphx)
        n_nodes = graphx.number_of_nodes()

    return graphx,n_nodes