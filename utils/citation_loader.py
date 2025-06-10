import numpy as np
import scipy.sparse as sp
import os
import sys
import networkx as nx
import pickle as pkl

from torch_geometric.datasets import Planetoid

'''
读取引文网络（Cora，citeseer，pubmed）的graph，features，labels数据
将labels 按照gt_community的格式写入文件
引文网络数据来源：
'''

def parse_index_file(filename):
    """解析索引文件"""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features

def citation_graph_reader(root,dataset_str):
    '''
    从指定的数据集中读取一个图并将其表示为networkx中的图对象。
      ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    :param dataset_str:数据集的文件名 例如cora
    '''
    names = ['graph']
    objects = []  # 用于存储从文件中读取的对象
    for i in range(len(names)):  # 这里只有1个元素，因此只会执行一次循环
        # ../data/meta/raw/ind.meta.graph
        path=os.path.join(root,dataset_str,'raw','ind.{0}.{1}'.format(dataset_str,names[i]))
        with open(path, 'rb') as f:
            if sys.version_info > (3, 0):  # 检查python版本信息是否大于3.9.
                objects.append(pkl.load(f, encoding='latin1'))  # 使用latin1编码从打开的文件'f'中加载对象并添加到Objects列表中
            else:
                objects.append(pkl.load(f))  # python2.x不支持编码参数
    graphx = nx.from_dict_of_lists(objects[0])  # 从objcts中提取第一个对象（字典）
    return graphx

def citation_feature_reader(root,dataset_str, compression=0):
    '''
    读取指定数据集中的节点特征
    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    '''
    names = ['x', 'tx', 'allx']
    objects = []
    for i in range(len(names)):
        path = os.path.join(root, dataset_str, 'raw', 'ind.{0}.{1}'.format(dataset_str, names[i]))
        with open(path, 'rb') as f:  # ./data/meta/ind.meta.x
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, tx, allx = tuple(objects)
    test_path = os.path.join(root, dataset_str, 'raw', 'ind.{0}.test.index'.format(dataset_str))
    test_idx_reorder = parse_index_file(test_path)
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':  # citeseer图中存在孤立节点
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended

    features = sp.vstack((allx, tx)).tolil() #lil_matrix:(2708,1433)
    features[test_idx_reorder, :] = features[test_idx_range, :]
    preprocess_features(features)
    features = features.tocoo().astype(np.float32) #coo_matrix:(2708,1433)
    features = features.toarray() #ndarray:(2708,1433)

    feature_list = []
    feature_list.append(features)

    return features #list:1

def citation_target_reader(root, dataset):
    """
    从raw数据集中读取节点的labels数据。
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    :param dataset: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """

    names = ['y', 'ty', 'ally']
    objects = []
    for i in range(len(names)):
        path = os.path.join(root, dataset, 'raw', 'ind.{0}.{1}'.format(dataset, names[i]))
        with open(path, 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    y, ty, ally = tuple(objects) #都是ndarray
    # 读取测试节点实例
    test_path = os.path.join(root, dataset, 'raw', 'ind.{0}.test.index'.format(dataset))
    test_idx_reorder = parse_index_file(test_path)
    test_idx_range = np.sort(test_idx_reorder)

    #如有必要，扩展测试标签，例如citeseer数据集
    if dataset == 'citeseer':
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)

        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended
    #将标签数组转换为标签索引。
    ally = np.argmax(ally, axis=1)
    ty = np.argmax(ty, axis=1)
    # 将所有标签合并（包含了训练和测试集的）
    labels = np.concatenate((ally, ty)) #ndaaray
    # 保证测试标签位置和测试数据的位置对应。此时
    labels[test_idx_reorder] = labels[test_idx_range]

    # labels = labels.reshape(-1, 1)
    return labels

def write_labels_to_file(root, dataset, labels):
    '''
    将节点ID按照label标签分组，即按照社区，写入文本文件
    :param root: 数据存储根目录
    :param dataset: 数据集名称
    :param labels: 标签，ndarray格式。
    :return:
    '''
    #按照label分组标签
    node_id_to_label = np.argsort(labels)
    sorted_labels = labels[node_id_to_label]

    with open(os.path.join(root, dataset,f'{dataset}_comms.txt'), 'w',encoding='utf-8') as f:
        #在第一行写入唯一标签
        unique_labels =np.unique(labels)
        f.write(' '.join(map(str,unique_labels))+'\n')

        #写入按标签分组的节点ID
        current_label = None
        for node_id,label in zip(node_id_to_label,sorted_labels):
            if label != current_label:
                if current_label is not None:
                    f.write('\n')
                current_label = label
                f.write(f"{node_id} ")
            else :
                f.write(f"{node_id} ")

def load_graph(root,dataset,attack,ptb_rate=None,type=None):
    '''
    太多地方用到了，写一个公共的避免多个地方进行修改 我就是直接复制过来的
    目前缺少对facebook数据集的读取
    :param args:
    :return:
    '''
    print(f"[load_graph] attack = {attack}")
    print('在load graph中')
    if attack == 'none':  # 使用原始数据
        print('在none里')
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
        elif dataset in ['cora_stb','cora_gsr']:
            path = os.path.join(root, dataset, attack,f'{dataset}_raw.npz')
            adj_csr_matrix = sp.load_npz(path)
            graphx = nx.from_scipy_sparse_array(adj_csr_matrix)
            print(graphx)
            n_nodes = graphx.number_of_nodes()
        elif dataset in ['fb107']:
            print('成功进入这里')
            graphx = nx.read_edgelist(f'{root}/{dataset}/{dataset}.edges', nodetype=int, data=False)
            print(graphx)
            n_nodes = graphx.number_of_nodes()
    elif attack in ['random_add','random_remove','random_flip','flipm','cdelm','cflipm','caddm','gaddm','gdelm','gflipm','del','add']:
        path = os.path.join(root, dataset, attack,
                            f'{dataset}_{attack}_{ptb_rate}.npz')
        adj_csr_matrix = sp.load_npz(path)
        graphx = nx.from_scipy_sparse_array(adj_csr_matrix)
        print(graphx)
        n_nodes = graphx.number_of_nodes()
    else:
        print('噪声类型不匹配')
        path = os.path.join(root, dataset, attack,
                            f'{dataset}_{attack}_{ptb_rate}.npz')
        adj_csr_matrix = sp.load_npz(path)
        graphx = nx.from_scipy_sparse_array(adj_csr_matrix)
        print(graphx)
        n_nodes = graphx.number_of_nodes()

    return graphx,n_nodes


if __name__ == '__main__':
    root_name = '../data'
    dataset_name ='pubmed'

    # 指定数据集名称和存储路径
    dataset = Planetoid(root=root_name, name=dataset_name)
    data = dataset[0]
    print(data.num_nodes)
    print(data.num_edges)
    print(data.x.shape)
    print(data.y.shape)
    print(data.edge_index.shape)


    labels = citation_target_reader(root_name,dataset_name)

    write_labels_to_file(root_name, dataset_name, labels)