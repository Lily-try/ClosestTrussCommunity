import os

import networkx as nx
import scipy.sparse as sp
from citation_loader import citation_graph_reader
'''
生成运行传统的算法所需的数据集准备
'''
def change_query(root,dataset,num=3,test_size=500,k=3,graphx=None):
    '''

    :param root:
    :param dataset:
    :param num: 这个是原始查询中pos节点的个数，与命名有关
    :param test_size:
    :param k: 其实clique中用不着k
    :param graphx:
    :return:
    '''
    test_path = os.path.join(root,dataset, f'{dataset}_{num}_test_{test_size}.txt')
    save_path = os.path.join(root,dataset,'tra', f'{k}/query1.txt')
    #获取图中所有节点
    # valid_nodes = set(graphx.nodes()) #获取有效节点集合
    valid_nodes = {node for node in graphx.nodes() if graphx.degree(node) > 0}  # 只有度数大于零的节点
    # 打开原始文件进行读取
    with open(test_path, 'r', encoding='utf-8') as infile:
        # 打开新的文件用于写入
        with open(save_path, 'w', encoding='utf-8') as outfile:
            for line in infile:
                # 按照逗号分隔每行，提取 q
                q, _ = line.strip().split(',', 1)
                # 只有在q节点是有效节点时才写入
                if int(q) in valid_nodes:
                    outfile.write(q + '\n')
    print(f"k-clique所需的信息 q 已存入{save_path}")

def change_query_k(root,dataset,num=3,test_size=500,k=2, graphx=None):
    #truss和core需要的格式
    test_path = os.path.join(root, dataset, f'{dataset}_{num}_test_{test_size}.txt')
    truss_path = os.path.join(root, dataset,'tra', f'{k}/truss_querynodes.txt')
    core_path = os.path.join(root, dataset,'tra', f'{k}/core_querynodes.txt')
    # 获取图中所有节点
    valid_nodes = set(graphx.nodes())  # 获取有效节点集合
    # 打开原始文件进行读取
    with open(test_path, 'r', encoding='utf-8') as infile:
        # 打开新的文件用于写入
        with open(truss_path, 'w', encoding='utf-8') as outfile:
            for line in infile:
                # 按照逗号分隔每行，提取 q
                q, _ = line.strip().split(',', 1)
                if int(q) in valid_nodes:
                    outfile.write(f'{q} {k}\n')

    with open(test_path, 'r', encoding='utf-8') as infile:
        # 打开新的文件用于写入
        with open(core_path, 'w', encoding='utf-8') as outfile:
            for line in infile:
                # 按照逗号分隔每行，提取 q
                q, _ = line.strip().split(',', 1)
                if int(q) in valid_nodes:
                    outfile.write(f'{q} {k}\n')
    print('k-truss和k-core所需的query信息正确存储')

def change_k_query(root,dataset,num=3,test_size=500,k=2, graphx=None):
    #k-ecc需要的格式
    test_path = os.path.join(root, dataset, f'{dataset}_{num}_test_{test_size}.txt')
    save_path = os.path.join(root, dataset,'tra', f'{k}/query_ecc.txt')
    # 获取图中所有节点
    valid_nodes = set(graphx.nodes())  # 获取有效节点集合
    # 打开原始文件进行读取
    with open(test_path, 'r', encoding='utf-8') as infile:
        # 打开新的文件用于写入
        with open(save_path, 'w', encoding='utf-8') as outfile:
            for line in infile:
                # 按照逗号分隔每行，提取 q
                q, _ = line.strip().split(',', 1)
                if int(q) in valid_nodes:
                    outfile.write(f'{q} {k}\n')
    print(f"k-ecc所需要的query格式已存入：{save_path}")

def change_graph_type(root,dataset,attack,k,ptb_rate=None,noise_level=None):
    if attack == 'none':  # 使用原始数据
        if dataset in ['cora', 'pubmed', 'citeseer']:
            graphx = citation_graph_reader(root, dataset)  # 读取图 nx格式的
            print(graphx)
            n_nodes = graphx.number_of_nodes()
    elif attack.startswith('random'):
        path = os.path.join(root, dataset, 'random',
                            f'{dataset}_{attack}_{ptb_rate}.npz')
        adj_csr_matrix = sp.load_npz(path)
        graphx = nx.from_scipy_sparse_array(adj_csr_matrix)
        print(graphx)
        n_nodes = graphx.number_of_nodes()
    elif attack =='add': #metaGC中自己注入随机噪声
        path = os.path.join(root, dataset, attack,
                            f'{dataset}_{attack}_{noise_level}.npz')
        adj_csr_matrix = sp.load_npz(path)
        graphx = nx.from_scipy_sparse_array(adj_csr_matrix)
        print(graphx)
        n_nodes = graphx.number_of_nodes()
    elif attack in ['del','gflipm']:
        path = os.path.join(root,dataset,attack,
                            f'{dataset}_{attack}_{ptb_rate}.npz')
        adj_csr_matrix = sp.load_npz(path)
        graphx = nx.from_scipy_sparse_array(adj_csr_matrix)
        print(graphx)
        n_nodes = graphx.number_of_nodes()
    #将graphx转成txt存储，便于传统方法的调用
    save_path = os.path.join(root,dataset,'tra',f'{k}/graph.txt')
    with open(save_path, "w") as f:
        for edge in graphx.edges():
            f.write(f"{edge[0]} {edge[1]}\n")
    print(f"Edges saved to {save_path}")
    return graphx
def read_graph(root,dataset,attack,ptb_rate=None,noise_level=None):
    if attack == 'none':  # 使用原始数据
        if dataset in ['cora', 'pubmed', 'citeseer']:
            graphx = citation_graph_reader(root, dataset)  # 读取图 nx格式的
            print(graphx)
            n_nodes = graphx.number_of_nodes()
    elif attack.startswith('random'):
        path = os.path.join(root, dataset, 'random',
                            f'{dataset}_{attack}_{ptb_rate}.npz')
        adj_csr_matrix = sp.load_npz(path)
        graphx = nx.from_scipy_sparse_array(adj_csr_matrix)
        print(graphx)
        n_nodes = graphx.number_of_nodes()
    elif attack =='add': #metaGC中自己注入随机噪声
        path = os.path.join(root, dataset, attack,
                            f'{dataset}_{attack}_{noise_level}.npz')
        adj_csr_matrix = sp.load_npz(path)
        graphx = nx.from_scipy_sparse_array(adj_csr_matrix)
        print(graphx)
        n_nodes = graphx.number_of_nodes()
    elif attack in ['del','gflipm']:
        path = os.path.join(root,dataset,attack,
                            f'{dataset}_{attack}_{ptb_rate}.npz')
        adj_csr_matrix = sp.load_npz(path)
        graphx = nx.from_scipy_sparse_array(adj_csr_matrix)
        print(graphx)
        n_nodes = graphx.number_of_nodes()
    return graphx

def load_FB(root,dataset,attack,k,ptb_rate=None,noise_level=None):
    max = 0
    edges = []
    '''********************1. 加载图数据******************************'''
    if attack == 'none':  # 使用原始数据
        if dataset in ['football', 'facebook_all']: #原文中的数据集
            path = os.path.join(root, dataset, f'{dataset}.txt')
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
        elif dataset.startswith(('fb', 'wfb','fa')): #读取ego-facebook的邻接矩阵
            graphx = nx.read_edgelist(f'{root}/{dataset}/{dataset}.edges', nodetype=int,data=False)
            print(graphx)
            n_nodes = graphx.number_of_nodes()
        else:
            raise ValueError(f"未识别的数据集类型：{dataset}")  # 处理没有匹配的情况
    elif attack == 'random':
        path = os.path.join(root, dataset, attack, f'{dataset}_{attack}_{type}_{ptb_rate}.npz')
        adj_csr_matrix = sp.load_npz(path)
        graphx = nx.from_scipy_sparse_array(adj_csr_matrix)
        print(graphx)
        n_nodes = graphx.number_of_nodes()
    elif attack =='add': #metaGC中自己注入随机噪声
        path = os.path.join(root, dataset, attack, f'{dataset}_{attack}_{noise_level}.npz')
        adj_csr_matrix = sp.load_npz(path)
        graphx = nx.from_scipy_sparse_array(adj_csr_matrix)
        print(graphx)
        n_nodes = graphx.number_of_nodes()
    elif attack in ['del','gflipm']:
        path = os.path.join(root,dataset,attack,
                            f'{dataset}_{attack}_{ptb_rate}.npz')
        adj_csr_matrix = sp.load_npz(path)
        graphx = nx.from_scipy_sparse_array(adj_csr_matrix)
        print(graphx)
        n_nodes = graphx.number_of_nodes()
    else:
        raise ValueError(f"未识别的"
                         f"攻击类型：{attack}")  # 处理没有匹配的攻击类型情况
    #将graphx转成txt存储，便于传统方法的调用
    save_path = os.path.join(root,dataset,'tra',f'{k}/graph.txt')
    with open(save_path, "w") as f:
        for edge in graphx.edges():
            f.write(f"{edge[0]} {edge[1]}\n")
    print(f"Edges saved to {save_path}")
    return graphx

if __name__ == '__main__':
    root = '../data'
    dataset = 'cora'
    attack = 'gflipm'  #none,add,del,random_remove,random_flip,random_add,gflipm,gdelm
    ptb_rate = 0.4 #除了add noise以外的都有这个参数
    noise_level = 2 #如果是add noise则有这个
    k =3  #生成所需要的k
    if dataset.startswith('fb'):
        graphx = load_FB(root,dataset,attack,k,ptb_rate=ptb_rate,noise_level=noise_level)
    else:
        graphx = change_graph_type(root, dataset, attack,k, ptb_rate=ptb_rate, noise_level=noise_level)
    # graphx = read_graph(root, dataset, attack, ptb_rate=ptb_rate, noise_level=noise_level)
    #k-clique
    change_query(root,dataset,num=3,test_size=500,k=k,graphx=graphx)
    # #truss和core需要的格式
    change_query_k(root,dataset,num=3,test_size=500,k=k,graphx=graphx)
    # #k-ecc
    change_k_query(root,dataset,num=3,test_size=500,k=k,graphx=graphx)