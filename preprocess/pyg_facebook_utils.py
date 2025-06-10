'''
这个coclep会直接out-of-memory
from pyg attributes datasets
将graph存储为.edges
将featurs存储为.feats
将comms存储。
'''
import os

import networkx as nx
import numpy as np
import torch
from torch_geometric.datasets import AttributedGraphDataset
from torch_geometric.utils import to_networkx
def encode_labels(labels):
    # 使用 argmax 找到每个节点的类别编号
    encoded_labels = torch.argmax(labels, dim=1)
    return encoded_labels
def reindex_labels(encoded_labels):
    # 获取实际存在的非空类别编号并重新编号
    unique_labels = torch.unique(encoded_labels)
    label_map = {old_label.item(): new_label for new_label, old_label in enumerate(unique_labels)}

    # 根据映射关系重新编号
    reindexed_labels = torch.tensor([label_map[label.item()] for label in encoded_labels], dtype=torch.long)
    return reindexed_labels, label_map

def write_labels_to_file(encoded_labels, file_path="../data/facebook/facebook.comms"):
    # Step 1: 重新编号
    reindexed_labels, label_map = reindex_labels(encoded_labels) #不再是one-hot的
    # Step 2: 分组节点并写入文件
    num_classes = len(label_map)  # 新的类别总数
    class_nodes = {i: [] for i in range(num_classes)}  # 初始化类别字典

    for node, label in enumerate(reindexed_labels):
        class_nodes[label.item()].append(node)  # 将节点编号加入对应类别的列表
    # 写入文件
    with open(file_path, 'w') as f:
        # 写入第一行类别编号
        f.write(" ".join(map(str, range(num_classes))) + "\n")
        # 写入每个类别的节点编号
        for i in range(num_classes):
            f.write(" ".join(map(str, class_nodes[i])) + "\n")

def Preprocess_FB(root_name,dataset_name):
    dataset = AttributedGraphDataset(root=root_name, name=dataset_name)
    data = dataset[0]

    edge_list_file = f'{root_name}/{dataset_name}/{dataset_name}.edges'
    feature_file = f'{root_name}/{dataset_name}/{dataset_name}.feat'

    #将edge_index保存为边列表
    edge_index = data.edge_index.numpy().T #转置为(source,target)格式
    np.savetxt(edge_list_file,edge_index,fmt='%d',delimiter=' ')
    print(f'Edges saved to {edge_list_file}')

    #保存节点特征
    features = data.x.numpy()
    np.savetxt(feature_file,features,fmt='%d',delimiter=' ')
    print(f'Features saved to {feature_file}')

    #读取并保存comms数据
    one_hot_labels = data.y  # pyg数据集中默认的labels是one-hot的
    labels = encode_labels(one_hot_labels)
    write_labels_to_file(labels)

    return labels

if __name__ == '__main__':
    # 加载 Facebook 数据集
    root_name = '../data'
    dataset_name = 'facebook'
    labels = Preprocess_FB(root_name,dataset_name) #将数据集存入文件
    edge_list_file = f'{root_name}/{dataset_name}/{dataset_name}.edges'
    feature_file = f'{root_name}/{dataset_name}/{dataset_name}.feats'
    #读取处理后的数据
    graph = nx.read_edgelist(edge_list_file, nodetype=int, data=False)
    # 读取节点特
    features = np.loadtxt(feature_file, dtype=float, delimiter=' ')
    # 打印信息
    print(f"Number of nodes: {graph.number_of_nodes()}")
    print(f"Number of edges: {graph.number_of_edges()}")
    print(f'networkx G :{graph}')
    print(f'节点特征维度：{features.shape}')
    print(f"Labels shape: {labels.shape}")
