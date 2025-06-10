'''
from snap
'''
import networkx as nx
import numpy as np
import torch

def Preprocess_EgoFb_woCenter(root, dataset):
    '''忽略中心节点的ego-facebook网络,这段代码处理出来的有bug所以不用这个。
    :param root:
    :param dataset:
    :return:
    '''
    '''1. 加载节点特征'''
    ego = dataset[3:]

    feat = np.genfromtxt("{}/{}/raw/{}.feat".format(root, dataset, ego), dtype=np.dtype(str))
    feat = feat.astype(np.int32)
    features = feat[:, 1:]
    # print("all feature num", features.sum(), features.sum() / features.shape[0])
    print(features.shape)
    # 将特征归一化
    # features = fnormalize(features)

    '''2.加载邻接矩阵'''
    node_map = {}
    #利用map将节点编号重新映射为从0开始
    for i in range(feat.shape[0]):  # feat.shape(0)节点个数
        print(f'原始编号: {feat[i,0]}, 映射后的编号: {i}')
        node_map[feat[i, 0]] = i  # 将feat[i,0]是第i个周围节点的原始编号，将其映射为从0开始的编号。
    #加载原始编号的边信息
    edges = np.genfromtxt("{}/{}/raw/{}.edges".format(root,dataset,ego), dtype=np.dtype(str))
    edges = edges.astype(np.int32) #转换成int32类型的numpy数组

    Edges = [] #重新映射节点
    for e in edges:  # 新编号后的边对应的节点编号添加到edge中
        Edges.append((node_map[e[0]], node_map[e[1]]))

    #创建networx图
    G = nx.Graph()
    G.add_edges_from(Edges)

    '''3.读取circles数据'''
    #加载原始编号的circles数据
    circles =[]
    with open('{}/{}/raw/{}.circles'.format(root,dataset, ego), 'r') as label_file:
        for line in label_file:
            line = line.strip().split('\t')
            circle_name = line[0][6:]
            members = list(map(int, line[1:]))
            circles.append((circle_name, members))
    #创建与新编号关联的circles
    mapped_circles = []
    for circle_name, members in circles:
        mapped_members = []
        for member in members:
            if member in node_map:
                mapped_members.append(node_map[member])
        mapped_circles.append(mapped_members)

    #将重新编号后的存入文件
    np.save(f'{root}/{dataset}/{dataset}_features.npy', features)

    nx.write_edgelist(G,f'{root}/{dataset}/{dataset}_adj.edges',data=False)

    with open(f'{root}/{dataset}/{dataset}_comms.txt', 'w',encoding='utf-8') as f:
        circle_names = [circle_name for circle_name, _ in circles]
        f.write(' '.join(circle_names)+'\n')
        for mapped_members in mapped_circles:
            mapped_members_str = ' '.join(map(str, mapped_members))
            f.write(mapped_members_str+'\n')



def fnormalize(mx):
    """Row-normalize sparse matrix"""
    mx = mx.transpose(0, 1)
    # print("mx shape", mx.shape)
    rowsum = mx.sum(1)
    rowsum[rowsum == 0] = 1
    # print("rowsum shape", rowsum.shape)
    # print("rowsum", rowsum[:24])
    mx = mx / rowsum[:, np.newaxis]
    mx = mx.transpose(0, 1)
    return mx

def ego_feature_reader(root,dataset):
    feature_path ="{}/{}/{}_features.npy".format(root, dataset,dataset)
    features = np.load(feature_path)
    return features

def ego_graph_reader(root,dataset):
    edges_path = "{}/{}/{}.edges".format(root, dataset,dataset)
    G = nx.read_edgelist(edges_path, nodetype=int)
    return G

def Preprocess_EgoFb(root, dataset):
    '''
    :param root:
    :param dataset:
    :return:
    '''
    '''1. 加载节点特征'''
    ego = dataset[2:]
    # 加载中心节点特征
    ego_feat = np.genfromtxt("{}/{}/raw/{}.egofeat".format(root, dataset, ego), dtype=np.dtype(str))
    # 加载周围节点特征
    feat = np.genfromtxt("{}/{}/raw/{}.feat".format(root, dataset, ego), dtype=np.dtype(str))
    # 将特征转为numpy数组并堆叠在一起
    ego_feat = ego_feat.astype(np.int32)
    feat = feat.astype(np.int32)

    features = np.vstack((ego_feat, feat[:, 1:])) #逻辑：feat的第一行就是编号为0

    # 将特征归一化
    # features = fnormalize(features)

    '''2.加载邻接矩阵'''
    node_map = {}
    #利用map将节点编号重新映射为从0开始
    node_map[int(ego)] = 0  # 将中心节点原始编号映射为0

    for i in range(feat.shape[0]):  # 遍历feat中的每个周围节点
        node_map[feat[i, 0]] = i+1  # 将feat[i,0]是第i个周围节点的原始编号，将其映射为从0开始的编号。

    #加载边信息
    edges = np.genfromtxt("{}/{}/raw/{}.edges".format(root,dataset,ego), dtype=np.dtype(str)) #加载周围节点的边信息
    edges = edges.astype(np.int32) #转换成int32类型的numpy数组

    Edges = [] #存储节点重新编码后的边信息
    for i in range(feat.shape[0]):  # 遍历feat中的每个节点
        # 添加中心节点ego到每个周围节点的边
        Edges.append((node_map[int(ego)], i + 1))
    for e in edges:  # 新编号后的边对应的节点编号添加到edge中
        Edges.append((node_map[e[0]], node_map[e[1]]))

    #创建networx图
    G = nx.Graph()
    G.add_edges_from(Edges)

    '''3.读取circles数据'''
    #加载circles数据并建议映射
    circles =[]
    with open('{}/{}/raw/{}.circles'.format(root,dataset, ego), 'r') as label_file:
        for line in label_file:
            line = line.strip().split('\t')
            circle_name = line[0][6:]
            members = list(map(int, line[1:]))
            circles.append((circle_name, members))
    #创建与新编号关联的circles
    mapped_circles = []
    for circle_name, members in circles:
        mapped_members = []
        for member in members:
            if member in node_map:
                mapped_members.append(node_map[member])
        mapped_circles.append(mapped_members)


    # np.save('{}/{}/{}_features.npy'.format(root,dataset,dataset), features)
    np.savetxt(f'{root}/{dataset}/{dataset}.feat',features,fmt='%d',delimiter=' ')

    nx.write_edgelist(G,'{}/{}/{}.edges'.format(root,dataset,dataset),data=False)

    with open('{}/{}/{}.comms'.format(root,dataset,dataset), 'w',encoding='utf-8') as f:
        circle_names = [circle_name for circle_name, _ in circles]
        f.write(' '.join(circle_names)+'\n')
        for mapped_members in mapped_circles:
            mapped_members_str = ' '.join(map(str, mapped_members))
            f.write(mapped_members_str+'\n')

def reindex_wfb_datas(root, dataset):
    '''
    不考虑中心节点的，对数据进行重新编号。
    :param root:
    :param dataset:
    :return:
    '''
    # 第一步：读取边文件并重新编号节点
    G = nx.Graph()
    original_to_new_id = {}  # 保存原始编号到新编号的映射
    current_id = 0  # 新编号的计数器

    # 读取边文件
    edge_file = f"{root}/{dataset}/raw/{dataset[3:]}.edges"
    with open(edge_file, 'r') as ef:
        for line in ef:
            source, target = map(int, line.strip().split())
            # 检查并分配新编号
            if source not in original_to_new_id: #确保每个原始编号只有一个映射，跳过已经映射过的节点
                original_to_new_id[source] = current_id
                current_id += 1
            if target not in original_to_new_id:
                original_to_new_id[target] = current_id
                current_id += 1

            # 将重新编号的边添加到图中
            G.add_edge(original_to_new_id[source], original_to_new_id[target])

    # 第二步：读取特征文件并重新编号节点特征数据
    n_nodes = len(original_to_new_id)
    print(f'Graph={G},n_nodes={n_nodes}')

    feature_dim = None  # 用于保存特征维度
    features = None
    feature_file = f"{root}/{dataset}/raw/{dataset[3:]}.feat"
    with open(feature_file, 'r') as ff:
        for line in ff:
            parts = line.strip().split() #字符串列表
            original_node_id = int(parts[0]) #pats[0]是原始节点ID

            # 检查节点是否在图中（可能有无边节点）
            if original_node_id in original_to_new_id: #查看这个节点是否在图中有边
                new_id = original_to_new_id[original_node_id] #获取该节点对应的新ID
                node_features = list(map(float, parts[1:])) #该节点的特征向量

                # 如果是第一次读取特征，需要设置特征维度feature_dim，以初始化特征矩阵
                if feature_dim is None:
                    feature_dim = len(node_features)
                    features = np.zeros((n_nodes, feature_dim), dtype=float)

                #将该节点的特征node_features赋值到特征矩阵的对应位置new_id
                features[new_id] = node_features

    # 第三步：读取 circles 文件并重新编号
    circles = []
    circles_file = f"{root}/{dataset}/raw/{dataset[3:]}.circles"
    with open(circles_file, 'r') as cf:
        for line in cf: #一个circle数据
            parts = line.strip().split()
            circle_name = parts[0]  #parts[0]是circle name
            original_circle_nodes = map(int, parts[1:])  # 获取原始节点编号

            # 将原始节点编号转换为新的编号
            new_circle_nodes = [
                original_to_new_id[node] for node in original_circle_nodes if node in original_to_new_id
            ]
            circles.append((circle_name, new_circle_nodes))

    return G, features, circles

def save_processed_wfb_data(root, dataset, G, features, circles):
    '''
    保存不包含中心节点的数据
    :param root:
    :param dataset:
    :param G:
    :param features:
    :param circles:
    :return:
    '''
    # 保存边数据
    edge_output = f"{root}/{dataset}/{dataset}.edges"
    nx.write_edgelist(G, edge_output, data=False)

    # 保存特征数据
    np.savetxt(f"{root}/{dataset}/{dataset}.feat", features, fmt='%d', delimiter=' ')

    # 保存圈子数据
    circle_output = f"{root}/{dataset}/{dataset}_comms.txt"
    with open(circle_output, 'w') as f:
        circle_indices =' '.join(str(i) for i in range(len(circles)))
        f.write(circle_indices+'\n')

        for _, nodes in circles:
            nodes_str = " ".join(map(str, nodes))
            f.write(f"{nodes_str}\n")

def load_features(feature_file):
    # 初始化一个空列表来存储所有特征数据
    features_list = []
    with open(feature_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            # 跳过第一个元素（节点编号），仅保留特征数据
            features = list(map(float, parts[1:]))
            features_list.append(features)
    # 将特征数据列表转换为 NumPy 数组
    features_array = np.array(features_list)

    return features_array


if __name__ == '__main__':
    # root = '../data'
    # dataset = 'fb107'
    # Preprocess_EgoFb(root, dataset)
    root = '../data'
    dataset = 'wfb107'
    egos =['fb0','fb107','fb348','fb414','fb686','fb698','fb1684','fb1912','fb3437','fb3980']
    '''1 不包含中心节点的读取并重新编号'''
    G, features, circles = reindex_wfb_datas(root, dataset)
    '''2 将重新编号后的数据集存入文件'''
    save_processed_wfb_data(root, dataset, G, features, circles)
    '''3 读取重新存储的重新编号的数据并检验'''
    edges_path = "{}/{}/{}.edges".format(root,dataset, dataset)
    graphx = nx.read_edgelist(edges_path, nodetype=int)
    print(graphx)

    feature_file = "{}/{}/{}.feat".format(root,dataset, dataset)
    feats_array = load_features(feature_file)
    nodes_feats = torch.tensor(feats_array, dtype=torch.float32)
    print('节点特征维度：',nodes_feats.shape[1])
    #加载circles数据

    #解决到底考不考虑中心节点

