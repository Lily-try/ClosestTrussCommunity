import argparse
import math
import os
import numpy as np
import networkx as nx
import scipy.sparse as sp

from utils.citation_loader import citation_graph_reader
from utils import citation_loader
from utils.load_utils import load_graph
'''
已知gt社区文件
生成查询任务并划分后存入文件中
'''
# def gen_queries_and_split(root, dataset, nq=350, np_min=5, np_max=30, threshold=0.8, split_ratios=(150, 100, 100), constrain='True'):
#     '''
#     这里是直接从已经生成的社区文件开始的。
#     :param root: 根目录
#     :param dataset: 数据集名称
#     :param nq: 要生成的查询数量。
#     :param np_min: 每个查询中的最小正例数量。
#     :param np_max: 每个查询中的最大正例数量。
#     :param split_ratios:训练集、验证集和测试集的划分比例
#     :return: 成comms,all_queries,train,val,test的txt文件。
#     '''
#
#     #可以直接读取的。
#     # 1.读取每个节点的label数据
#     # labels = citation_target_reader(root, dataset)  # 读取labels
#     # 2.将 label数据中获取comms列表并存入文件
#     # write_labels_to_file(root, dataset, labels)  # 将labels以comms的形式存入文件
#
#     print('从gt社区中生成查询任务并划分')
#     # 3.从comms文件中读取所有社区列表
#     communities = read_community_data(root, dataset)
#     # 4.从comms中生成查询任务,注意限制。
#     gen_all_queries(root, dataset, communities, nq, np_min, np_max,threshold,constrain)  # 生成并存储全部的查询任务
#     # 5.读取生成的所有查询任务
#     queries = read_queries_from_file(f'{root}/{dataset}/{dataset}_all_queries.txt')  # 读取查询任务
#     # 6. 将查询任务划分并写入文件。
#     split_and_write_queries(queries, root, dataset, split_ratios)  # 将查询任务分割并存入文件
#
#     print('————————测  试——————————————————')
#     dataset_root = os.path.join(root, dataset)
#     train_path = os.path.join(dataset_root, f'{dataset}_pos_train_{split_ratios[0]}.txt')
#     val_path = os.path.join(dataset_root, f'{dataset}_val_{split_ratios[1]}.txt')
#     test_path = os.path.join(dataset_root, f'{dataset}_test_{split_ratios[2]}.txt')
#     train_queries = read_queries_from_file(train_path)
#     validation_queries = read_queries_from_file(val_path)
#     test_queries = read_queries_from_file(test_path)
#     print("Train Queries:", train_queries[:5])  # 打印前5个训练查询任务作为示例
#     print("Validation Queries:", validation_queries[:5])  # 打印前5个验证查询任务作为示例
#     print("Test Queries:", test_queries[:5])  # 打印前5个测试查询任务作为示例
def largest_connected_components(adj):
    graph = nx.from_scipy_sparse_matrix(adj)
    largest_cc = max(nx.connected_components(graph), key=len)
    return list(largest_cc)
def largest_connected_components(adj, n_components=1):
    """S
	Parameterselect k largest connected components.
	----------
	adj : scipy.sparse.csr_matrix
		input adjacency matrix
	n_components : int
		n largest connected components we want to select
	"""
    _, component_indices = sp.csgraph.connected_components(adj)
    component_sizes = np.bincount(component_indices)
    components_to_keep = np.argsort(component_sizes)[::-1][:n_components]  # reverse order to sort descending
    nodes_to_keep = [
        idx for (idx, component) in enumerate(component_indices) if component in components_to_keep]
    print("Selecting {0} largest connected components".format(n_components))
    return nodes_to_keep

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
def read_queries_from_file(file_path):
    """
    从文件中读取查询任务。
    :param file_path: 包含查询任务的文件路径。
    :return: 查询任务的列表，每个任务是 (q, pos, comm)，其中 pos 和 comm 是整数列表。
    """
    queries = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 3:
                q = int(parts[0])
                pos = list(map(int, parts[1].split()))
                comm = list(map(int, parts[2].split()))
                queries.append((q, pos, comm))
            elif len(parts) == 2:  # 适用于测试集，可能只有 q 和 comm
                q = int(parts[0])
                comm = list(map(int, parts[1].split()))
                queries.append((q, [], comm))  # 空 pos 列表
    return queries
def write_queries_to_file(queries, file_path):
    """
    将生成好的查询任务写入文件。根据文件路径区分，测试集和验证集只写入 q 和 comm。
    :param queries: 包含查询任务的列表，每个任务是 (q, pos, comm)。
    :param file_path: 要写入的文件路径。
    """
    with open(file_path, 'w') as f:
        if 'train' in file_path or 'queries' in file_path:
            for q, pos, comm in queries:
                q_str = str(q)
                pos_str = ' '.join(map(str, pos))
                comm_str = ' '.join(map(str, comm))
                f.write(f"{q_str},{pos_str},{comm_str}\n")
        else: #验证集和测试集，只存储q和comm
            for q, _, comm in queries:
                q_str = str(q)
                comm_str = ' '.join(map(str, comm))
                f.write(f"{q_str},{comm_str}\n")

def write_onlynodes_to_file(queries, file_path):
    """
    将生成好的查询任务写入文件。根据文件路径区分，测试集和验证集只写入 q 和 comm。
    :param queries: 包含查询任务的列表，每个任务是 (q, pos, comm)。
    :param file_path: 要写入的文件路径。
    """
    with open(file_path, 'w') as f:
            for q, _, comm in queries:
                q_str = str(q)
                comm_str = ' '.join(map(str, comm))
                f.write(f"{q_str}\n")


def split_and_write_queries(queries, root,dataset, split_ratios=(150, 100, 100)):
    """
    将已经生成的所有查询任务划分为训练集、验证集和测试集，并分别写入文件。
    :param queries: 包含查询任务的列表，每个任务是 (q, pos, comm)。
    :param split_ratios: 元组，表示训练集、验证集和测试集的划分比例，缺省为 (150, 100, 100)。
    """
    dataset_root = os.path.join(root, dataset)
    train_path = os.path.join(dataset_root, f'{dataset}_pos_train_{split_ratios[0]}.txt')
    val_path = os.path.join(dataset_root, f'{dataset}_val_{split_ratios[1]}.txt')
    test_path = os.path.join(dataset_root, f'{dataset}_test_{split_ratios[2]}.txt')

    train_size, val_size, test_size = split_ratios
    assert sum(split_ratios) == len(queries), "Sum of split ratios must equal the total number of queries"
    # 划分任务
    train_queries = queries[:train_size]
    validation_queries = queries[train_size:train_size+val_size]
    test_queries = queries[train_size+val_size:train_size+val_size+test_size]

    # 写入训练集文件
    write_queries_to_file(train_queries, train_path)
    # 写入验证集文件
    write_queries_to_file(validation_queries, val_path)
    # 写入测试集文件
    write_queries_to_file(test_queries, test_path)

def gen_all_queries(root, dataset, communities, nq, np_min, np_max,threshold=0.8,constrain='True',valid_nodes=None):
    """
    根据range [np_min,np_max]从社区数据中生成全部的查询任务，存入对应的all_quries。
    :param communities: 从文件读取的社区字典。
    :param nq: 要生成的查询数量。
    :param np_min: 每个查询中的最小正例数量。
    :param np_max: 每个查询中的最大正例数量。
    :param constrain: 是否对于生成的查询任务间不重复进行限制。
    :return: 查询任务列表，每个任务是 (q, pos, comm)。
    """
    queries = []
    used_qs = set()
    used_pos_sets = set()
    community_labels = list(communities.keys())
    max_attempts = 10000
    attempts = 0
    pos_max = np_max #用于考虑小社区中np_max超过社区size的情况
    # 将字符串转换为布尔值
    constrain_bool = constrain.lower() in ['true', '1', 't', 'y', 'yes']

    while len(queries) < nq and attempts < max_attempts:
        community_label = np.random.choice(community_labels)
        community_nodes = communities[community_label]
        community_size = len(community_nodes)
        if community_size < np_min:
            continue  # 如果社区太小，跳过
        if math.ceil(community_size * 0.4) <= np_max: #如果np_max超过社区上限，则将np_max调低
            np_max = max(math.ceil(community_size * 0.3),np_min) #还需确保np_max >np_min 否则会出错
            # print(f'community_size: {community_size},np_max: {np_max}')
        else:
            np_max = pos_max #否则恢复，初始设置的np_max大小。

        num_pos = np.random.randint(np_min, min(np_max, len(community_nodes)) + 1)
        pos_nodes = tuple(np.random.choice(community_nodes, num_pos, replace=False))#从当前社区community_nodes中随机选择num_pos个节点
        # q = np.random.choice(pos_nodes) 从pos_nodes中随机选择1个作为查询节点。

        # 剩余节点作为可能的查询节点，从中随机选取q
        remaining_nodes = [node for node in community_nodes if node not in pos_nodes]
        if not remaining_nodes:
            attempts += 1
            continue  # 如果没有剩余节点，跳过此次循环
        q = np.random.choice(remaining_nodes)

        if valid_nodes is not None:  # 验证是不是有效的
            if q not in valid_nodes or any(p not in valid_nodes for p in pos_nodes) or any(
                    c not in valid_nodes for c in community_nodes):
                attempts += 1
                continue
        if not constrain_bool or (q not in used_qs and pos_nodes not in used_pos_sets):
            queries.append((q, pos_nodes, community_nodes))
            if constrain_bool:
                used_qs.add(q)
                used_pos_sets.add(pos_nodes)
        elif len(queries) >= threshold * nq:  # 当生成了足够多的唯一任务后，接受重复任务
            queries.append((q, pos_nodes, community_nodes))
        attempts += 1

    if len(queries) < nq: #如果未达到目标数量，生成剩余的任务，忽略重复约束
        print(f'Warning:gen {len(queries)}, Less than desired queries generated, generating remaining without constraints.')
        while len(queries) < nq:

            community_label = np.random.choice(community_labels)
            community_nodes = communities[community_label]
            if len(community_nodes) < np_min:
                continue
            num_pos = np.random.randint(np_min, min(np_max, len(community_nodes)) + 1)
            pos_nodes = tuple(np.random.choice(community_nodes, num_pos, replace=False))
            # q = np.random.choice(pos_nodes)
            remaining_nodes = [node for node in community_nodes if node not in pos_nodes]
            if not remaining_nodes:
                continue
            q = np.random.choice(remaining_nodes)
            if valid_nodes is not None: #避免选到孤立节点
                if q not in valid_nodes or any(p not in valid_nodes for p in pos_nodes) or any(
                        c not in valid_nodes for c in community_nodes):
                    continue
            queries.append((q, pos_nodes, community_nodes))

    #将生成的所有查询任务存入文件
    querys_path = os.path.join(root, dataset, f'{dataset}_all_queries.txt')
    write_queries_to_file(queries,querys_path)
    return queries

def gen_all_queries_with_ratio(root, dataset, communities, nq, ratio,threshold=0.8,constrain='True'):
    """
    从社区数据中生成全部的查询任务，存入对应的all_quries。
    :param communities: 从文件读取的社区字典。
    :param nq: 要生成的查询数量。
    :param np_min: 每个查询中的最小正例数量。
    :param np_max: 每个查询中的最大正例数量。
    :param constrain: 是否对于生成的查询任务间不重复进行限制。
    :return: 查询任务列表，每个任务是 (q, pos, comm)。
    """
    queries = []
    used_qs = set()
    used_pos_sets = set()
    community_labels = list(communities.keys())
    max_attempts = 10000
    attempts = 0
    # 将字符串转换为布尔值
    constrain_bool = constrain.lower() in ['true', '1', 't', 'y', 'yes']

    while len(queries) < nq and attempts < max_attempts:
        community_label = np.random.choice(community_labels)
        community_nodes = communities[community_label] #当前社区的节点数量
        community_size = len(community_nodes)
        if len(community_nodes) < 3:
            continue  # 如果社区太小，跳过
        #根据ratio计算当前社区所需的正例节点的数量
        num_pos = max(1,int(ratio*community_size))
        if num_pos>=community_size:
            num_pos = community_size - 1 #需要至少保留一个节点用于查询

        pos_nodes = tuple(np.random.choice(community_nodes, num_pos, replace=False))#从当前社区community_nodes中随机选择num_pos个节点

        # 剩余节点作为可能的查询节点，从中随机选取q
        remaining_nodes = [node for node in community_nodes if node not in pos_nodes]
        if not remaining_nodes:
            attempts += 1
            continue  # 如果没有剩余节点，跳过此次循环
        q = np.random.choice(remaining_nodes)

        if not constrain_bool or (q not in used_qs and pos_nodes not in used_pos_sets):
            queries.append((q, pos_nodes, community_nodes))
            if constrain_bool:
                used_qs.add(q)
                used_pos_sets.add(pos_nodes)
        elif len(queries) >= threshold * nq:  # 当生成了足够多的唯一任务后，接受重复任务
            queries.append((q, pos_nodes, community_nodes))
        attempts += 1

    if len(queries) < nq: #如果未达到目标数量，生成剩余的任务，忽略重复约束
        print(f'Warning:gen {len(queries)}, Less than desired queries generated, generating remaining without constraints.')
        while len(queries) < nq:
            community_label = np.random.choice(community_labels)
            community_nodes = communities[community_label]
            if len(community_nodes) < 3:
                continue
            num_pos =max(1,int(ratio*community_size))
            if num_pos>=community_size:
                num_pos = community_size -1

            pos_nodes = tuple(np.random.choice(community_nodes, num_pos, replace=False))

            remaining_nodes = [node for node in community_nodes if node not in pos_nodes]
            if not remaining_nodes:
                continue
            q = np.random.choice(remaining_nodes)
            queries.append((q, pos_nodes, community_nodes))

    #将生成的所有查询任务存入文件
    querys_path = os.path.join(root, dataset, f'{dataset}_all_queries_{ratio}.txt')
    write_queries_to_file(queries,querys_path)

def gen_all_queries_with_dis(root, dataset, communities,G, nq, ratio,threshold=0.8,constrain='True',min_distance=3):
    """限制不同任务的查询节点的距离
    从社区数据中生成全部的查询任务，存入对应的all_quries。
    :param communities: 从文件读取的社区字典。
    :param nq: 要生成的查询数量。
    :param ratio: 正例节点在地面真值社区中的比例。
    :param threshold:  当生成了足够多的唯一任务后，接受重复任务的比例。
    :param constrain: 是否对于生成的查询任务间不重复进行限制。
    :param min_distance: 测试集中的查询节点与训练集中查询节点之间的最小距离。
    :return: 查询任务列表，每个任务是 (q, pos, comm)。
    """
    queries = []
    used_qs = set()
    used_pos_sets = set()
    community_labels = list(communities.keys())
    max_attempts = 10000
    attempts = 0
    # 将字符串转换为布尔值
    constrain_bool = constrain.lower() in ['true', '1', 't', 'y', 'yes']

    while len(queries) < nq and attempts < max_attempts:
        community_label = np.random.choice(community_labels) #随机选择一个社区
        community_nodes = communities[community_label] #当前社区的节点
        community_size = len(community_nodes) #社区大小
        if community_size < 3:
            continue  # 如果社区太小，跳过

        #根据ratio计算当前社区所需的正例节点的数量
        num_pos = max(1,int(ratio*community_size))
        if num_pos>=community_size:
            num_pos = community_size - 1 #需要至少保留一个节点用于查询

        # 从当前社区community_nodes中随机选择num_pos个节点
        pos_nodes = tuple(np.random.choice(community_nodes, num_pos, replace=False))

        # 剩余节点作为可能的查询节点，从中随机选取一个q
        remaining_nodes = [node for node in community_nodes if node not in pos_nodes]
        if not remaining_nodes:
            attempts += 1
            continue  # 如果没有剩余节点，跳过此次循环
        q = np.random.choice(remaining_nodes)

        #检查q与已使用的查询节点之间的距离是否满足要求
        is_valid = True
        if constrain_bool:
            for used_q in used_qs:
                if nx.has_path(G,q,used_q):
                    distance = nx.shortest_path_length(G,q,used_q) #查询节点与使用过的q中的最短距离
                    if distance < min_distance:
                        is_valid = False
                        break #表示当前随机选的这个查询节点不合适，重新进行选取。

        if not constrain_bool or (q not in used_qs and pos_nodes not in used_pos_sets):
            queries.append((q, pos_nodes, community_nodes))
            if constrain_bool:
                used_qs.add(q)
                used_pos_sets.add(pos_nodes)
        elif len(queries) >= threshold * nq:  # 当生成了足够多的唯一任务后，接受重复任务
            queries.append((q, pos_nodes, community_nodes))
        attempts += 1

    if len(queries) < nq: #如果未达到目标数量，生成剩余的任务，忽略重复约束
        print(f'Warning:gen {len(queries)}, Less than desired queries generated, generating remaining without constraints.')
        while len(queries) < nq:
            community_label = np.random.choice(community_labels)
            community_nodes = communities[community_label]
            community_size = len(community_nodes)  # 社区大小
            if community_size < 3:
                continue
            num_pos =max(1,int(ratio*community_size))
            if num_pos>=community_size:
                num_pos = community_size -1

            pos_nodes = tuple(np.random.choice(community_nodes, num_pos, replace=False))

            remaining_nodes = [node for node in community_nodes if node not in pos_nodes]
            if not remaining_nodes:
                continue
            q = np.random.choice(remaining_nodes)

            #检查q与已使用的查询节点之间的距离是否满足要求
            is_valid = True
            for used_q in used_qs:
                if nx.has_path(G,q,used_q):
                    distance = nx.shortest_path_length(G,q,used_q)
                    if distance < min_distance:
                        is_valid = False
                        break
            if is_valid:
                queries.append((q, pos_nodes, community_nodes))
                used_qs.add(q)
            else:
                attempts += 1

    #将生成的所有查询任务存入文件
    querys_path = os.path.join(root, dataset, f'{dataset}_all_queries_{ratio}_{min_distance}.txt')
    write_queries_to_file(queries,querys_path)
def gen_queries_and_split_with_ratio(root, dataset, nq=350, pos_ratio=0.15, threshold=0.8, split_ratios=(150, 100, 100), constrain='True'):
    ''' 根据比例生成查询任务
    这里是直接从已经生成的社区文件开始的。
    :param root: 根目录
    :param dataset: 数据集名称
    :param nq: 要生成的查询数量。
    :param np_min: 每个查询中的最小正例数量。
    :param np_max: 每个查询中的最大正例数量。
    :param split_ratios:训练集、验证集和测试集的划分比例
    :return: 成comms,all_queries,train,val,test的txt文件。
    '''

    #可以直接读取的。
    # 1.读取每个节点的label数据
    # labels = citation_target_reader(root, dataset)  # 读取labels
    # 2.将 label数据中获取comms列表并存入文件
    # write_labels_to_file(root, dataset, labels)  # 将labels以comms的形式存入文件

    print('从gt社区中生成查询任务并划分')
    # 3.从comms文件中读取所有社区列表
    communities = read_community_data(root, dataset)


    # 4.从comms中生成查询任务，每个任务中的正例数量取决于pos_ratio
    gen_all_queries_with_ratio(root, dataset, communities, nq, pos_ratio,threshold,constrain)  # 生成并存储全部的查询任务

    # 5.读取生成的所有查询任务
    queries = read_queries_from_file(f'{root}/{dataset}/{dataset}_all_queries_{pos_ratio}.txt')  # 读取查询任务
    # 6. 将查询任务划分并写入文件。
    # split_and_write_queries(queries, root, dataset, split_ratios)  # 将查询任务分割并存入文件
    dataset_root = os.path.join(root, dataset)
    train_path = os.path.join(dataset_root, f'{dataset}_pos_train_{split_ratios[0]}_{pos_ratio}.txt')
    val_path = os.path.join(dataset_root, f'{dataset}_val_{split_ratios[1]}_{pos_ratio}.txt')
    test_path = os.path.join(dataset_root, f'{dataset}_test_{split_ratios[2]}_{pos_ratio}.txt')

    train_size, val_size, test_size = split_ratios
    assert sum(split_ratios) == len(queries), "Sum of split ratios must equal the total number of queries"
    # 划分任务
    train_queries = queries[:train_size]
    validation_queries = queries[train_size:train_size+val_size]
    test_queries = queries[train_size+val_size:train_size+val_size+test_size]

    # 写入训练集文件
    write_queries_to_file(train_queries, train_path)
    # 写入验证集文件
    write_queries_to_file(validation_queries, val_path)
    # 写入测试集文件
    write_queries_to_file(test_queries, test_path)

    print('————————测  试——————————————————')
    dataset_root = os.path.join(root, dataset)
    train_path = os.path.join(dataset_root, f'{dataset}_pos_train_{split_ratios[0]}.txt')
    val_path = os.path.join(dataset_root, f'{dataset}_val_{split_ratios[1]}.txt')
    test_path = os.path.join(dataset_root, f'{dataset}_test_{split_ratios[2]}.txt')
    train_queries = read_queries_from_file(train_path)
    validation_queries = read_queries_from_file(val_path)
    test_queries = read_queries_from_file(test_path)
    print("Train Queries:", train_queries[:5])  # 打印前5个训练查询任务作为示例
    print("Validation Queries:", validation_queries[:5])  # 打印前5个验证查询任务作为示例
    print("Test Queries:", test_queries[:5])  # 打印前5个测试查询任务作为示例


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=str, default='../data',help='dataset root path')
    # ,choices=['fb107','wfb107','cora_stb','cora_gsr','cora','cocs_gsr','cocs_stb','citeseer', 'pubmed','cocs','football','facebook','facebook_all','fb107','fb686','fb348','fb414','fb1684','wfb107']
    parser.add_argument('--dataset', type=str, default='citeseer_gsr', help='dataset name')
    #nq = train_size+val_size+test_size
    parser.add_argument('--nq',type=int,default=900,help='number of queries')
    parser.add_argument('--train_size',type=int,default=300,help='size of train set')
    parser.add_argument('--val_size',type=int,default=100,help='size of validation set')
    parser.add_argument('--test_size',type=int,default=500,help='size of test set')
    parser.add_argument('--type',type=str,default='num',choices=['ratio','range','dis','num'],help='ways to gen query')

    # 控制攻击方法、攻击类型和攻击率
    #choices=['none','meta', 'random_remove', 'random_flip', 'random_add', 'meta_attack', 'add', 'del','gflipm', 'gdelm', 'gaddm', 'cdelm', 'cflipm', 'delm', 'flipm']
    parser.add_argument('--attack', type=str, default='none')
    parser.add_argument('--ptb_rate', type=float, default=0.40, help='pertubation rate')

    #if type = range
    parser.add_argument('--np_min',type=int,default=5,help='min number of queries，egofb:3; citations or facebook_all:5; ')   #如果是ego-facebook数据集，将这个参数设置为3。5太大了
    parser.add_argument('--np_max',type=int,default=30,help='max number of queries, egofb:10;citations or facebook_all:30; ')  #如果是ego-facebook数据集，将这个参数设置为10，30太大了。
    parser.add_argument('--num',type=int,default=3,help='number of pos nodes') #if type =num
    #if type = ratio
    parser.add_argument('--pos_ratio',type=float,default=0.2,help='ratio of positive samples within each query task') #type=ratio时，pos相对于gt_comm的比例
    #if type = dis 考虑生成的训练节点的位置
    parser.add_argument('--min_distance', type=int, default=3, help='each query node min distance')  # type=dis
    #是否限制任务之间的不重复，默认是true
    parser.add_argument('--constrain',type=str,default='True',help='constrain queries') #是否要限制不重复
    parser.add_argument('--threshold', type=float, default=0.8, help='if constrain ==True, consider queries repeat num threshold') #constrain=True时，限制的比例
    args = parser.parse_args()

    split_ratios = (args.train_size, args.val_size, args.test_size)
    print('处理',args.dataset,'数据集咯')


    #可以直接读取的。
    # 1.读取每个节点的label数据
    # labels = citation_target_reader(root, dataset)  # 读取labels
    # 2.将 label数据中获取comms列表并存入文件
    # write_labels_to_file(root, dataset, labels)  # 将labels以comms的形式存入文件

    print('从gt社区中生成查询任务并划分')
    # 3.从comms文件中读取所有社区列表
    communities = read_community_data(args.root, args.dataset)

    # split_and_write_queries(queries, root, dataset, split_ratios)  # 将查询任务分割并存入文件
    if args.type=='ratio':
        # 4.从comms中生成查询任务，每个任务中的正例数量取决于pos_ratio
        gen_all_queries_with_ratio(args.root, args.dataset, communities, args.nq, args.pos_ratio, args.threshold,args.constrain)  # 生成并存储全部的查询任务
        # 5.读取生成的所有查询任务
        queries = read_queries_from_file(f'{args.root}/{args.dataset}/{args.dataset}_all_queries_{args.pos_ratio}.txt')  # 读取查询任务
        # 6. 将查询任务划分并写入文件。
        dataset_root = os.path.join(args.root, args.dataset)
        train_path = os.path.join(dataset_root, f'{args.dataset}_pos_train_{split_ratios[0]}_{args.pos_ratio}.txt')
        val_path = os.path.join(dataset_root, f'{args.dataset}_val_{split_ratios[1]}_{args.pos_ratio}.txt')
        test_path = os.path.join(dataset_root, f'{args.dataset}_test_{split_ratios[2]}_{args.pos_ratio}.txt')
    elif args.type=='dis':
        #读取图数据
        if args.dataset in ['cora', 'pubmed', 'citeseer']:
            graphx = citation_graph_reader(args.root, args.dataset)  # 读取图 nx格式的
            print(graphx)
        #生成所有查询任务
        gen_all_queries_with_dis(args.root, args.dataset, communities,graphx,args.nq,args.pos_ratio,args.threshold,args.constrain,args.min_distance)
        # 5.读取生成的所有查询任务
        queries = read_queries_from_file(f'{args.root}/{args.dataset}/{args.dataset}_all_queries_{args.pos_ratio}.txt')  # 读取查询任务
        # 6. 将查询任务划分并写入文件。
        dataset_root = os.path.join(args.root, args.dataset)
        train_path = os.path.join(dataset_root, f'{args.dataset}_pos_train_{split_ratios[0]}_{args.pos_ratio}_{args.min_distance}.txt')
        val_path = os.path.join(dataset_root, f'{args.dataset}_val_{split_ratios[1]}_{args.pos_ratio}_{args.min_distance}.txt')
        test_path = os.path.join(dataset_root, f'{args.dataset}_test_{split_ratios[2]}_{args.pos_ratio}_{args.min_distance}.txt')
    elif args.type=='range':  #这是目前使用的
        # gen_queries_and_split(args.root, args.dataset, args.nq, args.np_min, args.np_max, args.threshold, split_ratios, constrain=args.constrain)
        # 4.从comms中生成查询任务，每个任务中的正例数量取决于,np_min和np_max这个范围
        gen_all_queries(args.root, args.dataset, communities, args.nq, args.np_min, args.np_max, args.threshold, args.constrain)  # 生成并存储全部的查询任务
        # 5.读取生成的所有查询任务
        queries = read_queries_from_file(f'{args.root}/{args.dataset}/{args.dataset}_all_queries.txt')  # 读取查询任务
        # 6. 将查询任务划分并写入文件。
        dataset_root = os.path.join(args.root, args.dataset)
        train_path = os.path.join(dataset_root, f'{args.dataset}_pos_train_{split_ratios[0]}.txt')
        val_path = os.path.join(dataset_root, f'{args.dataset}_val_{split_ratios[1]}.txt')
        test_path = os.path.join(dataset_root, f'{args.dataset}_test_{split_ratios[2]}.txt')
    elif args.type == 'num':
        graphx, n_nodes = load_graph(args.root, args.dataset, args.attack, args.ptb_rate)
        valid_nodes = set(graphx.nodes())
        gen_all_queries(args.root, args.dataset, communities, args.nq, args.num, args.num, args.threshold,args.constrain,valid_nodes)  # 生成并存储全部的查询任务
        # 5.读取生成的所有查询任务
        queries = read_queries_from_file(f'{args.root}/{args.dataset}/{args.dataset}_all_queries.txt')  # 读取查询任务
        # 6. 将查询任务划分并写入文件。
        dataset_root = os.path.join(args.root, args.dataset)
        train_path = os.path.join(dataset_root, f'{args.dataset}_{args.num}_pos_train_{split_ratios[0]}.txt')
        val_path = os.path.join(dataset_root, f'{args.dataset}_{args.num}_val_{split_ratios[1]}.txt')
        test_path = os.path.join(dataset_root, f'{args.dataset}_{args.num}_test_{split_ratios[2]}.txt')
    elif args.type == 'trans': #选取部分社区用于生成查询
        pass


    #进行数据划分.用于生成不同比例的 默认生成900个查询任务，然后划分为：300,100,500
    train_size, val_size, test_size = split_ratios
    assert sum(split_ratios) == len(queries), "Sum of split ratios must equal the total number of queries"
    # 划分任务
    train_queries = queries[:train_size]
    validation_queries = queries[train_size:train_size+val_size]
    test_queries = queries[train_size+val_size:train_size+val_size+test_size]

    # 写入训练集文件
    write_queries_to_file(train_queries, train_path)
    # 写入验证集文件
    write_queries_to_file(validation_queries, val_path)
    # 写入测试集文件
    write_queries_to_file(test_queries, test_path)
    #只写入测试集的查询节点（用于传统方法）
    query_path = os.path.join(args.root, args.dataset, f'{args.dataset}_querynode.txt')
    print(f'File will be saved at: {query_path}')  # 打印路径
    write_onlynodes_to_file(test_queries,query_path)



    print('————————测  试——————————————————')
    train_queries = read_queries_from_file(train_path)
    validation_queries = read_queries_from_file(val_path)
    test_queries = read_queries_from_file(test_path)
    print("Train Queries:", train_queries[:5])  # 打印前5个训练查询任务作为示例
    print("Validation Queries:", validation_queries[:5])  # 打印前5个验证查询任务作为示例
    print("Test Queries:", test_queries[:5])  # 打印前5个测试查询任务作为示例

    #随机数种子是否需要考虑？