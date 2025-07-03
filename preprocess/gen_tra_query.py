import os
import random
import shutil

import networkx as nx
import scipy.sparse as sp
from citation_loader import citation_graph_reader
from load_utils import load_graph
from main_COCLEP import gen_new_queries_
from preprocess import relabel_graph

'''
生成运行传统的算法所需的数据集准备
'''
def change_query(root,dataset,attack,ptb_rate,num=3,test_size=500,graphx=None):
    '''

    :param root:
    :param dataset:
    :param num: 这个是原始查询中pos节点的个数，与命名有关
    :param test_size:
    :param k: 其实clique中用不着k
    :param graphx:
    :return:
    '''
    #准备graph
    # 读取攻击图
    if attack != 'none':
        target_dataset = f'{dataset}_{attack}_{ptb_rate}'
    else:
        target_dataset = dataset
    target_dir = os.path.join('..', "kclique","Dataset",target_dataset)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    #创建clique子目录
    clique_dir = os.path.join(target_dir, 'clique')
    if not os.path.exists(clique_dir):
        os.makedirs(clique_dir)
    else:
        # 删除目录下所有文件
        for filename in os.listdir(clique_dir):
            file_path = os.path.join(clique_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
    graphx, n_nodes = load_graph(root, dataset, attack, ptb_rate)
    # 判断是否存在自环并删除
    self_loops = list(nx.selfloop_edges(graphx))  # 找出所有自环
    if self_loops:
        print(f"Found {len(self_loops)} self-loops, removing them...")
        graphx.remove_edges_from(self_loops)

    #1.第一步：将graphx转成txt存储，便于传统方法的调用
    # save_path = os.path.join(root, dataset,f'graph.txt')
    save_path = os.path.join(target_dir,'graph.txt')

    with open(save_path, "w") as f:
        for edge in graphx.edges():
            f.write(f"{edge[0]} {edge[1]}\n")
    print(f"Edges saved to {save_path}")

    #测试集，第二步：将查询和gt社区存入
    test_path = os.path.join(root,dataset, f'{dataset}_{num}_test_{test_size}.txt')
    # save_path = os.path.join(root,dataset, f'query1.txt')
    save_path = os.path.join(target_dir,'query1.txt')
    #获取图中所有节点
    valid_nodes = {node for node in graphx.nodes() if graphx.degree(node) > 0}  # 只有度数大于零的节点
    # 打开原始文件进行读取
    ground_truths=[]
    with open(test_path, 'r', encoding='utf-8') as infile:
        # 打开新的文件用于写入
        with open(save_path, 'w', encoding='utf-8') as outfile:
            for line in infile:
                # 按照逗号分隔每行，提取 q
                q, comm = line.strip().split(',', 1)
                # 只有在q节点是有效节点时才写入
                if int(q) in valid_nodes:
                    outfile.write(q + '\n')
                    ground_truths.append((q, comm))
    print(f"k-clique所需的信息 q 已存入{save_path}")
    # target_path = os.path.join(root, dataset,f'comms.txt')
    target_path = os.path.join(target_dir,'comms.txt')
    with open(target_path, 'w') as f_comms:
        for q, comm in ground_truths:
            nodes = comm.strip().split()
            f_comms.write(f"{q}, {' '.join(nodes)}\n")
    print(f"Filtered comms saved to {target_path}")


def change_query_ctc(root,dataset,attack,ptb_rate,num=3,test_size=500,seed=32,qnum=3,max_queries=500):
    '''

    :param root:
    :param dataset:
    :param num: 这个是原始查询中pos节点的个数，与命名有关
    :param test_size:
    :param k: 其实clique中用不着k
    :param graphx:
    :return:
    '''

    if attack != 'none':
        target_dataset =f'{dataset}_{attack}_{ptb_rate}'
    else:
        target_dataset =dataset

    target_dir = os.path.join("..", "ClosestTrussCommunity", "Dataset", target_dataset)
    os.makedirs(target_dir, exist_ok=True)  # 确保目标目录存在
    #图
    graphx,n_nodes = load_graph(root,dataset,attack,ptb_rate)
    # 判断是否存在自环并删除
    self_loops = list(nx.selfloop_edges(graphx))  # 找出所有自环
    if self_loops:
        print(f"Found {len(self_loops)} self-loops, removing them...")
        graphx.remove_edges_from(self_loops)
    #将graphx转成txt存储，便于传统方法的调用
    print(f'读取的图中，节点数{graphx.number_of_nodes()}，边数{graphx.number_of_edges()}')
    save_path = os.path.join(target_dir,f'graph.txt')
    with open(save_path, "w") as f:
        for edge in graphx.edges():
            f.write(f"{edge[0]} {edge[1]}\n")
    print(f"Edges saved to {save_path}")

    #!!!在预处理的时候已经将节点编号都加1了
    with open(save_path, 'r') as f_in:
        edges = []
        max_node = 0
        for line in f_in:
            u, v = map(int, line.strip().split())
            u_new = u + 1
            v_new = v + 1
            edges.append((u_new, v_new))
            current_max = max(u_new, v_new)
            max_node = max(max_node, current_max)
        num_nodes = max_node
        num_edges = len(edges)
    if attack != 'none':
        new_path = os.path.join(target_dir,f'{dataset}_{attack}_{ptb_rate}.txt')
    else:
        new_path = os.path.join(target_dir,f'{dataset}.txt')
    with open(new_path, 'w') as f_out:
        f_out.write(f"{num_nodes} {num_edges}\n")
        for u, v in edges:
            f_out.write(f"{u} {v}\n")
    print(f'将节点编号加一后的数据已经存入:{new_path}')
    #测试集"
    test_path = os.path.join(root,dataset, f'{dataset}_{num}_test_{test_size}.txt')

    valid_nodes = {node for node in graphx.nodes() if graphx.degree(node) > 0}  # 只有度数大于零的节点是有效节点
    random.seed(seed)
    queries = []
    save_path = os.path.join(target_dir, 'query.txt')
    ground_truths = []
    with open(test_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            if max_queries is not None and len(queries) >= max_queries:
                break  # 控制最多生成的数量
            if not line.strip():
                continue
            q_str, comm_str = line.strip().split(',')
            q = int(q_str.strip())
            if q not in valid_nodes:
                continue #跳过度数不为0的
            # queries.append([q]) #只留下这一行是让查询中只包含查询节点
            comm = list(map(int, comm_str.strip().split()))
            comm = [n for n in comm if n in valid_nodes]
            if not comm or q not in comm:
                continue
            comm.remove(q)
            if len(comm) == 0:
                continue  # 跳过空社区
            ks = random.randint(1, min(qnum, len(comm))) #要抽取的个数，
            if ks == 1:
                sample =[q]
            else:
                sample = [q]
                sample.extend(random.sample(comm, ks-1))
            queries.append(sample)
            ground_truths.append((q, comm))

    with open(save_path, 'w') as f:
        f.write(f"{len(queries)}\n")   #这里是看是不是还应该将查询中的节点编号都加1
        for q in queries:
            f.write(f"{len(q)} {' '.join(map(str, q))}\n")
        # for q in queries: #测试是否需要将query中的节点编号也都加上1
        #     q_plus_1 = [node + 1 for node in q]
        #     f.write(f"{len(q_plus_1)} {' '.join(map(str, q_plus_1))}\n")
    print(f"Converted {len(queries)} queries saved to {save_path}")

    comms_path = os.path.join(target_dir, "comms.txt")
    with open(comms_path, 'w') as f_comms:
        for q, comm in ground_truths:
            f_comms.write(f"{q}, {' '.join(map(str, comm))}\n")
    print(f"Filtered comms saved to {comms_path}")

def change_query_photo(root,dataset,attack,ptb_rate,num=3,test_size=500,seed=32,qnum=3,max_queries=500):
    '''
        针对重新编号的
    :param root:
    :param dataset:
    :param num: 这个是原始查询中pos节点的个数，与命名有关
    :param test_size:
    :param k: 其实clique中用不着k
    :param graphx:
    :return:
    '''

    if attack != 'none':
        target_dataset =f'{dataset}_{attack}_{ptb_rate}'
    else:
        target_dataset =dataset

    target_dir = os.path.join("..", "ClosestTrussCommunity", "Dataset", target_dataset)
    os.makedirs(target_dir, exist_ok=True)  # 确保目标目录存在
    #图
    graphx,n_nodes = load_graph(root,dataset,attack,ptb_rate)
    graphx, mapping = relabel_graph(graphx)

    # 判断是否存在自环并删除
    self_loops = list(nx.selfloop_edges(graphx))  # 找出所有自环
    if self_loops:
        print(f"Found {len(self_loops)} self-loops, removing them...")
        graphx.remove_edges_from(self_loops)
    #将graphx转成txt存储，便于传统方法的调用
    save_path = os.path.join(target_dir,f'graph.txt')
    with open(save_path, "w") as f:
        for edge in graphx.edges():
            f.write(f"{edge[0]} {edge[1]}\n")
    print(f"Edges saved to {save_path}")

    #!!!在预处理的时候已经将节点编号都加1了
    with open(save_path, 'r') as f_in:
        edges = []
        max_node = 0
        for line in f_in:
            u, v = map(int, line.strip().split())
            u_new = u + 1
            v_new = v + 1
            edges.append((u_new, v_new))
            current_max = max(u_new, v_new)
            max_node = max(max_node, current_max)
        num_nodes = max_node
        num_edges = len(edges)
    if attack != 'none':
        new_path = os.path.join(target_dir,f'{dataset}_{attack}_{ptb_rate}.txt')
    else:
        new_path = os.path.join(target_dir,f'{dataset}.txt')
    with open(new_path, 'w') as f_out:
        f_out.write(f"{num_nodes} {num_edges}\n")
        for u, v in edges:
            f_out.write(f"{u} {v}\n")
    print(f'新编号后的数据已经存入:{new_path}')
    #测试集" 生成查询
    valid_nodes = set(graphx.nodes())
    gen_new_queries_(root,dataset,300,100,test_size,3,valid_nodes)
    test_path = os.path.join(root,dataset, f'{dataset}_{num}_test_{test_size}.txt')

    #获取图中有效点
    valid_nodes = {node for node in graphx.nodes() if graphx.degree(node) > 0}  # 只有度数大于零的节点
    random.seed(seed)
    queries = []
    # save_path = os.path.join(root, dataset,'tra',f'{k}/query.txt')
    save_path = os.path.join(target_dir, 'query.txt')
    ground_truths = []
    with open(test_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            if max_queries is not None and len(queries) >= max_queries:
                break  # 控制最多生成的数量
            if not line.strip():
                continue
            q_str, comm_str = line.strip().split(',')
            q = int(q_str.strip())
            if q not in valid_nodes:
                continue #跳过度数不为0的
            # queries.append([q]) #只留下这一行是让查询中只包含查询节点
            comm = list(map(int, comm_str.strip().split()))
            comm = [n for n in comm if n in valid_nodes]
            if not comm or q not in comm:
                continue
            comm.remove(q)
            if len(comm) == 0:
                continue  # 跳过空社区
            ks = random.randint(1, min(qnum, len(comm))) #要抽取的个数，
            if ks == 1:
                sample =[q]
            else:
                sample = [q]
                sample.extend(random.sample(comm, ks-1))
            queries.append(sample)
            ground_truths.append((q, comm))

    with open(save_path, 'w') as f:
        f.write(f"{len(queries)}\n")   #这里是看是不是还应该将查询中的节点编号都加1
        for q in queries:
            f.write(f"{len(q)} {' '.join(map(str, q))}\n")
        # for q in queries: #测试是否需要将query中的节点编号也都加上1
        #     q_plus_1 = [node + 1 for node in q]
        #     f.write(f"{len(q_plus_1)} {' '.join(map(str, q_plus_1))}\n")
    print(f"Converted {len(queries)} queries saved to {save_path}")

    comms_path = os.path.join(target_dir,"comms.txt")
    with open(comms_path, 'w') as f_comms:
        for q, comm in ground_truths:
            f_comms.write(f"{q}, {' '.join(map(str, comm))}\n")
    print(f"Filtered comms saved to {comms_path}")

def change_query_k(root,dataset,attack,ptb_rate,num=3,test_size=500,k=2, graphx=None):
    '''
    修改kcore的
    :param root:
    :param dataset:
    :param attack:
    :param num:
    :param test_size:
    :param k:
    :param graphx:
    :return:
    '''

    if attack != 'none':
        target_dataset = f'{dataset}_{attack}_{ptb_rate}'
    else:
        target_dataset = dataset
    target_dir = os.path.join('..', "kcore","Dataset",target_dataset)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    graphx, n_nodes = load_graph(root, dataset, attack, ptb_rate)
    # 判断是否存在自环并删除
    self_loops = list(nx.selfloop_edges(graphx))  # 找出所有自环
    if self_loops:
        print(f"Found {len(self_loops)} self-loops, removing them...")
        graphx.remove_edges_from(self_loops)

    #1.第一步：将graphx转成txt存储，便于传统方法的调用
    # save_path = os.path.join(root, dataset,f'graph.txt')
    save_path = os.path.join(target_dir,'graph.txt')
    with open(save_path, "w") as f:
        for edge in graphx.edges():
            f.write(f"{edge[0]} {edge[1]}\n")
    print(f"Edges saved to {save_path}")

    #truss和core需要的格式
    test_path = os.path.join(root, dataset, f'{dataset}_{num}_test_{test_size}.txt')
    truss_path = os.path.join(target_dir, 'truss_querynodes.txt')
    core_path = os.path.join(target_dir,'core_querynodes.txt')
    # 获取图中所有节点
    valid_nodes = set(graphx.nodes())  # 获取有效节点集合
    # 打开原始测试集文件进行读取
    with open(test_path, 'r', encoding='utf-8') as infile:
        # 打开新的文件用于写入
        with open(truss_path, 'w', encoding='utf-8') as outfile:
            for line in infile:
                # 按照逗号分隔每行，提取 q
                q, _ = line.strip().split(',', 1)
                if int(q) in valid_nodes:
                    outfile.write(f'{q} {k}\n')

    #获取图中所有节点
    # valid_nodes = {node for node in graphx.nodes() if graphx.degree(node) > 0}  # 只有度数大于零的节点
    ground_truths = []
    with open(test_path, 'r', encoding='utf-8') as infile:
        # 打开新的文件用于写入
        with open(core_path, 'w', encoding='utf-8') as outfile:
            for line in infile:
                # 按照逗号分隔每行，提取 q
                q, comm = line.strip().split(',', 1)
                if int(q) in valid_nodes:
                    outfile.write(f'{q} {k}\n')
                    ground_truths.append((q, comm))
    target_path = os.path.join(target_dir, 'comms.txt')
    with open(target_path, 'w') as f_comms:
        for q, comm in ground_truths:
            nodes = comm.strip().split() #正确拆分节点ID
            f_comms.write(f"{q}, {' '.join(nodes)}\n")
    print(f"Filtered comms saved to {target_path}")
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

def change_graph_type(root,dataset,attack,ptb_rate=None,k=None):
    #读取攻击图
    graphx,n_nodes = load_graph(root,dataset,attack,ptb_rate)
    # 判断是否存在自环并删除
    self_loops = list(nx.selfloop_edges(graphx))  # 找出所有自环
    if self_loops:
        print(f"Found {len(self_loops)} self-loops, removing them...")
        graphx.remove_edges_from(self_loops)

    #将graphx转成txt存储，便于传统方法的调用
    save_path = os.path.join(root,dataset,'tra',f'{k}/graph.txt')
    with open(save_path, "w") as f:
        for edge in graphx.edges():
            f.write(f"{edge[0]} {edge[1]}\n")
    print(f"Edges saved to {save_path}")

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
    dataset = 'cora'  #fb107，cora,citeseer,cocs
    attack = 'meta'  #none,add,random_add,gaddm,del,random_remove,random_flip,gflipm,gdelm
    ptb_rate = 0.4
    k =3  #生成所需要的k

    # if dataset.startswith('fb'):
    #     graphx = load_FB(root,dataset,attack,k,ptb_rate=ptb_rate)
    # else: #将攻击图转换成txt文件的格式存入文件
    #     graphx = change_graph_type(root, dataset, attack,ptb_rate=ptb_rate,k=k)


    #ctc-cora_stb，现在用的是5
    # change_query_ctc(root,dataset,attack,ptb_rate,num=3,test_size=500,qnum=5,max_queries=500)
    # change_query_ctc(root,'cora_stb','none',0.4,num=3,test_size=500,qnum=5,max_queries=500)
    # change_query_ctc(root,'cora_stb','random_add',0.4,num=3,test_size=500,qnum=5,max_queries=500)
    # change_query_ctc(root,'cora_stb','flipm',0.4,num=3,test_size=500,qnum=5,max_queries=500)
    # change_query_ctc(root,'cora_stb','cdelm',0.4,num=3,test_size=500,qnum=5,max_queries=500)

    #ctc-cora，现在用的是5
    # change_query_ctc(root,dataset,attack,ptb_rate,num=3,test_size=500,qnum=5,max_queries=500)
    # change_query_ctc(root,'cora','none',0.4,num=3,test_size=500,qnum=5,max_queries=500)
    # change_query_ctc(root,'cora','random_add',0.4,num=3,test_size=500,qnum=5,max_queries=500)
    # change_query_ctc(root,'cora','flipm',0.8,num=3,test_size=500,qnum=5,max_queries=500)
    # change_query_ctc(root,'cora','cdelm',0.8,num=3,test_size=500,qnum=5,max_queries=500)
    # change_query_ctc(root,'cora','meta',0.8,num=3,test_size=500,qnum=5,max_queries=500)

    # #ctc-citeseer，现在用的是5
    # change_query_ctc(root,dataset,attack,ptb_rate,num=3,test_size=500,qnum=5,max_queries=500)
    change_query_ctc(root,'citeseer','none',0.4,num=3,test_size=500,qnum=3,max_queries=500)
    change_query_ctc(root,'citeseer','random_add',0.4,num=3,test_size=500,qnum=3,max_queries=500)
    change_query_ctc(root,'citeseer','flipm',0.4,num=3,test_size=500,qnum=3,max_queries=500)
    change_query_ctc(root,'citeseer','cdelm',0.4,num=3,test_size=500,qnum=3,max_queries=500)
    change_query_ctc(root,'citeseer','meta',0.4,num=3,test_size=500,qnum=3,max_queries=500)


    #ctc-amazon，现在用的是5
    # change_query_ctc(root,dataset,attack,ptb_rate,num=3,test_size=500,qnum=5,max_queries=500)
    # change_query_ctc(root,'amazon','none',0.4,num=3,test_size=500,qnum=3,max_queries=500)
    # change_query_ctc(root,'amazon','random_add',0.4,num=3,test_size=500,qnum=3,max_queries=500)
    # change_query_ctc(root,'amazon','flipm',0.4,num=3,test_size=500,qnum=3,max_queries=500)
    # change_query_ctc(root,'amazon','cdelm',0.4,num=3,test_size=500,qnum=3,max_queries=500)
    # change_query_ctc(root,'amazon','meta',0.4,num=3,test_size=500,qnum=3,max_queries=500)


    # change_query_ctc(root,'dblp','none',0.4,num=3,test_size=500,qnum=3,max_queries=500)
    # change_query_ctc(root,'dblp','random_add',0.4,num=3,test_size=500,qnum=3,max_queries=500)
    # change_query_ctc(root,'dblp','flipm',0.4,num=3,test_size=500,qnum=3,max_queries=500)
    # change_query_ctc(root,'dblp','cdelm',0.4,num=3,test_size=500,qnum=3,max_queries=500)
    # change_query_ctc(root,'dblp','meta',0.4,num=3,test_size=500,qnum=3,max_queries=500)


    #ctc-facebook，现在用的是5
    # change_query_ctc(root,'facebook','none',0,num=3,test_size=500,qnum=5,max_queries=200) #没噪声是可以运行的
    # change_query_ctc(root,'facebook','random_add',0.2,num=3,test_size=500,qnum=5,max_queries=200)
    # change_query_ctc(root,'facebook','flipm',0.4,num=3,test_size=500,qnum=5,max_queries=200)
    # change_query_ctc(root,'facebook','cdelm',0.8,num=3,test_size=500,qnum=5,max_queries=200)
    # change_query_ctc(root,'facebook','meta',0.2,num=3,test_size=500,qnum=5,max_queries=200)

    #ctc-cora_gsr
    # change_query_ctc(root,'cora_gsr','none',0.4,num=3,test_size=500,qnum=5,max_queries=500)
    # change_query_ctc(root,'cora_gsr','random_add',0.8,num=3,test_size=500,qnum=5,max_queries=500)
    # change_query_ctc(root,'cora_gsr','flipm',0.8,num=3,test_size=500,qnum=5,max_queries=500)
    # change_query_ctc(root,'cora_gsr','cdelm',0.8,num=3,test_size=500,qnum=5,max_queries=500)



    #fb107_stb
    # change_query_ctc(root,'fb107_stb','none',0.4,num=3,test_size=500,qnum=1,max_queries=500)
    # change_query_ctc(root,'fb107_stb','flipm',0.4,num=3,test_size=500,qnum=1,max_queries=500)
    # change_query_ctc(root,'fb107_stb','cdelm',0.4,num=3,test_size=500,qnum=1,max_queries=500)

    #ctc-fb107_gsr
    # change_query_ctc(root,'fb107_gsr','none',0.4,num=3,test_size=500,qnum=1,max_queries=500)
    # change_query_ctc(root, 'fb107_gsr', 'random_add', 0.8, num=3, test_size=500, qnum=1, max_queries=500)
    # change_query_ctc(root,'fb107_gsr','flipm',0.8,num=3,test_size=500,qnum=1,max_queries=500)
    # change_query_ctc(root,'fb107_gsr','cdelm',0.8,num=3,test_size=500,qnum=1,max_queries=500)

    '''   k-clique'''

    #k-clique-cora数据集
    # change_query(root,'cora','none','0',num=3,test_size=500)
    # change_query(root,'cora','random_add','0.2',num=3,test_size=500)
    # change_query(root,'cora','random_add','0.4',num=3,test_size=500)
    # change_query(root,'cora','random_add','0.6',num=3,test_size=500)
    # change_query(root,'cora','random_add','0.8',num=3,test_size=500)
    # change_query(root,'cora','flipm','0.2',num=3,test_size=500)
    # change_query(root,'cora','flipm','0.4',num=3,test_size=500)
    # change_query(root,'cora','flipm','0.6',num=3,test_size=500)
    # change_query(root,'cora','flipm','0.8',num=3,test_size=500)
    # change_query(root,'cora','cdelm','0.2',num=3,test_size=500)
    # change_query(root,'cora','cdelm','0.4',num=3,test_size=500)
    # change_query(root,'cora','cdelm','0.6',num=3,test_size=500)
    # change_query(root,'cora','cdelm','0.8',num=3,test_size=500)
    # change_query(root, 'cora', 'meta', '0.2', num=3, test_size=500)
    # change_query(root, 'cora', 'meta', '0.4', num=3, test_size=500)
    # change_query(root, 'cora', 'meta', '0.6', num=3, test_size=500)
    # change_query(root, 'cora', 'meta', '0.8', num=3, test_size=500)

    # #k-clique-facebook数据集,仿佛死机了
    # change_query(root,'facebook','none','0.2',num=3,test_size=500)
    # change_query(root,'facebook','random_add','0.2',num=3,test_size=500)
    # change_query(root,'facebook','random_add','0.4',num=3,test_size=500)
    # change_query(root,'facebook','random_add','0.6',num=3,test_size=500)
    # change_query(root,'facebook','random_add','0.8',num=3,test_size=500)
    # change_query(root,'facebook','flipm','0.2',num=3,test_size=500)
    # change_query(root,'facebook','flipm','0.4',num=3,test_size=500)
    # change_query(root,'facebook','flipm','0.6',num=3,test_size=500)
    # change_query(root,'facebook','flipm','0.8',num=3,test_size=500)
    # change_query(root,'facebook','cdelm','0.2',num=3,test_size=500)
    # change_query(root,'facebook','cdelm','0.4',num=3,test_size=500)
    # change_query(root,'facebook','cdelm','0.6',num=3,test_size=500)
    # change_query(root,'facebook','cdelm','0.8',num=3,test_size=500)
    # change_query(root, 'facebook', 'meta', '0.2', num=3, test_size=500)
    # change_query(root, 'facebook', 'meta', '0.4', num=3, test_size=500)
    # change_query(root, 'facebook', 'meta', '0.6', num=3, test_size=500)
    # change_query(root, 'facebook', 'meta', '0.8', num=3, test_size=500)

    #kclique-fb107
    # change_query(root,'fb107','none','0.2',num=3,test_size=500)
    # change_query(root,'fb107','random_add','0.2',num=3,test_size=500)
    # change_query(root,'fb107','random_add','0.4',num=3,test_size=500)
    # change_query(root,'fb107','random_add','0.6',num=3,test_size=500)
    # change_query(root,'fb107','random_add','0.8',num=3,test_size=500)
    # change_query(root,'fb107','flipm','0.2',num=3,test_size=500)
    # change_query(root,'fb107','flipm','0.4',num=3,test_size=500)
    # change_query(root,'fb107','flipm','0.6',num=3,test_size=500)
    # change_query(root,'fb107','flipm','0.8',num=3,test_size=500)
    # change_query(root,'fb107','cdelm','0.2',num=3,test_size=500)
    # change_query(root,'fb107','cdelm','0.4',num=3,test_size=500)
    # change_query(root,'fb107','cdelm','0.6',num=3,test_size=500)
    # change_query(root,'fb107','cdelm','0.8',num=3,test_size=500)


    #ctc-photo，现在用的是5
    # change_query_ctc(root,dataset,attack,ptb_rate,num=3,test_size=500,qnum=5,max_queries=500)
    # change_query_ctc(root,'photo','none',0.4,num=3,test_size=500,qnum=5,max_queries=500)
    # change_query_ctc(root,'photo','random_add',0.4,num=3,test_size=500,qnum=5,max_queries=500)
    # change_query_ctc(root,'photo','flipm',0.4,num=3,test_size=500,qnum=5,max_queries=500)
    # change_query_ctc(root,'photo','cdelm',0.4,num=3,test_size=500,qnum=5,max_queries=500)
    # change_query_ctc(root,'photo','meta',0.4,num=3,test_size=500,qnum=5,max_queries=500)
    # change_query_photo(root,'photo','random_add',0.4,num=3,test_size=500,qnum=5,max_queries=500)

    #ctc-photo_stb，现在用的是5
    # change_query_ctc(root,dataset,attack,ptb_rate,num=3,test_size=500,qnum=5,max_queries=500)
    # change_query_ctc(root,'photo','none',0.4,num=3,test_size=500,qnum=5,max_queries=500)
    # change_query_ctc(root,'photo','random_add',0.4,num=3,test_size=500,qnum=5,max_queries=500)
    # change_query_ctc(root,'photo','flipm',0.4,num=3,test_size=500,qnum=5,max_queries=500)
    # change_query_ctc(root,'photo','cdelm',0.4,num=3,test_size=500,qnum=5,max_queries=500)
    # change_query_ctc(root,'photo','meta',0.4,num=3,test_size=500,qnum=5,max_queries=500)




    # # #truss和core需要的格式
    # change_query_k(root,'cora','none',0,num=3,test_size=500,k=3)
    knum=3

    #fb107
    # change_query_k(root, 'fb107', 'none', '0.2', num=3, test_size=500,k=knum)
    # change_query_k(root, 'fb107', 'random_add', '0.2', num=3, test_size=500,k=knum)
    # change_query_k(root, 'fb107', 'random_add', '0.4', num=3, test_size=500,k=knum)
    # change_query_k(root, 'fb107', 'random_add', '0.6', num=3, test_size=500,k=knum)
    # change_query_k(root, 'fb107', 'random_add', '0.8', num=3, test_size=500,k=knum)
    # change_query_k(root, 'fb107', 'flipm', '0.2', num=3, test_size=500,k=knum)
    # change_query_k(root, 'fb107', 'flipm', '0.4', num=3, test_size=500,k=knum)
    # change_query_k(root, 'fb107', 'flipm', '0.6', num=3, test_size=500,k=knum)
    # change_query_k(root, 'fb107', 'flipm', '0.8', num=3, test_size=500,k=knum)
    # change_query_k(root, 'fb107', 'cdelm', '0.2', num=3, test_size=500,k=knum)
    # change_query_k(root, 'fb107', 'cdelm', '0.4', num=3, test_size=500,k=knum)
    # change_query_k(root, 'fb107', 'cdelm', '0.6', num=3, test_size=500,k=knum)
    # change_query_k(root, 'fb107', 'cdelm', '0.8', num=3, test_size=500,k=knum)
    #facebook
    # change_query_k(root, 'facebook', 'none', '0', num=3, test_size=500,k=25)

    # change_query_k(root, 'facebook', 'random_add', '0.2', num=3, test_size=500,k=knum)
    # change_query_k(root, 'facebook', 'random_add', '0.4', num=3, test_size=500,k=knum)
    # change_query_k(root, 'facebook', 'random_add', '0.6', num=3, test_size=500,k=knum)
    # change_query_k(root, 'facebook', 'random_add', '0.8', num=3, test_size=500,k=knum)
    # change_query_k(root, 'facebook', 'flipm', '0.2', num=3, test_size=500,k=knum)
    # change_query_k(root, 'facebook', 'flipm', '0.4', num=3, test_size=500,k=knum)
    # change_query_k(root, 'facebook', 'flipm', '0.6', num=3, test_size=500,k=knum)
    # change_query_k(root, 'facebook', 'flipm', '0.8', num=3, test_size=500,k=knum)
    # change_query_k(root, 'facebook', 'cdelm', '0.2', num=3, test_size=500,k=knum)
    # change_query_k(root, 'facebook', 'cdelm', '0.4', num=3, test_size=500,k=knum)
    # change_query_k(root, 'facebook', 'cdelm', '0.6', num=3, test_size=500,k=knum)
    # change_query_k(root, 'facebook', 'cdelm', '0.8', num=3, test_size=500,k=knum)

    #cora
    knum = 3
    # change_query_k(root, 'cora', 'none', '0', num=3, test_size=500,k=knum)
    # change_query_k(root, 'cora', 'none', '0', num=3, test_size=500,k=1)

    # change_query_k(root, 'cora', 'random_add', '0.2', num=3, test_size=500,k=knum)
    # change_query_k(root, 'cora', 'random_add', '0.4', num=3, test_size=500,k=knum)
    # change_query_k(root, 'cora', 'random_add', '0.6', num=3, test_size=500,k=knum)
    # change_query_k(root, 'cora', 'random_add', '0.8', num=3, test_size=500,k=knum)
    # change_query_k(root, 'cora', 'flipm', '0.2', num=3, test_size=500,k=knum)
    # change_query_k(root, 'cora', 'flipm', '0.4', num=3, test_size=500,k=knum)
    # change_query_k(root, 'cora', 'flipm', '0.6', num=3, test_size=500,k=knum)
    # change_query_k(root, 'cora', 'flipm', '0.8', num=3, test_size=500,k=knum)
    # change_query_k(root, 'cora', 'cdelm', '0.2', num=3, test_size=500,k=knum)
    # change_query_k(root, 'cora', 'cdelm', '0.4', num=3, test_size=500,k=knum)
    # change_query_k(root, 'cora', 'cdelm', '0.6', num=3, test_size=500,k=knum)
    # change_query_k(root, 'cora', 'cdelm', '0.8', num=3, test_size=500,k=knum)

    # # #k-ecc
    # change_k_query(root,dataset,num=3,test_size=500,k=k,graphx=graphx)