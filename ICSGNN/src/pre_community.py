import networkx as nx
import numpy as np
import random
import os, sys
import gzip
import pathlib
import  tarfile
import scipy.sparse as sp
from citation_loader import citation_graph_reader
import json
p = os.path.dirname(os.path.dirname((os.path.abspath('__file__'))))
if p not in sys.path:
    sys.path.append(p)
import os.path as osp
def load_graph(args):
    if args.attack == 'none':
        if args.dataset in ['cora', 'citeseer', 'pubmed']:
            graph = citation_graph_reader(args.root, args.dataset)  # 读取图
    elif args.attack == 'random':
        path = os.path.join(args.root, args.dataset, args.attack,
                            f'{args.dataset}_{args.attack}_{args.type}_{args.ptb_rate}.npz')
        adj_csr_matrix = sp.load_npz(path)
        graph = nx.from_scipy_sparse_array(adj_csr_matrix)
    elif args.attack in ['del', 'gflipm', 'gdelm', 'add']:
        path = os.path.join(args.root, args.dataset, args.attack,
                            f'{args.dataset}_{args.attack}_{args.ptb_rate}.npz')
        adj_csr_matrix = sp.load_npz(path)
        graph = nx.from_scipy_sparse_array(adj_csr_matrix)
    return graph

def load_comms(root,dataset):
    file_path = os.path.join(root, dataset, f'{dataset}.comms')
    com_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        # 跳过第一行的标签列表
        lines = f.readlines()
        for line in lines[1:]:
            nodes = list(map(int, line.strip().split()))
            com_list.append(nodes)
    return com_list


def save_data_json(seed_list, train_node, labels,gtcomms, file_path):
    """
    以 JSON 格式存储数据（适用于更通用的格式）
    """
    data = {
        "seed_list": seed_list,
        "train_node": train_node,
        "labels": labels,
        "gtcomms":gtcomms
    }

    with open(file_path, "w") as f:
        json.dump(data, f)

    print(f"数据已成功保存到 {file_path}")

def load_data_json(file_path):
    """
    读取 JSON 格式的 seed_list, train_node, labels
    """
    with open(file_path, "r") as f:
        data = json.load(f)
    print(f"数据已成功从 {file_path} 读取")
    return data["seed_list"], data["train_node"], data["labels"],data["gtcomms"]

def my_pre_com(args,subgraph_list=[400], train_ratio=0.02,seed_cnt=20,cmty_size=30):
    '''
    :param data_set:
    :param subgraph_list:子图大小
    :param train_ratio:训练数据比例
    :param seed_cnt: 种子数量
    :param cmty_size: 社区大小
    '''

    #加载图数据
    graph = load_graph(args)

    #加载gt社区数据
    com_list = load_comms(args.root,args.dataset) #com_list[i]是索引为i的社区（节点列表）
    # 创建一个包含社区索引和社区大小的列表，并按照社区大小降序排序
    com_len=[(i,len(line)) for i,line in enumerate(com_list)]
    com_len.sort(key=lambda x:x[1],reverse=True)

    #遍历每一个子图大小，计算每个子图中用于训练的标签数量。
    for subgraph_size in subgraph_list:
        #所需标签数目：子图大小*比例/2
        numlabel = int(subgraph_size * train_ratio / 2)
        # 筛选出足够大的社区，以便有足够的节点用于训练和构建社区。
        ok_com_len=[(i,lens) for i,lens in com_len if lens>=(numlabel+cmty_size) ]

        # 初始化种子列表、训练节点列表、标签列表、错误种子列表和时间计数器。
        seed_list=[] #挑选的种子节点（查询节点）
        train_node=[] #存储用于训练的节点列表（包含正例和负例）
        labels=[] #存储种子所在的社区(在子图上的）
        gtcomms=[] #种子节点的gt社区
        error_seed=[] #存储无法满足训练需求的种子节点
        time=0
        while len(seed_list)<seed_cnt: #循环直到收集足够的种子节点。
            time+=1
            # 随机选择一个符合条件的社区作为种子社区
            seed_com_index=random.randint(0,len(ok_com_len)-1)
            seed_com=com_list[ok_com_len[seed_com_index][0]] #种子社区的节点列表
            #复制并随机打乱种子社区中的节点，选择一个未被使用的种子节点。
            seed_com_suff=seed_com[:]
            random.shuffle(seed_com_suff)
            seed_index=0
            seed=seed_com_suff[seed_index]
            while (seed in seed_list or seed in error_seed ) and (seed_index+1)<len(seed_com_suff):
                seed_index+=1
                seed=seed_com_suff[seed_index]
            # 如果种子节点已被使用，跳过当前循环。
            if(seed in seed_list or seed in error_seed ):
                continue

            # 从选定的种子节点开始，通过遍历其邻居节点来扩展子图，直到达到指定的子图大小subgraph_size。
            allNodes=[]
            allNodes.append(seed)
            pos = 0
            while pos < len(allNodes) and pos < subgraph_size and len(allNodes) < subgraph_size:
                cnode = allNodes[pos]
                for nb in graph.neighbors(cnode):
                    if nb not in allNodes and len(allNodes) < subgraph_size:
                        allNodes.append(nb)
                pos += 1

            #正例节点：计算种子社区与生成的子图的交集，检查是否满足预期的社区大小和训练标签数量。
            posNodes = []
            posNodes.append(seed)
            seed_com_intersection=list(set(seed_com).intersection(set(allNodes))) #子图中与seed在同一社区的节点集合。
            if(len(seed_com_intersection)< numlabel+cmty_size):
                error_seed.append(seed)
                continue

            # 从交集中除去种子节点，随机排序，并选择足够的正例节点。
            seed_com_intersection_noseed=seed_com_intersection[:]
            seed_com_intersection_noseed.remove(seed)
            random.shuffle(seed_com_intersection_noseed)
            posNodes.extend(seed_com_intersection_noseed[:numlabel-1])

            #负例节点：计算子图中不属于种子社区的节点作为负例，检查是否有足够的负例节点。
            negNodes=list(set(allNodes).difference(set(seed_com)))
            if(len(negNodes)< numlabel):
                error_seed.append(seed)
                continue
            random.shuffle(negNodes)
            negNodes=negNodes[:numlabel]

            # 将选定的种子节点、训练节点（正例和负例的组合）、标签添加到各自的列表中。
            seed_list.append(seed)
            train_node.append(posNodes+negNodes)
            labels.append(seed_com_intersection)
            gtcomms.append(seed_com)
        #打印错误种子的数量和最终的种子列表。
        print('error num:',len(error_seed),"seed_list:",seed_list)
        print('type of labels:',type(labels))

    save_path =f'{args.root}/{args.dataset}/ics/{args.dataset}_{args.attack}_{args.ptb_rate}_data.json'
    save_data_json(seed_list,train_node,labels,gtcomms,save_path)
    # 返回边信息、种子列表(查询节点）、训练节点和标签，这些数据可用于进一步的图学习任务。
    return seed_list,train_node,labels


def pre_com(data_set='com_dblp',subgraph_list=[400], train_ratio=0.02,seed_cnt=20,cmty_size=30):
    '''
    :param data_set:
    :param subgraph_list:子图大小
    :param train_ratio:训练数据比例
    :param seed_cnt: 种子数量
    :param cmty_size: 社区大小
    '''

    #加载图的边数据
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..','data',data_set)
    print(f"Load {data_set} edges")
    if(os.path.exists(path + '//edge.npy') == False):
        untar_snap_data(data_set[4:])
    new_edge=np.load(path+'//edges.npy').tolist()
    graph = nx.from_edgelist(new_edge)

    #加载gt社区信息并转换为列表
    print(f"Load {data_set} cmty")
    com_list=np.load(path+'//comms.npy',allow_pickle=True).tolist()

    # 创建一个包含社区索引和社区大小的列表，并按照社区大小降序排序
    com_len=[(i,len(line)) for i,line in enumerate(com_list)]
    com_len.sort(key=lambda x:x[1],reverse=True)

    #遍历每一个子图大小，计算每个子图中用于训练的标签数量。
    for subgraph_size in subgraph_list:
        numlabel = int(subgraph_size * train_ratio / 2) #标签数量，标签（正负）为子图大小乘以训练比例/2
        # 筛选出足够大的社区，以便有足够的节点用于训练和构建社区。
        ok_com_len=[(i,lens) for i,lens in com_len if lens>=(numlabel+cmty_size) ]

        # 初始化种子列表、训练节点列表、标签列表、错误种子列表和时间计数器。
        seed_list=[]
        train_node=[]
        labels=[]
        error_seed=[]
        time=0
        while len(seed_list)<seed_cnt: #循环直到收集足够的种子节点。
            time+=1
            seed_com_index=random.randint(0,len(ok_com_len)-1)
            seed_com=com_list[ok_com_len[seed_com_index][0]] #从符合条件的社区中随机选择一个社区作为种子社区。
            #复制并随机打乱种子社区中的节点，选择一个未被使用的种子节点。
            seed_com_suff=seed_com[:]
            random.shuffle(seed_com_suff)
            seed_index=0
            seed=seed_com_suff[seed_index]
            while (seed in seed_list or seed in error_seed ) and (seed_index+1)<len(seed_com_suff):
                seed_index+=1
                seed=seed_com_suff[seed_index]
            # 如果种子节点已被使用，跳过当前循环。
            if(seed in seed_list or seed in error_seed ): 
                continue

            # 从选定的种子节点开始，通过遍历其邻居节点来扩展子图，直到达到指定的子图大小。
            allNodes=[] 
            allNodes.append(seed)
            pos = 0
            while pos < len(allNodes) and pos < subgraph_size and len(allNodes) < subgraph_size:
                cnode = allNodes[pos]
                for nb in graph.neighbors(cnode):
                    if nb not in allNodes and len(allNodes) < subgraph_size:
                        allNodes.append(nb)
                pos += 1

            # 计算种子社区与生成的子图的交集，检查是否满足预期的社区大小和训练标签数量。
            posNodes = []
            posNodes.append(seed)
            seed_com_intersection=list(set(seed_com).intersection(set(allNodes)))
            if(len(seed_com_intersection)< numlabel+cmty_size):
                error_seed.append(seed)
                continue

            # 从交集中除去种子节点，随机排序，并选择足够的正例节点。
            seed_com_intersection_noseed=seed_com_intersection[:]
            seed_com_intersection_noseed.remove(seed)
            random.shuffle(seed_com_intersection_noseed)
            posNodes.extend(seed_com_intersection_noseed[:numlabel-1])

            #计算子图中不属于种子社区的节点作为负例，检查是否有足够的负例节点。
            negNodes=list(set(allNodes).difference(set(seed_com)))
            if(len(negNodes)< numlabel):
                error_seed.append(seed)
                continue
            random.shuffle(negNodes)
            negNodes=negNodes[:numlabel]

            # 将选定的种子节点、训练节点（正例和负例的组合）、标签添加到各自的列表中。
            seed_list.append(seed)
            train_node.append(posNodes+negNodes)
            labels.append(seed_com_intersection)
        #打印错误种子的数量和最终的种子列表。
        print('error:',len(error_seed),"seed_list:",seed_list)

    # 返回边信息、种子列表、训练节点和标签，这些数据可用于进一步的图学习任务。
    return new_edge,seed_list,train_node,labels

def load_facebook(seed):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data','facebook')
    print('Load facebook data')
    if(os.path.exists(path + f'//{str(seed)}.circles') == False):
        untart_facebook()
    file_circle= path + f'//{str(seed)}.circles'
    file_edges=path + f'//{str(seed)}.edges'
    file_egofeat=path + f'//{str(seed)}.egofeat'
    file_feat=path + f'//{str(seed)}.feat'
    edges=[]
    node=[]
    feature = {}
    with open(file_egofeat) as f:
        feature[seed] = [int(i) for i in f.readline().split()]
    with open(file_feat) as f:
        for line in f:
            line = [int(i) for i in line.split()]
            feature[int(line[0])] = line[1:]
            node.append(int(line[0]))
    with open(file_edges,'r') as f:
        for line in f:
            u,v=line.split()
            u=int(u)
            v=int(v)
            if(u in feature.keys() and v in feature.keys()):
                edges.append((u,v))

    for i in node:
        edges.append((seed, i))
    node=sorted(node+[seed])
    mapper = {n: i for i, n in enumerate(node)}
    edges=[(mapper[u],mapper[v]) for u,v in edges]
    node=[mapper[u] for u in node]

    features=[0]*len(node)
    for i in list(feature.keys()):
        features[mapper[i]]=feature[i]
    circle=[]
    with open(file_circle) as f:
        for line in f:
            line=line.split()
            line=[ mapper[int(i)] for i  in line[1:]]
            if(len(line)<8):continue
            circle.append(line)

    seed=mapper[seed]

    return edges,features,circle,seed

#读取snap网站提供的有gt社区的数据集。
def load_snap(data_set,com_size):
    print(f'Load {data_set} edge')
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', data_set)
    if(os.path.exists(path + '//edge.npy') == False):
        untar_snap_data(data_set[4:]) #将数据集解压缩。
    edges=np.load(path + '//edge.npy').tolist() #加载图数据
    print(f'Load {data_set} cmty')
    com_list = np.load(path + '//comms.npy', allow_pickle=True).tolist() #加载comm数据。
    com_list=[i for i in com_list if len(i)>=com_size]
    return edges,com_list

def untar_snap_data(name):
    """Load the snap comm datasets."""
    print(f'Untar {name} edge')
    root = pathlib.Path('raw')
    #print(root)  #震惊！直接读取的是zip数据集。
    with gzip.open(root / f'com-{name}.ungraph.txt.gz', 'rt') as fh:
        edges = fh.read().strip().split('\n')[4:]
    edges = [[int(i) for i in e.split()] for e in edges]
    nodes = {i for x in edges for i in x}
    mapping = {u: i for i, u in enumerate(sorted(nodes))}
    edges = [[mapping[u], mapping[v]] for u, v in edges]
    print(f'Untar {name} cmty')
    with gzip.open(root / f'com-{name}.top5000.cmty.txt.gz', 'rt') as fh:
        comms = fh.readlines()
    comms = [[mapping[int(i)] for i in x.split()] for x in comms]
    root = pathlib.Path()/'data'/f'com_{name}'
    root.mkdir(exist_ok=True, parents=True)
    np.save(root/'edges',edges)
    np.save(root/'comms',comms,allow_pickle=True)
    np.save(root/'map',mapping,allow_pickle=True)

def untart_facebook():
    print(f'Untar  facebook')
    tar = tarfile.open(osp.join(osp.dirname(osp.realpath(__file__)), '..', 'raw','facebook.tar.gz'))
    names = tar.getnames()
    path =osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data')
    for name in names:
        tar.extract(name,path)
    tar.close()