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

from load_utils import load_graph

p = os.path.dirname(os.path.dirname((os.path.abspath('__file__'))))
if p not in sys.path:
    sys.path.append(p)
import os.path as osp
# def load_graph(args):
#     if args.attack == 'none':
#         if args.dataset in ['cora', 'citeseer', 'pubmed']:
#             graph = citation_graph_reader(args.root, args.dataset)  # 读取图
#     elif args.attack == 'random':
#         path = os.path.join(args.root, args.dataset, args.attack,
#                             f'{args.dataset}_{args.attack}_{args.type}_{args.ptb_rate}.npz')
#         adj_csr_matrix = sp.load_npz(path)
#         graph = nx.from_scipy_sparse_array(adj_csr_matrix)
#     elif args.attack in ['del', 'gflipm', 'gdelm', 'add']:
#         path = os.path.join(args.root, args.dataset, args.attack,
#                             f'{args.dataset}_{args.attack}_{args.ptb_rate}.npz')
#         adj_csr_matrix = sp.load_npz(path)
#         graph = nx.from_scipy_sparse_array(adj_csr_matrix)
#     return graph

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
def find_node_community(com_list, query_node):
    '''
    从读取的comlist中可以查询任意一个节点所在的社区
    :param com_list:
    :param query_node:
    :return:
    '''
    for community in com_list:
        if query_node in community:
            return community
    return None  # 如果节点不在任何社区中


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

def load_seed_queries(file_path):
    """
    从cocle的测试/验证文件中读取 (q, comm)。
    文件格式形如：  q, v1 v2 v3 ...
    返回：
        seeds  : [q1, q2, ...]
        gtcoms : [[v1,v2,...], [u1,u2,...], ...]
    """
    seeds, gtcoms = [], []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) != 2:
                continue   # 跳过格式异常的行
            q = int(parts[0])
            comm = list(map(int, parts[1].split()))
            seeds.append(q)
            gtcoms.append(comm)
    return seeds, gtcoms

def build_subgraph_with_positives(graph, seed, seed_com,
                                  cmty_size, subgraph_size):
    seed_com = set(seed_com)
    pos_nodes = set([seed] +
                    random.sample(list(seed_com - {seed}),
                                  min(len(seed_com)-1, cmty_size)))
    allNodes  = list(pos_nodes)
    queue     = list(pos_nodes)
    while queue and len(allNodes) < subgraph_size:
        u = queue.pop(0)
        for v in graph.neighbors(u):
            if v not in allNodes:
                allNodes.append(v)
                queue.append(v)
                if len(allNodes) == subgraph_size: break

    # ✅ 强制包含所有正例：变量名必须一致！
    allNodes = list(set(allNodes) | pos_nodes)

    return allNodes, list(pos_nodes)

def my_pre_com(args,subgraph_list=[400], train_ratio=0.02,seed_cnt=20,cmty_size=30,test_query_file=None,old2new=None,graph=None):
    '''
    :param data_set:
    :param subgraph_list:子图大小
    :param train_ratio:训练数据比例
    :param seed_cnt: 种子数量  原文默认是20，我将其修改为500
    :param cmty_size: 社区大小
    '''
    """
        若 test_query_file 不为 None，则依次用文件中的 q 作为种子；
        否则按原逻辑随机挑选。
    """
    # #加载图数据。加载原始图（旧的编号）与社区
    if graph is None:#这没重新编号啊
        graph,n_nodes = load_graph(args.root,args.dataset,args.attack,args.ptb_rate)
    if old2new is None:
        valid_old_ids = sorted(graph.nodes())
        old2new = {old: new for new, old in enumerate(valid_old_ids)}
    #加载gt社区数据
    com_list = load_comms(args.root,args.dataset) #com_list[i]是索引为i的社区（节点列表）
    print('in my_pre_com，加载com_list成功')
    # —— 过滤掉不在图里的节点 —— #
    # com_list = [[v for v in comm if v in graph_nodes] for comm in com_list]
    # 社区列表同步映射编号
    com_list = [
        [old2new[v] for v in comm if v in old2new]  # 映射 + 过滤
        for comm in com_list
    ]
    com_list = [comm for comm in com_list if comm]  # 删空社区
    graph_nodes = set(graph.nodes())

    # ---------- 如果提供了测试集文件，预取 seeds ----------
    if test_query_file: #过滤掉不在图中的节点
        file_seeds, file_gt_comms = load_seed_queries(test_query_file)
        # 只取前 seed_cnt 个（如果想都用完就删掉 [:seed_cnt]）
        filtered_pairs = []
        for q,comm in zip(file_seeds,file_gt_comms):
            if q in old2new:
                q_new = old2new[q]
                comm_new = [old2new[v] for v in comm if v in old2new]
                if comm_new:
                    filtered_pairs.append((q_new, comm_new))
                # comm = [v for v in comm if v in graph_nodes]
                # if comm:
                #     filtered_pairs.append((q, comm))
        fixed_seed_pairs = filtered_pairs[:seed_cnt]
    else:
        fixed_seed_pairs = None  # 触发原随机逻辑
    #遍历每一个子图大小，计算每个子图中用于训练的标签数量。
    for subgraph_size in subgraph_list:
        #所需标签数目：子图大小*比例/2,这里算出来是4
        #修改成固定为3个pos，3geneg
        # numlabel = int(subgraph_size * train_ratio / 2)
        numlabel = 3

        # 初始化种子列表、训练节点列表、标签列表、错误种子列表和时间计数器。
        seed_list=[] #挑选的种子节点（查询节点）
        train_node=[] #存储用于训练的节点列表（包含正例和负例）
        labels=[] #存储种子所在的社区(在子图上的）
        gtcomms=[] #种子节点的gt社区
        error_seed=[] #存储无法满足训练需求的种子节点
        time=0
        print("fixed_seed_pairs 总数:", len(fixed_seed_pairs))
        if fixed_seed_pairs:
            candidates = fixed_seed_pairs
            print("示例:", fixed_seed_pairs[0])
        else: # 创建一个包含社区索引和社区大小的列表，并按照社区大小降序排序
            com_len = [(i, len(line)) for i, line in enumerate(com_list)]
            com_len.sort(key=lambda x: x[1], reverse=True)
            # 筛选出足够大的社区，以便有足够的节点用于训练和构建社区。
            ok_com_len = [(i, lens) for i, lens in com_len if lens >= (numlabel + cmty_size)]
            print(f'ok_com_len:{ok_com_len}')
            candidates = ok_com_len  # 这里仅为占位，真正的随机逻辑仍在 while 中
        idx = 0
        debug = {"pos不足": 0, "neg不足": 0, "重复": 0, "通过": 0}
        while len(seed_list)<seed_cnt and idx < len(candidates): #循环直到收集足够的种子节点（其实就是查询节点）。
            if fixed_seed_pairs:  # —— 顺序取文件里的 —
                seed, gt_comm = candidates[idx]
                seed_com = gt_comm
            else: #保持和之前一致的随机逻辑挑选
                time+=1
                # 随机选择一个符合条件的社区索引作为种子社区
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
                    debug["重复"] += 1;
                    idx+=1 #使用下一个candidate
                    continue
                # 将选定的种子节点、训练节点（正例和负例的组合）、标签添加到各自的列表中。
                gt_comm = find_node_community(com_list, seed)  # 获取该种子节点所在的全部社区

            #构建子图
            # 从选定的种子节点开始，通过遍历其邻居节点来扩展子图，直到达到指定的子图大小subgraph_size。
            if args.dataset == 'amazon':
                allNodes,posNodes= build_subgraph_with_positives(graph,seed,seed_com,cmty_size,subgraph_size)
                # >>> 添加下面三行，实现“裁剪到 numlabel（含 seed）” <<<
                extra_pos = list(set(posNodes) - {seed})
                random.shuffle(extra_pos)
                posNodes = [seed] + extra_pos[:numlabel - 1]
                # ----------------- 补这一行 -----------------
                seed_com_intersection = list(set(seed_com) & set(allNodes))
                print(f'amazon的seed_com_intersection大小为：{len(seed_com_intersection)}')
                # 若你只保证了 numlabel 个正例，可加断言确保满足
                assert len(seed_com_intersection) >= numlabel, \
                    f"[Bug] 子图正例不足: {len(seed_com_intersection)}"
            else:
                allNodes = [seed]
                pos = 0
                while pos < len(allNodes) and pos < subgraph_size and len(allNodes) < subgraph_size:
                    cnode = allNodes[pos]
                    for nb in graph.neighbors(cnode):
                        if nb not in allNodes and len(allNodes) < subgraph_size:
                            allNodes.append(nb)
                    pos += 1
                # 正例节点：计算种子社区与生成的子图的交集，检查是否满足预期的社区大小和训练标签数量。
                posNodes = []
                posNodes.append(seed)
                seed_com_intersection = list(set(seed_com).intersection(set(allNodes)))  # 子图中与seed在同一社区的节点集合。
                if (len(seed_com_intersection) < numlabel + cmty_size):
                    debug["pos不足"] += 1;
                    error_seed.append(seed)
                    idx += 1
                    continue
                # 从交集中除去种子节点，随机排序，并选择足够的正例节点。
                seed_com_intersection_noseed = [x for x in seed_com_intersection if x != seed]
                random.shuffle(seed_com_intersection_noseed)
                posNodes.extend(seed_com_intersection_noseed[:numlabel - 1])
            print("[Debug] 正例节点列表:", posNodes)
            print("[Debug] 子图节点 allNodes 包含正例吗？", all(node in allNodes for node in posNodes))
            # 保证 posNodes 全都在 allNodes 中（保险）
            if not all(x in allNodes for x in posNodes):
                error_seed.append(seed)
                idx += 1
                continue
            #负例节点：计算子图中不属于种子社区的节点作为负例，检查是否有足够的负例节点。
            negNodes=list(set(allNodes).difference(set(seed_com)))
            if(len(negNodes)< numlabel):
                debug["neg不足"] += 1
                error_seed.append(seed)
                idx += 1
                continue
            random.shuffle(negNodes)
            negNodes=negNodes[:numlabel]

            # ---------- 收集 ----------
            debug["通过"] += 1
            seed_list.append(seed)
            print(f'train_node中，posNodes{len(posNodes)}个，negNodes{len(negNodes)}个')
            train_node.append(posNodes+negNodes)
            labels.append(seed_com_intersection)
            # gtcomms.append(seed_com) #这是原本的只存了gtcomm
            gtcomms.append(gt_comm) #现在我要的是全部
            idx += 1  # 用下一个 candidate
        #打印错误种子的数量和最终的种子列表。
        print('error num:',len(error_seed),"seed_list:",seed_list)
        print('type of labels:',type(labels))
        print(debug)
    save_path = f'{args.root}/{args.dataset}/ics/'
    os.makedirs(save_path, exist_ok=True)
    save_path =f'{args.root}/{args.dataset}/ics/{args.dataset}_{args.attack}_{args.ptb_rate}_data.json'
    save_data_json(seed_list,train_node,labels,gtcomms,save_path)
    # 返回边信息、种子列表(查询节点）、训练节点和标签，这些数据可用于进一步的图学习任务。
    return seed_list,train_node,labels,gtcomms

# def my_pre_com(args,subgraph_list=[400], train_ratio=0.02,seed_cnt=20,cmty_size=30,test_query_file=None):
#     '''
#     :param data_set:
#     :param subgraph_list:子图大小
#     :param train_ratio:训练数据比例
#     :param seed_cnt: 种子数量  原文默认是20，我将其修改为500
#     :param cmty_size: 社区大小
#     '''
#     """
#         若 test_query_file 不为 None，则依次用文件中的 q 作为种子；
#         否则按原逻辑随机挑选。
#     """
#     #加载图数据
#     graph,n_nodes = load_graph(args.root,args.dataset,args.attack,args.ptb_rate)
#
#     #加载gt社区数据
#     com_list = load_comms(args.root,args.dataset) #com_list[i]是索引为i的社区（节点列表）
#     print('in my_pre_com，加载com_list成功')
#     # ---------- 如果提供了测试集文件，预取 seeds ----------
#     if test_query_file:
#         file_seeds, file_gt_comms = load_seed_queries(test_query_file)
#         # 只取前 seed_cnt 个（如果想都用完就删掉 [:seed_cnt]）
#         fixed_seed_pairs = list(zip(file_seeds, file_gt_comms))[:seed_cnt]
#     else:
#         fixed_seed_pairs = None  # 触发原随机逻辑
#
#
#
#     # 创建一个包含社区索引和社区大小的列表，并按照社区大小降序排序
#     com_len=[(i,len(line)) for i,line in enumerate(com_list)]
#     com_len.sort(key=lambda x:x[1],reverse=True)
#
#     #遍历每一个子图大小，计算每个子图中用于训练的标签数量。
#     for subgraph_size in subgraph_list:
#         #所需标签数目：子图大小*比例/2,这里算出来是4
#         #修改成固定为3个pos，3geneg
#         # numlabel = int(subgraph_size * train_ratio / 2)
#         numlabel = 3
#         # 筛选出足够大的社区，以便有足够的节点用于训练和构建社区。
#         ok_com_len=[(i,lens) for i,lens in com_len if lens>=(numlabel+cmty_size) ]
#         print(f'ok_com_len:{ok_com_len}')
#         # 初始化种子列表、训练节点列表、标签列表、错误种子列表和时间计数器。
#         seed_list=[] #挑选的种子节点（查询节点）
#         train_node=[] #存储用于训练的节点列表（包含正例和负例）
#         labels=[] #存储种子所在的社区(在子图上的）
#         gtcomms=[] #种子节点的gt社区
#         error_seed=[] #存储无法满足训练需求的种子节点
#         time=0
#         while len(seed_list)<seed_cnt: #循环直到收集足够的种子节点（其实就是查询节点）。
#             time+=1
#             # 随机选择一个符合条件的社区索引作为种子社区
#             seed_com_index=random.randint(0,len(ok_com_len)-1)
#             seed_com=com_list[ok_com_len[seed_com_index][0]] #种子社区的节点列表
#             #复制并随机打乱种子社区中的节点，选择一个未被使用的种子节点。
#             seed_com_suff=seed_com[:]
#             random.shuffle(seed_com_suff)
#             seed_index=0
#             seed=seed_com_suff[seed_index]
#             while (seed in seed_list or seed in error_seed ) and (seed_index+1)<len(seed_com_suff):
#                 seed_index+=1
#                 seed=seed_com_suff[seed_index]
#             # 如果种子节点已被使用，跳过当前循环。
#             if(seed in seed_list or seed in error_seed ):
#                 continue
#
#             # 从选定的种子节点开始，通过遍历其邻居节点来扩展子图，直到达到指定的子图大小subgraph_size。
#             allNodes=[]
#             allNodes.append(seed)
#             pos = 0
#             while pos < len(allNodes) and pos < subgraph_size and len(allNodes) < subgraph_size:
#                 cnode = allNodes[pos]
#                 for nb in graph.neighbors(cnode):
#                     if nb not in allNodes and len(allNodes) < subgraph_size:
#                         allNodes.append(nb)
#                 pos += 1
#
#             #正例节点：计算种子社区与生成的子图的交集，检查是否满足预期的社区大小和训练标签数量。
#             posNodes = []
#             posNodes.append(seed)
#             seed_com_intersection=list(set(seed_com).intersection(set(allNodes))) #子图中与seed在同一社区的节点集合。
#             if(len(seed_com_intersection)< numlabel+cmty_size):
#                 error_seed.append(seed)
#                 continue
#
#             # 从交集中除去种子节点，随机排序，并选择足够的正例节点。
#             seed_com_intersection_noseed=seed_com_intersection[:]
#             seed_com_intersection_noseed.remove(seed)
#             random.shuffle(seed_com_intersection_noseed)
#             posNodes.extend(seed_com_intersection_noseed[:numlabel-1])
#
#             #负例节点：计算子图中不属于种子社区的节点作为负例，检查是否有足够的负例节点。
#             negNodes=list(set(allNodes).difference(set(seed_com)))
#             if(len(negNodes)< numlabel):
#                 error_seed.append(seed)
#                 continue
#             random.shuffle(negNodes)
#             negNodes=negNodes[:numlabel]
#
#             # 将选定的种子节点、训练节点（正例和负例的组合）、标签添加到各自的列表中。
#             gt_comm = find_node_community(com_list,seed) #获取该种子节点所在的全部社区
#             seed_list.append(seed)
#             train_node.append(posNodes+negNodes)
#             labels.append(seed_com_intersection)
#             # gtcomms.append(seed_com) #这是原本的只存了gtcomm
#             gtcomms.append(gt_comm) #现在我要的是全部
#         #打印错误种子的数量和最终的种子列表。
#         print('error num:',len(error_seed),"seed_list:",seed_list)
#         print('type of labels:',type(labels))
#
#     save_path =f'{args.root}/{args.dataset}/ics/{args.dataset}_{args.attack}_{args.ptb_rate}_data.json'
#     save_data_json(seed_list,train_node,labels,gtcomms,save_path)
#     # 返回边信息、种子列表(查询节点）、训练节点和标签，这些数据可用于进一步的图学习任务。
#     return seed_list,train_node,labels,gtcomms


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