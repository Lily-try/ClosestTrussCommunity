from parser import parameter_parser
from citation_loader import citation_feature_reader, citation_target_reader, citation_graph_reader
from load_utils import load_graph
from src.subgraph import SubGraph
from src import utils
import torch
from torch_geometric.datasets import Reddit
from src.pre_community import *
import scipy.sparse as sp
import datetime

#加载数据
def load_data(args):
    '''
    Load data
    '''
    seed_list = None
    train_nodes = None
    labels= None
    if args.dataset in ['cora','citeseer','pubmed']:  #有属性，没有gt社区。数据集默认是cora
        graph = utils.graph_reader1(args.dataset) #读取图
        features = utils.feature_reader1(args.dataset) #读取特征
        target = utils.target_reader1(args.dataset) #所有节点的标签
    if args.dataset in ["reddit"]:#有属性和真实社区的reddit  #从pyg中读取的？？？
        path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', args.dataset)
        data = Reddit(path)[0]
        edge_index, features, target = data.edge_index.numpy(), data.x.numpy(), data.y.numpy()
        graph = utils.construct_graph(edge_index)
        target = target[:, np.newaxis]
    if args.dataset in ["dblp", 'amazon']: #没有属性，有gt社区。
        if args.iteration:
            edge, labels = load_snap(dataset='com_' + args.dataset, com_size=args.community_size)
            graph = nx.from_edgelist(edge)
            n = graph.number_of_nodes()
            features = np.array([np.random.normal(0, 1, 100) for _ in range(n)])
            target = None
        else: #在干嘛？
            edge,seed_list,train_nodes,labels=pre_com(dataset='com_'+args.dataset,subgraph_list=[args.subgraph_size],train_ratio=args.train_ratio,cmty_size=args.community_size,seed_cnt=args.seed_cnt)
            graph = nx.from_edgelist(edge)
            n = graph.number_of_nodes()
            features=np.array([np.random.normal(0, 1, 100) for _ in range(n)])
            target=None

    return graph,features,target,seed_list,train_nodes,labels

# def load_citations(args):
#     '''
#     这个是没有重新编号后的读取方式。
#     '''
#     seed_list = None
#     train_nodes = None
#     labels = None
#     # 1. 加载图数据
#     graph, n_nodes = load_graph(args.root, args.dataset, args.attack, args.ptb_rate)
#     print(f'图中的节点{graph.number_of_nodes()}, 图中的边{graph.number_of_edges()}')
#     #2. 加载特征数据
#     if args.dataset in ['cora','citeseer','pubmed']:
#         features = citation_feature_reader(args.root,args.dataset)
#     elif args.dataset in ['cocs','photo']:
#         with open(f'{args.root}/{args.dataset}/{args.dataset}.feats', "r") as f:
#             # 每行特征转换为列表，然后堆叠为 ndarray
#             features = np.array([list(map(float, line.strip().split())) for line in f])
#             print(f'{args.dataset}的节点特征shape:{features.shape}')
#     elif args.dataset in ['fb107','wfb107']:  # 不加入中心节点
#         feats_array = np.loadtxt(f'{args.root}/{args.dataset}/{args.dataset}.feat', delimiter=' ', dtype=np.float32)
#         print(type(feats_array))
#         # nodes_feats = fnormalize(feats_array)  # 将特征进行归一化
#         features = torch.from_numpy(feats_array)
#         node_in_dim = features.shape[1]
#     elif args.dataset in ['cora_stb', 'cora_gsr']:
#         features = citation_feature_reader(args.root, args.dataset[:-4])  # numpy.ndaaray:(2708,1433)
#         features = torch.from_numpy(features)  # 转换成tensor
#         node_in_dim = features.shape[1]
#     elif args.dataset in ['fb107_gsr', 'fb107_stb']:
#         feats_array = np.loadtxt(f'{args.root}/{args.dataset[:-4]}/{args.dataset[:-4]}.feat', delimiter=' ',
#                                  dtype=np.float32)
#         print(type(feats_array))
#         # nodes_feats = fnormalize(feats_array)  # 将特征进行归一化
#         features = torch.from_numpy(feats_array)
#         node_in_dim = features.shape[1]
#     elif args.dataset in ['facebook']:  # 读取pyg中的特征数据
#         feats_array = np.loadtxt(f'{args.root}/{args.dataset}/{args.dataset}.feat', dtype=float, delimiter=' ')
#         features = torch.tensor(feats_array, dtype=torch.float32)
#         node_in_dim = features.shape[1]
#     print(f'节点特征的维度:{features.shape}')
#     #3.加载标签数据
#     if args.dataset in ['cora', 'citeseer', 'pubmed']:
#         target = citation_target_reader(args.root,args.dataset)  # 所有节点的标签
#         target = target.reshape(-1, 1)
#         print(f'标签的维度：{target.shape}')
#     elif args.dataset in ['cocs','fb107','photo','facebook']:  # 加载共同作者数据
#         with open(f'{args.root}/{args.dataset}/{args.dataset}.comms', 'r') as f:
#             lines = f.readlines()
#         lines = lines[1:]# 跳过第一行的社区编号（你可以用也可以不用，反正是从0开始按行编号）
#         # 统计最大节点ID以确定labels数组大小
#         max_node_id = -1
#         community_nodes = []
#         for line in lines:
#             nodes = list(map(int, line.strip().split()))
#             if nodes:
#                 community_nodes.append(nodes)
#                 max_node_id = max(max_node_id, max(nodes))
#         # labels = np.zeros(max_node_id + 1, dtype=int)  #初始化label数组，默认是0
#         labels = np.zeros(n_nodes, dtype=int)  #初始化label数组，默认是0
#         for comm_id, nodes in enumerate(community_nodes):
#             for node_id in nodes:
#                 labels[node_id] = comm_id
#         target =labels.reshape(-1,1)  #id为索引位置的节点对应的标签（社区）编号。
#         print(f'标签的维度：{labels.shape}, {target.shape}')
#     elif args.dataset in ['cocs_stb','fb107_stb','fb107_gsr','cora_gsr','cora_stb','cocs_gsr','photo_gsr','photo_stb']:  # 加载共同作者数据
#         with open(f'{args.root}/{args.dataset[:-4]}/{args.dataset[:-4]}.comms', 'r') as f:
#             lines = f.readlines()
#         lines = lines[1:]# 跳过第一行的社区编号（你可以用也可以不用，反正是从0开始按行编号）
#         # 统计最大节点ID以确定labels数组大小
#         max_node_id = -1
#         community_nodes = []
#         for line in lines:
#             nodes = list(map(int, line.strip().split()))
#             if nodes:
#                 community_nodes.append(nodes)
#                 max_node_id = max(max_node_id, max(nodes))
#         labels = np.zeros(max_node_id + 1, dtype=int)  #初始化label数组，默认是0
#         for comm_id, nodes in enumerate(community_nodes):
#             for node_id in nodes:
#                 labels[node_id] = comm_id
#         target =labels.reshape(-1,1)  #id为索引位置的节点对应的标签（社区）编号。
#         print(f'标签的维度：{labels.shape}, {target.shape}')
#
#
#     #加载训练数据(seed,train_nodes(pos,neg))
#     file_path = f'{args.root}/{args.dataset}/ics/{args.dataset}_{args.attack}_{args.ptb_rate}_data.json'
#     if os.path.exists(file_path):#说明之前已经调用了对应的my_pre_com方法。
#         print('使用已有的训练数据')
#         seed_list, train_nodes, labels, gtcomms = load_data_json(file_path)
#     else: #新生成训练数据,并存储到json，这里我已经设定是3个正标签，3个负标签了
#         path_test = os.path.join(args.root, args.dataset, f'{args.dataset}_{args.test_path}_{args.test_size}.txt')
#         print(f'正在使用{path_test}的数据重新生成训练数据')
#         seed_list, train_nodes, labels, gtcomms = my_pre_com(args, subgraph_list=[args.subgraph_size],train_ratio=args.train_ratio,seed_cnt=args.seed_cnt,cmty_size=args.community_size,test_query_file=path_test)
#
#     return graph, features, target, seed_list, train_nodes, labels,gtcomms
def build_labels_from_comms(comms_path: str,
                            old2new: dict,
                            n_nodes: int,
                            skip_first_line: bool = True):
    """
    读 .comms 文件 → 丢弃图外节点 → old_id → new_id → 生成 labels, comms_cleaned
    """
    comms_cleaned = []
    with open(comms_path, "r") as f:
        if skip_first_line:
            next(f)                          # 跳过首行（社区总数）
        for line in f:
            nodes = [int(x) for x in line.split()]
            mapped = [old2new[x] for x in nodes if x in old2new]
            if mapped:                       # 全无效社区直接丢掉
                comms_cleaned.append(mapped)

    labels = np.full(n_nodes, -1, dtype=np.int32)
    for cid, nodes in enumerate(comms_cleaned):
        labels[nodes] = cid                 # numpy 花式索引
    target = labels.reshape(-1, 1)
    return labels, target, comms_cleaned
def load_citations(args):
    '''
    这个是重新编号后的读取方式
    '''
    seed_list = None
    train_nodes = None
    labels = None
    # 1. 加载图数据
    graph, n_nodes = load_graph(args.root, args.dataset, args.attack, args.ptb_rate)
    print(f'图中的节点{graph.number_of_nodes()}, 图中的边{graph.number_of_edges()}')
    #2. 加载特征数据
    if args.dataset in ['cora','citeseer','pubmed']:
        feats_raw = citation_feature_reader(args.root,args.dataset)
    elif args.dataset in ['cocs','photo']:
        with open(f'{args.root}/{args.dataset}/{args.dataset}.feats', "r") as f:
            # 每行特征转换为列表，然后堆叠为 ndarray
            feats_raw = np.array([list(map(float, line.strip().split())) for line in f])
            print(f'{args.dataset}的节点特征shape:{feats_raw.shape}')
    elif args.dataset in ['dblp','amazon']:
        with open(f'{args.root}/{args.dataset}/{args.dataset}.feats', "r") as f:
            first_line = f.readline().strip().split()
            num_nodes, feat_dim = int(first_line[0]), int(first_line[1])
            feat_dict = {}
            for line in f:
                parts = line.strip().split()
                node_id = int(parts[0])
                feats = list(map(float, parts[1:]))
                feat_dict[node_id] = feats
            # 根据 node id 顺序填充
            nodes_feats = np.zeros((num_nodes, feat_dim), dtype=np.float32)
            for node_id, feats in feat_dict.items():
                nodes_feats[node_id] = feats
            feats_raw = torch.from_numpy(nodes_feats)
            node_in_dim = feats_raw.shape[1]
    elif args.dataset in ['fb107','wfb107']:  # 不加入中心节点
        feats_array = np.loadtxt(f'{args.root}/{args.dataset}/{args.dataset}.feat', delimiter=' ', dtype=np.float32)
        print(type(feats_array))
        # nodes_feats = fnormalize(feats_array)  # 将特征进行归一化
        feats_raw = torch.from_numpy(feats_array)
        node_in_dim = feats_raw.shape[1]
    elif args.dataset in ['cora_stb', 'cora_gsr']:
        feats_raw = citation_feature_reader(args.root, args.dataset[:-4])  # numpy.ndaaray:(2708,1433)
        feats_raw = torch.from_numpy(feats_raw)  # 转换成tensor
        node_in_dim = feats_raw.shape[1]
    elif args.dataset in ['fb107_gsr', 'fb107_stb']:
        feats_array = np.loadtxt(f'{args.root}/{args.dataset[:-4]}/{args.dataset[:-4]}.feat', delimiter=' ',
                                 dtype=np.float32)
        print(type(feats_array))
        # nodes_feats = fnormalize(feats_array)  # 将特征进行归一化
        feats_raw = torch.from_numpy(feats_array)
        node_in_dim = feats_raw.shape[1]
    elif args.dataset in ['facebook']:  # 读取pyg中的特征数据
        feats_array = np.loadtxt(f'{args.root}/{args.dataset}/{args.dataset}.feat', dtype=float, delimiter=' ')
        feats_raw = torch.tensor(feats_array, dtype=torch.float32)
        node_in_dim = feats_raw.shape[1]

    print(f'没有编号前的节点特征的维度:{feats_raw.shape}')


    #（a）仅保留出现在grap中的节点
    valid_old_ids = sorted(graph.nodes())  # e.g. [0,1,...,7534]
    #创建了旧编号到新的编号的映射关系
    old2new = {old: new for new, old in enumerate(valid_old_ids)}
    new2old = np.array(valid_old_ids)  # new2old[new_id] = old_id
    n_nodes = len(valid_old_ids)
    # （b）重新编号图，使节点 0..n_nodes-1 连续
    graph = nx.relabel_nodes(graph, old2new, copy=True)
    # （c）裁剪并重排特征
    features = torch.tensor(feats_raw[valid_old_ids],dtype=torch.float32)  # (7535, 745)
    print(f'编号后的节点特征的维度:{features.shape}')

    #3.加载标签数据
    if args.dataset in ['cora', 'citeseer', 'pubmed']:
        target = citation_target_reader(args.root,args.dataset)  # 所有节点的标签
        target = target.reshape(-1, 1)
        print(f'标签的维度：{target.shape}')
    elif args.dataset in ['cocs','fb107','photo','facebook','dblp','amazon']:  # 加载共同作者数据
        comm_path = f"{args.root}/{args.dataset}/{args.dataset}.comms"
        labels, target, comms_cleaned = build_labels_from_comms(
            comm_path, old2new, n_nodes, skip_first_line=True)
        print(f'标签的维度：{labels.shape}, {target.shape}')
    elif args.dataset in ['cocs_stb','fb107_stb','fb107_gsr','cora_gsr','cora_stb','cocs_gsr','photo_gsr','photo_stb']:  # 加载共同作者数据
        base = args.dataset[:-4]  # 去掉后缀 _stb / _gsr
        comm_path = f"{args.root}/{base}/{base}.comms"
        labels, target, comms_cleaned = build_labels_from_comms(
            comm_path, old2new, n_nodes, skip_first_line=True)
        print(f'标签的维度：{labels.shape}, {target.shape}')
    else:
        raise ValueError(f"Unknown dataset {args.dataset} feats")

    #加载训练数据(seed,train_nodes(pos,neg))
    file_path = f'{args.root}/{args.dataset}/ics/{args.dataset}_{args.attack}_{args.ptb_rate}_data.json'
    if os.path.exists(file_path):#说明之前已经调用了对应的my_pre_com方法。
        print('使用已有的训练数据')
        seed_list, train_nodes, labels, gtcomms = load_data_json(file_path)
    else: #新生成训练数据,并存储到json，这里我已经设定是3个正标签，3个负标签了
        path_test = os.path.join(args.root, args.dataset, f'{args.dataset}_{args.test_path}_{args.test_size}.txt')
        print(f'正在使用{path_test}的数据重新生成训练数据')
        seed_list, train_nodes, labels, gtcomms = my_pre_com(args, subgraph_list=[args.subgraph_size],train_ratio=args.train_ratio,seed_cnt=args.seed_cnt,cmty_size=args.community_size,test_query_file=path_test,old2new=old2new,graph=graph)

    return graph, features, target, seed_list, train_nodes, labels,gtcomms,new2old
def get_res_path(resroot,args,method):
    '''
    根据args创建需要的结果路径
    :param args:
    :return:
    '''
    if args.attack == 'meta':
        return f'{resroot}{args.dataset}/{args.dataset}_{args.attack}_{args.ptb_rate}_{method}_res.txt'
    elif args.attack == 'random':
        return f'{resroot}{args.dataset}/{args.dataset}_{args.attack}_{args.type}_{args.ptb_rate}_{method}_res.txt'
    elif args.attack in  ['del','gflipm','gdelm','add','random_add','random_remove','random_flip','flipm','cdelm']:
        return f'{resroot}{args.dataset}/{args.dataset}_{args.attack}_{args.ptb_rate}_{method}_res.txt'
    else:
        return f'{resroot}{args.dataset}/{args.dataset}_{method}_res.txt'

def get_comm_path(resroot,args,method):
    '''
    根据args创建存储找到的社区的路径
    :param args:
    :return:

    '''
    if not os.path.exists(f'{resroot}{args.dataset}/comm'):
        os.makedirs(f'{resroot}{args.dataset}/comm')  # results/coclep/cora/
    if args.attack in  ['del','gflipm','gdelm','add','random_add','random_remove','random_flip','flipm','cdelm','meta']:
        return f'{resroot}{args.dataset}/comm/{args.dataset}_{args.attack}_{args.ptb_rate}_{method}_res.txt'
    else: #如果没有攻击的话
        return f'{resroot}{args.dataset}/comm/{args.dataset}_{method}_res.txt'
def main():
    '''
    Parsing command line parameters
    '''
    args = parameter_parser() #获取所需的配置
    #在需要生成随机数据的实验中，每次实验都需要生成数据。设置随机种子是为了确保每次生成固定的随机数，这就使得每次实验结果显示一致了，有利于实验的比较和改进。使得每次运行该 .py 文件时生成的随机数相同。
    torch.manual_seed(args.seed) #为 CPU和CPU设置 种子 用于生成随机数，以使得结果是确定的。
    random.seed(args.seed) #为random库设置种子
    if(args.iteration==True): #Whether to start iteration. Default is False 这个应该是测试的时候测试单个迭代的情况的;默认是false
        run_iteration(args)
        return
    if args.dataset == 'ics-facebook':
        run_facebook(args)
    else: #默认执行此方法
        if args.count==2:
            res1 = run(args)
            res2 = run(args)
            avg_result = {
                "F1": (res1["F1"] + res2["F1"]) / 2,
                "Pre": (res1["Pre"] + res2["Pre"]) / 2,
                "Rec": (res1["Rec"] + res2["Rec"]) / 2,
                "TimeSingle": (res1["TimeSingle"] + res2["TimeSingle"]) / 2,
                "TimeTotal": (res1["TimeTotal"] + res2["TimeTotal"]) / 2,
                "method": res1["method"]  # 默认两次方法一致
            }
            output = get_res_path('../results/ICSGNN/', args, avg_result["method"])
            with open(output, 'a+', encoding='utf-8') as fh:
                current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                line = (
                    f"Average\n"
                    f"args: {args}\n"
                    f'Dataset: {args.dataset}, Attack: {args.attack}, ptb_rate: {args.ptb_rate}\n'
                    f"Avg Single Using Time: {avg_result['TimeSingle']}\n"
                    f"Avg Total Using Time: {avg_result['TimeTotal']}\n"
                    f"Avg F1: {avg_result['F1']}\n"
                    f"Avg Pre: {avg_result['Pre']}\n"
                    f"Avg Rec: {avg_result['Rec']}\n"
                    f"current_time: {current_time}\n"
                    "----------------------------------------\n"
                )
                fh.write(line)
            print("Averaged Results:")
            print(line)
        else:
            run(args)


def run_facebook(args):
    '''
    Run the search at random community on each of Facebook's Ego Graphs
    In order to keep consistent with the paper experiment,
    the subgraph-size and train-ratio on Facebook are fixed as ego-graphs' size
    and 8/ ego-graphs' size. And there is no Rking loss in the paper experiment.
    '''
    seedlist = [0, 107, 348, 414, 686, 698, 1684, 1912, 3437, 3980]
    for seed in seedlist:
        args.ego = 'facebook_' + str(seed)
        edges, features, circle, seed = load_facebook(seed=seed)
        features = np.array(features)
        graph = nx.from_edgelist(edges)
        target = None
        args.subgraph_size=len(graph.nodes)
        args.train_ratio=8/args.subgraph_size
        subg = SubGraph(args, graph, features, target)
        utils.tab_printer(args)
        itr = 0
        while itr< args.seed_cnt:
            print("%d "% (itr)+20*'*')
            print("\nCurrent Seed Node is %d" % seed)
            label=circle[random.randint(0, len(circle) - 1)]
            random.shuffle(label)
            negnode=list(set(graph.nodes).difference(set(label+[seed])))
            random.shuffle(negnode)
            numlabel=int(args.subgraph_size*args.train_ratio/2)
            trian_node=label[:numlabel-1]+[seed]+negnode[:numlabel]
            print(trian_node,label,seed)
            isOK= subg.community_search(seed,trian_node,label+[seed])
            itr+=isOK
        subg.methods_result()

def run(args):
    '''
    Randomly selected seeds run the community search
    The SNAP dataset is pre-processed,Randomly select a community in which the node label 1 and the other node label 0
    是否考虑seed_list和train_nodes
    '''
    #加载
    # graph, features, target, seed_list, train_nodes, labels = load_data(args)
    graph, features, target, seed_list, train_nodes, labels,gtcomms,new2old = load_citations(args)
    subg = SubGraph(args, graph, features, target) #
    utils.tab_printer(args)
    trian_node = None
    label = None
    seed_comm = None
    itr = 0

    F1 = 0.0
    Pre = 0.0
    Rec = 0.0
    Using_time_A=0.0
    Using_time_Avg=0.0
    comm_path = get_comm_path('../Case/icsgnn/', args,"Greedy-G")  # 指定保存路径
    print(f'找到的社区结果将存入{comm_path}中')
    print(f'seed_list的大小为{len(seed_list)}')
    Qsize = len(seed_list) if seed_list is not None else args.seed_cnt

    with open(comm_path, 'w', encoding='utf-8') as f_comm:
        while itr < Qsize:  # 默认是20,遍历种子节点(这就是一个query的查询结果吗？）
            seed_new = random.randint(0, len(subg.graph.nodes) - 1)
            if (seed_list is not None):  # 用于控制是随机挑选种子，还是使用预定义的种子。
                seed_new = seed_list[itr]
                trian_node = train_nodes[itr]
                label = labels[itr]
                seed_comm = gtcomms[itr]
            print(f"*********第{itr}个查询节点,查询点为*****************")
            print("  Current Seed Node is %d" % seed_new)
            f1, pre, rec, using_time, method, isOK, cnodes_new = subg.community_search(seed_new, trian_node, label,seed_comm)  # 对于一个节点的时间。
            # ---------------------③ 映射回旧编号 ---------------------
            # new2old 是 np.ndarray，所以直接花式索引即可来恢复原始的编号
            cnodes_old = new2old[np.asarray(cnodes_new, dtype=int)]
            seed_old = int(new2old[seed_new])  # 查询点也要换回旧编号

            if isOK:  # 仅在有效结果下保存社区
                print('正在存储社区结果')
                # comm_find = list(set(cnodes))  # 去重后转换为列表
                # line = str(seed) + "," + " ".join(str(u) for u in comm_find)
                comm_find_old = list(set(cnodes_old))
                line = str(seed_old) + "," + " ".join(str(u) for u in comm_find_old)
                f_comm.write(line + "\n")
            F1 += f1
            Pre += pre
            Rec += rec
            Using_time_A += using_time
            itr += isOK
    # 将运行结果存入文件
    subg.methods_result()

    F1 = F1/Qsize
    Pre = Pre/Qsize
    Rec = Rec/Qsize
    Using_time_Avg = Using_time_A/Qsize #平均每个种子节点的耗时
    print(f'Test_set Avg：F1 = {F1}, Pre = {Pre}, Rec = {Rec}')
    output = get_res_path('../results/ICSGNN/',args,method)
    with open(output, 'a+', encoding='utf-8') as fh:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # f"best_comm_threshold: {s_}, best_validation_Avg_F1: {f1_}\n"
        line = (
            f"args: {args}\n"
            f'Dataset:{args.dataset},Attack:{args.attack}, ptb_rate:{args.ptb_rate}\n'
            f"Using_time_Avg: {Using_time_Avg}\n"
            f"Using_time: {Using_time_A}\n"
            f"F1: {F1}\n"
            f"Pre: {Pre}\n"
            f"Rec: {Rec}\n"
            # f"nmi_score: {nmi_score}\n"
            # f"ari_score: {ari_score}\n"
            # f"jac_score: {jac_score}\n"
            f"current_time: {current_time}\n"
            "----------------------------------------\n"
        )
        fh.write(line)
        fh.close()

    return {
        "F1": F1 ,
        "Pre": Pre,
        "Rec": Rec,
        "TimeSingle": Using_time_Avg,
        "TimeTotal": Using_time_A,
        "method": method
    }


def run_iteration(args):
    '''
    Run community search with iteration
    '''
    graph, features, target, _ ,_ ,com_list= load_data(args)
    utils.tab_printer(args)
    subg = SubGraph(args, graph, features, target)
    itr = 0
    while itr < args.seed_cnt:
        if args.dataset in ['dblp','amazon']:
                random.shuffle(com_list)
                com_max = com_list[0]
                target =[ 1 if i in com_max else 0 for i in range(len(graph.nodes))]
                subg.target = np.array(target)[:, np.newaxis]
                seed = com_max[random.randint(0, len(com_max) - 1)]
        else:
                seed = random.randint(0, len(graph.nodes) - 1)
        print("%d " % (itr) + 20 * '*')
        print("\nCurrent Seed Node is %d" % seed)
        isOK = subg.community_search_iteration(seed)
        itr += isOK
    subg.methods_result()

if __name__ == "__main__":	
    main()
