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

def load_citations(args):
    '''
    我是直接只有graph，features和target的/
    '''
    seed_list = None
    train_nodes = None
    labels = None
    #第一步：加载特征数据
    if args.dataset in ['cora','citeseer','pubmed']:
        features = citation_feature_reader(args.root,args.dataset)
    elif args.dataset in ['cocs']:
        with open(f'{args.root}/{args.dataset}/{args.dataset}.feats', "r") as f:
            # 每行特征转换为列表，然后堆叠为 ndarray
            nodes_feats = np.array([list(map(float, line.strip().split())) for line in f])
            print('cocs的节点特征shape:',nodes_feats.shape)

    #加载标签数据
    if args.dataset in ['cora', 'citeseer', 'pubmed']:
        target = citation_target_reader(args.root,args.dataset)  # 所有节点的标签
        target = target.reshape(-1, 1)
    elif args.dataset in ['cocs']:  # 加载共同作者数据
        with open(f'{args.root}/{args.dataset}/{args.dataset}.comms', 'r') as f:
            lines = f.readlines()
        lines = lines[1:]# 跳过第一行的社区编号（你可以用也可以不用，反正是从0开始按行编号）

        # 统计最大节点ID以确定labels数组大小
        max_node_id = -1
        community_nodes = []
        for line in lines:
            nodes = list(map(int, line.strip().split()))
            if nodes:
                community_nodes.append(nodes)
                max_node_id = max(max_node_id, max(nodes))
        labels = np.zeros(max_node_id + 1, dtype=int)  #初始化label数组，默认是0
        for comm_id, nodes in enumerate(community_nodes):
            for node_id in nodes:
                labels[node_id] = comm_id
        target =labels.reshape(-1,1)  #id为索引位置的节点对应的标签（社区）编号。

    #加载图数据
    graph,n_nodes = load_graph(args.root,args.dataset,args.attack,args.ptb_rate)

    #加载训练数据(seed,train_nodes(pos,neg))
    file_path = f'{args.root}/{args.dataset}/ics/{args.dataset}_{args.attack}_{args.ptb_rate}_data.json'
    if os.path.exists(file_path):#说明之前已经调用了对应的my_pre_com方法。
        print('使用已有的训练数据')
        seed_list, train_nodes, labels, gtcomms = load_data_json(file_path)
    else: #新生成训练数据,并存储到json，这里我已经设定是3个正标签，3个负标签了
        print('正在重新生成训练数据')
        seed_list, train_nodes, labels, gtcomms = my_pre_com(args, subgraph_list=[args.subgraph_size],train_ratio=args.train_ratio,seed_cnt=args.seed_cnt,cmty_size=args.community_size)

    return graph, features, target, seed_list, train_nodes, labels,gtcomms
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
    if args.dataset == 'facebook':
        run_facebook(args)
    else: #默认执行此方法
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
    graph, features, target, seed_list, train_nodes, labels,gtcomms = load_citations(args)
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
    while itr < args.seed_cnt: #默认是20,遍历种子节点(这就是一个query的查询结果吗？）
        seed = random.randint(0, len(subg.graph.nodes) - 1)
        if (seed_list is not None):  # 用于控制是随机挑选种子，还是使用预定义的种子。
            seed = seed_list[itr]
            trian_node = train_nodes[itr]
            label = labels[itr]
            seed_comm = gtcomms[itr]
        print(f"*********第{itr}个查询节点,查询点为*****************")
        print("  Current Seed Node is %d" % seed)
        f1,pre,rec,using_time,method,isOK = subg.community_search(seed, trian_node, label,seed_comm)  #对于一个节点的时间。
        F1 +=f1
        Pre+=pre
        Rec+=rec
        Using_time_A+=using_time
        itr += isOK
    # 将运行结果存入文件
    subg.methods_result()

    F1 = F1/args.seed_cnt
    Pre = Pre/args.seed_cnt
    Rec = Rec/args.seed_cnt
    Using_time_Avg = Using_time_A/args.seed_cnt #平均每个种子节点的耗时
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
