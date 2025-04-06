from ICSGNN.parser import parameter_parser
from citation_loader import citation_feature_reader, citation_target_reader, citation_graph_reader
from src.subgraph import SubGraph
from src import utils
import torch
from torch_geometric.datasets import Reddit
from src.pre_community import *
import scipy.sparse as sp


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
    #加载特征数据
    if args.dataset in ['cora','citeseer','pubmed']:
        features = citation_feature_reader(args.root,args.dataset)
    #加载标签数据
    if args.dataset in ['cora', 'citeseer', 'pubmed']:
        target = citation_target_reader(args.root,args.dataset)  # 所有节点的标签
        target = target.reshape(-1, 1)
    #加载图数据
    graph = load_graph(args)

    #加载训练数据(seed,train_nodes(pos,neg))
    file_path = f'{args.root}/{args.dataset}/ics/{args.dataset}_{args.attack}_{args.ptb_rate}_data.json'
    if os.path.exists(file_path):
        seed_list, train_nodes, labels, gtcomms = load_data_json(file_path)
    else:
        seed_list, train_nodes, labels, gtcomms = my_pre_com(args, subgraph_list=[args.subgraph_size],train_ratio=args.train_ratio,seed_cnt=args.seed_cnt,cmty_size=args.community_size)

    return graph, features, target, seed_list, train_nodes, labels

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
    graph, features, target, seed_list, train_nodes, labels = load_citations(args)
    subg = SubGraph(args, graph, features, target) #
    utils.tab_printer(args)
    trian_node = None
    label = None
    itr = 0
    while itr < args.seed_cnt: #默认是20,遍历种子节点
        seed = random.randint(0, len(subg.graph.nodes) - 1)
        if (seed_list is not None):  # 用于控制是随机挑选种子，还是使用预定义的种子。
            seed = seed_list[itr]
            trian_node = train_nodes[itr]
            label = labels[itr]
        print("%d " % (itr) + 20 * '*')
        print("\nCurrent Seed Node is %d" % seed)
        isOK = subg.community_search(seed, trian_node, label)
        itr += isOK
    #将运行结果存入文件
    subg.methods_result()


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
