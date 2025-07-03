# This is a sample Python script.
import argparse
import datetime
import os

import networkx as nx
import scipy.sparse as sp
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import add_remaining_self_loops

from scipy.sparse import csr_matrix

from utils.citation_loader import citation_graph_reader,citation_target_reader,citation_feature_reader
from models.EmbLearner import EmbLearner
from models.COCLE import COCLE
from models.EmbLearnerWithWeights import EmbLearnerwithWeights
from models.EmbLearnerWithoutHyper import EmbLearnerWithoutHyper
from utils.load_utils import load_data, hypergraph_construction, load_graph
from utils.log_utils import get_logger, get_log_path
from utils.cocle_val_utils import f1_score_, NMI_score, ARI_score, JAC_score, get_res_path, get_model_path, cal_pre, \
    get_comm_path, get_Size_res_path
import copy
'''
测试训练集验证集测试集大小的影响
'''
def validation(val,nodes_feats, model, edge_index, edge_index_aug):
    scorelists = []
    for q, comm in val:
        h = model((q, None, edge_index, edge_index_aug, nodes_feats))
        # 计算余弦相似度
        sim=F.cosine_similarity(h[q].unsqueeze(0),h,dim=1) #(115,)
        #使用 torch.sigmoid 将相似度值转换为概率，然后使用 squeeze(0) 移除多余的维度，
        # 并将结果转移到 CPU，最后转换为 NumPy 数组并转换为 Python 列表。
        simlists = torch.sigmoid(sim.squeeze(0)).to(
            torch.device('cpu')).numpy().tolist()  # torch.sigmoid(simlists).numpy().tolist()
        #将结果存储在scorelists中
        scorelists.append([q, comm, simlists]) #记录该样本的测试结果
    s_ = 0.1 #阈值？？
    f1_m = 0.0 #记录最大的样本得分
    s_m = s_ #记录可以取的最大的社区阈值
    while(s_<=0.9): #结束循环后得到的是从0.1按照0.05的步长不断增加社区阈值可以得到的最大的平均f1值f1_m和最优的s_取值s_m。
        f1_x = 0.0
        # print("------------------------------", s_) #s_是什么？？
        for q, comm, simlists in scorelists:
            comm_find = []
            for i, score in enumerate(simlists):#i是每个节点的编号；score是q与每个节点的相似得分。
                if score >=s_ and i not in comm_find:
                    comm_find.append(i)

            comm_find = set(comm_find)
            comm_find = list(comm_find)
            comm = set(comm)
            comm = list(comm)
            f1, pre, rec = f1_score_(comm_find, comm)
            f1_x= f1_x+f1 #累加此样本的f1得分
        f1_x = f1_x/len(val) #总的f1得分除以验证集样本数量
        if f1_m<f1_x: #如果此社区阈值下得到的平均f1得分更高
            f1_m = f1_x
            s_m = s_
        s_ = s_+0.05 #将s_进行增大。
    logger.info(f'best threshold: {s_m}, validation_set Avg F1: {f1_m}')
    return s_m, f1_m

def loadQuerys(dataset, root, train_n, val_n, test_n, train_path, test_path, val_path):
    '''
    加载数据的训练集、验证集和测试集
    这里改的是输入的train_path等就是最终的path
    :param dataset:
    :param root:
    :param train_n:
    :param val_n:
    :param test_n:
    :param train_path:
    :param test_path:
    :param val_path:
    :return:
    '''
    # path_train = os.path.join(root,dataset,f'{dataset}_{train_path}_{train_n}.txt')
    if not os.path.isfile(train_path):
        raise Exception("No such file: %s" % train_path)
    print(f'训练记录经wei:{train_path}')
    train_lists = []
    for line in open(train_path, encoding='utf-8'):
        q, pos, comm = line.split(",")
        q = int(q)
        pos = pos.split(" ")
        pos_ = [int(x) for x in pos if int(x)!=q]
        comm = comm.split(" ")
        comm_ = [int(x) for x in comm]
        if len(train_lists)>=train_n:
            break
        train_lists.append((q, pos_, comm_))

    # path_test = os.path.join(root,dataset,f'{dataset}_{test_path}_{test_n}.txt')
    if not os.path.isfile(test_path):
        raise Exception("No such file: %s" % test_path)
    test_lists = []
    for line in open(test_path, encoding='utf-8'):
        q, comm = line.split(",")
        q = int(q)
        comm = comm.split(" ")
        comm_ = [int(x) for x in comm]
        if len(test_lists)>=test_n:
            break
        test_lists.append((q, comm_))
    # path_val = os.path.join(root,dataset,f'{dataset}_{val_path}_{val_n}.txt')
    if not os.path.isfile(val_path):
        raise Exception("No such file: %s" % val_path)
    val_lists = []
    for line in open(val_path, encoding='utf-8'):
        q, comm = line.split(",")
        q = int(q)
        comm = comm.split(" ")
        comm_ = [int(x) for x in comm]
        if len(val_lists)>=val_n:
            break
        val_lists.append((q, comm_))

    return train_lists, val_lists, test_lists

def read_queries_from_file(file_path):
    """
    从文件中读取查询任务。和query_utils一样的
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
    和query_utils一样的
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
def validation_pre(val,nodes_feats, model, edge_index, edge_index_aug):
    '''
    改为选择precision最优的结果
    :param val:
    :param nodes_feats:
    :param model:
    :param edge_index:
    :param edge_index_aug:
    :return:
    '''
    scorelists = []
    for q, comm in val:
        h = model((q, None, edge_index, edge_index_aug, nodes_feats))
        # 计算余弦相似度
        sim = F.cosine_similarity(h[q].unsqueeze(0), h, dim=1)  # (115,)
        # 使用 torch.sigmoid 将相似度值转换为概率，然后使用 squeeze(0) 移除多余的维度，
        # 并将结果转移到 CPU，最后转换为 NumPy 数组并转换为 Python 列表。
        simlists = torch.sigmoid(sim.squeeze(0)).to(
            torch.device('cpu')).numpy().tolist()  # torch.sigmoid(simlists).numpy().tolist()
        # 将结果存储在scorelists中
        scorelists.append([q, comm, simlists])  # 记录该样本的测试结果
    s_ = 0.1  # 阈值？？
    pre_m = 0.0
    s_m = s_  # 记录可以取的最大的社区阈值
    while (s_ <= 0.9):  # 结束循环后得到的是从0.1按照0.05的步长不断增加社区阈值可以得到的最大的平均f1值f1_m和最优的s_取值s_m。
        pre_x = 0.0
        # print("------------------------------", s_) #s_是什么？？
        for q, comm, simlists in scorelists:
            comm_find = []
            for i, score in enumerate(simlists):  # i是每个节点的编号；score是q与每个节点的相似得分。
                if score >= s_ and i not in comm_find:
                    comm_find.append(i)

            comm_find = set(comm_find)
            comm_find = list(comm_find)
            comm = set(comm)
            comm = list(comm)
            f1, pre, rec = f1_score_(comm_find, comm)
            pre_x = pre_x + pre  # 累加此样本的f1得分
        pre_x = pre_x / len(val)  # 总的f1得分除以验证集样本数量
        if pre_m < pre_x:  # 如果此社区阈值下得到的平均f1得分更高
            pre_m = pre_x
            s_m = s_
        s_ = s_ + 0.05  # 将s_进行增大。
    logger.info(f'best threshold: {s_m}, validation_set Avg Pre: {pre_m}')
    return s_m, pre_m

def load_citations(args):
    '''
    修改了加载查询的部分，
    :param args:
    :return:
    '''
    '''********************1. 加载图数据******************************'''
    graphx,n_nodes = load_graph(args.root,args.dataset,args.attack,args.ptb_rate)

    # calAA_start = datetime.datetime.now()
    # # 计算aa指标
    # aa_indices = nx.adamic_adar_index(graphx)
    # # 初始化 Adamic-Adar 矩阵
    # aa_matrix = np.zeros((n_nodes, n_nodes))
    # # 计算 Adamic-Adar 指数
    # for u, v, p in aa_indices:
    #     aa_matrix[u, v] = p
    #     aa_matrix[v, u] = p  # 因为是无向图，所以也需要填充对称位置
    # # 转换为张量
    # aa_tensor = torch.tensor(aa_matrix, dtype=torch.float32)
    # logger.info(f'calAA_time = {datetime.datetime.now() - calAA_start}')

    src = []
    dst = []
    for id1, id2 in graphx.edges:
        src.append(id1)
        dst.append(id2)
        src.append(id2)
        dst.append(id1)
    # 这两行是获得存储成稀疏矩阵的格式，加权模型中使用
    num_nodes = graphx.number_of_nodes()
    adj_matrix = csr_matrix(([1] * len(src), (src, dst)), shape=(num_nodes, num_nodes))
    # 构建超图
    calhyper_start = datetime.datetime.now()
    edge_index = torch.tensor([src, dst])
    edge_index_aug, egde_attr = hypergraph_construction(edge_index, n_nodes, k=args.k)  # 构建超图
    edge_index = add_remaining_self_loops(edge_index, num_nodes=n_nodes)[0]
    logger.info(f'Cal Hyper_time = {datetime.datetime.now() - calhyper_start}')
    '''2:************************加载训练数据**************************'''
    if args.dataset.startswith('stb_'):
        dataset = args.dataset[4:]
    else:
        dataset = args.dataset
    logger.info('正在加载训练数据')
    queries = read_queries_from_file(f'{args.root}/{args.dataset}/{args.dataset}_all_queries.txt')  # 读取查询任务
    split_ratios = (args.train_size, args.val_size, args.test_size)
    train_size, val_size, test_size = split_ratios
    assert sum(split_ratios) == len(queries), "Sum of split ratios must equal the total number of queries"
    # 划分任务
    train_queries = queries[:train_size]
    validation_queries = queries[train_size:train_size + val_size]
    test_queries = queries[train_size + val_size:train_size + val_size + test_size]
    #定义路径,把比例都加上就不会重复了
    dataset_root = os.path.join(args.root, args.dataset)
    train_path = os.path.join(dataset_root, f'size/{args.dataset}_{args.attack}_{args.ptb_rate}_{args.pos_size}_pos_train_{split_ratios[0]}_{split_ratios[1]}_{split_ratios[2]}.txt')
    val_path = os.path.join(dataset_root, f'size/{args.dataset}_{args.attack}_{args.ptb_rate}_{args.pos_size}_val_{split_ratios[0]}_{split_ratios[1]}_{split_ratios[2]}.txt')
    test_path = os.path.join(dataset_root, f'size/{args.dataset}_{args.attack}_{args.ptb_rate}_{args.pos_size}_test_{split_ratios[0]}_{split_ratios[1]}_{split_ratios[2]}.txt')

    # 写入训练集文件
    write_queries_to_file(train_queries, train_path)
    # 写入验证集文件
    write_queries_to_file(validation_queries, val_path)
    # 写入测试集文件
    write_queries_to_file(test_queries, test_path)

    train, val, test = loadQuerys(dataset, args.root, args.train_size, args.val_size, args.test_size,
                                  train_path, val_path, test_path)
    logger.info('加载训练数据完成')
    '3.*************加载特征数据************'
    logger.info('正在加载特征数据')
    if args.dataset in ['cora','pubmed','citeseer']:
        nodes_feats = citation_feature_reader(args.root, dataset)  # numpy.ndaaray:(2708,1433)
        nodes_feats = torch.from_numpy(nodes_feats)  # 转换成tensor
        node_in_dim = nodes_feats.shape[1]
        print(f'{args.dataset}的feats dtype: {nodes_feats.dtype}')
    elif args.dataset in ['cora_stb','cora_gsr','citeseer_stb','citeseer_gsr']:
        nodes_feats = citation_feature_reader(args.root, dataset[:-4])  # numpy.ndaaray:(2708,1433)
        nodes_feats = torch.from_numpy(nodes_feats)  # 转换成tensor
        node_in_dim = nodes_feats.shape[1]
    elif args.dataset in ['fb107_gsr','fb107_stb']:
        feats_array = np.loadtxt(f'{args.root}/{args.dataset[:-4]}/{args.dataset[:-4]}.feat', delimiter=' ', dtype=np.float32)
        print(type(feats_array))
        # nodes_feats = fnormalize(feats_array)  # 将特征进行归一化
        nodes_feats = torch.from_numpy(feats_array)
        node_in_dim = nodes_feats.shape[1]
    elif args.dataset in ['cocs','photo']:
        with open(f'{args.root}/{args.dataset}/{dataset}.feats', "r") as f:
            # 每行特征转换为列表，然后堆叠为 ndarray,注意要是float32
            nodes_feats = np.array([list(map(float, line.strip().split())) for line in f],dtype=np.float32)
            print(f'{args.dataset}的nodes_feats.dtype = {nodes_feats.dtype}')
            print(f'{args.dataset}的节点特征shape:', nodes_feats.shape)
            nodes_feats = torch.from_numpy(nodes_feats)  # 转换成tensor
            node_in_dim = nodes_feats.shape[1]
    elif args.dataset in ['fb107','wfb107']:  # 不加入中心节点
        feats_array = np.loadtxt(f'{args.root}/{args.dataset}/{args.dataset}.feat', delimiter=' ', dtype=np.float32)
        print(type(feats_array))
        # nodes_feats = fnormalize(feats_array)  # 将特征进行归一化
        nodes_feats = torch.from_numpy(feats_array)
        node_in_dim = nodes_feats.shape[1]
    elif args.dataset in ['facebook']:  # 读取pyg中的特征数据
        feats_array = np.loadtxt(f'{args.root}/{args.dataset}/{args.dataset}.feat', dtype=float, delimiter=' ')
        nodes_feats = torch.tensor(feats_array, dtype=torch.float32)
        node_in_dim = nodes_feats.shape[1]
    elif args.dataset in ['football']:
        path_feat = args.root + args.dataset + '/' + args.feats_path
        if not os.path.isfile(path_feat):
            raise Exception("No such file: %s" % path_feat)
        feats_node = {}
        count = 1
        for line in open(path_feat, encoding='utf-8'):
            if count == 1:
                node_n_, node_in_dim = line.split()
                node_in_dim = int(node_in_dim)
                count = count + 1
            else:
                emb = [float(x) for x in line.split()]
                id = int(emb[0])
                emb = emb[1:]
                feats_node[id] = emb
        nodes_feats = []

        for i in range(0, n_nodes):
            if i not in feats_node:
                nodes_feats.append([0.0] * node_in_dim)
            else:
                nodes_feats.append(feats_node[i])
        nodes_feats = torch.tensor(nodes_feats)
    else:
        print('加载节点特征失败，数据集不匹配')
    print('加载节点特征完成完成')
    return nodes_feats, train, val, test, node_in_dim, n_nodes, edge_index, edge_index_aug, adj_matrix  #, aa_tensor



def Community_Search(args,logger):

    preprocess_start = datetime.datetime.now()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'device: {device}')

    #加载数据并移动到device
    # nodes_feats, train, val, test, node_in_dim, n_nodes, edge_index, edge_index_aug, adj_matrix, aa_th = load_citations(args)
    nodes_feats, train, val, test, node_in_dim, n_nodes, edge_index, edge_index_aug, adj_matrix = load_citations(args)
    logger.info(f'load_time = {datetime.datetime.now() - preprocess_start}, train len = {len(train)}')
    nodes_feats = nodes_feats.to(device)
    edge_index = edge_index.to(device)
    edge_index_aug = edge_index_aug.to(device)

    #创建节点嵌入学习模型
    if args.method == 'EmbLearner':
        embLearner = EmbLearner(node_in_dim, args.hidden_dim, args.num_layers, args.drop_out, args.tau, device,args.alpha, args.lam, args.k)  # COCLEP中的模型

    elif args.method == '':
        embLearner = EmbLearnerWithoutHyper(node_in_dim, args.hidden_dim, args.num_layers, args.drop_out, args.tau,device, args.alpha, args.lam, args.k)  # 去掉COCLEP中的超图视图，但得到的结果很差

    elif args.method == 'COCLE':  #这个是初始最默认的算法
        embLearner = COCLE(node_in_dim, args.hidden_dim, args.num_layers, args.drop_out, args.tau, device, args.alpha, args.lam, args.k) #COCLEP中的模型，目前和EmbLearner是一样的

    elif args.method == 'EmbLearnerwithWeights': #将这个作为我的
        embLearner = EmbLearnerwithWeights(node_in_dim, args.hidden_dim,args.num_layers,args.drop_out,args.tau,device,args.alpha,args.lam,args.k) #传入edge_weight参数的模型
    else:
        raise ValueError(f'method {args.method} not supported')

    logger.info(f'embLearner: {args.method}')

    emb_optim = torch.optim.Adam(embLearner.parameters(), lr=args.lr,weight_decay=args.weight_decay)
    embLearner.to(device)
    #1.预处理时间
    pre_process_time = (datetime.datetime.now() - preprocess_start).seconds
    logger.info('start trainning')
    val_best_f1 = 0.  # 记录最优的结果
    val_bst_pre = 0.
    val_bst_f1_ep = 0  # 记录取得最优结果的epoch
    val_bst_pre_ep = 0  # 记录取得最优结果的epoch
    val_bst_f1_model = copy.deepcopy(embLearner.state_dict())  # 存储最优的模型
    val_bst_pre_model = copy.deepcopy(embLearner.state_dict())  # 存储最优的模型
    val_epochs_time=0.0 #记录整个epoch下的时间
    #需要一个时间记录运行到最优的epoch后花费的时间
    #模型训练阶段
    # logger.info(torch.cuda.memory_summary(device=None, abbreviated=True))
    # torch.cuda.empty_cache()
    # torch.cuda.ipc_collect()
    # logger.info(torch.cuda.memory_summary(device=None, abbreviated=True))

    train_start = datetime.datetime.now()  # 记录模型训练的开始时间
    for epoch in range(args.epoch_n):
        embLearner.train()
        start = datetime.datetime.now()
        loss_b = 0.0
        i = 0
        for q, pos, comm in train:
            if len(pos) == 0:
                i = i + 1
                continue
            # 前馈
            loss,h = embLearner((q, pos, edge_index, edge_index_aug, nodes_feats))
            loss_b = loss_b + loss.item()  # 累积批次中的损失
            loss.backward()
            if (i + 1) % args.batch_size == 0:
                emb_optim.step()
                emb_optim.zero_grad()
            i = i + 1
        epoch_time = (datetime.datetime.now() - start).seconds  # 运行每个epoch的时间
        logger.info(f'epoch_loss = {loss_b}, epoch = {epoch}, epoch_time = {epoch_time}')
        # 每轮训练完成后，记录验证集上的准确度，选择最优的结果
        embLearner.eval()
        with torch.no_grad():
            val_start = datetime.datetime.now()
            s_,f1_ = validation(val,nodes_feats,embLearner,edge_index, edge_index_aug)
            sp_,pre_ = validation_pre(val,nodes_feats,embLearner,edge_index, edge_index_aug)
            val_time = (datetime.datetime.now() - val_start).seconds  #这两个验证的时间
            val_epochs_time= val_epochs_time+val_time
        if f1_ > val_best_f1:
            val_best_f1 = f1_
            val_bst_f1_ep = epoch
            val_bst_f1_model = copy.deepcopy(embLearner.state_dict())  # 拷贝的是最佳的权重
            # logger.info(f"Type of val_best_model: {type(val_bst_f1_model)}")
            val_bst_f1_time=(datetime.datetime.now() - train_start).seconds -val_epochs_time #当前这个epoch的时间减去训练后的时间,将验证的时间减去？？
        if pre_ > val_bst_pre:
            val_bst_pre = pre_
            val_bst_pre_ep = epoch
            val_bst_pre_model = copy.deepcopy(embLearner.state_dict())  # 拷贝的是最佳的权重
            # logger.info(f"Type of val_best_pre_model: {type(val_bst_pre_model)}")
            val_bst_pre_time=(datetime.datetime.now() - train_start).seconds -val_epochs_time #当前这个epoch的时间减去训练后的时间,将验证的时间减去？？
    #2.整个100个的训练时间
    training_time = (datetime.datetime.now() - train_start).seconds-val_epochs_time #将验证的时间减去
    logger.info(f'===best F1 at epoch {val_bst_f1_ep}, Best F1:{val_best_f1} ===,Best epoch time:{val_bst_f1_time}')
    logger.info(f'===best Pre at epoch {val_bst_pre_ep}, Best Precision:{val_bst_pre} ===,Best epoch time:{val_bst_pre_time}')
    logger.info(f'trainning time = {training_time},validate time ={val_epochs_time}')

    bst_model_path = get_model_path('./results/coclep/res_model/size/',args)
    torch.save(val_bst_f1_model, f'{bst_model_path}_f1.pkl')  # 存储最优的模型
    torch.save(val_bst_pre_model,f'{bst_model_path}_pre.pkl') #存储最优pre的模型

    #评估阶段
    logger.info(f'#################### Starting evaluation######################')
    #目前是加载具有最优pre的模型
    if args.val_type == 'pre':
        embLearner.load_state_dict(torch.load(f'{bst_model_path}_pre.pkl'))  # 加载模型
    else:
        embLearner.load_state_dict(torch.load(f'{bst_model_path}_f1.pkl'))  # 加载模型
    embLearner.eval()

    F1 = 0.0
    Pre = 0.0
    Rec = 0.0

    nmi_score = 0.0
    ari_score = 0.0
    jac_score = 0.0
    count = 0.0

    eval_start = datetime.datetime.now()

    with torch.no_grad():
        #使用验证集数据找打最佳阈值s_
        if args.val_type == 'f1':
            s_, f1_ = validation(val, nodes_feats, embLearner, edge_index, edge_index_aug)
            logger.info(f'evaluation time = {datetime.datetime.now() - eval_start}, best s_={s_}, best val f1_={f1_}')
        elif args.val_type == 'pre':
            s_, pre_ = validation_pre(val, nodes_feats, embLearner, edge_index, edge_index_aug)
            logger.info(f'evaluation time = {datetime.datetime.now() - eval_start}, best s_={s_}, best val pre_={pre_}')
        val_running_time = (datetime.datetime.now() - eval_start).seconds  # 结束了测试运行的时间
        #开始测试
        logger.info(f'#################### starting test  ####################')
        test_start = datetime.datetime.now()
        # 将找到的社区结果存入文件
        comm_path = get_comm_path('./results/coclep/', args)
        logger.info(f'找到的社区将被存入{comm_path}')
        with open(comm_path, 'a', encoding='utf-8') as f:
            for q, comm in test:
                h = embLearner((q, None, edge_index, edge_index_aug, nodes_feats))
                count = count + 1
                sim = F.cosine_similarity(h[q].unsqueeze(0), h, dim=1)
                simlists = torch.sigmoid(sim.squeeze(0)).to(torch.device('cpu')).numpy().tolist()

                comm_find = []
                for i, score in enumerate(simlists):
                    if score >= s_ and i not in comm_find:  # 此时的阈值已经是前面找到的最优的阈值了
                        comm_find.append(i)

                comm_find = set(comm_find)
                comm_find = list(comm_find)
                #将找到的社区存入文件。
                line = str(q) + "," + " ".join(str(u) for u in comm_find)
                f.write(line + "\n")
                comm = set(comm)
                comm = list(comm)
                f1, pre, rec = f1_score_(comm_find, comm)
                F1 = F1 + f1  # 累加每个样本的F1,pre和rec
                Pre = Pre + pre
                Rec = Rec + rec
                nmi = NMI_score(comm_find, comm, n_nodes)  # 计算当前样本的NMI
                nmi_score = nmi_score + nmi  # 将当前样本的NMI累加

                ari = ARI_score(comm_find, comm, n_nodes)  # 计算当前样本的ARI
                ari_score = ari_score + ari  # 将当前样本的ARI累加

                jac = JAC_score(comm_find, comm, n_nodes)  # 计算当前样本的JAC
                jac_score = jac_score + jac  # 将当前样本的JAC累加
    # 结束了测试阶段，计算测试集上的平均F1,Pre和Rec并打印
    test_running_time = (datetime.datetime.now() - test_start).seconds  # 结束了测试运行的时间

    F1 = F1 / len((test))
    Pre = Pre / len((test))
    Rec = Rec / len((test))
    nmi_score = nmi_score / len(test)
    ari_score = ari_score / len(test)
    jac_score = jac_score / len(test)
    logger.info(f'Test time = {test_running_time}')
    logger.info(f'Test_set Avg：F1 = {F1}, Pre = {Pre}, Rec = {Rec}, s = {s_}')
    logger.info(f'Test_set Avg NMI = {nmi_score}, ARI = {ari_score}, JAC = {jac_score}')

    # 存储测试结果
    # output = get_res_path('./results/coclep/', args)
    output = get_Size_res_path('./results/coclep/', args)
    os.makedirs(os.path.dirname(output), exist_ok=True)

    with open(output, 'a+',encoding='utf-8') as fh:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # f"best_comm_threshold: {s_}, best_validation_Avg_F1: {f1_}\n"
        #这里都是单次运行的时间
        line = (
            f"args: {args}\n"
            f"val_type:{args.val_type}"
            f"bst_f1_epoch:{val_bst_f1_ep}, bst_ep_time:{val_bst_f1_time}, bst_ep_f1:{val_best_f1}\n"
            f"bst_pre_epoch:{val_bst_pre_ep}, bst_ep_time:{val_bst_pre_time}, bst_ep_f1:{val_bst_pre}\n"
            f"best_comm_threshold: {s_}\n"
            f"pre_process_time: {pre_process_time}\n"
            f"training_time: {training_time}\n"
            f"val_time:{val_running_time}"
            f"test_running_time: {test_running_time}\n"
            f"F1: {F1}\n"
            f"Pre: {Pre}\n"
            f"Rec: {Rec}\n"
            f"nmi_score: {nmi_score}\n"
            f"ari_score: {ari_score}\n"
            f"jac_score: {jac_score}\n"
            f"current_time: {current_time}\n"
            "----------------------------------------\n"
        )
        fh.write(line)
        fh.close()
    return F1, Pre, Rec, nmi_score, ari_score, jac_score, pre_process_time, training_time,val_time, test_running_time


'''用这个做验证'''
def Val_Community_Search(args,logger):

    preprocess_start = datetime.datetime.now()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'device: {device}')

    #加载数据并移动到device
    nodes_feats, train, val, test, node_in_dim, n_nodes, edge_index, edge_index_aug, adj_matrix = load_citations(args)
    logger.info(f'load_time = {datetime.datetime.now() - preprocess_start}, train len = {len(train)}')
    nodes_feats = nodes_feats.to(device)
    edge_index = edge_index.to(device)
    edge_index_aug = edge_index_aug.to(device)

    #创建节点嵌入学习模型
    if args.method == 'EmbLearner':
        embLearner = EmbLearner(node_in_dim, args.hidden_dim, args.num_layers, args.drop_out, args.tau, device,args.alpha, args.lam, args.k)  # COCLEP中的模型

    elif args.method == '':
        embLearner = EmbLearnerWithoutHyper(node_in_dim, args.hidden_dim, args.num_layers, args.drop_out, args.tau,device, args.alpha, args.lam, args.k)  # 去掉COCLEP中的超图视图，但得到的结果很差

    elif args.method == 'COCLE':  #这个是初始最默认的算法
        embLearner = COCLE(node_in_dim, args.hidden_dim, args.num_layers, args.drop_out, args.tau, device, args.alpha, args.lam, args.k) #COCLEP中的模型，目前和EmbLearner是一样的

    elif args.method == 'EmbLearnerwithWeights': #将这个作为我的
        embLearner = EmbLearnerwithWeights(node_in_dim, args.hidden_dim,args.num_layers,args.drop_out,args.tau,device,args.alpha,args.lam,args.k) #传入edge_weight参数的模型
    else:
        raise ValueError(f'method {args.method} not supported')

    logger.info(f'embLearner: {args.method}')

    emb_optim = torch.optim.Adam(embLearner.parameters(), lr=args.lr,weight_decay=args.weight_decay)
    embLearner.to(device)


    logger.info(f'#################### Starting evaluation######################')
    #加载模型参数
    bst_model_path = get_model_path('./results/coclep/res_model/size/',args)
    #目前是加载具有最优pre的模型
    if args.val_type == 'pre':
        embLearner.load_state_dict(torch.load(f'{bst_model_path}_pre.pkl'))  # 加载模型
    else:
        embLearner.load_state_dict(torch.load(f'{bst_model_path}_f1.pkl'))  # 加载模型
    embLearner.eval()

    F1 = 0.0
    Pre = 0.0
    Rec = 0.0

    nmi_score = 0.0
    ari_score = 0.0
    jac_score = 0.0
    count = 0.0

    eval_start = datetime.datetime.now()

    with torch.no_grad():
        #使用验证集数据找打最佳阈值s_
        if args.val_type == 'f1':
            s_, f1_ = validation(val, nodes_feats, embLearner, edge_index, edge_index_aug)
            logger.info(f'evaluation time = {datetime.datetime.now() - eval_start}, best s_={s_}, best val f1_={f1_}')
        elif args.val_type == 'pre':
            s_, pre_ = validation_pre(val, nodes_feats, embLearner, edge_index, edge_index_aug)
            logger.info(f'evaluation time = {datetime.datetime.now() - eval_start}, best s_={s_}, best val pre_={pre_}')
        val_running_time = (datetime.datetime.now() - eval_start).seconds  # 结束了测试运行的时间
        #开始测试
        logger.info(f'#################### starting test  ####################')
        test_start = datetime.datetime.now()
        # 将找到的社区结果存入文件
        comm_path = get_comm_path('./results/coclep/', args)
        logger.info(f'找到的社区将被存入{comm_path}')
        with open(comm_path, 'a', encoding='utf-8') as f:
            for q, comm in test:
                h = embLearner((q, None, edge_index, edge_index_aug, nodes_feats))
                count = count + 1
                sim = F.cosine_similarity(h[q].unsqueeze(0), h, dim=1)
                simlists = torch.sigmoid(sim.squeeze(0)).to(torch.device('cpu')).numpy().tolist()

                comm_find = []
                for i, score in enumerate(simlists):
                    if score >= s_ and i not in comm_find:  # 此时的阈值已经是前面找到的最优的阈值了
                        comm_find.append(i)

                comm_find = set(comm_find)
                comm_find = list(comm_find)
                #将找到的社区存入文件。
                line = str(q) + "," + " ".join(str(u) for u in comm_find)
                f.write(line + "\n")
                comm = set(comm)
                comm = list(comm)
                f1, pre, rec = f1_score_(comm_find, comm)
                F1 = F1 + f1  # 累加每个样本的F1,pre和rec
                Pre = Pre + pre
                Rec = Rec + rec
                nmi = NMI_score(comm_find, comm, n_nodes)  # 计算当前样本的NMI
                nmi_score = nmi_score + nmi  # 将当前样本的NMI累加

                ari = ARI_score(comm_find, comm, n_nodes)  # 计算当前样本的ARI
                ari_score = ari_score + ari  # 将当前样本的ARI累加

                jac = JAC_score(comm_find, comm, n_nodes)  # 计算当前样本的JAC
                jac_score = jac_score + jac  # 将当前样本的JAC累加
    # 结束了测试阶段，计算测试集上的平均F1,Pre和Rec并打印
    test_running_time = (datetime.datetime.now() - test_start).seconds  # 结束了测试运行的时间

    F1 = F1 / len((test))
    Pre = Pre / len((test))
    Rec = Rec / len((test))
    nmi_score = nmi_score / len(test)
    ari_score = ari_score / len(test)
    jac_score = jac_score / len(test)
    logger.info(f'Test time = {test_running_time}')
    logger.info(f'Test_set Avg：F1 = {F1}, Pre = {Pre}, Rec = {Rec}, s = {s_}')
    logger.info(f'Test_set Avg NMI = {nmi_score}, ARI = {ari_score}, JAC = {jac_score}')

    # 存储测试结果
    output = get_Size_res_path('./results/coclep/', args)
    os.makedirs(os.path.dirname(output), exist_ok=True)

    with open(output, 'a+',encoding='utf-8') as fh:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # f"best_comm_threshold: {s_}, best_validation_Avg_F1: {f1_}\n"
        #这里都是单次运行的时间
        line = (
            f"args: {args}\n"
            f"val_type:{args.val_type}"
            f"best_comm_threshold: {s_}\n"
            f"pre_process_time: {pre_process_time}\n"
            f"val_time:{val_running_time}"
            f"test_running_time: {test_running_time}\n"
            f"F1: {F1}\n"
            f"Pre: {Pre}\n"
            f"Rec: {Rec}\n"
            f"nmi_score: {nmi_score}\n"
            f"ari_score: {ari_score}\n"
            f"jac_score: {jac_score}\n"
            f"current_time: {current_time}\n"
            "----------------------------------------\n"
        )
        fh.write(line)
        fh.close()
    return F1, Pre, Rec, nmi_score, ari_score, jac_score, pre_process_time,val_time, test_running_time


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 重复次数，论文中常用的重复5次取平均
    parser.add_argument('--count', type=int, default=1)
    parser.add_argument('--root', type=str, default='./data')
    # parser.add_argument('--res_root', type=str, default='./results/', help='result path')
    # parser.add_argument("--log", action='store_true', help='run prepare_data or not')
    parser.add_argument("--log", type=bool,default=True, help='run prepare_data or not')
    # 训练完毕的模型的存储路径
    parser.add_argument('--method',type=str,default='COCLE',choices=['EmbLearner','COCLE','EmbLearnerWithoutHyper','EmbLearnerwithWeights'])
    parser.add_argument('--model_path', type=str, default='CS')
    parser.add_argument('--m_model_path', type=str, default='META')

    # 数据集选项
    parser.add_argument('--dataset', type=str, default='citeseer')
    parser.add_argument('--pos_size', type=int, default=3) #每个训练查询顶点给出的标签数
    # 训练集、验证集、测试集大小，以及相应的文件路径，节点特征存储路径
    parser.add_argument('--train_size', type=int, default=300)
    parser.add_argument('--val_size', type=int, default=100)
    parser.add_argument('--test_size', type=int, default=500)
    parser.add_argument('--train_path', type=str, default='pos_train')
    parser.add_argument('--test_path', type=str, default='test')
    parser.add_argument('--val_path', type=str, default='val')
    parser.add_argument('--feats_path', type=str, default='feats.txt')
    parser.add_argument('--val_type', type=str, default='f1',help='pre or f1 to val')
    # 控制攻击方法、攻击类型和攻击率
    #choices=['none','meta', 'random_remove','random_flip','random_add', 'meta_attack','add','del','gflipm','gdelm','gaddm','cdelm','cflipm','delm','flipm']
    parser.add_argument('--attack', type=str, default='none')
    parser.add_argument('--type', type=str, default='add', help='random attack type', choices=['add', 'remove', 'flip'])
    parser.add_argument('--noise_level', type=int, default=3, choices=[1, 2, 3], help='noisy level')
    parser.add_argument('--ptb_rate', type=float, default=0.30, help='pertubation rate')

    # 模型batch大小，隐藏层维度，训练epoch数，drop_out，学须率lr，权重衰减weight_decay
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--epoch_n', type=int, default=10)
    parser.add_argument('--drop_out', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.001)  # 原文默认的是0.001，调整大一些0.1。
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    # 注意力系数tau，不同损失函的比率，超图跳数k
    parser.add_argument('--tau', type=float, default=0.2)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--lam', type=float, default=0.2)
    # 超图的跳数，论文中使用的是1
    parser.add_argument('--k', type=int, default=2)

    # 权重计算模型及其学习率
    parser.add_argument('--mw_net', type=str, default='MLP', choices=['MLP', 'GCN'], help='type of meta-weighted model')
    parser.add_argument('--m_lr', type=float, default=0.005, help='learning rate of meta model')

    # 更新图的阈值。pa加边阈值，pd删边阈值。
    parser.add_argument('--pa', type=float, default=0.7)
    parser.add_argument('--pd', type=float, default=0.3)
    parser.add_argument('--n_p', type=int, default=5, help='number of positive pairs per node')
    parser.add_argument("--n_n", type=int, default=5, help='number of negitive pairs per node')
    parser.add_argument('--sigma', type=float, default=100,
                        help='the parameter to control the variance of sample weights in rec loss')
    parser.add_argument('--t_delete', type=float, default=0.1,
                        help='threshold of eliminating the edges')
    parser.add_argument('--gamma', type=float, default=0.01,
                        help='weight of rec loss')

    args = parser.parse_args()
    if args.log:
        log_path = get_log_path('./log/coclep/', args)
        logger = get_logger(log_path)
        print(f'save logger to {log_path}')
    else:
        logger = get_logger()

    # 预处理时间，模型训练时间，测试时间
    pre_process_time_A, train_model_running_time_A,val_time_A, test_running_time_A = 0.0, 0.0,0.0, 0.0
    count = 0
    F1lists = []
    Prelists = []
    Reclists = []
    nmi_scorelists = []
    ari_scorelists = []
    jac_scorelists = []

    for i in range (args.count):
        count = count + 1
        # logger.info('='*20)
        now = datetime.datetime.now()
        logger.info(f'##第 {count} 次执行, Starting Time: {now.strftime("%Y-%m-%d %H:%M:%S")}')

        #执行社区搜索
        F1, Pre, Rec, nmi_score, ari_score, jac_score, pre_process_time, train_model_running_time,val_time,test_running_time = \
            Community_Search(args,logger)

        # 打印结束时间
        logger.info(f'##第{count}次Finishing Time: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        running_time = (datetime.datetime.now() - now).seconds
        # 打印总的运行时间
        logger.info(f'##第{count}次Running Time(s): {running_time}')
        print('= ' * 20)

        F1lists.append(F1)
        Prelists.append(Pre)
        Reclists.append(Rec)
        nmi_scorelists.append(nmi_score)
        ari_scorelists.append(ari_score)
        jac_scorelists.append(jac_score)
        # 累计预处理时间、训练时间和测试时间
        pre_process_time_A = pre_process_time_A + pre_process_time
        train_model_running_time_A = train_model_running_time_A + train_model_running_time
        val_time_A = val_time_A+val_time
        test_running_time_A = test_running_time_A + test_running_time

    # 计算count次数的各个评价指标的均值和方差
    F1_std = np.std(F1lists)
    F1_mean = np.mean(F1lists)
    Pre_std = np.std(Prelists)
    Pre_mean = np.mean(Prelists)
    Rec_std = np.std(Reclists)
    Rec_mean = np.mean(Reclists)
    nmi_std = np.std(nmi_scorelists)
    nmi_mean = np.mean(nmi_scorelists)
    ari_std = np.std(ari_scorelists)
    ari_mean = np.mean(ari_scorelists)
    jac_std = np.std(jac_scorelists)
    jac_mean = np.mean(jac_scorelists)

    # 计算平均每次社区搜索的各个时间
    pre_process_time_A = pre_process_time_A / float(args.count)
    train_model_running_time_A = train_model_running_time_A / float(args.count)
    val_time_A = val_time_A / float(args.count)
    test_running_time_A = test_running_time_A / float(args.count) #除以平均次数
    single_query_time = test_running_time_A/float(args.test_size)  #除以测试集大小得到测试时间
    # 将得到的结果进行存储，此时存储的是多次的average的各个指标。
    # 存储测试结果
    # output = get_res_path('./results/coclep/', args)

    output = get_Size_res_path('./results/coclep/', args)
    os.makedirs(os.path.dirname(output), exist_ok=True)

    with open(output, 'a+',encoding='utf-8') as fh:  # 记录的是 count 次的各个平均结果
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = (
            f"average {args}\n"
            f"pre_process_time: {pre_process_time_A}\n"
            f"train_model_running_time: {train_model_running_time_A}\n"
            f"val_time_A: {val_time_A}\n"
            f"test_running_time: {test_running_time_A}\n"
            f"single_query_time: {single_query_time}\n"
            f"F1 mean: {F1_mean}\n"
            f"F1 std: {F1_std}\n"
            f"Pre mean: {Pre_mean}\n"
            f"Pre std: {Pre_std}\n"
            f"Rec mean: {Rec_mean}\n"
            f"Rec std: {Rec_std}\n"
            f"nmi_score mean: {nmi_mean}\n"
            f"nmi std: {nmi_std}\n"
            f"ari_score mean: {ari_mean}\n"
            f"ari std: {ari_std}\n"
            f"jac mean: {jac_mean}\n"
            f"jac std: {jac_std}\n"
            f"current_time: {current_time}\n"
            "----------------------------------------\n"
        )
        fh.write(line)
        fh.close()