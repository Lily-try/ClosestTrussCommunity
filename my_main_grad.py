# This is a sample Python script.
import argparse
import copy
import datetime
import os

import networkx as nx
import scipy.sparse as sp
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import add_remaining_self_loops,to_undirected

from scipy.sparse import csr_matrix

from config import get_config
from models.EmbLearnerDyKnn import EmbLearnerDyKNN
from models.EmbLearnerKnn import EmbLearnerKNN
from models.EmbLearnerKnnGrad import EmbLearnerKNNGrad
from utils.citation_loader import citation_graph_reader,citation_target_reader,citation_feature_reader
from models.EmbLearner import EmbLearner
from models.COCLE import COCLE
from models.EmbLearnerWithWeights import EmbLearnerwithWeights
from models.EmbLearnerWithoutHyper import EmbLearnerWithoutHyper
from utils.load_utils import load_data, hypergraph_construction, loadQuerys
from utils.log_utils import get_logger, get_log_path
from utils.val_utils import f1_score_, NMI_score, ARI_score, JAC_score, get_res_path, get_model_path, cal_pre

'''
使用引文网络相关的数据集
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
    # logger.info(f'best threshold: {s_m}, validation_set Avg F1: {f1_m}')
    return s_m, f1_m

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
        sim=F.cosine_similarity(h[q].unsqueeze(0),h,dim=1) #(115,)
        #使用 torch.sigmoid 将相似度值转换为概率，然后使用 squeeze(0) 移除多余的维度，
        # 并将结果转移到 CPU，最后转换为 NumPy 数组并转换为 Python 列表。
        simlists = torch.sigmoid(sim.squeeze(0)).to(
            torch.device('cpu')).numpy().tolist()  # torch.sigmoid(simlists).numpy().tolist()
        #将结果存储在scorelists中
        scorelists.append([q, comm, simlists]) #记录该样本的测试结果
    s_ = 0.1 #阈值？？
    pre_m = 0.0 #记录最大的样本得分
    s_m = s_ #记录可以取的最大的社区阈值
    while(s_<=0.9): #结束循环后得到的是从0.1按照0.05的步长不断增加社区阈值可以得到的最大的平均f1值f1_m和最优的s_取值s_m。
        pre_x = 0.0
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
            pre = cal_pre(comm_find, comm) #float precision
            pre_x= pre_x+pre #累加此样本的f1得分
        pre_x = pre_x/len(val) #总的f1得分除以验证集样本数量
        if pre_m<pre_x: #如果此社区阈值下得到的平均f1得分更高
            pre_m = pre_x
            s_m = s_
        s_ = s_+0.05 #将s_进行增大。
    logger.info(f'best threshold: {s_m}, validation_set Avg Pre: {pre_m}')
    return s_m, pre_m

def load_citations(args):
    '1:************************加载训练数据**************************'
    train, val, test = loadQuerys(args.dataset, args.root, args.train_size, args.val_size, args.test_size,
                                  args.train_path, args.test_path, args.val_path)
    '2.*************加载特征数据************'
    if args.dataset in ['cora', 'pubmed','citeseer']:
        nodes_feats = citation_feature_reader(args.root, args.dataset)  # numpy.ndaaray:(2708,1433)
        nodes_feats = torch.from_numpy(nodes_feats)  # 转换成tensor
        node_in_dim = nodes_feats.shape[1]
        # print(f'{args.dataset}的feats dtype: {nodes_feats.dtype}')
    elif args.dataset in ['cocs']: #加载共同作者数据
        with open(f'{args.root}/{args.dataset}/{args.dataset}.feats', "r") as f:
            # 每行特征转换为列表，然后堆叠为 ndarray
            nodes_feats = np.array([list(map(float, line.strip().split())) for line in f])
            print('cocs的节点特征shape:',nodes_feats.shape)
            nodes_feats = torch.from_numpy(nodes_feats)  # 转换成tensor
            node_in_dim = nodes_feats.shape[1]
    '''3.********************加载图数据******************************'''
    if args.attack == 'none':  # 使用原始数据
        if args.dataset in ['cora', 'pubmed', 'citeseer']:
            graphx = citation_graph_reader(args.root, args.dataset)  # 读取图 nx格式的
            print(graphx)
            n_nodes = graphx.number_of_nodes()
        elif args.dataset in ['cocs']:
            graphx = nx.Graph()
            with open(f'{args.root}/{args.dataset}/{args.dataset}.edges', "r") as f:
                for line in f:
                    node1,node2 = map(int,line.strip().split())
                    graphx.add_edge(node1,node2)
            print(f'{args.dataset}:',graphx)
            n_nodes = graphx.number_of_nodes()
    elif args.attack == 'random':
        path = os.path.join(args.root, args.dataset, args.attack,
                            f'{args.dataset}_{args.attack}_{args.type}_{args.ptb_rate}.npz')
        adj_csr_matrix = sp.load_npz(path)
        graphx = nx.from_scipy_sparse_array(adj_csr_matrix)
        print(graphx)
        n_nodes = graphx.number_of_nodes()
    elif args.attack in ['del','gflipm','gdelm','add']:
        path = os.path.join(args.root, args.dataset, args.attack,
                            f'{args.dataset}_{args.attack}_{args.ptb_rate}.npz')
        adj_csr_matrix = sp.load_npz(path)
        graphx = nx.from_scipy_sparse_array(adj_csr_matrix)
        print(graphx)
        n_nodes = graphx.number_of_nodes()

    # 计算 Adamic-Adar 指数
    aa_indices = nx.adamic_adar_index(graphx) #计算aa指标
    aa_matrix = np.zeros((n_nodes, n_nodes)) #初始化 Adamic-Adar 矩阵
    for u, v, p in aa_indices:
        aa_matrix[u, v] = p
        aa_matrix[v, u] = p  # 因为是无向图，所以也需要填充对称位置
    aa_tensor = torch.tensor(aa_matrix, dtype=torch.float32)# 转换为张量

    #将邻接矩阵转成edge_index
    src = []
    dst = []
    for id1, id2 in graphx.edges:
        src.append(id1)
        dst.append(id2)
        src.append(id2)
        dst.append(id1)
    # 这两行是获得存储成稀疏矩阵的格式，加权模型中使用
    adj_matrix = csr_matrix(([1] * len(src), (src, dst)), shape=(n_nodes, n_nodes))
    edge_index = torch.tensor([src, dst])#得到edge_index
    edge_index = add_remaining_self_loops(edge_index, num_nodes=n_nodes)[0] #添加自环
    return nodes_feats, train, val, test, node_in_dim, n_nodes, edge_index, adj_matrix, aa_tensor

def load_CS(args):
    '''TransZero中的数据'''
    data_list = torch.load("../cs_dgl.pt")

    #处理gt社区
    labels = data_list[2]
    print(labels, torch.min(labels), torch.max(labels))
    num_class = torch.max(labels) - torch.min(labels) + 1
    print(num_class)
    communities = [[i for i in range(labels.shape[0]) if labels[i] == j] for j in range(num_class)]
    print(communities, len(communities))

    #处理邻接矩阵
    adj = data_list[0]
    coalesced_tensor = adj.coalesce() #将稀疏邻接矩阵标准化
    index = coalesced_tensor.indices()
    edge_file = open("cocs.edges", "w")
    for i in range(index.shape[1]):
        if index[0][i].item() != index[1][i].item(): #将自环过滤掉了
            edge_file.write(str(index[0][i].item())) #index[0]是起点的索引
            edge_file.write(" ")
            edge_file.write(str(index[1][i].item())) #index[1]是终点的索引
            edge_file.write("\n")


def construct_augG(aug,nodes_feats,edge_index,n_nodes):
    '''
    根据args.aug构建对应的增强视图
    :param aug:
    :param nodes_feats:
    :param edge_index:
    :param n_nodes:
    :return:
    '''
    logger.info(f'增强图类型为：{args.aug}')
    if aug == 'hyper':  # 构建超图
        edge_index_aug, egde_attr = hypergraph_construction(edge_index, n_nodes, k=args.k)

        # 检查 edge_index_aug 的形状
        print("Shape of edge_index_aug:", edge_index_aug.shape)
        # 检查 edge_index_aug 的数据类型
        print("Data type of edge_index_aug:", edge_index_aug.dtype)
        # 检查 edge_index_aug 的张量类型
        print("Tensor type of edge_index_aug:", type(edge_index_aug))

        # 构建超图，这里自环被添加了
    elif aug in ['knn','dyknn']:  # 构建knn图
        sim = F.normalize(nodes_feats).mm(F.normalize(nodes_feats).T).fill_diagonal_(0.0)
        dst = sim.topk(10, 1)[1]  # 找到每个节点的k个最近邻节点的索引，这里的k先直接指定为10
        src = torch.arange(nodes_feats.size(0)).unsqueeze(1).expand_as(sim.topk(10, 1)[1])
        #确保dst和src在与edge_index相同的设备上
        device = edge_index.device
        src = src.to(device)
        dst = dst.to(device)

        logger.info(f"src device: {src.device}, dst device: {dst.device}, edge_index device: {edge_index.device}")
        edge_index_aug = torch.stack([src.reshape(-1), dst.reshape(-1)])
        edge_index_aug = to_undirected(edge_index_aug)
        edge_index_aug = add_remaining_self_loops(edge_index_aug, num_nodes=n_nodes)[0]        #添加自环
        # # 检查 edge_index_aug 的形状
        # print("Shape of edge_index_aug:", edge_index_aug.shape) #cora torch.Size([2, 43614])
        # # 检查 edge_index_aug 的数据类型
        # print("Data type of edge_index_aug:", edge_index_aug.dtype) #torch.int64
        # # 检查 edge_index_aug 的张量类型
        # print("Tensor type of edge_index_aug:", type(edge_index_aug)) #torch.Tensor
    else:
        logger.error('请指定使用的aug类型')
        return None
    return edge_index_aug
def update_adj(adj_grad,h):


    pass

def Community_Search(args,logger):

    preprocess_start = datetime.datetime.now()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'device: {device}')

    #加载数据并移动到device
    nodes_feats, train, val, test, node_in_dim, n_nodes, edge_index, adj_matrix, aa_th = load_citations(args)
    nodes_feats = nodes_feats.to(device)
    edge_index = edge_index.to(device)
    # edge_weights = torch.nn.Parameter(torch.ones(edge_index.size(1)),device = device)
    edge_weights = torch.nn.Parameter(torch.ones(edge_index.size(1), device=device))
    # 为每条边分配一个可训练的权重,用于计算损失相对于邻接矩阵的梯度
    #是不是应该随机初始化呢
    #根据初始的特征矩阵获取增强的edge_index
    edge_index_aug = construct_augG(args.aug,nodes_feats,edge_index,n_nodes)
    if edge_index_aug !=None:
        edge_index_aug = edge_index_aug.to(device)
    logger.info(f'load_time = {datetime.datetime.now() - preprocess_start}, train len = {len(train)}')

    #创建节点嵌入学习模型
    if args.log:
        logger.info(f'===使用的embLearner为: {args.method}======')
    if args.method == 'EmbLearner':
        embLearner = EmbLearner(node_in_dim, args.hidden_dim, args.num_layers, args.drop_out, args.tau, device,args.alpha, args.lam, args.k)  # COCLEP中的模型
    elif args.method == 'EmbLearnerWithoutHyper':
        embLearner = EmbLearnerWithoutHyper(node_in_dim, args.hidden_dim, args.num_layers, args.drop_out, args.tau,device, args.alpha, args.lam, args.k)  # 去掉COCLEP中的超图视图，但得到的结果很差
    elif args.method == 'EmbLearnerwithHyper':
        embLearner = COCLE(node_in_dim, args.hidden_dim, args.num_layers, args.drop_out, args.tau, device, args.alpha, args.lam, args.k) #COCLEP中的模型，目前和EmbLearner是一样的
    elif args.method in ['knn','dyknn']: #只是将hypergraph换成了knn图
        embLearner = EmbLearnerKNN(node_in_dim, args.hidden_dim,args.num_layers,args.drop_out,args.tau,device,args.alpha,args.lam,args.k) #COCLEP中的模型，目前和EmbLearner是一样的
    elif args.method == 'EmbLearnerwithWeights': #将这个作为我的
        embLearner = EmbLearnerwithWeights(node_in_dim, args.hidden_dim,args.num_layers,args.drop_out,args.tau,device,args.alpha,args.lam,args.k) #传入edge_weight参数的模型
    elif args.method == 'Grad': #将这个作为我的
        embLearner = EmbLearnerKNNGrad(node_in_dim, args.hidden_dim,args.num_layers,args.drop_out,args.tau,device,args.alpha,args.lam,args.k) #
    else:
        raise ValueError(f'method {args.method} not supported')

    emb_optim = torch.optim.Adam(embLearner.parameters(), lr=args.lr,weight_decay=args.weight_decay)
    embLearner.to(device)
    pre_process_time = (datetime.datetime.now() - preprocess_start).seconds

    logger.info('start trainning')

    val_bst_f1 = 0. #记录最优的结果
    val_bst_pre = 0.
    val_bst_f1_ep = 0 #记录取得最优结果的epoch
    val_bst_pre_ep = 0  # 记录取得最优结果的epoch
    val_bst_f1_model = copy.deepcopy(embLearner.state_dict()) #存储最优的模型
    val_bst_pre_model = copy.deepcopy(embLearner.state_dict())  # 存储最优的模型
    #模型训练阶段
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
            loss,h = embLearner((q, pos, edge_index, edge_index_aug, nodes_feats),edge_weights)
            # adj_grad = torch.autograd.grad(loss,edge_weights, retain_graph=False)[0]
            # print('测试adj_grad的类型',type(adj_grad),'adj_grad的形状',adj_grad.shape)
            loss_b = loss_b + loss.item()  # 累积批次中的损失
            loss.backward()
            print("edge_weights.requires_grad:", edge_weights.requires_grad)
            print("edge_weights is leaf:", edge_weights.is_leaf)
            adj_grad = edge_weights.grad
            print('edge_index的shape:',edge_index.shape)#[2,13264]
            print('测试adj_grad的类型',type(adj_grad),'adj_grad的形状',adj_grad.shape) #因为edge_index中添加了自环，因此这里面也包含自环 13264
            if (i + 1) % args.batch_size == 0:
                if args.aug == 'dyknn':  # 让knn图使用训练过程中的节点嵌入变化。放在这里就是每经过一个批次的数据更新一下创建的knn图。
                    edge_index_aug = construct_augG(args.aug, h, edge_index, n_nodes)
                    edge_index_aug.to('cuda')
                emb_optim.step()
                emb_optim.zero_grad()
            i = i + 1
        epoch_time = (datetime.datetime.now() - start).seconds  # 运行每个epoch的时间
        logger.info(f'epoch_loss = {loss_b}, epoch = {epoch}, epoch_time = {epoch_time}')

        #每轮训练完成后，记录验证集上的准确度，用于早停
        embLearner.eval()
        check_start = datetime.datetime.now()

        with torch.no_grad():
            val_start = datetime.datetime.now()
            s_, f1_ = validation(val, nodes_feats, embLearner, edge_index, edge_index_aug)
            sp_, pre_ = validation_pre(val, nodes_feats, embLearner, edge_index, edge_index_aug)
            val_time = (datetime.datetime.now() - train_start).seconds  # 当前这个epoch的时间减去训练后的时间
        if f1_ > val_bst_f1:
            val_bst_f1 = f1_
            val_bst_f1_ep = epoch
            val_bst_f1_model = copy.deepcopy(embLearner.state_dict())  # 拷贝的是最佳的权重
            logger.info(f"Type of val_bst_model: {type(val_bst_f1_model)}")
            val_bst_f1_time = (datetime.datetime.now() - train_start).seconds - val_time  # 当前这个epoch的时间减去训练后的时间,将验证的时间减去？？
        if pre_ > val_bst_pre:
            val_bst_pre = pre_
            val_bst_pre_ep = epoch
            val_bst_pre_model = copy.deepcopy(embLearner.state_dict())  # 拷贝的是最佳的权重
            logger.info(f"Type of val_bst_pre_model: {type(val_bst_pre_model)}")
            val_bst_pre_time = (datetime.datetime.now() - train_start).seconds - val_time  # 当前这个epoch的时间减去训练后的时间,将验证的时间减去？？
    #运行完所有epoch的时间
    training_time = (datetime.datetime.now() - train_start).seconds-val_time #不额外记录early_stop所耗费的时间
    logger.info(f'===best F1 at epoch {val_bst_f1_ep}, Best F1:{val_bst_f1} ===,Best epoch time:{val_bst_f1_time}')
    logger.info(f'===best Pre at epoch {val_bst_pre_ep}, Best Precision:{val_bst_pre} ===,Best epoch time:{val_bst_pre_time}')
    logger.info(f'trainning time = {training_time}')

    # model_path = './results/res_model/' + args.dataset + '_' + args.model_path + '.pkl'
    # m_model_path = './results/res_model/' + args.dataset + '_' + args.m_model_path + '.pkl'
    # 存储在验证集上表现最优的模型
    bst_model_path = get_model_path('./results/res_model/',args)
    torch.save(val_bst_f1_model, f'{bst_model_path}_f1.pkl')  # 存储最优的模型
    torch.save(val_bst_pre_model,f'{bst_model_path}_pre.pkl') #存储最优pre的模型
    '''*********************************************评估阶段***************************************'''
    logger.info(f'#################### Starting evaluation######################')
    #从训练好的路径中加载模型
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
        #开始测试
        logger.info(f'#################### starting test  ####################')
        for q,comm in test:
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
            comm = set(comm)
            comm = list(comm)
            f1, pre, rec = f1_score_(comm_find, comm)
            F1 = F1 + f1  # 累加每个样本的F1,pre和rec
            Pre = Pre + pre
            Rec = Rec + rec
            # print(f'--{count}--')
            # print(f"第{count}个样本Res：q = {q},f1 = {f1}, pre = {pre}, rec = {rec}")
            # print(f"到{count}个样本时的Avg Res：F1 ={F1 / count}, Pre = {Pre / count}, Rec = {Rec / count}")

            nmi = NMI_score(comm_find, comm, n_nodes)  # 计算当前样本的NMI
            nmi_score = nmi_score + nmi  # 将当前样本的NMI累加

            ari = ARI_score(comm_find, comm, n_nodes)  # 计算当前样本的ARI
            ari_score = ari_score + ari  # 将当前样本的ARI累加

            jac = JAC_score(comm_find, comm, n_nodes)  # 计算当前样本的JAC
            jac_score = jac_score + jac  # 将当前样本的JAC累加

    # 结束了测试阶段，计算测试集上的平均F1,Pre和Rec并打印
    test_running_time = (datetime.datetime.now() - now).seconds  # 结束了测试运行的时间
    F1 = F1 / len((test))
    Pre = Pre / len((test))
    Rec = Rec / len((test))
    nmi_score = nmi_score / len(test)
    ari_score = ari_score / len(test)
    jac_score = jac_score / len(test)
    logger.info(f'Test time = {test_running_time}')
    logger.info(f'Test_set Avg：F1 = {F1}, Pre = {Pre}, Rec = {Rec}, s = {s_}')
    logger.info(f'Test_set Avg NMI = {nmi_score}, ARI = {ari_score}, JAC = {jac_score}')

    # 这里存储的是本次(共count次）的结果
    output = get_res_path('./results/',args)
    with open(output, 'a+',encoding='utf-8') as fh:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = (
            f"args: {args}\n"
            f"bst_f1_epoch:{val_bst_f1_ep}, bst_ep_time:{val_bst_f1_time}, bst_ep_f1:{val_bst_f1}\n"
            f"bst_pre_epoch:{val_bst_pre_ep}, bst_ep_time:{val_bst_pre_time}, bst_ep_f1:{val_bst_pre}\n"
            f"best_comm_threshold: {s_}, best_validation_Avg_F1: {f1_}\n"
            f"pre_process_time: {pre_process_time}\n"
            f"training_time: {training_time}\n"
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
    return F1, Pre, Rec, nmi_score, ari_score, jac_score, pre_process_time, training_time, test_running_time


def Community_Val(args,logger):

    preprocess_start = datetime.datetime.now()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'device: {device}')

    #加载数据并移动到device
    nodes_feats, train, val, test, node_in_dim, n_nodes, edge_index, adj_matrix, aa_th = load_citations(args)
    if args.aug !='none':
        edge_index_aug = construct_augG(args.aug,nodes_feats,edge_index,n_nodes)
        edge_index_aug = edge_index_aug.to(device)
    nodes_feats = nodes_feats.to(device)
    edge_index = edge_index.to(device)
    logger.info(f'load_time = {datetime.datetime.now() - preprocess_start}, train len = {len(train)}')

    #创建节点嵌入学习模型
    if args.log:
        logger.info(f'===使用的embLearner为: {args.method}======')
    if args.method == 'EmbLearner':
        embLearner = EmbLearner(node_in_dim, args.hidden_dim, args.num_layers, args.drop_out, args.tau, device,args.alpha, args.lam, args.k)  # COCLEP中的模型
    elif args.method == 'EmbLearnerWithoutHyper':
        embLearner = EmbLearnerWithoutHyper(node_in_dim, args.hidden_dim, args.num_layers, args.drop_out, args.tau,device, args.alpha, args.lam, args.k)  # 去掉COCLEP中的超图视图，但得到的结果很差
    elif args.method == 'EmbLearnerwithHyper':
        embLearner = COCLE(node_in_dim, args.hidden_dim, args.num_layers, args.drop_out, args.tau, device, args.alpha, args.lam, args.k) #COCLEP中的模型，目前和EmbLearner是一样的
    elif args.method == 'knn':
        embLearner = EmbLearnerKNN(node_in_dim, args.hidden_dim,args.num_layers,args.drop_out,args.tau,device,args.alpha,args.lam,args.k) #COCLEP中的模型，目前和EmbLearner是一样的
    elif args.method == 'EmbLearnerwithWeights': #将这个作为我的
        embLearner = EmbLearnerwithWeights(node_in_dim, args.hidden_dim,args.num_layers,args.drop_out,args.tau,device,args.alpha,args.lam,args.k) #传入edge_weight参数的模型
    else:
        raise ValueError(f'method {args.method} not supported')
    emb_optim = torch.optim.Adam(embLearner.parameters(), lr=args.lr,weight_decay=args.weight_decay)
    embLearner.to(device)
    pre_process_time = (datetime.datetime.now() - preprocess_start).seconds

    logger.info(f'#################### Starting evaluation######################')
    #从训练好的路径中加载模型
    model_path = get_model_path('./results/res_model/',args)
    embLearner.load_state_dict(torch.load(model_path,weights_only=True))  # 加载模型
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
        s_, f1_ = validation(val, nodes_feats, embLearner, edge_index, edge_index_aug)
        logger.info(f'Evaluation time = {datetime.datetime.now() - eval_start}, best s_={s_}, best val f1_={f1_}')
        #开始测试
        logger.info(f'#################### starting test  ####################')
        for q,comm in test:
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
            comm = set(comm)
            comm = list(comm)
            f1, pre, rec = f1_score_(comm_find, comm)
            F1 = F1 + f1  # 累加每个样本的F1,pre和rec
            Pre = Pre + pre
            Rec = Rec + rec
            # print(f'--{count}--')
            # print(f"第{count}个样本Res：q = {q},f1 = {f1}, pre = {pre}, rec = {rec}")
            # print(f"到{count}个样本时的Avg Res：F1 ={F1 / count}, Pre = {Pre / count}, Rec = {Rec / count}")

            nmi = NMI_score(comm_find, comm, n_nodes)  # 计算当前样本的NMI
            nmi_score = nmi_score + nmi  # 将当前样本的NMI累加

            ari = ARI_score(comm_find, comm, n_nodes)  # 计算当前样本的ARI
            ari_score = ari_score + ari  # 将当前样本的ARI累加

            jac = JAC_score(comm_find, comm, n_nodes)  # 计算当前样本的JAC
            jac_score = jac_score + jac  # 将当前样本的JAC累加

    # 结束了测试阶段，计算测试集上的平均F1,Pre和Rec并打印
    test_running_time = (datetime.datetime.now() - now).seconds  # 结束了测试运行的时间
    F1 = F1 / len((test))
    Pre = Pre / len((test))
    Rec = Rec / len((test))
    nmi_score = nmi_score / len(test)
    ari_score = ari_score / len(test)
    jac_score = jac_score / len(test)
    logger.info(f'Test time = {test_running_time}')
    logger.info(f'Test_set Avg：F1 = {F1}, Pre = {Pre}, Rec = {Rec}, s = {s_}')
    logger.info(f'Test_set Avg NMI = {nmi_score}, ARI = {ari_score}, JAC = {jac_score}')

    # 这里存储的是本次(共count次）的结果
    output = get_res_path('./results/',args)
    with open(output, 'a+',encoding='utf-8') as fh:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = (
            f"args: {args}\n"
            f"best_comm_threshold: {s_}, best_validation_Avg_F1: {f1_}\n"
            f"pre_process_time: {pre_process_time}\n"
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
    return F1, Pre, Rec, nmi_score, ari_score, jac_score, pre_process_time, test_running_time



# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    #加载配置信息
    args = get_config()
    if args.log:
        log_path = get_log_path('./log/',args)
        logger = get_logger(log_path)
        print(f'save logger to {log_path}')
    else: #不指定文件，即不创建日志文件了
        logger = get_logger()

    # 预处理时间，模型训练时间，测试时间
    pre_process_time_A, train_model_running_time_A, test_running_time_A = 0.0, 0.0, 0.0
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
        F1, Pre, Rec, nmi_score, ari_score, jac_score, pre_process_time, train_model_running_time, test_running_time = \
            Community_Search(args,logger)

        # #不训练模型，直接进行验证
        # F1, Pre, Rec, nmi_score, ari_score, jac_score, pre_process_time, test_running_time = Community_Val(args, logger)
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
    test_running_time_A = test_running_time_A / float(args.count)

    # 将count次的平均结果存入文件
    output = get_res_path('./results/',args)
    with open(output, 'a+',encoding = 'utf-8') as fh:  # 记录的是 count 次的各个平均结果
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = (
            f"AVERAGE： {args}\n"
            f"pre_process_time: {pre_process_time_A}\n"
            f"train_model_running_time: {train_model_running_time_A}\n"
            f"test_running_time: {test_running_time_A}\n"
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
