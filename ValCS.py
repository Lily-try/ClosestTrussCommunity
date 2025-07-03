# This is a sample Python script.
import argparse
import datetime
import os

import networkx as nx
import pandas as pd
import scipy.sparse as sp
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from scipy.stats import ttest_ind, ttest_rel
from torch_geometric.utils import add_remaining_self_loops

from scipy.sparse import csr_matrix

from utils.citation_loader import citation_graph_reader,citation_target_reader,citation_feature_reader
from models.EmbLearner import EmbLearner
from models.COCLE import COCLE
from models.EmbLearnerWithWeights import EmbLearnerwithWeights
from models.EmbLearnerWithoutHyper import EmbLearnerWithoutHyper
from utils.load_utils import load_data, hypergraph_construction, loadQuerys, load_graph
from utils.log_utils import get_logger, get_log_path
from utils.cocle_val_utils import f1_score_, NMI_score, ARI_score, JAC_score, get_res_path, get_model_path, cal_pre, \
    get_comm_path

import torch, random, itertools, numpy as np
from collections import defaultdict
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
plt.switch_backend("Agg")
import copy
'''
使用引文网络相关的数据集
这个是coclep源码
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
    train, val, test = loadQuerys(dataset, args.root, args.train_size, args.val_size, args.test_size,
                                  args.train_path, args.test_path, args.val_path)
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

'''用这个做验证'''
def Val_Community_Search(args,logger):

    preprocess_start = datetime.datetime.now()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'device: {device}')

    #加载数据并移动到device
    nodes_feats, train, val, test, node_in_dim, n_nodes, edge_index, edge_index_aug, adj_matrix = load_citations(args)

    # 归一化一次原始特征，后面直接用
    X_norm = F.normalize(nodes_feats, p=2, dim=1).to(device)  # (N, d_x)



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
    bst_model_path = get_model_path('./results/coclep/res_model/',args)
    #目前是加载具有最优pre的模型
    if args.val_type == 'pre':
        embLearner.load_state_dict(torch.load(f'{bst_model_path}_pre.pkl'))  # 加载模型
    else:
        embLearner.load_state_dict(torch.load(f'{bst_model_path}_f1.pkl'))  # 加载模型
    embLearner.eval()

    eval_start = datetime.datetime.now()
    # intra_sum, inter_sum = 0.0, 0.0
    # intra_cnt, inter_cnt = 0, 0
    # all_nodes = torch.arange(n_nodes, device=device)

    # ----------------------- 初始化两个统计器 -----------------------
    intra_sum_H = inter_sum_H = 0.0
    intra_cnt_H = inter_cnt_H = 0
    intra_sum_X = inter_sum_X = 0.0
    intra_cnt_X = inter_cnt_X = 0

    intra_sum_H_sig = inter_sum_H_sig = 0.0
    intra_cnt_H_sig = inter_cnt_H_sig = 0
    intra_sum_X_sig = inter_sum_X_sig = 0.0
    intra_cnt_X_sig = inter_cnt_X_sig = 0

    pos_scores_raw, neg_scores_raw = [], []  # 点积 / 余弦
    pos_scores_sig, neg_scores_sig = [], []  # sigmoid 后

    all_nodes = torch.arange(n_nodes, device=device)

    with torch.no_grad():
        #使用验证集数据找打最佳阈值s_
        if args.val_type == 'f1':
            s_, f1_ = validation(val, nodes_feats, embLearner, edge_index, edge_index_aug)
            logger.info(f'evaluation time = {datetime.datetime.now() - eval_start}, best s_={s_}, best val f1_={f1_}')
        elif args.val_type == 'pre':
            s_, pre_ = validation_pre(val, nodes_feats, embLearner, edge_index, edge_index_aug)
            logger.info(f'evaluation time = {datetime.datetime.now() - eval_start}, best s_={s_}, best val pre_={pre_}')
        val_running_time = (datetime.datetime.now() - eval_start).seconds  # 结束了测试运行的时间
        logger.info(f'验证结束，用时：val_running_time')
        logger.info(f'#################### starting test  ####################')
        for q, comm in test:
            h = embLearner((q, None, edge_index, edge_index_aug, nodes_feats))
            # h = F.normalize(h, p=2, dim=1)

            comm_idx = torch.tensor(comm, device=h.device, dtype=torch.long)
            out_idx = all_nodes[~torch.isin(all_nodes, comm_idx)]

            # ---- (1) 社区内两两相似 ----
            if len(comm_idx) > 1:
                h_c = h[comm_idx]  # (m,d)
                h_c = F.normalize(h_c, p=2, dim=1) #
                sims = torch.mm(h_c, h_c.T)
                sims_sig = torch.sigmoid(sims)  # ★ 新增 sigmoid 映射

                iu = torch.triu_indices(len(comm_idx), len(comm_idx), offset=1)
                intra_sum_H += sims[iu[0], iu[1]].sum().item()
                intra_cnt_H += iu.size(1)

                intra_sum_H_sig += sims_sig[iu[0], iu[1]].sum().item()
                intra_cnt_H_sig += iu.size(1)

            # ---- (2) 社区↔外部 ----
            h_out = h[out_idx]  # (n,d)
            h_out = F.normalize(h_out, p=2, dim=1)
            h_c = h[comm_idx]
            h_c = F.normalize(h_c, p=2, dim=1)
            sims2 = torch.mm(h_c, h_out.T)  # (m,n)
            sims2_sig = torch.sigmoid(sims2)  # ★ 新增 sigmoid 映射

            inter_sum_H += sims2.sum().item()
            inter_cnt_H += sims2.numel()

            inter_sum_H_sig += sims2_sig.sum().item()
            inter_cnt_H_sig += sims2_sig.numel()

            # ---------- 2) 原始特征 X ----------
            X_c = X_norm[comm_idx]  # (m, d_x)
            X_out = X_norm[out_idx]  # (n, d_x)

            if len(comm_idx) > 1:
                sims_x = torch.mm(X_c, X_c.T)
                sims_x_sig = torch.sigmoid(sims_x)
                iu_x = torch.triu_indices(len(comm_idx), len(comm_idx), offset=1)
                intra_sum_X += sims_x[iu_x[0], iu_x[1]].sum().item()
                intra_cnt_X += iu_x.size(1)
                intra_sum_X_sig += sims_x_sig[iu_x[0], iu_x[1]].sum().item()
                intra_cnt_X_sig += iu_x.size(1)

            sims2_x = torch.mm(X_c, X_out.T)
            sims2_x_sig = torch.sigmoid(sims2_x)
            inter_sum_X += sims2_x.sum().item()
            inter_cnt_X += sims2_x.numel()

            inter_sum_X_sig += sims2_x_sig.sum().item()
            inter_cnt_X_sig += sims2_x_sig.numel()
        # ----------------------- 计算平均值 -----------------------
        μ_intra_H = intra_sum_H / intra_cnt_H
        μ_inter_H = inter_sum_H / inter_cnt_H
        μ_intra_X = intra_sum_X / intra_cnt_X
        μ_inter_X = inter_sum_X / inter_cnt_X

        μ_intra_H_sig = intra_sum_H_sig / intra_cnt_H_sig
        μ_inter_H_sig = inter_sum_H_sig / inter_cnt_H_sig
        μ_intra_X_sig = intra_sum_X_sig / intra_cnt_X_sig
        μ_inter_X_sig = inter_sum_X_sig / inter_cnt_X_sig

        logger.info(f"H:  μ_intra={μ_intra_H:.4f}, μ_inter={μ_inter_H:.4f}")
        logger.info(f"H(sigmoid):  μ_intra={μ_intra_H_sig:.4f}, μ_inter={μ_inter_H_sig:.4f}")
        logger.info(f"X:  μ_intra={μ_intra_X:.4f}, μ_inter={μ_inter_X:.4f}")
        logger.info(f"X(sigmoid):  μ_intra={μ_intra_X_sig:.4f}, μ_inter={μ_inter_X_sig:.4f}")

        return μ_intra_H, μ_inter_H, μ_intra_X, μ_inter_X


def Val_Community_Search_zhifang(args,logger):

    preprocess_start = datetime.datetime.now()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'device: {device}')

    #加载数据并移动到device
    nodes_feats, train, val, test, node_in_dim, n_nodes, edge_index, edge_index_aug, adj_matrix = load_citations(args)

    # 归一化一次原始特征，后面直接用
    X_norm = F.normalize(nodes_feats, p=2, dim=1).to(device)  # (N, d_x)



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
    bst_model_path = get_model_path('./results/coclep/res_model/',args)
    #目前是加载具有最优pre的模型
    if args.val_type == 'pre':
        embLearner.load_state_dict(torch.load(f'{bst_model_path}_pre.pkl'))  # 加载模型
    else:
        embLearner.load_state_dict(torch.load(f'{bst_model_path}_f1.pkl'))  # 加载模型
    embLearner.eval()

    eval_start = datetime.datetime.now()
    # intra_sum, inter_sum = 0.0, 0.0
    # intra_cnt, inter_cnt = 0, 0
    # all_nodes = torch.arange(n_nodes, device=device)

    # ----------------------- 初始化两个统计器 -----------------------
    intra_sum_H = inter_sum_H = 0.0
    intra_cnt_H = inter_cnt_H = 0
    intra_sum_X = inter_sum_X = 0.0
    intra_cnt_X = inter_cnt_X = 0

    intra_sum_H_sig = inter_sum_H_sig = 0.0
    intra_cnt_H_sig = inter_cnt_H_sig = 0
    intra_sum_X_sig = inter_sum_X_sig = 0.0
    intra_cnt_X_sig = inter_cnt_X_sig = 0

    pos_scores_raw, neg_scores_raw = [], []  # 点积 / 余弦
    pos_scores_sig, neg_scores_sig = [], []  # sigmoid 后

    all_nodes = torch.arange(n_nodes, device=device)

    with torch.no_grad():
        #使用验证集数据找打最佳阈值s_
        if args.val_type == 'f1':
            s_, f1_ = validation(val, nodes_feats, embLearner, edge_index, edge_index_aug)
            logger.info(f'evaluation time = {datetime.datetime.now() - eval_start}, best s_={s_}, best val f1_={f1_}')
        elif args.val_type == 'pre':
            s_, pre_ = validation_pre(val, nodes_feats, embLearner, edge_index, edge_index_aug)
            logger.info(f'evaluation time = {datetime.datetime.now() - eval_start}, best s_={s_}, best val pre_={pre_}')
        val_running_time = (datetime.datetime.now() - eval_start).seconds  # 结束了测试运行的时间
        logger.info(f'验证结束，用时：val_running_time')
        logger.info(f'#################### starting test  ####################')
        for q, comm in test:
            h = embLearner((q, None, edge_index, edge_index_aug, nodes_feats))
            # h = F.normalize(h, p=2, dim=1) #方便直接用点积=余弦
            comm_idx = torch.tensor(comm, device=h.device, dtype=torch.long)
            out_idx = all_nodes[~torch.isin(all_nodes, comm_idx)]

            # ---- (1) 社区内两两相似 ----
            if len(comm_idx) > 1:
                h_c = h[comm_idx]  # (m,d)
                h_c = F.normalize(h_c, p=2, dim=1) #
                sims = torch.mm(h_c, h_c.T)
                sims_sig = torch.sigmoid(sims)  # ★ 新增 sigmoid 映射

                iu = torch.triu_indices(len(comm_idx), len(comm_idx), offset=1)

                # 用于画直方图
                pos_raw = sims[iu[0], iu[1]]  # (m·(m-1)/2, )
                pos_scores_raw.extend(pos_raw.tolist())
                pos_scores_sig.extend(torch.sigmoid(pos_raw).tolist())

                intra_sum_H += sims[iu[0], iu[1]].sum().item()
                intra_cnt_H += iu.size(1)

                intra_sum_H_sig += sims_sig[iu[0], iu[1]].sum().item()
                intra_cnt_H_sig += iu.size(1)

            # ---- (2) 社区↔外部 ----
            h_out = h[out_idx]  # (n,d)
            h_out = F.normalize(h_out, p=2, dim=1)
            h_c = h[comm_idx]
            h_c = F.normalize(h_c, p=2, dim=1)
            sims2 = torch.mm(h_c, h_out.T)  # (m,n)
            #用于画直方图
            neg_raw = sims2.flatten()
            neg_scores_raw.extend(neg_raw.tolist())
            neg_scores_sig.extend(torch.sigmoid(neg_raw).tolist())

            sims2_sig = torch.sigmoid(sims2)  # ★ 新增 sigmoid 映射

            inter_sum_H += sims2.sum().item()
            inter_cnt_H += sims2.numel()

            inter_sum_H_sig += sims2_sig.sum().item()
            inter_cnt_H_sig += sims2_sig.numel()

            # ---------- 2) 原始特征 X ----------
            X_c = X_norm[comm_idx]  # (m, d_x)
            X_out = X_norm[out_idx]  # (n, d_x)

            if len(comm_idx) > 1:
                sims_x = torch.mm(X_c, X_c.T)
                sims_x_sig = torch.sigmoid(sims_x)
                iu_x = torch.triu_indices(len(comm_idx), len(comm_idx), offset=1)
                intra_sum_X += sims_x[iu_x[0], iu_x[1]].sum().item()
                intra_cnt_X += iu_x.size(1)
                intra_sum_X_sig += sims_x_sig[iu_x[0], iu_x[1]].sum().item()
                intra_cnt_X_sig += iu_x.size(1)

            sims2_x = torch.mm(X_c, X_out.T)
            sims2_x_sig = torch.sigmoid(sims2_x)
            inter_sum_X += sims2_x.sum().item()
            inter_cnt_X += sims2_x.numel()

            inter_sum_X_sig += sims2_x_sig.sum().item()
            inter_cnt_X_sig += sims2_x_sig.numel()
        # ----------------------- 计算平均值 -----------------------
        μ_intra_H = intra_sum_H / intra_cnt_H
        μ_inter_H = inter_sum_H / inter_cnt_H
        μ_intra_X = intra_sum_X / intra_cnt_X
        μ_inter_X = inter_sum_X / inter_cnt_X

        μ_intra_H_sig = intra_sum_H_sig / intra_cnt_H_sig
        μ_inter_H_sig = inter_sum_H_sig / inter_cnt_H_sig
        μ_intra_X_sig = intra_sum_X_sig / intra_cnt_X_sig
        μ_inter_X_sig = inter_sum_X_sig / inter_cnt_X_sig

        logger.info(f"H:  μ_intra={μ_intra_H:.4f}, μ_inter={μ_inter_H:.4f}")
        logger.info(f"H(sigmoid):  μ_intra={μ_intra_H_sig:.4f}, μ_inter={μ_inter_H_sig:.4f}")
        logger.info(f"X:  μ_intra={μ_intra_X:.4f}, μ_inter={μ_inter_X:.4f}")
        logger.info(f"X(sigmoid):  μ_intra={μ_intra_X_sig:.4f}, μ_inter={μ_inter_X_sig:.4f}")

        # -------- (3) 绘图(示例) --------
        print('开始绘图')

        # ---------- 1) 保存原始打分（4 组） ----------
        os.makedirs("Visual/tongji", exist_ok=True)

        np.savetxt("Visual/tongji/pos_scores_raw.txt", np.array(pos_scores_raw), fmt="%.6f")
        np.savetxt("Visual/tongji/neg_scores_raw.txt", np.array(neg_scores_raw), fmt="%.6f")
        np.savetxt("Visual/tongji/pos_scores_sig.txt", np.array(pos_scores_sig), fmt="%.6f")
        np.savetxt("Visual/tongji/neg_scores_sig.txt", np.array(neg_scores_sig), fmt="%.6f")

        logger.info("✅ 已导出原始分数字符串到 Visual/tongji/*.txt")

        # ---------- 2) 如需提前算好直方图 ----------
        # 这里用 50 个 bin（-1~1），你可按需修改
        bins = np.linspace(-1, 1, 51)  # 50 bins => 51 个分割点
        centers = 0.5 * (bins[:-1] + bins[1:])  # bin 中心

        hist_pos_raw, _ = np.histogram(pos_scores_raw, bins=bins)
        hist_neg_raw, _ = np.histogram(neg_scores_raw, bins=bins)

        hist_pos_sig, _ = np.histogram(pos_scores_sig, bins=np.linspace(0, 1, 51))
        hist_neg_sig, _ = np.histogram(neg_scores_sig, bins=np.linspace(0, 1, 51))

        # 保存 raw 直方图
        df_raw = pd.DataFrame({
            "bin_center": centers,
            "pos_count": hist_pos_raw,
            "neg_count": hist_neg_raw
        })
        df_raw.to_csv("Visual/tongji/hist_raw.csv", index=False)

        # 保存 sigmoid 直方图（注意中心 0~1）
        df_sig = pd.DataFrame({
            "bin_center": 0.5 * (np.linspace(0, 1, 51)[:-1] + np.linspace(0, 1, 51)[1:]),
            "pos_count": hist_pos_sig,
            "neg_count": hist_neg_sig
        })
        df_sig.to_csv("Visual/tongji/hist_sigmoid.csv", index=False)

        logger.info("✅ 已导出直方图计数到 export/hist_*.csv")

        # plot_histogram_save(
        #     pos_scores_raw, neg_scores_raw,
        #     title="Raw cosine similarity distribution",
        #     xlabel="cosine similarity", bins=50,
        #     save_path="Visual/tongji/raw_sim_hist.png"  # 👈 指定文件名
        # )
        #
        # plot_histogram_save(
        #     pos_scores_sig, neg_scores_sig,
        #     title="Sigmoid-mapped similarity distribution",
        #     xlabel="sigmoid(sim)", bins=50,
        #     save_path="Visual/tongji/sigmoid_sim_hist.png"
        # )
        # # plot_histogram(
        #     pos_scores_raw, neg_scores_raw,
        #     title="Raw cosine similarity distribution",
        #     xlabel="cosine similarity", bins=50
        # )
        # plot_histogram(
        #     pos_scores_sig, neg_scores_sig,
        #     title="Sigmoid-mapped similarity distribution",
        #     xlabel="sigmoid(sim)", bins=50
        # )
        return μ_intra_H, μ_inter_H, μ_intra_X, μ_inter_X


def plot_histogram_save(pos, neg, title, xlabel, bins=50, save_path=None):
    plt.figure(figsize=(6, 4))
    plt.hist(pos, bins=bins, alpha=0.6, label="intra (positive)", density=True)
    plt.hist(neg, bins=bins, alpha=0.6, label="inter (negative)", density=True)
    plt.xlabel(xlabel)
    plt.ylabel("density")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    if save_path is None:          # 如果没给路径就拼一个
        safe_title = title.lower().replace(" ", "_")
        save_path = f"{safe_title}.png"

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()                    # 释放内存 / 不阻塞


def plot_histogram(pos, neg, title, xlabel, bins=50):
    plt.figure(figsize=(6, 4))
    plt.hist(pos, bins=bins, alpha=0.6, label="intra (positive)", density=True)
    plt.hist(neg, bins=bins, alpha=0.6, label="inter (negative)", density=True)
    plt.xlabel(xlabel)
    plt.ylabel("density")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

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
    parser.add_argument('--dataset', type=str, default='cora')
    # 训练集、验证集、测试集大小，以及相应的文件路径，节点特征存储路径
    parser.add_argument('--train_size', type=int, default=300)
    parser.add_argument('--val_size', type=int, default=100)
    parser.add_argument('--test_size', type=int, default=500)
    parser.add_argument('--train_path', type=str, default='3_pos_train')
    parser.add_argument('--test_path', type=str, default='3_test')
    parser.add_argument('--val_path', type=str, default='3_val')
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

    μ_intra_H, μ_inter_H, μ_intra_X, μ_inter_X = Val_Community_Search_zhifang(args, logger)
    print(f"{args.dataset}_{args.attack}_{args.ptb_rate}：μ_intra_H = {μ_intra_H:.4f},  μ_inter = {μ_inter_H:.4f}")
    print(f"{args.dataset}_{args.attack}_{args.ptb_rate}：μ_intra_x = {μ_intra_X:.4f},  μ_inter = {μ_inter_X:.4f}")
