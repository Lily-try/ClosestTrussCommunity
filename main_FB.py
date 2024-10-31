# This is a sample Python script.
import argparse
import datetime
import os
import scipy.sparse as sp
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from torch_geometric.datasets import AttributedGraphDataset
from torch_geometric.utils import add_remaining_self_loops, to_networkx

from models.EmbLearner import EmbLearner
from models.EmbLearnerWithHyper import EmbLearnerwithHyper
from models.EmbLearnerWithWeights import EmbLearnerwithWeights
from models.EmbLearnerWithoutHyper import EmbLearnerWithoutHyper
from utils.load_utils import load_data, hypergraph_construction, loadQuerys
from utils.log_utils import get_logger
from utils.val_utils import f1_score_, NMI_score, ARI_score, JAC_score, validation
from utils.ego_utils import *
'''
使用facebook相关的数据集
'''


def load_FB(args):
    max = 0
    edges = []
    '''********************1. 加载图数据******************************'''
    if args.attack == 'none':  # 使用原始数据
        if args.dataset in ['football', 'facebook_all']: #原文中的数据集
            path = os.path.join(args.root, args.dataset, f'{args.dataset}.txt')
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
        elif args.dataset.startswith(('fb', 'wfb','fa')): #读取ego-facebook的邻接矩阵
            graphx = nx.read_edgelist(f'{args.root}/{args.dataset}/{args.dataset}.edges', nodetype=int,data=False)
            print(graphx)
            n_nodes = graphx.number_of_nodes()
        else:
            raise ValueError(f"未识别的数据集类型：{args.dataset}")  # 处理没有匹配的情况
    elif args.attack == 'random':
        path = os.path.join(args.root, args.dataset, args.attack, f'{args.dataset}_{args.attack}_{args.type}_{args.ptb_rate}.npz')
        adj_csr_matrix = sp.load_npz(path)
        graphx = nx.from_scipy_sparse_array(adj_csr_matrix)
        print(graphx)
        n_nodes = graphx.number_of_nodes()
    elif args.attack =='add': #metaGC中自己注入随机噪声
        path = os.path.join(args.root, args.dataset, args.attack, f'{args.dataset}_{args.attack}_{args.noise_level}.npz')
        adj_csr_matrix = sp.load_npz(path)
        graphx = nx.from_scipy_sparse_array(adj_csr_matrix)
        print(graphx)
        n_nodes = graphx.number_of_nodes()
    else:
        raise ValueError(f"未识别的"
                         f"攻击类型：{args.attack}")  # 处理没有匹配的攻击类型情况

    # 计算aa指标
    aa_indices = nx.adamic_adar_index(graphx)
    # 初始化 Adamic-Adar 矩阵
    aa_matrix = np.zeros((n_nodes, n_nodes))
    # 计算 Adamic-Adar 指数
    for u, v, p in aa_indices:
        aa_matrix[u, v] = p
        aa_matrix[v, u] = p  # 因为是无向图，所以也需要填充对称位置
    # 转换为张量
    aa_tensor = torch.tensor(aa_matrix, dtype=torch.float32)

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
    edge_index = torch.tensor([src, dst])
    edge_index_aug, egde_attr = hypergraph_construction(edge_index, n_nodes, k=args.k)  # 构建超图
    edge_index = add_remaining_self_loops(edge_index, num_nodes=n_nodes)[0]

    '3.*************加载特征数据************'
    if args.dataset.startswith(('fb', 'wfb','fa')): #不加入中心节点
        # feature_file = "{}/{}/{}.feat".format(args.root, args.dataset, args.dataset)
        # feats_array = load_features(feature_file)
        feats_array = np.loadtxt(f'{args.root}/{args.dataset}/{args.dataset}.feat',delimiter=' ', dtype=np.float32)
        print(type(feats_array))
        nodes_feats = fnormalize(feats_array)  # 将特征进行归一化
        nodes_feats = torch.from_numpy(feats_array)
        node_in_dim = nodes_feats.shape[1]
    elif args.dataset in ['facebook']: #读取pyg中的特征数据
        feats_array = np.loadtxt(f'{args.root}/{args.dataset}/{args.dataset}.feat', dtype=float, delimiter=' ')
        nodes_feats = torch.tensor(feats_array, dtype=torch.float32)
        node_in_dim = nodes_feats.shape[1]

    '''2:************************加载训练数据**************************'''
    train, val, test = loadQuerys(args.dataset, args.root, args.train_size, args.val_size, args.test_size,
                                  args.train_path, args.test_path, args.val_path)

    return nodes_feats, train, val, test, node_in_dim, n_nodes, edge_index, edge_index_aug, adj_matrix, aa_tensor

def Community_Search(args,logger):

    preprocess_start = datetime.datetime.now()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'device: {device}')

    #加载数据并移动到device
    nodes_feats, train, val, test, node_in_dim, n_nodes, edge_index, edge_index_aug, adj_matrix, aa_th = load_FB(args)

    logger.info(f'load_time = {datetime.datetime.now() - preprocess_start}, train len = {len(train)}')
    nodes_feats = nodes_feats.to(device)
    edge_index = edge_index.to(device)
    edge_index_aug = edge_index_aug.to(device)

    #创建节点嵌入学习模型
    if args.method == 'EmbLearner':
        embLearner = EmbLearner(node_in_dim, args.hidden_dim, args.num_layers, args.drop_out, args.tau, device,args.alpha, args.lam, args.k)  # COCLEP中的模型

    elif args.method == 'EmbLearnerWithoutHyper':
        embLearner = EmbLearnerWithoutHyper(node_in_dim, args.hidden_dim, args.num_layers, args.drop_out, args.tau,device, args.alpha, args.lam, args.k)  # 去掉COCLEP中的超图视图，但得到的结果很差

    elif args.method == 'EmbLearnerwithHyper':
        embLearner = EmbLearnerwithHyper(node_in_dim, args.hidden_dim,args.num_layers,args.drop_out,args.tau,device,args.alpha,args.lam,args.k) #COCLEP中的模型，目前和EmbLearner是一样的

    elif args.method == 'EmbLearnerwithWeights':
        embLearner = EmbLearnerwithWeights(node_in_dim, args.hidden_dim,args.num_layers,args.drop_out,args.tau,device,args.alpha,args.lam,args.k) #传入edge_weight参数的模型
    else:
        raise ValueError(f'method {args.method} not supported')

    logger.info(f'embLearner: {args.method}')

    emb_optim = torch.optim.Adam(embLearner.parameters(), lr=args.lr,weight_decay=args.weight_decay)
    embLearner.to(device)

    pre_process_time = (datetime.datetime.now() - preprocess_start).seconds
    logger.info('start trainning')
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
            loss,h = embLearner((q, pos, edge_index, edge_index_aug, nodes_feats))
            loss_b = loss_b + loss.item()  # 累积批次中的损失
            loss.backward()
            if (i + 1) % args.batch_size == 0:
                emb_optim.step()
                emb_optim.zero_grad()
            i = i + 1
        epoch_time = (datetime.datetime.now() - start).seconds  # 运行每个epoch的时间
        logger.info(f'epoch_loss = {loss_b}, epoch = {epoch}, epoch_time = {epoch_time}')

    training_time = (datetime.datetime.now() - train_start).seconds
    logger.info(f'trainning time = {training_time}')

    # model_path = './results/res_model/' + args.dataset + '_' + args.model_path + '.pkl'
    # m_model_path = './results/res_model/' + args.dataset + '_' + args.m_model_path + '.pkl'
    model_dir = './results/res_model/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if args.attack == 'meta':
        model_path = f'{model_dir}{args.dataset}_{args.attack}_{args.ptb_rate}_cs.pkl'
    elif args.attack == 'random':
        model_path = f'{model_dir}{args.dataset}_{args.attack}_{args.type}_{args.ptb_rate}_cs.pkl'
    elif args.attack =='add':
        model_path = f'{model_dir}{args.dataset}_{args.attack}_{args.noise_level}_cs.pkl'
    else:
        model_path = args.res_root + args.dataset + '_res.txt'

    torch.save(embLearner.state_dict(), model_path)  #存储训练好的模型

    #评估阶段
    logger.info(f'#################### Starting evaluation######################')
    embLearner.load_state_dict(torch.load(model_path))  # 加载模型
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
        logger.info(f'evaluation time = {datetime.datetime.now() - eval_start}, best threshold s_={s_}, best val f1_={f1_}')

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

    # 存储测试结果
    if args.attack == 'meta':
        output = args.res_root + args.dataset + f'_{args.attack}_{args.ptb_rate}_res.txt'
    elif args.attack == 'random':
        output = args.res_root + args.dataset + f'_{args.attack}_{args.type}_{args.ptb_rate}_res.txt'
    elif args.attack =='add':
        output = args.res_root + args.dataset + f'_{args.attack}_{args.noise_level}_res.txt'
    else:
        output = args.res_root + args.dataset + '_res.txt'
    with open(output, 'a+') as fh:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = (
            f"args: {args}\n"
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

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 重复次数，论文中常用的重复5次取平均
    parser.add_argument('--count', type=int, default=1)
    parser.add_argument('--root', type=str, default='./data')
    parser.add_argument('--res_root', type=str, default='./results/', help='result path')
    # parser.add_argument("--log", action='store_true', help='run prepare_data or not')
    parser.add_argument("--log", type=bool,default=True, help='run prepare_data or not')
    # 训练完毕的模型的存储路径
    parser.add_argument('--method',type=str,default='EmbLearner',choices=['EmbLearner','EmbLearnerWithoutHyper','EmbLearnerwithWeights'])
    parser.add_argument('--model_path', type=str, default='CS')
    parser.add_argument('--m_model_path', type=str, default='META')

    # 数据集选项
    parser.add_argument('--dataset', type=str, default='football')
    # 训练集、验证集、测试集大小，以及相应的文件路径，节点特征存储路径
    parser.add_argument('--train_size', type=int, default=300)
    parser.add_argument('--val_size', type=int, default=100)
    parser.add_argument('--test_size', type=int, default=500)
    parser.add_argument('--train_path', type=str, default='pos_train.txt')
    parser.add_argument('--test_path', type=str, default='test.txt')
    parser.add_argument('--val_path', type=str, default='val.txt')
    parser.add_argument('--feats_path', type=str, default='feats.txt')

    # 控制攻击方法、攻击类型和攻击率
    parser.add_argument('--attack', type=str, default='none', choices=['none', 'random', 'meta_attack','add'])
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
        if args.attack == 'meta':
            log_path = './log/' + args.dataset + f'_{args.attack}_{args.ptb_rate}_{args.method}.log'
        elif args.attack == 'random':
            log_path = './log/' + args.dataset + f'_{args.attack}_{args.type}_{args.ptb_rate}_{args.method}.log'
        elif args.attack == 'add':
            log_path = './log/'  + args.dataset + f'_{args.attack}_{args.noise_level}_{args.method}.log'
        else:
            log_path = './log/' + args.dataset +f'_{args.method}.log'
        # logger = get_logger('./log/' + args.attack + '/' + 'ours_' + args.dataset + '_' + str(args.ptb_rate) + '.log')
        logger = get_logger(log_path)
    else:
        logger = get_logger('./log/try.log')

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

    # 将得到的结果进行存储，此时存储的是多次的average的各个指标。
    # output = args.res_root + args.dataset + '_res.txt'
    # 存储测试结果
    if args.attack == 'meta':
        output = args.res_root + args.dataset + f'_{args.attack}_{args.ptb_rate}_res.txt'
    elif args.attack == 'random':
        output = args.res_root + args.dataset + f'_{args.attack}_{args.type}_{args.ptb_rate}_res.txt'
    elif args.attack =='add':
        output = args.res_root + args.dataset + f'_{args.attack}_{args.noise_level}_res.txt'
    else:
        output = args.res_root + args.dataset + '_res.txt'
    with open(output, 'a+') as fh:  # 记录的是 count 次的各个平均结果
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = (
            f"average {args}\n"
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
