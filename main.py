# This is a sample Python script.
import argparse
import datetime
import os

import numpy as np
import torch
import torch.nn.functional as F

from models.EmbLearner import EmbLearner
from models.EmbLearnerWithHyper import EmbLearnerwithHyper
from models.EmbLearnerWithWeights import EmbLearnerwithWeights
from models.EmbLearnerWithoutHyper import EmbLearnerWithoutHyper
from utils.load_utils import load_data
from utils.log_utils import get_logger
from utils.val_utils import f1_score_, NMI_score, ARI_score, JAC_score


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

def Community_Search(args,logger):

    preprocess_start = datetime.datetime.now()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'device: {device}')

    #加载数据并移动到device
    nodes_feats, train, val, test, node_in_dim, n_nodes, edge_index, edge_index_aug, adj_matrix, aa_th = load_data(args)
    logger.info(f'load_time = {datetime.datetime.now() - preprocess_start}, train len = {len(train)}')
    nodes_feats = nodes_feats.to(device)
    edge_index = edge_index.to(device)
    edge_index_aug = edge_index_aug.to(device)

    #创建节点嵌入学习模型
    # embLearner = EmbLearnerwithHyper(node_in_dim, args.hidden_dim,args.num_layers,args.drop_out,args.tau,device,args.alpha,args.lam,args.k) #COCLEP中的模型
    embLearner = EmbLearner(node_in_dim, args.hidden_dim,args.num_layers,args.drop_out,args.tau,device,args.alpha,args.lam,args.k) #COCLEP中的模型
    # embLearner = EmbLearnerWithoutHyper(node_in_dim, args.hidden_dim,args.num_layers,args.drop_out,args.tau,device,args.alpha,args.lam,args.k) #去掉COCLEP中的超图视图，但得到的结果很差
    # embLearner = EmbLearnerwithWeights(node_in_dim, args.hidden_dim,args.num_layers,args.drop_out,args.tau,device,args.alpha,args.lam,args.k) #传入edge_weight参数

    emb_optim = torch.optim.Adam(embLearner.parameters(), lr=args.lr,weight_decay=args.weight_decay)
    embLearner.to(device)

    pre_process_time = (datetime.datetime.now() - preprocess_start).seconds

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
        logger.info(f'evaluation time = {datetime.datetime.now() - eval_start}, best s_={s_}, best val f1_={f1_}')
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
        # logger = get_logger('./log/' + args.attack + '/' + 'ours_' + args.dataset + '_' + str(args.ptb_rate) + '.log')
        logger = get_logger('./log/'+args.dataset+'_cocle.log')
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
