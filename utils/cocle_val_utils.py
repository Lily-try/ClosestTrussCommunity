import os

import numpy as np
import torch
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, jaccard_score
import torch.nn.functional as F
from utils.log_utils import get_logger
'''
评价社区搜索效果的指标
'''

def f1_score_(comm_find, comm):
    '''

    :param comm_find:TP+FP， 所有预测为正（是社区成员的元素集合），
    :param comm: 实际的社区成员集合
    :return:
    '''

    lists = [x for x in comm_find if x in comm] #TP（将正类预测为正类）交集，同时在com_find和comm中出现的元素。
    if len(lists) == 0:
        return 0.0, 0.0, 0.0
    pre = len(lists) * 1.0 / len(comm_find) #pre = TP/(TP+FP) = TP/comm_find
    rec = len(lists) * 1.0 / len(comm) #recall = TP/(TP+FN) = TP/comm
    #ACC= (TP+TN)/(TP+TN+FP+FN)
    f1 = 2 * pre * rec / (pre + rec) # F1=2*P*R/(P+R)
    return f1, pre, rec
def cal_pre(comm_find, comm):
    '''

    :param comm_find:TP+FP， 所有预测为正（是社区成员的元素集合），
    :param comm: 实际的社区成员集合
    :return:
    '''

    lists = [x for x in comm_find if x in comm] #TP（将正类预测为正类）交集，同时在com_find和comm中出现的元素。
    if len(lists) == 0:
        return 0.0
    pre = len(lists) * 1.0 / len(comm_find) #pre = TP/(TP+FP) = TP/comm_find
    return pre

def NMI_score(comm_find, comm, n_nodes):

    truthlabel = np.zeros((n_nodes), dtype=int)
    truthlabel[comm] = 1
    prelabel = np.zeros((n_nodes), dtype=int)
    prelabel[comm_find] = 1
    score = normalized_mutual_info_score(truthlabel, prelabel)
    #print("q, nmi:", score)
    return score

def ARI_score(comm_find, comm, n_nodes):

    truthlabel = np.zeros((n_nodes), dtype=int)
    truthlabel[comm] = 1
    prelabel = np.zeros((n_nodes), dtype=int)
    prelabel[comm_find] = 1
    score = adjusted_rand_score(truthlabel, prelabel)
    #print("q, ari:", score)

    return score

def JAC_score(comm_find, comm, n_nodes):
    truthlabel = np.zeros((n_nodes), dtype=int)
    truthlabel[comm] = 1
    prelabel = np.zeros((n_nodes), dtype=int)
    prelabel[comm_find] = 1
    score = jaccard_score(truthlabel, prelabel)
    #print("q, jac:", score)
    return score

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
    return s_m, f1_m

def get_res_path(resroot,args):
    '''
    根据args创建需要的结果路径
    :param args:
    :return:
    '''
    if args.attack == 'meta':
        return f'{resroot}{args.dataset}/{args.dataset}_{args.attack}_{args.ptb_rate}_{args.method}_res.txt'
    # elif args.attack == 'random':
    #     return f'{resroot}{args.dataset}/{args.dataset}_{args.attack}_{args.type}_{args.ptb_rate}_{args.method}_res.txt'
    # elif args.attack =='add':
    #     return f'{resroot}{args.dataset}_{args.aug}_{args.attack}_{args.noise_level}_{args.method}_res.txt'
    elif args.attack in  ['del','gflipm','gdelm','add','random_remove','random_add','random_flip','gaddm','cdelm','cflipm','delm','flipm']:
        return f'{resroot}{args.dataset}/{args.dataset}_{args.attack}_{args.ptb_rate}_{args.method}_res.txt'
    else:
        return f'{resroot}{args.dataset}/{args.dataset}_{args.method}_res.txt'

def get_model_path(model_dir,args):
    '''
    返回存储最佳模型的路径
    :param args:
    :return:
    '''
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if args.attack == 'meta':
        return f'{model_dir}{args.dataset}/{args.dataset}_{args.attack}_{args.ptb_rate}_{args.method}'
    # elif args.attack == 'random': #random attack
    #     return f'{model_dir}{args.dataset}/{args.dataset}_{args.attack}_{args.type}_{args.ptb_rate}_{args.method}'
    # elif args.attack =='add': #noisy graph
    #     return f'{model_dir}{args.dataset}_{args.attack}_{args.noise_level}_{args.method}.pkl'
    elif args.attack in  ['random_add','random_remove','random_flip','del','add','gflipm','gdelm','gaddm','cdelm','cflipm','delm','flipm']: #incomplete graph
        return f'{model_dir}{args.dataset}/{args.dataset}_{args.attack}_{args.ptb_rate}_{args.method}'
    else: #不要攻击
        return f'{model_dir}{args.dataset}/{args.dataset}_{args.method}'
