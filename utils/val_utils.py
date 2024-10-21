import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, jaccard_score

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