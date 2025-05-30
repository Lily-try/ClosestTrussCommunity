import os
import pathlib
import pickle
import time

import networkx as nx
from deeprobust.graph.global_attack import Random
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
import argparse

'''
基于random attack的插入边
'''

# 开始计时
start_time = time.time()

from utils import citation_loader
from preprocess import txt_utils

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--dataset', type=str, default='cocs', choices=['cora','citeseer','cocs','football','facebook_all','cora_ml', 'polblogs', 'pubmed'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.4,  help='pertubation rate')
parser.add_argument('--type', type=str, default='add',  help=
'attack type',choices=['add','remove','flip'])
parser.add_argument('--root', type=str, default='../data',  help='data store root')
#配置
args = parser.parse_args()
args.cuda = torch.cuda.is_available()
print('cuda: %s' % args.cuda)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#固定随机数种子
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

#加载数据集,只需要csr_matricx:(2485,2485)}格式的邻接矩阵即可
dataset = args.dataset
if dataset in ['football','facebook_all']:
   adj= txt_utils.load_txt_adj(args.root, dataset)
    # 读取features
    # 读取labels
if dataset in ['cora', 'citeseer', 'pubmed']:#引文网络，deeprobust本身就有的
    #从pyg中的原始数据集中读取数据
    graph = citation_loader.citation_graph_reader(args.root, args.dataset)  # 读取图 nx格式的
    print(graph)
    adj = nx.adjacency_matrix(graph)  # 转换为CSR格式的稀疏矩阵
    #弃用下面两行从deeprobust库中读取数据集的操作。
    # data = Dataset(root=os.path.join('../data',dataset), name=args.dataset)
    # adj, features, labels = data.adj, data.features, data.labels
# if dataset in ['dblp','amazon']:#sanp数据集上的
#     edge, labels = snap_utils.load_snap(args.root, data_set='com_' + dataset, com_size=3)  # edge是list:1049866
    #将edge转换成csr_matrix
elif dataset in ['cocs']:
    graphx = nx.Graph()
    with open(f'{args.root}/{args.dataset}/{args.dataset}.edges', "r") as f:
        for line in f:
            node1, node2 = map(int, line.strip().split())
            graphx.add_edge(node1, node2)
    print(f'{args.dataset}:', graphx)
    adj = nx.adjacency_matrix(graphx)  # 转换为CSR格式的稀疏矩阵
#使用randomAttack进行攻击
model = Random()

n_perturbations = int(args.ptb_rate * (adj.sum()//2))
'''
#!!!!注意事项，这里的attack的type默认是ADD攻击，即注入噪声边
type: str,perturbation type. Could be 'add', 'remove' or 'flip'.
'''
model.attack(adj, n_perturbations,type=args.type)
modified_adj = model.modified_adj

#存储攻击后的adj
modified_adj = modified_adj.tocsr() #lil_matraix:(2485,2485)}-->csr_matrix
#存储成npz格式.
path = os.path.join(args.root, args.dataset,f'random_{args.type}')
name=f'{args.dataset}_random_{args.type}_{args.ptb_rate}'
sp.save_npz(os.path.join(path, name), modified_adj)

# 结束计时
end_time = time.time()
execution_time = end_time - start_time
# 打印总用时
print(f'Total execution time: {execution_time:.4f} seconds')

# 日志文件路径
log_file_path ='../data/log/execution_log.txt'
# 创建或追加到日志文件
with open(log_file_path, 'a') as log_file:
    log_file.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write(f"Script: {__file__}\n")
    log_file.write(f"Arguments: Seed={args.seed}, Dataset={args.dataset}, Perturbation Rate={args.ptb_rate}\n")
    log_file.write(f"Execution Time: {execution_time:.4f} seconds\n")
    log_file.write("--------------------------------------------------------\n")

# 存储修改后的邻接矩阵
# name =pathlib.Path("%s_%s_%s.pkl" % (args.dataset, 'random', args.ptb_rate))
# path = os.path.join(args.root, args.dataset,name)
# with open(path, 'wb') as f:
#     pickle.dump(modified_adj, f)

# with open(os.path.join(path), 'rb') as f:
#     modified_adj = pickle.load(f)
# modified_adj = [i for i in modified_adj]  #就变成list了，依旧可以转换成nx的格式。