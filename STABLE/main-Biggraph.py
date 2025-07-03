import torch.nn as nn
import time
import argparse

from citation_loader import citation_feature_reader, citation_graph_reader
# from scipy.sparse import csr_matrix

from utils import *
from model import GCN, LogReg
from copy import deepcopy
import scipy
from robcon import get_contrastive_emb
import networkx as nx
import os

''''
想办法在大图上使用
'''

def load_features(root,dataset):
    if dataset in ['cora', 'pubmed', 'citeseer']:
        nodes_feats = citation_feature_reader(root, dataset)  # numpy.ndaaray:(2708,1433)
        nodes_feats = scipy.sparse.csr_matrix(nodes_feats)
    elif dataset in ['cocs', 'photo', 'dblp', 'physics', 'reddit', 'texas', 'wisconsin']:
        with open(f'{root}/{dataset}/{dataset}.feats', "r") as f:
            # 每行特征转换为列表，然后堆叠为 ndarray
            nodes_feats = np.array([list(map(float, line.strip().split())) for line in f])
            nodes_feats = scipy.sparse.csr_matrix(nodes_feats)
    elif dataset.startswith(('fb', 'wfb', 'fa')):  # 不加入中心节点
        feats_array = np.loadtxt(f'{args.root}/{args.dataset}/{args.dataset}.feat', delimiter=' ', dtype=np.float32)
        nodes_feats = scipy.sparse.csr_matrix(feats_array)
        # nodes_feats = fnormalize(feats_array)  # 将特征进行归一化

    else: #这个是源码自己使用的
        nodes_feats = scipy.sparse.load_npz('./ptb_graphs/%s_features.npz' % (dataset))
    print(f'{dataset} nodes_feats type:{type(nodes_feats)}, nodes_feats shape:{nodes_feats.shape}')
    return nodes_feats

def load_adj(root,dataset,attack): #torch.Tenso
    if attack == 'none':  # 使用原始数据
        if args.dataset in ['cora', 'pubmed', 'citeseer']:
            graphx = citation_graph_reader(args.root, args.dataset)  # 读取图 nx格式的
            print(graphx)
            n_nodes = graphx.number_of_nodes()
        elif args.dataset in ['cocs','photo','dblp']:
            graphx = nx.Graph()
            with open(f'{args.root}/{args.dataset}/{args.dataset}.edges', "r") as f:
                for line in f:
                    node1, node2 = map(int, line.strip().split())
                    graphx.add_edge(node1, node2)
            print(f'{args.dataset}:', graphx)
            n_nodes = graphx.number_of_nodes()
        elif args.dataset in ['fb107']:
            graphx = nx.read_edgelist(f'{root}/{dataset}/{dataset}.edges', nodetype=int, data=False)
            print(graphx)
            n_nodes = graphx.number_of_nodes()
    elif args.attack in ['del','random_remove','random_add','random_flip','flipm','cdelm','cflipm','gflipm', 'gdelm', 'add','gaddm']:
        path = os.path.join(args.root, args.dataset, args.attack,
                            f'{args.dataset}_{args.attack}_{args.ptb_rate}.npz')
        adj_csr_matrix = sp.load_npz(path)
        graphx = nx.from_scipy_sparse_array(adj_csr_matrix)
        print(graphx)
        n_nodes = graphx.number_of_nodes()
    else:
        # perturbed_adj = torch.load('./ptb_graphs/%s/%s_%s_%s.pt' % (args.attack, args.attack, args.dataset, args.ptb_rate))
        print('噪声类型不匹配')
        path = os.path.join(args.root, args.dataset, args.attack,
                            f'{args.dataset}_{args.attack}_{args.ptb_rate}.npz')
        adj_csr_matrix = sp.load_npz(path)
        graphx = nx.from_scipy_sparse_array(adj_csr_matrix)
        print(graphx)
        n_nodes = graphx.number_of_nodes()
    return graphx


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='../data')
    # choices=['fb107','cora','cocs','cora_ml', 'citeseer', 'polblogs', 'pubmed'],
    parser.add_argument('--dataset', type=str, default='cora', help='dataset')
    #choices=['none','gflipm', 'gdelm', 'gaddm','flipm','cdelm','cflipm','delm','del','add', 'random_remove', 'random_add','random_flip', 'mettack'],
    parser.add_argument('--attack', type=str, default='flipm',help='attack method')
    parser.add_argument('--ptb_rate', type=float, default=0.4, help='pertubation rate')
    parser.add_argument('--threshold', type=float, default=1, help='threshold')
    parser.add_argument('--jt', type=float, default=0.03, help='jaccard threshold')
    parser.add_argument('--cos', type=float, default=0.1, help='cosine similarity threshold')
    parser.add_argument('--k', type=int, default=3, help='add k neighbors')
    parser.add_argument('--alpha', type=float, default=0.3, help='add k neighbors')
    parser.add_argument('--beta', type=float, default=2, help='the weight of selfloop')
    parser.add_argument("--log", action='store_true', help='run prepare_data or not')

    start_time = time.time() #记录开始时间
    args = parser.parse_args()
    if args.log:
        logger = get_logger(f'./log/{args.attack}/{args.dataset}_stb_{args.attack}_{args.ptb_rate}.log')
    else:
        logger = get_logger('./log/try.log')

    if args.attack == 'nettack':
        args.ptb_rate = int(args.ptb_rate)

    # 设置随机数种子
    seed = int(time.time())
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info(args)

    #加载数据
    dataset = args.dataset
    ptb_rate = args.ptb_rate
    #①加载节点特征  feature type:<class 'scipy.sparse._csr.csr_matrix'>, feature shape:(2485, 1433)
    # features = load_features(args.root, args.dataset)
    feat_sp = load_features(args.root, args.dataset)

    feat_coo = feat_sp.tocoo()
    features = torch.sparse_coo_tensor(
        indices=torch.vstack((torch.LongTensor(feat_coo.row),
                              torch.LongTensor(feat_coo.col))),
        values=torch.FloatTensor(feat_coo.data),
        size=feat_coo.shape,
        device=device,
        dtype=torch.float16  # 若有 fp16 支持
    ).coalesce()

    print(f'feature type:{type(features)}, feature shape:{features.shape}')
    #② 加载扰动邻接矩阵 perturbed_adj type:<class 'torch.Tensor'>, perturbed_adj shape:torch.Size([2485, 2485])
    graphx = load_adj(args.root,args.dataset,args.attack)
    #显示增加孤立节点，避免邻接矩阵和节点特征矩阵的维度不匹配
    total_nodes = features.shape[0]
    graphx.add_nodes_from(range(total_nodes))
    # adj = nx.to_scipy_sparse_array(graphx, format='csr')

    # adj_matrix = nx.adjacency_matrix(graphx)   #将其转换成tensor

    adj_sp = nx.adjacency_matrix(graphx).tocoo()
    perturbed_adj = torch.sparse_coo_tensor(
        indices=torch.vstack((torch.LongTensor(adj_sp.row),
                              torch.LongTensor(adj_sp.col))),
        values=torch.ones(adj_sp.nnz, dtype=torch.float16, device=device),
        size=adj_sp.shape,
        device=device,
    ).coalesce()

    # if args.dataset == 'dblp':
    #     # 将 scipy.sparse 矩阵转换为 PyTorch 的稀疏张量
    #     adj_matrix_coo = adj_matrix.tocoo()  # 转为 COO 格式，便于构建稀疏张量
    #     indices = torch.tensor([adj_matrix_coo.row, adj_matrix_coo.col], dtype=torch.long)
    #     values = torch.tensor(adj_matrix_coo.data, dtype=torch.float32)
    #     shape = torch.Size(adj_matrix_coo.shape)
    #     perturbed_adj = torch.sparse_coo_tensor(indices, values, shape)
    # else:
    #     dense_matrix = adj_matrix.toarray()  # 将稀疏矩阵转换为稠密矩阵
    #     perturbed_adj = torch.tensor(dense_matrix, dtype=torch.float32)  # 将稠密矩阵转换为 PyTorch 张量

    print(f'perturbed_adj type:{type(perturbed_adj)}, perturbed_adj shape:{perturbed_adj.shape}')
    # 加载标签
    # labels = np.load('./ptb_graphs/%s_labels.npy' % (args.dataset))
    # n_nodes = features.shape[0]
    # n_class = labels.max() + 1

    #csr_matrix:(2485,2485) of numpy float32
    perturbed_adj_sparse = to_scipy(perturbed_adj) #将扰动后的邻接矩阵 perturbed_adj 转换为 scipy 稀疏矩阵形式，以便更高效地进行操作。

    #####图预处理(粗略去噪）得到adj_pre，adj_delete记录了被预处理删除的边
    logger.info('===start preprocessing the graph===')
    if args.dataset == 'polblogs':
        args.jt = 0 #如果选择的数据集是 polblogs，则将 args.jt 设为 0
    adj_pre,removed_cnt_pre = preprocess_adj(feat_sp, perturbed_adj_sparse, logger, threshold=args.jt)
    adj_delete = perturbed_adj_sparse - adj_pre
    # _, features = to_tensor(perturbed_adj_sparse, features) #将邻接矩阵和特征矩阵转换为张量（tensor），以便使用深度学习框架（如 PyTorch）进行后续操作。

    # _, features = to_tensor(perturbed_adj_sparse, features)
    # features = features.coalesce()  # 保持稀疏格式
    # features = features.to(device, torch.float16)  # 半精度再省 2× 显存

    ######将开始获取对比学习嵌入
    logger.info('===start getting contrastive embeddings===')
    # embeds, _ = get_contrastive_emb(logger, adj_pre, features.unsqueeze(dim=0).to_dense(), adj_delete=adj_delete,
    #                                 lr=0.001, weight_decay=0.0, nb_epochs=10000, beta=args.beta,dataset=args.dataset)

    embeds, _ = get_contrastive_emb(logger, adj_pre, features, adj_delete=adj_delete,
                                    lr=0.001, weight_decay=0.0, nb_epochs=10000, beta=args.beta,dataset=args.dataset)

    if embeds.dim() == 3:
        embeds = embeds.squeeze(dim=0) #去掉第一个维度即最终的节点嵌入
    embeds = embeds.to('cpu')
    # embeds = to_scipy(embeds) #将 embeds 转换为 scipy 稀疏矩阵。

    embeds = sp.csr_matrix(embeds.cpu().numpy())

    #利用得到的节点嵌入清理图得到adj_clean，（清理即剪枝，余弦相似度（args.cos）作为剪枝的阈值）
    adj_clean,removed_cnt_clean = preprocess_adj(embeds, perturbed_adj_sparse, logger, jaccard=False, threshold=args.cos)


    #添加topk的邻居
    embeds = torch.FloatTensor(embeds.todense()).to(device)
    adj_clean = sparse_mx_to_sparse_tensor(adj_clean).to_dense().to(device)  # 转为 dense Tensor
    adj_temp = adj_clean.clone()
    added_edges = get_reliable_neighbors(adj_temp, embeds, k=args.k, degree_threshold=args.threshold)
    # Step 3: 把 PyTorch dense Tensor 转回 Scipy 稀疏矩阵并保存
    adj_clean_np = adj_temp.cpu().numpy()
    adj_clean_sp = sp.csr_matrix(adj_clean_np)


    #将清理后的图存入文件
    if args.attack == 'none':
        clean_path = f'{args.root}/{args.dataset}_stb/{args.attack}/{args.dataset}_stb_raw.npz'
    else:
        clean_path = f'{args.root}/{args.dataset}_stb/{args.attack}/{args.dataset}_stb_{args.attack}_{args.ptb_rate}.npz'
    os.makedirs(os.path.dirname(clean_path), exist_ok=True)
    logger.info(f'stb graphs is saved at {clean_path} ')
    # sp.save_npz(clean_path,adj_clean)
    sp.save_npz(clean_path,adj_clean_sp)

    #结束时间
    end_time = time.time()
    duration = end_time - start_time

    # 构建记录内容
    run_info = f"""[Run Info]
    Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}
    Dataset: {args.dataset}
    Attack: {args.attack}
    Perturbation Rate: {args.ptb_rate}
    Del_Edges:{removed_cnt_pre}
    Del_clenEdges:{removed_cnt_clean}
    Add_Edges:{added_edges}
    Jaccard Threshold: {args.jt}
    Cosine Threshold: {args.cos}
    K Neighbors: {args.k}
    Alpha: {args.alpha}
    Beta: {args.beta}
    Runtime: {duration:.2f} seconds
    """

    # 保存到文件
    # log_file = './run_time_log.txt'
    log_file = f'./run_logs/{args.attack}/{args.dataset}_{args.attack}_{args.ptb_rate}_run_time.txt'
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, 'a') as f:
        f.write(run_info)

    print(f'[INFO] Runtime info saved to {log_file}')