import os.path as osp
import sys
from datetime import datetime
from time import time

sys.path.append((osp.abspath(osp.dirname(__file__)).split('src')[0] + 'src'))
import os
import numpy as np
import scipy.sparse as sp
from utils import *


def dgl_to_scipy_adj(g):
    src, dst = g.edges()
    num_nodes = g.num_nodes()
    adj = sp.coo_matrix(
        (np.ones(len(src)), (src.cpu().numpy(), dst.cpu().numpy())),
        shape=(num_nodes, num_nodes)
    )
    return adj
def get_edge_set(graph):
    """返回图的无向边集合（去重后）,用于记录修改的边"""
    src, dst = graph.edges()
    edges = set()
    for u, v in zip(src.tolist(), dst.tolist()):
        if u != v:
            edges.add(tuple(sorted((u, v))))
    return edges


def log_time_to_file(log_path, label, duration):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'a') as f:
        f.write(f"{label}: {duration:.4f} seconds\n")
@time_logger
def train_GSR(args):
    # ! Init Environment
    exp_init(args.seed if hasattr(args, 'seed') else 0, args.gpu)

    # ! Import packages
    # Note that the assignment of GPU-ID must be specified before torch/dgl is imported.
    import torch as th
    import dgl
    from utils.data_utils import preprocess_data
    from utils.early_stopper import EarlyStopping

    from models.GSR.GSR import GSR_pretrain, GSR_finetune, para_copy
    from models.GSR.config import GSRConfig
    from models.GSR.data_utils import get_pretrain_loader, get_structural_feature
    from models.GSR.cl_utils import MemoryMoCo, moment_update, NCESoftmaxLoss
    from models.GSR.trainer import FullBatchTrainer
    from models.GSR.trainGSR import train_GSR
    from models.GSR.PolyLRDecay import PolynomialLRDecay
    from time import time


    # ! Config
    cf = GSRConfig(args)
    cf.device = th.device("cuda:0" if args.gpu >= 0 else "cpu")

    t_data_load = time()
    # ！加载图 在utils/data_utils文件中
    g, features, cf.n_feat, cf.n_class, labels, train_x, val_x, test_x,num_train_nodes = \
        preprocess_data(cf.root,cf.dataset,cf.attack,cf.ptb_rate, cf.train_percentage)
    #存入预处理时间
    print(f"[Time] Data loading and preprocessing: {time() - t_data_load:.2f} seconds")
    t_data_load = time()-t_data_load
    feat = {'F': features, 'S': get_structural_feature(g, cf)}
    cf.feat_dim = {v: feat.shape[1] for v, feat in feat.items()}
    supervision = SimpleObject({'train_x': train_x, 'val_x': val_x, 'test_x': test_x, 'labels': labels})


    t_train_start = time()
    # ! Train Init
    print(f'{cf}\nStart training..')
    p_model = GSR_pretrain(g, cf).to(cf.device)
    # ! Train Phase 1: Pretrain
    if cf.p_epochs > 0:
        # os.remove(cf.pretrain_model_ckpt)  # Debug Only
        if os.path.exists(cf.pretrain_model_ckpt):

            p_model.load_state_dict(th.load(cf.pretrain_model_ckpt, map_location=cf.device))
            print(f'Pretrain embedding loaded from {cf.pretrain_model_ckpt}')
        else:
            print(f'>>>> PHASE 1 - Pretraining and Refining Graph Structure <<<<<')
            views = ['F', 'S']
            optimizer = th.optim.Adam(
                p_model.parameters(), lr=cf.prt_lr, weight_decay=cf.weight_decay)
            if cf.p_schedule_step > 1:
                scheduler_poly_lr_decay = PolynomialLRDecay(optimizer, max_decay_steps=cf.p_schedule_step,
                                                            end_learning_rate=0.0001, power=2.0)
            # Construct virtual relation triples
            p_model_ema = GSR_pretrain(g, cf).to(cf.device)
            moment_update(p_model, p_model_ema, 0)  # Copy
            moco_memories = {v: MemoryMoCo(cf.n_hidden, cf.nce_k,  # Single-view contrast
                                           cf.nce_t, device=cf.device).to(cf.device)
                             for v in views}
            criterion = NCESoftmaxLoss(cf.device)
            pretrain_loader = get_pretrain_loader(g.cpu(), cf)

            for epoch_id in range(cf.p_epochs):
                for step, (input_nodes, edge_subgraph, blocks) in enumerate(pretrain_loader):
                    t0 = time()
                    blocks = [b.to(cf.device) for b in blocks]
                    edge_subgraph = edge_subgraph.to(cf.device)
                    # input_feature = {v: feat[v][input_nodes].to(cf.device) for v in views}
                    input_feature = {v: feat[v][input_nodes.cpu()].to(cf.device) for v in views}

                    # ===================Moco forward=====================
                    p_model.train()

                    q_emb = p_model(edge_subgraph, blocks, input_feature, mode='q')
                    std_dict = {v: round(q_emb[v].std(dim=0).mean().item(), 4) for v in ['F', 'S']}
                    print(f"Std: {std_dict}")

                    if std_dict['F'] == 0 or std_dict['S'] == 0:
                        print(f'\n\n????!!!! Same Embedding Epoch={epoch_id}Step={step}\n\n')
                        # q_emb = p_model(edge_subgraph, blocks, input_feature, mode='q')

                    with th.no_grad():
                        k_emb = p_model_ema(edge_subgraph, blocks, input_feature, mode='k')
                    intra_out, inter_out = [], []

                    for tgt_view, memory in moco_memories.items():
                        for src_view in views:
                            if src_view == tgt_view:
                                intra_out.append(memory(
                                    q_emb[f'{tgt_view}'], k_emb[f'{tgt_view}']))
                            else:
                                inter_out.append(memory(
                                    q_emb[f'{src_view}->{tgt_view}'], k_emb[f'{tgt_view}']))

                    # ===================backward=====================
                    # ! Self-Supervised Learning
                    intra_loss = th.stack([criterion(out_) for out_ in intra_out]).mean()
                    inter_loss = th.stack([criterion(out_) for out_ in inter_out]).mean()
                    # ! Loss Fusion
                    loss_tensor = th.stack([intra_loss, inter_loss])
                    intra_w = float(cf.intra_weight)
                    loss_weights = th.tensor([intra_w, 1 - intra_w], device=cf.device)
                    loss = th.dot(loss_weights, loss_tensor)
                    # ! Semi-Supervised Learning
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    moment_update(p_model, p_model_ema, cf.momentum_factor)
                    print_log({'Epoch': epoch_id, 'Batch': step, 'Time': time() - t0,
                               'intra_loss': intra_loss.item(), 'inter_loss': inter_loss.item(),
                               'overall_loss': loss.item()})

                    if cf.p_schedule_step > 1:
                        scheduler_poly_lr_decay.step()

                epochs_to_save = P_EPOCHS_SAVE_LIST + ([1, 2, 3, 4] if args.dataset == 'arxiv' else [])
                if epoch_id + 1 in epochs_to_save:
                    # Convert from p_epochs to current p_epoch checkpoint
                    ckpt_name = cf.pretrain_model_ckpt.replace(f'_pi{cf.p_epochs}', f'_pi{epoch_id + 1}')
                    th.save(p_model.state_dict(), ckpt_name)
                    print(f'Model checkpoint {ckpt_name} saved.')

            th.save(p_model.state_dict(), cf.pretrain_model_ckpt)

    train_time = time()-t_train_start
    print(f'Training time',train_time)

    # ! Train Phase 2: Graph Structure Refine 进行图结构细化
    print(f'>>>> PHASE 2 - Graph Structure Refine <<<<< ')
    t_refine_start = time()
    if cf.p_epochs <= 0 or cf.add_ratio + cf.rm_ratio == 0:
        print('Use original graph!')
        g_new = g
    else:
        if os.path.exists(cf.refined_graph_file):
            print(f'Refined graph loaded from {cf.refined_graph_file}')
            g_new = dgl.load_graphs(cf.refined_graph_file)[0][0]
        else:
            g_new = p_model.refine_graph(g, feat)
            dgl.save_graphs(cf.refined_graph_file, [g_new])
    #将优化后的图邻接矩阵存入文件
    adj_clean = dgl_to_scipy_adj(g_new)
    if cf.attack == 'none':
        clean_path = f'{cf.root}/{cf.dataset}_gsr/{cf.attack}/{cf.dataset}_gsr_raw.npz'
    else:
        clean_path = f'{cf.root}/{cf.dataset}_gsr/{cf.attack}/{cf.dataset}_gsr_{cf.attack}_{cf.ptb_rate}.npz'
    os.makedirs(os.path.dirname(clean_path), exist_ok=True)
    sp.save_npz(clean_path, adj_clean)

    # 计算边修改数量
    old_edges = get_edge_set(g)
    new_edges = get_edge_set(g_new)
    added_edges = new_edges - old_edges  #添加的边
    removed_edges = old_edges - new_edges #删除的边
    num_added = len(added_edges)
    num_removed = len(removed_edges)
    num_modified = num_added + num_removed
    print(f"[Graph Changes] Added edges: {num_added}, Removed edges: {num_removed}, Total changes: {num_modified}")

    refine_time = time() - t_refine_start
    print(f"[Time] Graph refinement: {refine_time:.2f} seconds")

    # 存入时间
    log_path = f'{cf.root}/{cf.dataset}_gsr/{cf.attack}/{cf.dataset}_gsr_{cf.attack}_{cf.ptb_rate}_time_stats.txt'
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'a+', encoding='utf-8') as f:
        line = (
            f'load_time:{t_data_load}\n'
            f'train_time:{train_time}\n'
            f'refine_time:{refine_time}\n'
            f'train+refine_time:{train_time + refine_time}\n'
            f'num_train_nodes:{num_train_nodes}\n'
            f'train_size:{train_x.shape},val_size:{val_x.shape}, test_size{test_x.shape}\n'
            f'added_edges:{num_added}\n'
            f'removed_edges:{num_removed}\n'
            f'add+removed_edges:{num_modified}\n'
        )
        f.write(line)
        f.close()


    # ! Train Phase 3:  Node Classification
    classification_start = time()
    f_model = GSR_finetune(cf).to(cf.device)
    print(f_model)
    # Copy parameters
    if cf.p_epochs > 0:
        para_copy(f_model, p_model.encoder.F, paras_to_copy=['conv1.weight', 'conv1.bias'])
    optimizer = th.optim.Adam(f_model.parameters(), lr=cf.lr, weight_decay=cf.weight_decay)
    stopper = EarlyStopping(patience=cf.early_stop, path=cf.checkpoint_file) if cf.early_stop else None
    del g, feat, p_model
    th.cuda.empty_cache()

    print(f'>>>> PHASE 3 - Node Classification <<<<< ')
    trainer_func = FullBatchTrainer
    trainer = trainer_func(model=f_model, g=g_new, features=features, sup=supervision, cf=cf,
                           stopper=stopper, optimizer=optimizer, loss_func=th.nn.CrossEntropyLoss())
    trainer.run()
    trainer.eval_and_save()

    classification_time = time() - classification_start

    # 存入时间
    log_path = f'{cf.root}/{cf.dataset}_gsr/{cf.attack}/{cf.dataset}_gsr_{cf.attack}_{cf.ptb_rate}_time_stats.txt'
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'a+', encoding='utf-8') as f:
        line = (
            f'classification_time:{classification_time}\n'
        )
        f.write(line)
        f.close()

    return cf


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training settings")
    # dataset = 'citeseer'
    # dataset = 'arxiv'
    dataset = 'cora'
    # dataset = 'flickr'
    # dataset = 'airport'
    # dataset = 'blogcatalog'
    # ! Settings
    parser.add_argument("-g", "--gpu", default=1, type=int, help="GPU id to use.")
    parser.add_argument('-r','--root', type=str, default='data')
    # parser.add_argument("-d", "--dataset", type=str, default=dataset)
    #choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed','cocs','facebook','reddit'],
    parser.add_argument('-d','--dataset', type=str, default='cora', help='dataset')
    parser.add_argument('-a','--attack', type=str, default='del',choices=['none','random_add','random_remove','random_flip','flipm','cdelm','gflipm', 'gdelm', 'gaddm','del','add', 'random', 'random_attack', 'mettack'],help='attack method')
    parser.add_argument('-p','--ptb_rate', type=float, default=0.3, help='pertubation rate')
    parser.add_argument("-t", "--train_percentage", default=0.1, type=float)  #用于训练的比例
    parser.add_argument("-e", "--early_stop", default=100, type=int)
    parser.add_argument("--epochs", default=1000, type=int)
    parser.add_argument("--seed", default=0)
    args = parser.parse_args()
    #
    # if '192.168.0' in get_ip():
    #     args.gpu = -1
    #     args.dataset = args.dataset if args.dataset != 'arxiv' else 'cora'
    # ! Train
    run_start = time()
    cf = train_GSR(args)
    running_time = time() - run_start

    # 获取当前时间字符串
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('my running_time:', running_time)
    log_path = f'{cf.root}/{cf.dataset}_gsr/{cf.attack}/{cf.dataset}_gsr_{cf.attack}_{cf.ptb_rate}_time_stats.txt'
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'a+', encoding='utf-8') as f:
        line = (
            f'current_time:{current_time}\n'
            f'my_running_time:{running_time}\n'
            '--------------------------------------------\n'
            '\n'
        )
        f.write(line)
        f.close()


# python /home/zja/PyProject/MGSL/src/models/GSR/trainGSR.py -darxiv
