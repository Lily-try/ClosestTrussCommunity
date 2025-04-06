'''相关的参数配置'''
import argparse

def get_config():
    parser = argparse.ArgumentParser()
    # 重复次数，论文中常用的重复5次取平均
    parser.add_argument('--count', type=int, default=1)
    parser.add_argument('--root', type=str, default='./data')
    # parser.add_argument('--res_root', type=str, default='./results/', help='result path')
    # parser.add_argument("--log", action='store_true', help='run prepare_data or not')
    parser.add_argument("--log", type=bool,default=True, help='run prepare_data or not')
    # 训练完毕的模型的存储路径
    parser.add_argument('--method',type=str,default='Grad',choices=['EmbLearner','EmbLearnerWithoutHyper','EmbLearnerwithWeights','knn','dyknn','Grad'])
    parser.add_argument('--model_path', type=str, default='CS')
    parser.add_argument('--m_model_path', type=str, default='META')

    # 数据集选项
    parser.add_argument('--dataset', type=str, default='cora')
    # 训练集、验证集、测试集大小，以及相应的文件路径，节点特征存储路径
    parser.add_argument('--train_size', type=int, default=300)
    parser.add_argument('--val_size', type=int, default=100)
    parser.add_argument('--test_size', type=int, default=500)
    #根据命名的不同，来控制我想使用的标签比例
    parser.add_argument('--train_path', type=str, default='3_pos_train')
    parser.add_argument('--test_path', type=str, default='3_test')
    parser.add_argument('--val_path', type=str, default='3_val')
    parser.add_argument('--feats_path', type=str, default='feats.txt')
    #增强类型
    parser.add_argument('--aug',type=str,default='knn',choices=['knn','hyper','none','dyknn'])
    parser.add_argument('--topk',type=int,default=10)
    # 控制攻击方法、攻击类型和攻击率
    parser.add_argument('--attack', type=str, default='none', choices=['none', 'random', 'meta_attack','add','del','gflipm','gdelm'])
    parser.add_argument('--type', type=str, default='add', help='random attack type', choices=['add', 'remove', 'flip'])
    parser.add_argument('--noise_level', type=int, default=2, choices=[1, 2, 3], help='noisy level')
    parser.add_argument('--ptb_rate', type=float, default=0.30, help='pertubation rate')

    # 模型batch大小，隐藏层维度，训练epoch数，drop_out，学须率lr，权重衰减weight_decay
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--epoch_n', type=int, default=10) #对于大的数据集需要设置为10
    parser.add_argument('--drop_out', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.001)  # 原文默认的是0.001，调整大一些0.1。
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--patience', default=10, type=int, help='patience for early stopping (default 10)') #早停的容忍度
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

    return parser.parse_args()