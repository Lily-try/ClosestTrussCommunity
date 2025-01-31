import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.utils as utils
'''
这个是很久以前自己想使用的
'''
class EWNet(nn.Module):
    def __init__(self,embedding_dim, hidden_dim,edge_index,features,args,device='cpu'):
        super(EWNet, self).__init__()
        # 2个线性层处理链接特征
        self.linear_link_1 = nn.Linear(embedding_dim,hidden_dim)  # Linear(in_features = 1433,out_features = 64,bias =True)
        self.linear_link_2 = nn.Linear(hidden_dim, hidden_dim)  # Linear(64,64)

        # 用于处理Adamaic-Adar (AA)索引特征
        self.linear_aa_1 = nn.Linear(embedding_dim, hidden_dim)  # Linear(in_features = 1433,out_features = 64,bias =True)
        self.linear_aa_2 = nn.Linear(hidden_dim, hidden_dim)  # Linear(64,64)

        #随机初始化权重参数。
        self.emb_weight = nn.Parameter(torch.rand(1))
        self.structure_weight = nn.Parameter(torch.rand(1))

        # 定义激活函数
        self.relu = nn.ReLU()  # ReLU()
        self.sigmoid = nn.Sigmoid()  # Sigmoid()
        self.softmax = nn.Softmax()
        self.args = args
        self.device = device
        self.pre_weights = None #初始化估计的节点对权重
        self.poten_edge=self.get_poten_edge(edge_index,features,args.n_p)  #函数末尾将其移到了设备上。
        self.features_diff = torch.cdist(features, features, 2)  # 计算特征间的距离

    def get_poten_edge(self, edge_index, features, n_p):  # 传入的是所有的边tensor，传出的是潜在可能的边tensor
        '''
            # 根据节点特征相似度计算并返回潜在的边，以增强图的连接性。只在初始时调用嘛？
        :param edge_index:
        :param features:
        :param n_p:
        :return:
        '''
        if n_p == 0:
            return edge_index  # 如果没有额外潜在边，返回原边索引
        poten_edges = []  # 存储潜在边的列表
        for i in range(len(features)):  # 遍历每个节点
            # 计算与所有节点的相似度
            sim = torch.div(torch.matmul(features[i], features.T), features[i].norm() * features.norm(dim=1))
            _, indices = sim.topk(n_p)  # 获取前n_p个最相似的节点索引
            poten_edges.append([i, i])  # 添加自环
            indices = set(indices.cpu().numpy())  # 将tensor转换为集合
            indices.update(edge_index[1, edge_index[0] == i])  # 合并现有的边索引
            for j in indices:
                if j > i:
                    pair = [i, j]
                    poten_edges.append(pair)  # 只添加有序对，避免重复
        # 转换为tensor的格式
        poten_edges = torch.as_tensor(poten_edges).T
        # 进一步转换为无向边并移到指定设备上
        poten_edges = utils.to_undirected(poten_edges, len(features)).to(self.device)

        print('poten_edge: ', poten_edges.size())

        return poten_edges

    def forward(self, emb, edge_index, aa, comm=None):  # comm，即q所在的社区内的节点。
        '''

        :param emb: 节点嵌入
        :param edge_index: 边索引
        :param aa: 结构相似性指标
        :param comm: 可选的社区信息
        :return:
        '''
        # 节点在“链接”上下文中的特征表示。一个矩阵，每行表示一个节点的特征。
        # 属性相似性到底是输入X还是输入H，这个要调整init时的emb维度，这里先用H
        latent_link = self.relu(self.linear_link_1(emb))
        latent_link = self.linear_link_2(latent_link)  # （2708，64）
        # aa结构相似性指标
        latent_aa = self.relu(self.linear_aa_1(emb))
        latent_aa = self.linear_aa_2(latent_aa)  # （2708，64）

        # 计算节点间的链接相似度，得到相似度矩阵
        link_context = self.sigmoid(latent_link @ latent_link.T)  # （2708，2708）
        # 乘以aa，将节点的特征相似性和aa指标（网络结构相似性）结合起来了
        aa_context = self.sigmoid((latent_aa @ latent_aa.T) * aa)  # （2708，2708）

        # 计算链接相似性和结构相似性的重构损失
        rec_loss = self.reconstruct_loss(edge_index, link_context, aa_context, comm)

        # 从矩阵中提取感兴趣的边的相似性分数
        edges_link_scores = link_context[self.poten_edge[0], self.poten_edge[1]]
        edges_aa_scores = aa_context[self.poten_edge[0], self.poten_edge[1]]

        # 对两种得分进行加权求和并应用sigmoid激活函数
        pre_weights = self.sigmoid(self.emb_weight * edges_link_scores + self.structure_weight * edges_aa_scores)
        pre_weights = F.relu(pre_weights)  # 保证权重非负

        # 将小于删除阈值的权重设置为0
        pre_weights = torch.where(pre_weights < self.args.t_delete, torch.tensor(0.0, device=pre_weights.device),pre_weights)
        # self.pre_weights[self.pre_weights < self.args.t_delete] = 0.0

        # 更新模型内部的预测权重pre_weights
        self.pre_weights = pre_weights

        return rec_loss

    def reconstruct_loss(self, edge_index, link_context, aa_context, comm=None):

        # 1. 随机采样图中不存在的边作为负样本边
        num_nodes = link_context.shape[0]  # 节点总数
        # 随机生成负采样边（图中不存在的边），用于损失计算
        randn = utils.negative_sampling(edge_index, num_nodes=num_nodes, num_neg_samples=self.args.n_n * num_nodes)
        randn = randn[:, randn[0] < randn[1]]  # 确保边有序
        edge_index = edge_index[:, edge_index[0] < edge_index[1]]  # 过滤保证有序

        # 从矩阵中提取感兴趣的边的相似性分数
        neg_link_scores = link_context[randn[0], randn[1]]
        neg_aa_scores = aa_context[randn[0], randn[1]]
        # 加权求和并应用sigmoid激活函数
        neg = self.sigmoid(self.emb_weight * neg_link_scores + self.structure_weight * neg_aa_scores)
        # 计算负边损失
        neg_loss = torch.exp(torch.pow(self.features_diff[randn[0], randn[1]] / self.args.sigma, 2)) @ F.mse_loss(
            neg, torch.zeros_like(neg), reduction='none')

        # 2. 图中已经存在的边（正样本）的损失
        pos_link_scores = link_context[edge_index[0], edge_index[1]]
        pos_aa_scores = aa_context[edge_index[0], edge_index[1]]
        pos = self.sigmoid(self.emb_weight * pos_link_scores + self.structure_weight * pos_aa_scores)
        # 计算正边损失-考虑的是查询节点的。
        pos_loss = torch.exp(
            -torch.pow(self.features_diff[edge_index[0], edge_index[1]] / self.args.sigma, 2)) @ F.mse_loss(pos,
                                                                                                            torch.ones_like(
                                                                                                                pos),
                                                                                                            reduction='none')

        # 计算总的重构损失，还有一个元数据集上的损失这里没有加上来。
        rec_loss = (pos_loss + neg_loss) * num_nodes / (randn.shape[1] + edge_index.shape[1])
        return rec_loss