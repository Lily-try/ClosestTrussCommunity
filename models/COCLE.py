import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, HypergraphConv
from torch_geometric.nn.inits import glorot
from torch_scatter import scatter

'''
与COCLEP保持相同设置
'''

class MLP(nn.Module):
    '''
    双层MLP，用于将节点嵌入映射到不同空间
    '''
    def __init__(self, in_dim, out_dim):
        super(MLP, self).__init__()
        self.fcs = nn.Sequential(
            nn.Linear(in_dim, out_dim), #线性层
            nn.ReLU(),    #激活函数
            nn.Linear(out_dim, out_dim), #激活函数
        )

    def forward(self, x):
        return self.fcs(x)


class Contra(nn.Module):
    '''
    根据模型输出的节点嵌入计算对比损失
    '''
    def __init__(self, device):
        super(Contra, self).__init__()
        self.device =device
    def forward(self, h, h_aug, tau, train, alpha, lam, edge_index):
        if self.training ==False:
            return h #不需要进行对比训练，直接返回模型得到的节点嵌入
        #获取训练数据
        q,pos = train

        #视图内对比:计算h[q]和h向量之间的余弦相似度
        h_q = h[q].unsqueeze(0)  # h_q=(1,256) 确保h[q]是二维的（执行乘法操作要求如此）
        intra_cosine_sim = F.cosine_similarity(h_q, h, dim=1)  # 计算余弦相似度sim1=(115,)
        intra_cosine_sim = torch.exp(intra_cosine_sim / tau)

        #增强视图的视图内对比：h_aug[q]和h_aug 向量之间的余弦相似度
        aug_intra_cosine_sim = F.cosine_similarity(h_aug[q].unsqueeze(0), h_aug, dim=1)  # sim_aug_1=(115,)
        aug_intra_cosine_sim = torch.exp(aug_intra_cosine_sim / tau)

        #原始-增强视图间对比：h[q] 和 h_aug 之间的相似度
        inter_cosine_sim = F.cosine_similarity(h[q].unsqueeze(0), h_aug, dim=1)
        inter_cosine_sim = torch.exp(inter_cosine_sim / tau)

        #增强-原始视图间对比： h_aug[q] 和 h 之间的相似度
        aug_inter_cosine_sim = F.cosine_similarity(h_aug[q].unsqueeze(0), h, dim=1)
        aug_inter_cosine_sim = torch.exp(aug_inter_cosine_sim / tau)

        #正样本掩码
        mask_p = [False] * h.shape[0]  # (115,)
        mask_p = torch.tensor(mask_p)
        mask_p.to(self.device)
        mask_p[pos] = True
        mask_p[q] = False

        # 创建用于存储更新后的节点嵌入向量
        intra_loss = torch.tensor([0.0]).to(self.device)  # (1,)
        aug_intra_loss = torch.tensor([0.0]).to(self.device)
        inter_loss = torch.tensor([0.0]).to(self.device)
        aug_inter_loss = torch.tensor([0.0]).to(self.device)

        if len(pos) !=0:
            #视图内对比学习
            intra_loss = intra_cosine_sim.squeeze(0)[mask_p] / (torch.sum(intra_cosine_sim.squeeze(0)))  # intra_loss=(3,)
            intra_loss = -torch.log(intra_loss).mean()  # 低阶视图内对比损失 标量?()，值是4.7057...
            aug_intra_loss = aug_intra_cosine_sim.squeeze(0)[mask_p] / (torch.sum(aug_intra_cosine_sim.squeeze(0)))
            aug_intra_loss = -torch.log(aug_intra_loss).mean()  # 高阶视图内对比损失

            #视图间对比损失
            inter_loss = inter_cosine_sim.squeeze(0)[mask_p] / (torch.sum(inter_cosine_sim.squeeze(0)))
            inter_loss = -torch.log(inter_loss).mean()
            aug_inter_loss = aug_inter_cosine_sim.squeeze(0)[mask_p] / (torch.sum(aug_inter_cosine_sim.squeeze(0)))
            aug_inter_loss = -torch.log(aug_inter_loss).mean()

        total_intra_loss = 0.5*(intra_loss+aug_intra_loss) #视图内对比学习
        total_inter_loss = 0.5*(inter_loss+aug_inter_loss) #视图内对比学习

        # 查询节点本身的嵌入的视图间对比损失函数进行组合得到loss_unsup
        z_unsup = -torch.log(inter_cosine_sim.squeeze(0)[q] / torch.sum(inter_cosine_sim.squeeze(0)))
        z_aug_unsup = -torch.log(aug_inter_cosine_sim.squeeze(0)[q] / torch.sum(aug_inter_cosine_sim.squeeze(0)))
        loss_unsup = 0.5 * z_unsup + 0.5 * z_aug_unsup

        #计算总的loss函数
        loss = (total_intra_loss + alpha * total_inter_loss) + lam * loss_unsup
        return loss

class COCLE(nn.Module):
    '''
    进行节点嵌入学习的神经网络模型
    '''
    def __init__(self, node_in_dim, hidden_dim, num_layers, dropout, tau, device, alpha, lam, k):
        super(COCLE, self).__init__()
        self.num_layers = num_layers #GCN的层数
        self.dropout = dropout  # 丢弃率
        self.tau = tau #
        self.alpha = alpha #视图间损失的比重
        self.lam = lam #查询节点视图间损失的比重
        self.k = k  # 增强视图的跳数
        # self.use_hypergraph =use_hypergraph
        self.device = device

        self.contra = Contra(device)  # 计算对比损失的对象实例

        #L层原始视图query encoder：创建原始视图的神经网络层
        self.query_layers = nn.ModuleList()#原始图的3层
        self.query_layers.append(GCNConv(1, hidden_dim)) #第一层GCNConv(1,256)
        for _ in range(num_layers - 1):
            self.query_layers.append(GCNConv(hidden_dim, hidden_dim)) #添加剩余的2层, 2*GCNConv(256,256)

        #L层原始视图graph encoder：
        self.layers = nn.ModuleList()
        self.layers.append(GCNConv(node_in_dim, hidden_dim))  # 第一层的输入维度是node_in_dim,GCNConV(1,256)
        for _ in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))  # 2*GCNConv(256,256)

        #L：获得上一层的hf
        self.last_fusion_layers = nn.ModuleList()
        self.last_fusion_layers.append(GCNConv(hidden_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.last_fusion_layers.append(GCNConv(hidden_dim, hidden_dim))

        #计算query和h的注意力系数
        self.query_atts = []
        self.atts = []
        for _ in range(num_layers):
            #query encoder参数初始化
            query_att = nn.Parameter(torch.empty(size=(hidden_dim, 1)), requires_grad=True)  # 可学习的参数，
            glorot(query_att)
            #graph encoder参数初始化
            att = nn.Parameter(torch.empty(size=(hidden_dim, 1)), requires_grad=True)
            glorot(att)
            self.query_atts.append(query_att)
            self.atts.append(att)

        #初始没有前一层的hf，因此融合x和q作为前一层hf
        self.first_query_att = nn.Parameter(torch.empty(size=(hidden_dim, 1)), requires_grad=True)
        glorot(self.first_query_att)
        self.first_att = nn.Parameter(torch.empty(size=(hidden_dim, 1)), requires_grad=True)
        glorot(self.first_att)

        # 增强视图query encoder：
        self.aug_query_layers = nn.ModuleList()  # 超图的L层
        self.aug_query_layers.append(HypergraphConv(1, hidden_dim))  # 第一层GCNConv(1,256)
        for _ in range(num_layers - 1):
            self.aug_query_layers.append(HypergraphConv(hidden_dim, hidden_dim))  # 添加剩余的2层, 2*GCNConv(256,256)
        # 增强视图graph encoder：
        self.aug_layers = nn.ModuleList()
        self.aug_layers.append(HypergraphConv(node_in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.aug_layers.append(HypergraphConv(hidden_dim, hidden_dim))
        # 增强视图融合前一层的haugf
        self.aug_last_fusion_layers = nn.ModuleList()
        self.aug_last_fusion_layers.append(HypergraphConv(hidden_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.aug_last_fusion_layers.append(HypergraphConv(hidden_dim, hidden_dim))

        #query和graph的注意力系数
        self.aug_query_atts = []  # 超图中查询节点模型的参数
        self.aug_atts = []  # 超图中的所有节点特征的模型参数
        for _ in range(num_layers):
            # query encoder参数初始化
            att_weighthq = nn.Parameter(torch.empty(size=(hidden_dim, 1)), requires_grad=True)
            glorot(att_weighthq)
            # graph encoder
            att_weighth = nn.Parameter(torch.empty(size=(hidden_dim, 1)), requires_grad=True)
            glorot(att_weighth)
            self.aug_query_atts.append(att_weighthq)
            self.aug_atts.append(att_weighth)

        # 初始没有前一层的hf
        self.aug_first_query_att = nn.Parameter(torch.empty(size=(hidden_dim, 1)), requires_grad=True)
        glorot(self.aug_first_query_att)
        self.aug_first_att = nn.Parameter(torch.empty(size=(hidden_dim, 1)), requires_grad=True)
        glorot(self.aug_first_att)  # 一种参数初始化的方式

        self.mlp1 = MLP(hidden_dim, hidden_dim)

        #query encoder全连接层
        self.linerquerys = torch.nn.Linear(1, hidden_dim)
        #graph encoder 全连接层
        self.linerfeats = torch.nn.Linear(node_in_dim, hidden_dim)

    def q_att_layer(self, x, layer):
        '''
        将输入张量x与self.att_weightsq[layer]的权重矩阵相乘
        :param x:
        :param layer:
        :return:
        '''
        return torch.matmul(x, self.query_atts[layer].to(self.device))

    def att_layer(self, x, layer):
        return torch.matmul(x, self.atts[layer].to(self.device))

    def first_q_att_layer(self, x):
        '''
        这个只是用于第0层时，给query的注意力系数
        :param x:
        :return:
        '''
        return torch.matmul(x, self.first_query_att.to(self.device))
    def first_att_layer(self, x):
        '''
        第0层X的注意力系数
        :param x:
        :return:
        '''
        return torch.matmul(x, self.first_att.to(self.device))


    def aug_q_att_layer(self, x, layer):
        return torch.matmul(x, self.aug_query_atts[layer].to(self.device))
    def aug_first_q_att_layer(self, x):
        return torch.matmul(x, self.aug_first_query_att.to(self.device))
    def aug_att_layer(self, x, layer):
        return torch.matmul(x, self.aug_atts[layer].to(self.device))
    def aug_first_att_layer(self, x):
        return torch.matmul(x, self.aug_first_att.to(self.device))

    def hyperedge_representation(self, x, edge_index):
        '''
        将节点嵌入转换成超边表示
        :param x:
        :param edge_index:
        :return:
        '''
        #h = self.mlp2(x)
        h = x#torch.tanh(self.att(x))  将节点特征分配给h
        edges = h[edge_index[0]] #从h中提取与超边相关的节点特征？？？
        nodes = h[edge_index[1]] #从h中提取与超边连接的其他的节点特征？？？

        sim = torch.exp(torch.cosine_similarity(edges, nodes)) #计算相似度，然后通过指数函数转化

        denominator = scatter(sim, edge_index[1], dim=0, reduce='sum') #对'sim'进行汇总，以计算每个超边的分母
        denominator = denominator[edge_index[1]] #将分母按照edge_index[1]超边重新排列，以便与每个超边关联
        sim = (sim/denominator).unsqueeze(1) #将sim初一分母denominator,并将结果加入到一个新的维度，得到归一化的相似性得分。

        edges_ = x[edge_index[0]]
        edges_ = sim * (edges_)

        hyperedge = scatter(edges_, edge_index[1], dim=0, reduce='sum') #hyperedge = torch.cat([x, hyperedge], 1)

        return hyperedge

    def compute_loss(self,train):
        '''

        :param train:
        :return:
        '''
        #获取训练数据
        loss = None
        q, pos, edge_index, edge_index_aug, feats = train
        querys = torch.zeros(feats.shape[0], 1).to(self.device)
        querys[q] = 1.0  # 与特征维度相同的0张量，并将查询节点索引位置置为1

        '***********  第0层************************'
        # normal encoder
        hq = F.relu(self.query_layers[0](querys, edge_index)).to(self.device)  # hq=115*256 query_encoder[0]
        h = F.relu(self.layers[0](feats, edge_index)).to(self.device)  # h=115*256  graph_encoder[0]
        # 将hq和h分别通过第0个注意力层，并在列上拼接起来
        atten_co = torch.cat([self.q_att_layer(hq, 0), self.att_layer(h, 0)], 1) #fusion_attention[0]
        atten_co = F.softmax(atten_co, dim=1).unsqueeze(2)  # atten_co=(115,2,1)
        # 融合query encoder和graph encoder，并通过注意力得分进行加权
        hf = torch.stack([hq, h], dim=1)  # hf=(115,2,256)
        hf = atten_co * hf
        hf = torch.sum(hf, dim=1)  # 沿着第1维度求和（聚合）hf=(115,256)

        #aug encoder
        h_augq = F.relu(
            self.aug_query_layers[0](querys, edge_index_aug))  # querys(115,1)---GCNConV(1,256)-->h_augq(115,256)
        h_aug = F.relu(self.aug_layers[0](feats, edge_index_aug))  # feats(115,1) --GCNConV(1,256)-->h_aug(115,256)
        atten_coh = torch.cat([self.aug_q_att_layer(h_augq, 0), self.aug_att_layer(h_aug, 0)], 1)  # (115,2)
        atten_coh = F.softmax(atten_coh, dim=1).unsqueeze(2)  # atten_coh=(115,2,1)
        h_augf = torch.stack([h_augq, h_aug], dim=1)  # h_augf(115,2,256)
        h_augf = atten_coh * h_augf
        h_augf = torch.sum(h_augf, dim=1)  # h_augf=(115,256)

        # 对初始的查询特征和节点特征进行线性变换
        querys = self.linerquerys(querys)  # querys=(115,256)
        feats = self.linerfeats(feats)  # feats=(115,256)
        # 经过线性变换的querys和feats直接分别通过(256,1)的注意力层
        atten_co_ = torch.cat([self.first_q_att_layer(querys), self.first_att_layer(feats)], 1)
        atten_co_ = F.softmax(atten_co_, dim=1).unsqueeze(2)  # atten_co_=(115,2,1)
        hf_ = torch.stack([querys, feats], dim=1)  # hf_=(115,2,256)
        hf_ = atten_co_ * hf_
        hf_ = torch.sum(hf_, dim=1)  # hf_=(115,256)
        # 原始图第0层的嵌入：将hf_通过f这个第0个卷积层，然后和前面的hf相加，得到本层最终的hf
        hf = F.relu(hf + self.last_fusion_layers[0](hf_, edge_index))  # fusion_encoder[0]

        # 增强图：将经过线性变换的querys和feats直接分别通过(256,1)的注意力层
        atten_coh_ = torch.cat([self.aug_first_q_att_layer(querys), self.aug_first_att_layer(feats)],
                               1)  # atten_coh_=（115，2）
        atten_coh_ = F.softmax(atten_coh_, dim=1).unsqueeze(2)  # atten_coh_=(115,2,1)
        hfh_ = torch.stack([querys, feats], dim=1)  # hfh_=(115,2,256)
        hfh_ = atten_coh_ * hfh_
        hfh_ = torch.sum(hfh_, dim=1)  # hfh_=(115,256)

        h_augf = F.relu(h_augf + self.aug_last_fusion_layers[0](hfh_, edge_index_aug))


        '***************剩余n-2层********************'
        for _ in range(self.num_layers - 2):
            #将h^(l)进行dropout
            hq = F.dropout(hq, training=self.training, p=self.dropout)
            h = F.dropout(h, training=self.training, p=self.dropout)
            hf = F.dropout(hf, training=self.training, p=self.dropout)
            h_augq = F.dropout(h_augq, training=self.training, p=self.dropout)
            h_aug = F.dropout(h_aug, training=self.training, p=self.dropout)
            h_augf = F.dropout(h_augf, training=self.training, p=self.dropout)

            #进入l+1层
            hq = F.relu(self.query_layers[_ + 1](hq, edge_index)) # query_encoder[_+1]
            h = F.relu(self.layers[_ + 1](h, edge_index)) #graph_encoder[_+1]
            atten_co = torch.cat([self.q_att_layer(hq, _ + 1), self.att_layer(h, _ + 1)], 1)  # (115,2) fusion_attension[l+1]
            atten_co = F.softmax(atten_co, dim=1).unsqueeze(2)  # (115,2,1)
            hfx = torch.stack([hq, h], dim=1)  # hfx=(115,2,256)
            hfx = atten_co * hfx
            hfx = torch.sum(hfx, dim=1)  # hfx=(115,256)

            #   hf[_+1]= hfx[l+1] + (hf[l]-->GCN)
            hf = F.relu(hfx + self.last_fusion_layers[_ + 1](hf, edge_index))

            # 在增强超图中执行和前面类似的操作
            h_augq = F.relu(self.aug_query_layers[_ + 1](h_augq, edge_index_aug))
            h_aug = F.relu(self.aug_layers[_ + 1](h_aug, edge_index_aug))
            atten_coh = torch.cat([self.aug_q_att_layer(h_augq, _ + 1), self.aug_att_layer(h_aug, _ + 1)], 1)
            atten_coh = F.softmax(atten_coh, dim=1).unsqueeze(2)
            h_augfx = torch.stack([h_augq, h_aug], dim=1)
            h_augfx = atten_coh * h_augfx
            h_augfx = torch.sum(h_augfx, dim=1)
            h_augf = F.relu(h_augfx + self.aug_last_fusion_layers[_ + 1](h_augf, edge_index_aug))

        hq = F.dropout(hq, training=self.training, p=self.dropout)
        h = F.dropout(h, training=self.training, p=self.dropout)
        hf = F.dropout(hf, training=self.training, p=self.dropout)
        h_augq = F.dropout(h_augq, training=self.training, p=self.dropout)
        h_aug = F.dropout(h_aug, training=self.training, p=self.dropout)
        h_augf = F.dropout(h_augf, training=self.training, p=self.dropout)

        '*********最后一层******'
        hq = self.query_layers[self.num_layers - 1](hq, edge_index)
        h = self.layers[self.num_layers - 1](h, edge_index)
        atten_co = torch.cat([self.q_att_layer(hq, self.num_layers - 1), self.att_layer(h, self.num_layers - 1)], 1) #（115，2）
        atten_co = F.softmax(atten_co, dim=1).unsqueeze(2) #（115，2，1）
        hfx = torch.stack([hq, h], dim=1) #（115，2，256）
        hfx = atten_co * hfx
        hfx = torch.sum(hfx, dim=1) #（115，256）
        hf = hfx + self.last_fusion_layers[self.num_layers - 1](hf, edge_index) #（115，256）

        h_augq = self.aug_query_layers[self.num_layers - 1](h_augq, edge_index_aug)
        h_aug = self.aug_layers[self.num_layers - 1](h_aug, edge_index_aug)
        atten_coh = torch.cat(
            [self.aug_q_att_layer(h_augq, self.num_layers - 1), self.aug_att_layer(h_aug, self.num_layers - 1)],
            1)
        atten_coh = F.softmax(atten_coh, dim=1).unsqueeze(2)
        h_augfx = torch.stack([h_augq, h_aug], dim=1)
        h_augfx = atten_coh * h_augfx
        h_augfx = torch.sum(h_augfx, dim=1)
        h_augf = h_augfx + self.aug_last_fusion_layers[self.num_layers - 1](h_augf, edge_index_aug)


        #通过mlp映射
        h_ = self.mlp1(hf)  # h_=(115,256)
        h_auge = self.hyperedge_representation(h_augf, edge_index_aug)  # h_auge=(115,256)
        # h_auge = self.lineraugh(h_auge)#'''
        h_auge = self.mlp1(h_auge)  # h_auge=(115,256)

        if loss is None:
            # 调用contra模型，执行forward计算对比损失
            loss = self.contra(h_, h_auge, self.tau, (q, pos), self.alpha, self.lam, edge_index)
        else: #累加上一个训练任务的对比损失
            loss = loss + self.contra(h_, h_auge, self.tau, (q, pos), self.alpha, self.lam, edge_index)

        return loss,h_

    def valiates(self,train):
        '''
        模型不是训练阶段，直接使用原始视图的GNN层进行前馈获得节点嵌入并返回
        :param train:
        :return:
        '''
        #获取数据
        q,pos,edge_index,edge_index_aug,feats =train
        querys = torch.zeros(feats.shape[0], 1).to(self.device)
        querys[q] = 1.0

        '**第0层**'
        hq = F.relu(self.query_layers[0](querys, edge_index))  # 使用第1个图卷积层layersq[0]处理查询节点hq
        h = F.relu(self.layers[0](feats, edge_index))  # 使用第1个图卷积层layersq[0]处理节点特征h
        # 计算注意力权重，将hq和q分别通过attetion_layerq和attetion_layer计算得到，然后拼接在一起
        atten_co = torch.cat([self.q_att_layer(hq, 0), self.att_layer(h, 0)], 1)
        atten_co = F.softmax(atten_co, dim=1).unsqueeze(2)  # 使用softmax将注意力权重归一化
        hf = torch.stack([hq, h], dim=1)
        hf = atten_co * hf
        hf = torch.sum(hf, dim=1)  # 对h

        querys = self.linerquerys(querys)  # 对查询节点querys应用linerquerys进行线性变换。
        feats = self.linerfeats(feats)  # 对节点特征feats应用linerfeats进行线性变换。
        atten_co_ = torch.cat([self.first_q_att_layer(querys), self.first_att_layer(feats)], 1)
        atten_co_ = F.softmax(atten_co_, dim=1).unsqueeze(2)
        hf_ = torch.stack([querys, feats], dim=1)
        hf_ = atten_co_ * hf_
        hf_ = torch.sum(hf_, dim=1)  #再次计算注意力权重和融合信息得到hf
        hf = F.relu(hf + self.last_fusion_layers[0](hf_, edge_index))

        '***中间层的卷积操作***'
        for _ in range(self.num_layers - 2):  # 循环执行后续层的图卷积操作，每层都包括计算注意力权重、融合信息，然后将结果传给下一层
            hq = F.dropout(hq, training=self.training, p=self.dropout)
            h = F.dropout(h, training=self.training, p=self.dropout)
            hf = F.dropout(hf, training=self.training, p=self.dropout)

            hq = F.relu(self.query_layers[_ + 1](hq, edge_index))
            h = F.relu(self.layers[_ + 1](h, edge_index))

            atten_co = torch.cat([self.q_att_layer(hq, _ + 1), self.att_layer(h, _ + 1)], 1)
            atten_co = F.softmax(atten_co, dim=1).unsqueeze(2)
            hfx = torch.stack([hq, h], dim=1)
            hfx = atten_co * hfx
            hfx = torch.sum(hfx, dim=1)
            hf = F.relu(hfx + self.last_fusion_layers[_ + 1](hf, edge_index))

        hq = F.dropout(hq, training=self.training, p=self.dropout)
        h = F.dropout(h, training=self.training, p=self.dropout)
        hf = F.dropout(hf, training=self.training, p=self.dropout)

        '************最后一层的卷积操作*********'
        hq = self.query_layers[self.num_layers - 1](hq, edge_index)
        h = self.layers[self.num_layers - 1](h, edge_index)
        atten_co = torch.cat(
            [self.q_att_layer(hq, self.num_layers - 1), self.att_layer(h, self.num_layers - 1)], 1)
        atten_co = F.softmax(atten_co, dim=1).unsqueeze(2)
        hfx = torch.stack([hq, h], dim=1)
        hfx = atten_co * hfx
        hfx = torch.sum(hfx, dim=1)
        hf = hfx + self.last_fusion_layers[self.num_layers - 1](hf, edge_index)

        h_ = self.mlp1(hf)  # 最后应用mlp计算hf的最终表示。

        return h_

    def forward(self, train):
        if self.training ==False:
            h_ = self.valiates(train)
            return h_
        #否则计算对比损失
        loss,h_ = self.compute_loss(train)

        return loss,h_









