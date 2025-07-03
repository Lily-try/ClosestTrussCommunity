import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv, GCNConv, APPNP
from torch_geometric.nn.inits import glorot
from torch_scatter import scatter

#带有1个隐藏层的模型
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLP, self).__init__()
        self.fcs = nn.Sequential(
            nn.Linear(in_dim, out_dim), #线性层
            nn.ReLU(),    #激活函数
            nn.Linear(out_dim, out_dim), #激活函数
        )

    def forward(self, x):
        return self.fcs(x)

#对比学习
class Contra(nn.Module):
    def __init__(self, device):
        super(Contra, self).__init__()
        self.device = device  #初始化设备，指定cpu/gpu

    def forward(self, h, h_aug, tau, train, alpha, lam, edge_index):
        '''
        :param h: 模型得到的原始视图中各个节点的嵌入
        :param h_aug: 模型得到的增强视图中各个节点的嵌入
        :param tau: 注意力的温度系数
        :param train: 训练集数据（q,pos)
        :param alpha: 视图间对比损失的权重系数
        :param lam: 查询节点对比损失的权重系数
        :param edge_index: 返回计算的对比损失
        :return:返回计算的对比损失
        '''
        if self.training == False:  #不是训练阶段，则直接返回当前的h，不再进行对比学习训练了。
            return h #包含了所有节点的嵌入向量

        q, pos = train #获取训练集数据

        # 计算h[q]和h向量之间的余弦相似度  #h[q]原本是1维的，unsqueeze(0)会将其转换成二维的行向量
        h_q = h[q].unsqueeze(0)  #h_q=(1,256) 确保h[q]是二维的（执行乘法操作要求如此）
        sim1 = F.cosine_similarity(h_q, h, dim=1)  # 计算余弦相似度sim1=(115,)
        sim1 = torch.exp(sim1 / tau)  # 应用温度系数

        # 计算 h_aug[q]和h_aug 向量之间的余弦相似度
        sim_aug1 = F.cosine_similarity(h_aug[q].unsqueeze(0), h_aug, dim=1)#sim_aug_1=(115,)
        sim_aug1 = torch.exp(sim_aug1 / tau)

        # 计算 h[q] 和 h_aug 之间的相似度
        sim2 = F.cosine_similarity(h[q].unsqueeze(0), h_aug, dim=1)
        sim2 = torch.exp(sim2 / tau)

        # 计算 h_aug[q] 和 h 之间的相似度，类似于 sim2 但交换了 h 和 h_aug 的位置
        sim_aug2 = F.cosine_similarity(h_aug[q].unsqueeze(0), h, dim=1)
        sim_aug2 = torch.exp(sim_aug2 / tau)

        # 创建正样本掩码，正样本位置为True
        mask_p = [False] * h.shape[0] #(115,)
        mask_p = torch.tensor(mask_p)
        mask_p.to(self.device)
        mask_p[pos] = True
        mask_p[q] = False

        # 创建存储更新后的节点嵌入向量
        z1q = torch.tensor([0.0]).to(self.device) #(1,)
        z_aug1q = torch.tensor([0.0]).to(self.device)
        z2q = torch.tensor([0.0]).to(self.device)
        z_aug2q = torch.tensor([0.0]).to(self.device)

        #unsqueeze(0)在张量的指定位置（这里是0，即最前面）插入一个维度，例如形状为[n]的一维张量，unsqueeze(0)后形状变为[1,n]的二维张量
        #squeeze(0)移除张量中所有大小为1的维度，或者指定位置的大小为1的维度。例如形状为[1,n]的二维张量，squeeze(0)后变为形状为[n]的1维张量
        if len(pos) != 0: #计算标签内巡视
            # tmp1=sim1.squeeze(0) #tmp1=(115,),确保sim1是一维的
            #tmp11=sim1.squeeze(0)[mask_p] #sim1中所有mask_p为True的元素，即pos元素的sim。tmp11=(3,)
            # tmp12=torch.sum(sim1.squeeze(0)) #一个标量值
            z1q = sim1.squeeze(0)[mask_p] / (torch.sum(sim1.squeeze(0))) #z1q=(3,)
            z1q = -torch.log(z1q).mean() #低阶视图内对比损失 标量?()，值是4.7057...
            z_aug1q = sim_aug1.squeeze(0)[mask_p] / (torch.sum(sim_aug1.squeeze(0)))
            z_aug1q = -torch.log(z_aug1q).mean() #高阶视图内对比损失

            # 两个视图间对比损失
            z2q = sim2.squeeze(0)[mask_p] / (torch.sum(sim2.squeeze(0)))
            z2q = -torch.log(z2q).mean()
            z_aug2q = sim_aug2.squeeze(0)[mask_p] / (torch.sum(sim_aug2.squeeze(0)))
            z_aug2q = -torch.log(z_aug2q).mean()
        loss_intra = 0.5*(z1q+z_aug1q) #视图内对比学习
        loss_inter = 0.5*(z2q+z_aug2q) #视图内对比学习

        # 查询节点的视图间对比损失函数进行组合得到loss_unsup
        z_unsup = -torch.log(sim2.squeeze(0)[q]/torch.sum(sim2.squeeze(0)))
        z_aug_unsup = -torch.log(sim_aug2.squeeze(0)[q]/torch.sum(sim_aug2.squeeze(0)))
        loss_unsup = 0.5 * z_unsup + 0.5 * z_aug_unsup

        # 最终的loss函数
        loss = (loss_intra+ alpha *loss_inter) + lam * loss_unsup #+ loss2r * lc

        return loss#+loss_c

class ConRC(nn.Module):
    def __init__(self, node_in_dim, hidden_dim, num_layers, dropout, tau, device, alpha, lam, k):
        super(ConRC, self).__init__()
        #各种参数初始化
        self.tau = tau
        self.alpha = alpha
        self.lam = lam
        self.dropout = dropout
        self.num_layers = num_layers
        self.device = device
        self.k = k #文献中的r
        self.contra = Contra(device)  #创建用于计算对比损失的对象实例

        #原始图中的查询节点一系列神经网络层
        self.layersq = nn.ModuleList()#原始图的3层
        self.layersq.append(GCNConv(1, hidden_dim)) #第一层GCNConv(1,256)
        for _ in range(num_layers - 1):
            self.layersq.append(GCNConv(hidden_dim, hidden_dim)) #添加剩余的2层, 2*GCNConv(256,256)

        #超图中的查询节点一系列神经网络层
        self.layershq = nn.ModuleList() #超图的L层
        self.layershq.append(HypergraphConv(1, hidden_dim))#第一层GCNConv(1,256)
        for _ in range(num_layers - 1):
            self.layershq.append(HypergraphConv(hidden_dim, hidden_dim)) #添加剩余的2层, 2*GCNConv(256,256)

        # 原始图中的节点特征一系列神经网络层
        self.layers = nn.ModuleList()
        self.layers.append(GCNConv(node_in_dim, hidden_dim)) #第一层的输入维度是node_in_dim,GCNConV(1,256)
        for _ in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_dim, hidden_dim)) #2*GCNConv(256,256)

        # 超图中的节点特征的一系列神经网络层
        self.layersh = nn.ModuleList()
        self.layersh.append(HypergraphConv(node_in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layersh.append(HypergraphConv(hidden_dim, hidden_dim))

        #原始图中的融合的一系列神经网络层
        self.layersf = nn.ModuleList()
        self.layersf.append(GCNConv(hidden_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layersf.append(GCNConv(hidden_dim, hidden_dim))

        #超图中融合的一系列神经网路层
        self.layersfh = nn.ModuleList()
        self.layersfh.append(HypergraphConv(hidden_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layersfh.append(HypergraphConv(hidden_dim, hidden_dim))

        #一个mlp层，用于处理隐藏维度到隐藏维度的转换
        self.mlp1 = MLP(hidden_dim, hidden_dim)

        #初始化一系列权重矩阵，用于在模型的前向传播中计算注意力
        self.att_weightsq = [] # 原始图查询节点的模型参数
        self.att_weights = []   # 原始图中的所有节点特征的模型参数
        self.att_weighthsq = [] # 超图中查询节点模型的参数
        self.att_weighths = []  # 超图中的所有节点特征的模型参数
        for _ in range(num_layers): #循环次数由模型层数决定，初始化各个层的权重参数
            #创建权重，并使用glorot，一种权重初始化方法，用有助于在训练初始阶段为权重赋予合适的值，以加速收敛。
            att_weightq = nn.Parameter(torch.empty(size=(hidden_dim, 1)), requires_grad=True) #可学习的参数，
            glorot(att_weightq)
            # 原图中的权重参数
            att_weight = nn.Parameter(torch.empty(size=(hidden_dim, 1)), requires_grad=True)
            glorot(att_weight)
            # 超图中查询节点的权重参数
            att_weighthq = nn.Parameter(torch.empty(size=(hidden_dim, 1)), requires_grad=True)
            glorot(att_weighthq)
            # 超图中的权重参数
            att_weighth = nn.Parameter(torch.empty(size=(hidden_dim, 1)), requires_grad=True)
            glorot(att_weighth)
            #将初始化的各个权重参数添加到对应的列表中
            self.att_weightsq.append(att_weightq)
            self.att_weighthsq.append(att_weighthq)
            self.att_weights.append(att_weight)
            self.att_weighths.append(att_weighth)

        #没懂，为什么又创建了4个参数呢？
        self.att_weightq_ = nn.Parameter(torch.empty(size=(hidden_dim, 1)), requires_grad=True)
        glorot(self.att_weightq_)
        self.att_weight_ = nn.Parameter(torch.empty(size=(hidden_dim, 1)), requires_grad=True)
        glorot(self.att_weight_)
        self.att_weighthq_ = nn.Parameter(torch.empty(size=(hidden_dim, 1)), requires_grad=True)
        glorot(self.att_weighthq_)
        self.att_weighth_ = nn.Parameter(torch.empty(size=(hidden_dim, 1)), requires_grad=True)
        glorot(self.att_weighth_) #一种参数初始化的方式

        #查询节点的全连接层
        self.linerquerys = torch.nn.Linear(1, hidden_dim)
        #所有节点特征的全连接层
        self.linerfeats = torch.nn.Linear(node_in_dim, hidden_dim)

    def reset_parameters(self): #打印参数被重置了reset
        print("reset")

    #原始图中的查询节点注意力：将输入张量x和名为self.att_weightsq[layer]的权重矩阵相乘
    def attetion_layerq(self, x, layer):
        return torch.matmul(x, self.att_weightsq[layer].to(self.device))

    # 超图中的查询节点注意力：将输入张量x和名为self.att_weightsq[layer]的权重矩阵相乘
    def attetion_layerhq(self, x, layer):
        return torch.matmul(x, self.att_weighthsq[layer].to(self.device))

    #q_ 原始图中的注意力：(256,1)
    def attetion_layerq_(self, x):
        return torch.matmul(x, self.att_weightq_.to(self.device))

    # 超图中的注意力：
    def attetion_layerhq_(self, x):
        return torch.matmul(x, self.att_weighthq_.to(self.device))

    # 原始图中的节点特征点注意力：将输入张量x和名为self.att_weights[layer]的权重矩阵相乘
    def attetion_layer(self, x, layer):
        return torch.matmul(x, self.att_weights[layer].to(self.device))

    # 超图中的节点特征点注意力：将输入张量x和名为self.att_weighths[layer]的权重矩阵相乘
    def attetion_layerh(self, x, layer):
        return torch.matmul(x, self.att_weighths[layer].to(self.device))
    #_ 原始图中对节点特征的注意力，只需要1个就够了？
    def attetion_layer_(self, x):
        return torch.matmul(x, self.att_weight_.to(self.device))

    def attetion_layerh_(self, x):
        return torch.matmul(x, self.att_weighth_.to(self.device))

    #融合各个节点的表示得到超边的表示。
    def hyperedge_representation(self, x, edge_index):
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

    #实际上是模型训练时的前向传播计算损失的过程，在forward方法中调用
    def compute_loss(self, train):
        loss = None
        q, pos, edge_index, edge_index_aug, feats = train
        #feats=115*1
        #querys=115*1
        querys = torch.zeros(feats.shape[0], 1).to(self.device)
        querys[q] = 1.0  #与特征维度相同的0张量，并将查询节点索引位置置为1

        # 使用第一层对查询节点和特征进行处理，并使用ReLU激活函数
        hq = F.relu(self.layersq[0](querys, edge_index)).to(self.device) #hq=115*256
        h = F.relu(self.layers[0](feats, edge_index)).to(self.device) #h=115*256
        #将hq和h分别通过第0个注意力层，并在列上拼接起来
        # temphq=self.attetion_layerq(hq, 0) #temphq =115*1
        # temph= self.attetion_layer(h, 0) #temph =115*1
        atten_co = torch.cat([self.attetion_layerq(hq, 0), self.attetion_layer(h, 0)], 1)
        atten_co = F.softmax(atten_co, dim=1).unsqueeze(2) #atten_co=(115,2,1)
        # 结合查询节点和特征的隐藏表示，并通过注意力得分进行加权
        hf = torch.stack([hq, h], dim=1) #hf=(115,2,256)
        hf = atten_co * hf
        hf = torch.sum(hf, dim=1) #沿着第1维度求和（聚合）hf=(115,256)

        # 在增强图中进行和前面类似的操作，得到融合后的超图中的节点表示
        h_augq = F.relu(self.layershq[0](querys, edge_index_aug)) #querys(115,1)---GCNConV(1,256)-->h_augq(115,256)
        h_aug = F.relu(self.layersh[0](feats, edge_index_aug)) #feats(115,1) --GCNConV(1,256)-->h_aug(115,256)
        atten_coh = torch.cat([self.attetion_layerhq(h_augq, 0), self.attetion_layerh(h_aug, 0)], 1) #(115,2)
        atten_coh = F.softmax(atten_coh, dim=1).unsqueeze(2) #atten_coh=(115,2,1)
        h_augf = torch.stack([h_augq, h_aug], dim=1) #h_augf(115,2,256)
        h_augf = atten_coh * h_augf
        h_augf = torch.sum(h_augf, dim=1) #h_augf=(115,256)

        # 对查询节点和节点特征进行线性变换，作用是什么
        querys = self.linerquerys(querys) #querys=(115,256)
        feats = self.linerfeats(feats) #feats=(115,256)

        #原始图：将经过线性变换的querys和feats直接分别通过(256,1)的注意力层
        atten_co_ = torch.cat([self.attetion_layerq_(querys), self.attetion_layer_(feats)], 1)
        atten_co_ = F.softmax(atten_co_, dim=1).unsqueeze(2) #atten_co_=(115,2,1)
        hf_ = torch.stack([querys, feats], dim=1) #hf_=(115,2,256)
        hf_ = atten_co_ * hf_
        hf_ = torch.sum(hf_, dim=1)#hf_=(115,256)

        #将hf_通过f这个第0个卷积层，然后和前面的hf相加，得到最终的hf
        hf = F.relu(hf + self.layersf[0](hf_, edge_index))

        #增强图：将经过线性变换的querys和feats直接分别通过(256,1)的注意力层
        atten_coh_ = torch.cat([self.attetion_layerhq_(querys), self.attetion_layerh_(feats)], 1) #atten_coh_=（115，2）
        atten_coh_ = F.softmax(atten_coh_, dim=1).unsqueeze(2)#atten_coh_=(115,2,1)
        hfh_ = torch.stack([querys, feats], dim=1) #hfh_=(115,2,256)
        hfh_ = atten_coh_ * hfh_
        hfh_ = torch.sum(hfh_, dim=1) #hfh_=(115,256)

        h_augf = F.relu(h_augf + self.layersfh[0](hfh_, edge_index_aug))

        # 应用多层神经网络，每层后都使用dropout和ReLU。由于num_layers=3，因此这个循环只执行了一次。
        for _ in range(self.num_layers - 2):

            hq = F.dropout(hq, training=self.training, p=self.dropout)
            h = F.dropout(h, training=self.training, p=self.dropout)
            hf = F.dropout(hf, training=self.training, p=self.dropout)
            h_augq = F.dropout(h_augq, training=self.training, p=self.dropout)
            h_aug = F.dropout(h_aug, training=self.training, p=self.dropout)
            h_augf = F.dropout(h_augf, training=self.training, p=self.dropout)

            #将hq和h分别通过剩余的2层GCNConV(256,256)
            hq = F.relu(self.layersq[_+1](hq, edge_index))
            h = F.relu(self.layers[_+1](h, edge_index))
            # temhq=self.attetion_layerq(hq, _+1) #temhq=(115,1)
            # temph=self.attetion_layer(h, _+1) #temph=(115,1)
            atten_co = torch.cat([self.attetion_layerq(hq, _+1), self.attetion_layer(h, _+1)], 1) #(115,2)
            atten_co = F.softmax(atten_co, dim=1).unsqueeze(2) #(115,2,1)
            hfx = torch.stack([hq, h], dim=1) #hfx=(115,2,256)
            hfx = atten_co * hfx
            hfx = torch.sum(hfx, dim=1) #hfx=(115,256)
            #一部分进入了下一层卷积的和一部分没有的
            hf = F.relu(hfx + self.layersf[_+1](hf, edge_index))

            #在增强超图中执行和前面类似的操作
            h_augq = F.relu(self.layershq[_+1](h_augq, edge_index_aug))
            h_aug = F.relu(self.layersh[_+1](h_aug, edge_index_aug))
            atten_coh = torch.cat([self.attetion_layerhq(h_augq, _+1), self.attetion_layerh(h_aug, _+1)], 1)
            atten_coh = F.softmax(atten_coh, dim=1).unsqueeze(2)
            h_augfx = torch.stack([h_augq, h_aug], dim=1)
            h_augfx = atten_coh * h_augfx
            h_augfx = torch.sum(h_augfx, dim=1)
            h_augf = F.relu(h_augfx + self.layersfh[_+1](h_augf, edge_index_aug))

        hq = F.dropout(hq, training=self.training, p=self.dropout)
        h = F.dropout(h, training=self.training, p=self.dropout)
        hf = F.dropout(hf, training=self.training, p=self.dropout)
        h_augq = F.dropout(h_augq, training=self.training, p=self.dropout)
        h_aug = F.dropout(h_aug, training=self.training, p=self.dropout)
        h_augf = F.dropout(h_augf, training=self.training, p=self.dropout)

        #原始图：通过最后一层（第2层）卷积。
        hq = self.layersq[self.num_layers - 1](hq, edge_index)
        h = self.layers[self.num_layers - 1](h, edge_index)
        atten_co = torch.cat([self.attetion_layerq(hq, self.num_layers-1), self.attetion_layer(h, self.num_layers-1)], 1) #（115，2）
        atten_co = F.softmax(atten_co, dim=1).unsqueeze(2) #（115，2，1）
        hfx = torch.stack([hq, h], dim=1) #（115，2，256）
        hfx = atten_co * hfx
        hfx = torch.sum(hfx, dim=1) #（115，256）
        hf = hfx + self.layersf[self.num_layers - 1](hf, edge_index) #（115，256）

        # 超图：通过最后一层（第2层）卷积。
        h_augq = self.layershq[self.num_layers - 1](h_augq, edge_index_aug)
        h_aug = self.layersh[self.num_layers - 1](h_aug, edge_index_aug)
        atten_coh = torch.cat([self.attetion_layerhq(h_augq, self.num_layers-1), self.attetion_layerh(h_aug, self.num_layers-1)], 1)
        atten_coh = F.softmax(atten_coh, dim=1).unsqueeze(2)
        h_augfx = torch.stack([h_augq, h_aug], dim=1)
        h_augfx = atten_coh * h_augfx
        h_augfx = torch.sum(h_augfx, dim=1)
        h_augf = h_augfx + self.layersfh[self.num_layers - 1](h_augf, edge_index_aug)

        h_ = self.mlp1(hf) #h_=(115,256)
        h_auge = self.hyperedge_representation(h_augf, edge_index_aug) #h_auge=(115,256)
        #h_auge = self.lineraugh(h_auge)#'''
        h_auge = self.mlp1(h_auge)  # h_auge=(115,256)

        if loss is None:
            #调用contra模型，执行forward
            loss = self.contra(h_, h_auge, self.tau, (q, pos), self.alpha, self.lam, edge_index)
        else:
            loss = loss + self.contra(h_, h_auge, self.tau, (q, pos), self.alpha, self.lam, edge_index)


        return loss

    def forward(self, train):
        # 不是训练状态
        if self.training==False:
            q, pos, edge_index, edge_index_aug, feats = train
            querys = torch.zeros(feats.shape[0], 1).to(self.device)
            querys[q] = 1.0  #querys全0的查询节点掩码，将查询节点q对应的位置设为1

            hq = F.relu(self.layersq[0](querys, edge_index)) #使用第1个图卷积层layersq[0]处理查询节点hq
            h = F.relu(self.layers[0](feats, edge_index))    #使用第1个图卷积层layersq[0]处理节点特征h
            #计算注意力权重，将hq和q分别通过attetion_layerq和attetion_layer计算得到，然后拼接在一起
            atten_co = torch.cat([self.attetion_layerq(hq, 0), self.attetion_layer(h, 0)], 1)
            atten_co = F.softmax(atten_co, dim=1).unsqueeze(2) #使用softmax将注意力权重归一化
            hf = torch.stack([hq, h], dim=1)
            hf = atten_co * hf
            hf = torch.sum(hf, dim=1)#对hq和h应用注意力权重融合信息，并保存在hf中。

            querys = self.linerquerys(querys) #对查询节点querys应用linerquerys进行线性变换。
            feats = self.linerfeats(feats)   #对节点特征feats应用linerfeats进行线性变换。

            atten_co_ = torch.cat([self.attetion_layerq_(querys), self.attetion_layer_(feats)], 1)
            atten_co_ = F.softmax(atten_co_, dim=1).unsqueeze(2)
            hf_ = torch.stack([querys, feats], dim=1)
            hf_ = atten_co_ * hf_
            hf_ = torch.sum(hf_, dim=1)  #再次计算注意力权重和融合信息得到hf
            hf = F.relu(hf + self.layersf[0](hf_, edge_index))

            for _ in range(self.num_layers - 2): #循环执行后续层的图卷积操作，每层都包括计算注意力权重、融合信息，然后将结果传给下一层
                hq = F.dropout(hq, training=self.training, p=self.dropout)
                h = F.dropout(h, training=self.training, p=self.dropout)
                hf = F.dropout(hf, training=self.training, p=self.dropout)

                hq = F.relu(self.layersq[_ + 1](hq, edge_index))
                h = F.relu(self.layers[_ + 1](h, edge_index))
                atten_co = torch.cat([self.attetion_layerq(hq, _ + 1), self.attetion_layer(h, _ + 1)], 1)
                atten_co = F.softmax(atten_co, dim=1).unsqueeze(2)
                hfx = torch.stack([hq, h], dim=1)
                hfx = atten_co * hfx
                hfx = torch.sum(hfx, dim=1)
                hf = F.relu(hfx + self.layersf[_ + 1](hf, edge_index))

            hq = F.dropout(hq, training=self.training, p=self.dropout)
            h = F.dropout(h, training=self.training, p=self.dropout)
            hf = F.dropout(hf, training=self.training, p=self.dropout)

            hq = self.layersq[self.num_layers - 1](hq, edge_index)
            h = self.layers[self.num_layers - 1](h, edge_index)
            atten_co = torch.cat(
                [self.attetion_layerq(hq, self.num_layers - 1), self.attetion_layer(h, self.num_layers - 1)], 1)
            atten_co = F.softmax(atten_co, dim=1).unsqueeze(2)
            hfx = torch.stack([hq, h], dim=1)
            hfx = atten_co * hfx
            hfx = torch.sum(hfx, dim=1)
            hf = hfx + self.layersf[self.num_layers - 1](hf, edge_index)

            h_ = self.mlp1(hf) #最后应用mlp计算hf的最终表示。
            return h_
        #进行模型的训练
        loss = self.compute_loss(train)

        return loss

