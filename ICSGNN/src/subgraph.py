import torch
import random
import numpy as np
import networkx as nx
from src.clustergcn import ClusterGCNTrainer
from src.community import LocalCommunity
import time
import datetime


class SubGraph(object):
    def __init__(self, args, graph, features, target):
        self.args = args
        self.graph = graph
        self.features = features
        self.target = target
        self._set_sizes()
        self.methods = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.time_map = {}
        self.rankloss = 0
        self.posforrank = []
        self.negforrank = []
        self.cntmap = {}

    def _set_sizes(self):
        self.feature_count = self.features.shape[1]
        self.class_count = 2
        self.clusters = [0]
        self.cluster_membership = {node: 0 for node in self.graph.nodes()}

    def build_local_candidate_iteration(self, seed):
        '''
        Build subgraphs with iteration
        '''
        posNodes = []
        negNodes = []
        allNodes = [] #里面放的是seed
        length = self.args.subgraph_size
        if (len(self.allnode) == 1):
            allNodes.append(seed)
            numLabel = int(length * self.args.train_ratio / 2)
            pos = 0
            while pos < len(allNodes) and pos < length and len(allNodes) < length:
                cnode = allNodes[pos]
                for nb in self.graph.neighbors(cnode):
                    if nb not in allNodes and len(allNodes) < length:
                        allNodes.append(nb)
                pos = pos + 1
        else:
            numLabel = self.args.possize
            allNodes = self.allnode[:]
        print("The length of list is %d" % len(allNodes))
        print("The degree of seed is %d" % self.graph.degree(seed))
        for i in self.oldpos:
            if i not in allNodes:
                allNodes.append(i)
                print("pos not in subgraph")
            cnt = self.args.upsize
            for nb in self.graph.neighbors(i):
                if (cnt == 0): break
                if (nb not in allNodes):
                    allNodes.append(nb)
                    cnt -= 1
        self.allnode = allNodes
        seedLabel = self.target[seed]
        for node in allNodes:
            if (node == seed):
                continue
            if self.target[node] == seedLabel and node not in self.oldpos:
                posNodes.append(node)
            elif self.target[node] != seedLabel and node not in self.oldneg:
                negNodes.append(node)

        random.shuffle(posNodes)
        random.shuffle(negNodes)
        print('extern pos size', len(posNodes))
        print('extern neg size', len(negNodes))
        posNodes = posNodes[:numLabel]
        negNodes = negNodes[:numLabel]
        if (len(posNodes) < numLabel):
            print('e1')
            return 0
        if (seed not in self.oldpos):
            posNodes[0] = seed
        if (len(negNodes) < numLabel):
            print('e2')
            return 0

        posNodes = self.oldpos + posNodes
        negNodes = self.oldneg + negNodes
        print("Positive Nodes are ")
        print(posNodes)
        print("Negative Nodes are ")
        print(negNodes)
        self.oldpos = posNodes[:]
        self.oldneg = negNodes[:]
        negNodes = []
        for i in self.oldneg:
            if (i in allNodes):
                negNodes.append(i)
            else:
                print('neg不在里面')


        self.sg_nodes = {}
        self.sg_edges = {}
        self.sg_train_nodes = {}
        self.sg_test_nodes = {}
        self.sg_features = {}
        self.sg_targets = {}

        self.subgraph = nx.Graph()
        for i in range(len(allNodes)):
            for j in range(i):
                if ((allNodes[i], allNodes[j]) in self.graph.edges) or ((allNodes[j], allNodes[i]) in self.graph.edges):
                    self.subgraph.add_edge(allNodes[i], allNodes[j])

        print("size of nodes %d size of edges %d" % (len(self.subgraph.nodes), len(self.subgraph.edges)))
        self.sg_nodes[0] = [node for node in sorted(self.subgraph.nodes())]
        self.sg_predProbs = [0.0] * len(self.sg_nodes[0])
        self.sg_predLabels = [0] * len(self.sg_nodes[0])
        self.mapper = {node: i for i, node in enumerate(sorted(self.sg_nodes[0]))}
        self.rmapper = {i: node for i, node in enumerate(sorted(self.sg_nodes[0]))}
        self.sg_edges[0] = [[self.mapper[edge[0]], self.mapper[edge[1]]] for edge in self.subgraph.edges()] + [
            [self.mapper[edge[1]], self.mapper[edge[0]]] for edge in self.subgraph.edges()]
        self.sg_posNodes = [self.mapper[node] for node in posNodes]
        self.sg_negNodes = [self.mapper[node] for node in negNodes]
        allNodes1 = [self.mapper[node] for node in allNodes]
        self.sg_train_nodes[0] = self.sg_posNodes + self.sg_negNodes
        self.sg_test_nodes[0] = list(set(allNodes1).difference(set(self.sg_train_nodes[0])))
        self.sg_test_nodes[0] = sorted(self.sg_test_nodes[0])
        self.sg_train_nodes[0] = sorted(self.sg_train_nodes[0])
        self.sg_features[0] = self.features[self.sg_nodes[0], :]
        self.sg_targets[0] = self.target[self.sg_nodes[0], :]
        self.sg_targets[0] = self.sg_targets[0] == seedLabel
        self.sg_targets[0] = self.sg_targets[0].astype(int)


        print("Value 0 %d, Value 1 %d" % (sum(self.sg_targets[0] == 0), sum(self.sg_targets[0] == 1)))
        for x in self.sg_posNodes:
            self.sg_predProbs[x] = 1.0
            self.sg_predLabels[x] = 1
            if self.sg_targets[0][x] != 1.0:
                print("wrong1")
        for x in self.sg_negNodes:
            self.sg_predProbs[x] = 0.0
            self.sg_predLabels[x] = 0
            if self.sg_targets[0][x] != 0:
                print("wrong0")

        self.transfer_edges_and_nodes()
        self.TOPK_SIZE = int(self.args.community_size)
    def build_local_candidate(self, seed, trian_node, label):
        '''
        构建以种子节点为中心的局部子图用于社区搜索。
        Build subgraphs
        '''
        allNodes = [] #存储局部子图的所有节点，初始只有seed
        allNodes.append(seed)
        posNodes = set() #存储与seed标签相同的正样本节点
        negNodes = set() #存储与seed标签不同的负样本节点

        length = self.args.subgraph_size #控制子图大小
        numLabel = int(length * self.args.train_ratio / 2) # 训练集中正负样本的数量，按 train_ratio 确定
        pos = 0 #allNodes 中当前遍历的位置。

        #通过 BFS 扩展 seed 邻居，构建 allNodes
        while pos < len(allNodes) and pos < length and len(allNodes) < length:
            cnode = allNodes[pos]
            for nb in self.graph.neighbors(cnode):
                if nb not in allNodes and len(allNodes) < length: #若邻居不在allnodes且未超出大小限制，则加入
                    allNodes.append(nb)
                    if(nb!=seed and self.target is not None): #按照target进行划分。
                        if(self.target[nb] == self.target[seed]):
                            posNodes.add(nb)
                        else:
                            negNodes.add(nb)
            pos = pos + 1
        posNodes=list(posNodes)
        negNodes=list(negNodes)
        print("The length of allNodes list is %d" % len(allNodes))
        print("The degree of seed is %d" % self.graph.degree(seed))
        #训练集分配
        if (trian_node is not None): #按照train_node来分配正负样本
            posNodes = trian_node[:numLabel]
            negNodes = trian_node[numLabel:]
        else: #否则随机打乱选取正负样本
            seedLabel = self.target[seed]
            posNodes.append(seed)
            if(len(posNodes+[seed])<numLabel or len(negNodes)<numLabel):
                return 0
            random.shuffle(posNodes)
            random.shuffle(negNodes)
            posNodes=[seed]+posNodes[:numLabel-1] #numlabel-1+seed(1),组成numlabel个正样本
            negNodes=negNodes[:numLabel] #numlabel个负样本
        print("Positive Nodes are ")
        print(posNodes)
        print("Negative Nodes are ")
        print(negNodes)


        self.sg_nodes = {}
        self.sg_edges = {}
        self.sg_train_nodes = {}
        self.sg_test_nodes = {}
        self.sg_features = {}
        self.sg_targets = {}
        #生成子图
        self.subgraph = nx.Graph()
        for i in range(len(allNodes)): #遍历allnodes间的边构建subgraph
            for j in range(i):
                if ((allNodes[i], allNodes[j]) in self.graph.edges) or ((allNodes[j], allNodes[i]) in self.graph.edges):
                    self.subgraph.add_edge(allNodes[i], allNodes[j])

        print("size of nodes %d size of edges %d" % (len(self.subgraph.nodes), len(self.subgraph.edges)))
        #处理索引的映射
        self.sg_nodes[0] = [node for node in sorted(self.subgraph.nodes())]
        self.sg_predProbs = [0.0] * len(self.sg_nodes[0])
        self.sg_predLabels = [0] * len(self.sg_nodes[0])
        self.mapper = {node: i for i, node in enumerate(sorted(self.sg_nodes[0]))}
        self.rmapper = {i: node for i, node in enumerate(sorted(self.sg_nodes[0]))}
        self.sg_edges[0] = [[self.mapper[edge[0]], self.mapper[edge[1]]] for edge in self.subgraph.edges()] + [
            [self.mapper[edge[1]], self.mapper[edge[0]]] for edge in self.subgraph.edges()]
        self.sg_posNodes = [self.mapper[node] for node in posNodes]
        self.sg_negNodes = [self.mapper[node] for node in negNodes]

        allNodes1 = [self.mapper[node] for node in allNodes]

        #生成训练集和测试集
        self.sg_train_nodes[0] = self.sg_posNodes + self.sg_negNodes #训练集（正负样本）
        self.sg_test_nodes[0] = list(set(allNodes1).difference(set(self.sg_train_nodes[0]))) #其余节点作为测试集

        self.sg_test_nodes[0] = sorted(self.sg_test_nodes[0])
        self.sg_train_nodes[0] = sorted(self.sg_train_nodes[0])
        #提取子图中节点对应的节点特征
        self.sg_features[0] = self.features[self.sg_nodes[0], :]
        #计算目标标签
        if (label is None): #如果为空，则根据seedlabel生成
            self.sg_targets[0] = self.target[self.sg_nodes[0], :] #提取局部子图节点的原始标签
            self.sg_targets[0] = self.sg_targets[0] == seedLabel #生成二分类标签
            self.sg_targets[0] = self.sg_targets[0].astype(int) #转换成整数
        else: #不为空则直接使用label作为目标标签
            self.sg_targets[0] = [0] * len(allNodes1) #初始化为全0
            for i in label: #遍历label
                self.sg_targets[0][self.mapper[i]] = 1 #将label指定的节点为1（正类）
            self.sg_targets[0] = np.array(self.sg_targets[0])
            self.sg_targets[0] = self.sg_targets[0][:, np.newaxis]
            self.sg_targets[0] = self.sg_targets[0].astype(int)

        print("Value 0 %d, Value 1 %d" % (sum(self.sg_targets[0] == 0), sum(self.sg_targets[0] == 1)))
        #设置预测概率
        for x in self.sg_posNodes:
            self.sg_predProbs[x] = 1.0
            self.sg_predLabels[x] = 1
            if self.sg_targets[0][x] != 1.0:
                print("wrong")   #!!!!!!!!!!!!!!!!!!
        for x in self.sg_negNodes:
            self.sg_predProbs[x] = 0.0
            self.sg_predLabels[x] = 0
            if self.sg_targets[0][x] != 0:
                print("wrong")

        self.transfer_edges_and_nodes()
        self.TOPK_SIZE = self.args.community_size  #限制了找的社区大小


    def transfer_edges_and_nodes(self):
        '''
        Transfering the data to PyTorch format.
        '''
        for cluster in self.clusters:
            self.sg_nodes[cluster] = torch.LongTensor(self.sg_nodes[cluster]).to(self.device)
            self.sg_edges[cluster] = torch.LongTensor(self.sg_edges[cluster]).t().to(self.device)
            self.sg_train_nodes[cluster] = torch.LongTensor(self.sg_train_nodes[cluster]).to(self.device)
            self.sg_test_nodes[cluster] = torch.LongTensor(self.sg_test_nodes[cluster]).to(self.device)
            self.sg_features[cluster] = torch.FloatTensor(self.sg_features[cluster]).to(self.device)
            self.sg_targets[cluster] = torch.LongTensor(self.sg_targets[cluster]).to(self.device)


    def community_search(self, seed, trian_node, label,seed_comm):

        '''
        :param seed:初始节点，作为社区搜索的种子节点
        :param trian_node: 训练节点
        :param label: 节点标签
        :param seed_comm:种子（查询）节点所在的社区
        分为使用排名损失和不使用排名损失的社区搜索
        GNN training subgraph, heuristic search community without/with rking loss
        '''

        self.rankloss = 0 #排名损失初始化为0
        #构建以seed为中心的局部候选子图，如果无法构建打印错误信息并退出
        isOK = self.build_local_candidate(seed, trian_node, label)  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if isOK == 0:#如果构建局部候选子图失败，则直接返回0退出。
            print("cannot build a local subgraph")
            return 0
        for round in range(2): #迭代2轮循环，逐步优化社区搜索
            keepLayers = self.args.layers.copy() #备份当前层的GNN配置

            gcn_trainer = ClusterGCNTrainer(self.args, self) #用于GCN训练和测试
            begin_time = time.time()
            #返回节点权重nodeweight,预测标签predlabels，模型性能指标f1score
            #这个子图中的预测的权重，标签，和计算的f1-score
            nodeweight, predlabels, f1score = gcn_trainer.train_test_community()

            #时间和计数统计
            if 'gcn' not in self.time_map: #更新时间记录time_map和调用次数记录cntmap
                self.time_map['gcn'] = time.time() - begin_time
                self.cntmap['gcn']=1
            else:
                self.time_map['gcn'] = time.time() - begin_time + self.time_map['gcn']
                self.cntmap['gcn'] +=1

            #恢复之前备份的GNN层配置
            self.args.layers = keepLayers

            #更新预测概率和标签
            lc = LocalCommunity(self.args, self) #用于启发式社区搜索
            #补充测试节点的预测结果（因为正负样本节点初始的时候就赋值了）
            for i in range(len(self.sg_test_nodes[0])):
                self.sg_predProbs[self.sg_test_nodes[0][i]] = nodeweight[i].item()
                self.sg_predLabels[self.sg_test_nodes[0][i]] = predlabels[i].item()
            if(self.rankloss == 1):
                prefix='With_rking_loss'
            else:
                prefix="Without_rking_loss"

            #进行启发式的社区搜索
            #1.只用BFS
            # begin_time = time.time()
            # cnodes,topk = lc.locate_community_BFS_only(seed)
            # method = f'{prefix} BFS Only'
            # f1,pre,rec,using_time,res0 = lc.my_evaluate_community(cnodes,topk,seed_comm,method, time.time() - begin_time)

            #2.BFS Swap
            # begin_time = time.time()
            # cnodes,topk = lc.locate_community_BFS(seed)
            # method = f'{prefix}_BSF_Swap'
            # f1, pre, rec,using_time, res0 = lc.my_evaluate_community(cnodes,topk,seed_comm,method, time.time() - begin_time)

            #3.Greedy T
            # begin_time = time.time()
            # cnodes,topk = lc.locate_community_greedy(seed)
            # method = f'{prefix}_Greedy-T'
            # f1,pre,rec,using_time,res0 = lc.my_evaluate_community(cnodes,topk,seed_comm,method,time.time() - begin_time)

            #4. Greedy G
            begin_time = time.time()
            cnodes,topk = lc.locate_community_greedy_graph_prepath(seed)
            method = f'{prefix}_Greedy-G'
            f1,pre,rec,using_time,res0 = lc.my_evaluate_community(cnodes,topk,seed_comm,method,time.time() - begin_time)

            # begin_time = time.time()
            # cnodes, topk = lc.locate_community_BFS_only(seed)
            # lc.evaluate_community(topk, prefix + " BSF Only",time.time() - begin_time)
            # begin_time = time.time()
            # topk = lc.locate_community_BFS(seed)
            # lc.evaluate_community(topk, prefix + " BSF Swap", time.time() - begin_time)
            #
            # begin_time = time.time()
            # topk = lc.locate_community_greedy(seed)
            # lc.evaluate_community(topk, prefix + " Greedy-T", time.time() - begin_time)
            #
            # begin_time = time.time()
            # topk = lc.locate_community_greedy_graph_prepath(seed)
            # lc.evaluate_community(topk, prefix + " Greedy-G", time.time() - begin_time)

            #获取正负样本对用于排名损失
            self.posforrank, self.negforrank = self.getPNpairs()
            self.rankloss = 1

        #这个1是isOK
        return f1,pre,rec,using_time,method,1


    def getPNpairs(self):
        '''
        Get rking loss pair
        '''
        probs = self.sg_predProbs.copy()
        for x in self.sg_train_nodes[0]:
            probs[x] = 2
        for i in range(len(self.sg_targets[0])):
            if self.sg_targets[0][i] == 0:
                probs[i] = 2
        posIdx = np.argsort(np.array(probs))[0:int(self.args.train_ratio * self.args.subgraph_size / 2)]
        probs = self.sg_predProbs.copy()
        for x in self.sg_train_nodes[0]:
            probs[x] = -2
        for i in range(len(self.sg_targets[0])):
            if self.sg_targets[0][i] == 1:
                probs[i] = -2
        negIdx = np.argsort(-np.array(probs))[0:int(self.args.train_ratio * self.args.subgraph_size / 2)]
        return posIdx, negIdx

    def community_search_iteration(self,seed):
        '''
        GNN training subgraph, heuristic search community  with iteration without rking loss
        '''
        self.seed = seed
        self.oldpos = []
        self.oldneg = []
        self.allnode = [seed]
        for round in range(self.args.round):
            seed = self.seed
            isOK = self.build_local_candidate_iteration(seed)
            if isOK == 0:
                print("cannot build a local subgraph")
                return 0
            keepLayers = self.args.layers.copy()
            gcn_trainer = ClusterGCNTrainer(self.args, self)
            begin_time = time.time()
            nodeweight, predlabels, f1score = gcn_trainer.train_test_community()
            if 'gcn' not in self.time_map:
                self.time_map['gcn'] = time.time() - begin_time
                self.cntmap['gcn']=1
            else:
                self.time_map['gcn'] = time.time() - begin_time + self.time_map['gcn']
                self.cntmap['gcn'] +=1
            self.args.layers = keepLayers
            lc = LocalCommunity(self.args, self)

            for i in range(len(self.sg_test_nodes[0])):
                self.sg_predProbs[self.sg_test_nodes[0][i]] = nodeweight[i].item()
                self.sg_predLabels[self.sg_test_nodes[0][i]] = predlabels[i].item()

            prefix=str(round)+' Round'
            begin_time = time.time()
            cnodes,topk = lc.locate_community_BFS_only(seed)
            lc.evaluate_community(topk, prefix + " BSF Only",time.time() - begin_time)


            begin_time = time.time()
            cnodes,topk = lc.locate_community_BFS(seed)
            lc.evaluate_community(topk, prefix + " BSF Swap", time.time() - begin_time)




            begin_time = time.time()
            cnodes,topk = lc.locate_community_greedy(seed)
            lc.evaluate_community(topk, prefix + " Greedy-T", time.time() - begin_time)



            begin_time = time.time()
            cnodes,topk = lc.locate_community_greedy_graph_prepath(seed)
            lc.evaluate_community(topk, prefix + " Greedy-G", time.time() - begin_time)

        return 1
    def methods_result(self):
        '''
        save result
        '''
        file_handle = open('results.txt', mode='a+')
        now = datetime.datetime.now()
        sTime = now.strftime("%Y-%m-%d %H:%M:%S")
        file_handle.write(sTime + "\n")
        # 将args中的关键字存入
        args = vars(self.args)
        keys = sorted(args.keys())
        keycontent = [[k.replace("_", " ").capitalize(), args[k]] for k in keys]
        for x in keycontent:
            file_handle.writelines(str(x) + "\n")
        file_handle.write("gcn time %f \n" % (self.time_map['gcn'] / self.cntmap['gcn']))
        print("gcn time %f " % (self.time_map['gcn'] / self.cntmap['gcn']))
        for method in self.methods:
            if isinstance(self.methods[method], list):
                pre = self.methods[method][0] / self.cntmap[method]
                rpre= self.methods[method][1] / self.cntmap[method] #without posnode
                times = self.time_map[method] / self.cntmap[method] #花费的时间
                print(
                    "%s Method achieve the average precision= %f precision without posnode = %f using %d seeds with avgtime=%f s " % (
                        method, pre, rpre,self.cntmap[method], times))
                file_handle.writelines(
                    "%s Method achieve the average precision= %f precision without posnode = %f using %d seeds with time=%f s\n" % (
                        method, pre, rpre,self.cntmap[method], times))

        file_handle.writelines("\n")
        file_handle.close()
