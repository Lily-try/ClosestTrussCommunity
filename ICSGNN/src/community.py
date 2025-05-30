import networkx as nx



class LocalCommunity(object):

    def __init__(self, args, upgraph):
        self.args = args
        self.upgraph = upgraph
    def evaluate_community(self, top_index, name,using_time):
        '''
            评估搜索到的社区的质量
            :param top_index   预测为正的索引
            :param name: 当前使用的方法
        '''
        y_pred = [0] * len(top_index) #预测的标签数组   top_index和target都是是<class 'list'>，是TP+FP
        y_true = [0] * len(top_index) #真实标签数组
        ok = 0 #正确预测的样本数量
        pos = 0 #top_index中的节点是否是正样本节点的计数
        #获取真实标签
        target = self.upgraph.sg_targets[0].cpu().detach().numpy().tolist()

        for i in range(len(top_index)):
            y_pred[i] = self.upgraph.sg_predLabels[top_index[i]]
            y_true[i] = target[top_index[i]][0]
            if top_index[i] in self.upgraph.sg_posNodes or top_index[i] in self.upgraph.posforrank:
                pos+=1
            if(y_true[i]==1):#TP（由于pos被提前给定了，所以会在减去pos）
                ok+=1
        results= []
        pre=0
        results.append(ok/len(top_index)) #直接计算的精度是results[0]
        if (len(top_index) != pos):
            pre=(ok-pos)/(len(top_index)-pos)
        results.append(pre) #消除了提前给定的pos_nodes的影响是result[1]
        if name not in self.upgraph.methods:
            self.upgraph.methods[name]= results
            self.upgraph.time_map[name] =  using_time
            self.upgraph.cntmap[name]=1
        else:
            self.upgraph.methods[name] =[self.upgraph.methods[name][i]+results[i] for i in range(len(results))]
            self.upgraph.time_map[name] += using_time
            self.upgraph.cntmap[name]+=1
        print(name + " Precision={:.4f}  Precision without posnode={:.4f} using {:.4f}s".format(results[0],results[1],using_time))
        return results[0]

    def my_evaluate_community(self,cnodes,top_index,seed_comm, name,using_time):
        '''
            评估搜索到的社区的质量
            :param top_index   预测为正的索引
            :param seed_comm 当前样本的实际社区
            :param name: 当前使用的方法
        '''
        y_pred = [0] * len(top_index) #这30个节点的预测的标签数组   top_index和target都是是<class 'list'>，是TP+FP
        y_true = [0] * len(top_index) #这30个节点的真实标签数组
        ok = 0 #正确预测的样本数量
        pos = 0 #top_index中的节点是否是正样本节点的计数
        #获取当前这个子图中所有节点的真实标签
        target = self.upgraph.sg_targets[0].cpu().detach().numpy().tolist()
        for i in range(len(top_index)): #得到这30个节点的预测标签和真实标签
            y_pred[i] = self.upgraph.sg_predLabels[top_index[i]]
            y_true[i] = target[top_index[i]][0]
            if top_index[i] in self.upgraph.sg_posNodes or top_index[i] in self.upgraph.posforrank:
                pos+=1
            if(y_true[i]==1):#TP（由于pos被提前给定了，所以会在减去pos）
                ok+=1

        lists = [x for x in cnodes if x in seed_comm]  # TP（将正类预测为正类）交集，同时在com_find和comm中出现的元素。
        if len(lists) == 0:
            print('都是0')
            return 0.0, 0.0, 0.0
        pre = (len(lists)-pos) * 1.0 / (len(cnodes)-pos)  # pre = TP/(TP+FP) = TP/comm_find
        rec = len(lists) * 1.0 / len(seed_comm)  # recall = TP/(TP+FN) = TP/comm
        # ACC= (TP+TN)/(TP+TN+FP+FN)
        f1 = 2 * pre * rec / (pre + rec)  # F1=2*P*R/(P+R)
        print(name + "Precision={:.4f},Recall{:.4f},F1-score{:.4f},using {:.4f}s".format(pre,rec,f1,using_time))


        results= []
        pre=0
        #计算精度（未取出提前给定的pos_nodes的影响）results[0]
        results.append(ok/len(top_index))
        #消除了提前给定的pos_nodes的影响是result[1]
        if (len(top_index) != pos):
            pre=(ok-pos)/(len(top_index)-pos)
        results.append(pre)
        if name not in self.upgraph.methods:
            self.upgraph.methods[name]= results
            self.upgraph.time_map[name] =  using_time
            self.upgraph.cntmap[name]=1
        else:
            self.upgraph.methods[name] =[self.upgraph.methods[name][i]+results[i] for i in range(len(results))]
            self.upgraph.time_map[name] += using_time
            self.upgraph.cntmap[name]+=1
        # print(name + " Precision={:.4f}  Precision without posnode={:.4f} using {:.4f}s".format(results[0],results[1],using_time))
        return f1,pre,rec,using_time,results[0]


    def my_eval(self,top_index,name,using_time):
        '''

        :param top_index: 找到的索引
        :param name: 姓名
        :param using_time:
        :return:
        '''
        y_pred = [0] * len(top_index) #预测的标签数组   top_index和target都是是<class 'list'>，是TP+FP
        y_true = [0] * len(top_index) #真实标签数组
        ok = 0 #正确预测的样本数量
        pos = 0 #top_index中的节点是否是正样本节点的计数
        #获取真实标签
        target = self.upgraph.sg_targets[0].cpu().detach().numpy().tolist()

        for i in range(len(top_index)):
            y_pred[i] = self.upgraph.sg_predLabels[top_index[i]]
            y_true[i] = target[top_index[i]][0]
            if top_index[i] in self.upgraph.sg_posNodes or top_index[i] in self.upgraph.posforrank:
                pos+=1
            if(y_true[i]==1):#TP（由于pos被提前给定了，所以会在减去pos）
                ok+=1
        results= []
        pre=0
        results.append(ok/len(top_index)) #直接计算的精度
        if (len(top_index) != pos):
            pre=(ok-pos)/(len(top_index)-pos)
        results.append(pre) #消除了提前给定的pos_nodes的影响


        if name not in self.upgraph.methods:
            self.upgraph.methods[name]= results
            self.upgraph.time_map[name] =  using_time
            self.upgraph.cntmap[name]=1
        else:
            self.upgraph.methods[name] =[self.upgraph.methods[name][i]+results[i] for i in range(len(results))]
            self.upgraph.time_map[name] += using_time
            self.upgraph.cntmap[name]+=1
        print(name + " Precision={:.4f}  Precision without posnode={:.4f} using {:.4f}s".format(results[0],results[1],using_time))

        return results



    def locate_community_BFS_only(self, seed):
        '''
        Search community using bfs only
        '''
        cnodes = []
        cnodes.append(seed)
        pos =0
        while pos < len(cnodes) and pos < self.upgraph.TOPK_SIZE and len(cnodes) < self.upgraph.TOPK_SIZE:
            cnode = cnodes[pos]
            for nb in self.upgraph.subgraph.neighbors(cnode):
                if nb not in cnodes and len(cnodes) < self.upgraph.TOPK_SIZE:
                    cnodes.append(nb)
            pos = pos + 1

        topk = [self.upgraph.mapper[node] for node in cnodes]
        return cnodes,topk #返回了前 TOPK_SIZE 个社区节点的索引列表。


    def locate_community_BFS(self, seed):
        '''
        Search community using bfs with swap
        '''
        cnodes = []
        cnodes.append(seed)
        pos =0
        while pos < len(cnodes) and pos < self.upgraph.TOPK_SIZE and len(cnodes) < self.upgraph.TOPK_SIZE:
            cnode = cnodes[pos]
            for nb in self.upgraph.subgraph.neighbors(cnode):
                if nb not in cnodes and len(cnodes) < self.upgraph.TOPK_SIZE:
                    cnodes.append(nb)
            pos = pos + 1
        for pos in range(len(cnodes)):
            cnode = cnodes[pos]
            for nb in self.upgraph.subgraph.neighbors(cnode):
                 pos1= pos+1
                 while pos1<len(cnodes) and nb not in cnodes:
                    next = cnodes[pos1]

                    if self.upgraph.sg_predProbs[self.upgraph.mapper[nb]]>self.upgraph.sg_predProbs[self.upgraph.mapper[next]]:
                        cnodes[pos1] = nb
                    pos1 = pos1 +1
        topk = [self.upgraph.mapper[node] for node in cnodes]
        return cnodes,topk


    def locate_community_greedy(self, seed):
        '''
        Search community using  greedy-T
        '''
        cnodes = [ ]
        parents =[0] * len(self.upgraph.subgraph.nodes)
        cnodes.append(seed)
        parents[cnodes.index(seed)] =-1
        pos =0
        while pos < len(cnodes) :
            cnode = cnodes[pos]
            for nb in self.upgraph.subgraph.neighbors(cnode):
                if nb not in cnodes:
                    cnodes.append(nb)
                    parents[cnodes.index(nb)]=cnodes.index(cnode)
            pos = pos + 1
        topkidx=[]
        topkidx.append(0)
        for _ in range(self.upgraph.TOPK_SIZE):
            if(len(topkidx)==self.upgraph.TOPK_SIZE):break
            probs = [-1.0] * len(self.upgraph.subgraph.nodes)
            hops = [1] * len(self.upgraph.subgraph.nodes)
            for i in range(len(cnodes)):
                cnode = cnodes[i]
                if i in topkidx:
                    continue
                prob = self.upgraph.sg_predProbs[self.upgraph.mapper[cnode]]
                while parents[cnodes.index(cnode)] != -1:
                    cnode = cnodes[parents[cnodes.index(cnode)]]
                    if cnodes.index(cnode) in topkidx:
                        break
                    prob = prob + self.upgraph.sg_predProbs[self.upgraph.mapper[cnode]]
                    hops[i] = hops[i]+1
                probs[i] = prob
            for i in range(len(probs)):
                probs[i] = probs[i]/hops[i]
            maxValueIdx = probs.index(max(probs))
            if len(topkidx)<self.upgraph.TOPK_SIZE:
                topkidx.append(maxValueIdx)
            while parents[maxValueIdx] != -1:
                maxValueIdx = parents[maxValueIdx]
                if maxValueIdx not in topkidx and len(topkidx)<self.upgraph.TOPK_SIZE:
                    topkidx.append(maxValueIdx)
        topk =[self.upgraph.mapper[cnodes[idx]] for idx in topkidx]
        topk_raw_nodes = [cnodes[idx] for idx in topkidx]
        # print('len tok_raw_nodes:',len(topk_raw_nodes))
        # print('len nnodes:',len(cnodes))
        return topk_raw_nodes,topk


    def locate_community_greedy_graph_prepath(self, seed):
        '''
        Search community using  greedy-G
        '''
        cnodes = []
        cnodes.append(seed)
        pos = 0
        while pos < len(cnodes):
            cnode = cnodes[pos]
            for nb in self.upgraph.subgraph.neighbors(cnode):
                if nb not in cnodes:
                    cnodes.append(nb)
            pos = pos + 1
        topkidx = []
        topkidx.append(seed)
        for iter in range(self.upgraph.TOPK_SIZE):
            if (len(topkidx) == self.upgraph.TOPK_SIZE): break
            candidates = [-1.0] * 3 * self.upgraph.TOPK_SIZE
            for num in range(len(topkidx)):
                candidates[num] = 1.0
            probs = [-1.0] * len(self.upgraph.subgraph.nodes)
            hops = [1] * len(self.upgraph.subgraph.nodes)
            paths = self.get_all_path(self.upgraph.subgraph, topkidx,cnodes)
            for i in range(len(cnodes)):
                cnode = cnodes[i]
                if cnode in topkidx:
                    continue
                prob = self.upgraph.sg_predProbs[self.upgraph.mapper[cnode]]
                if prob < min(candidates):
                    continue
                for x in paths[i]:
                    prob = prob + self.upgraph.sg_predProbs[self.upgraph.mapper[x]]
                probs[i] = prob
                hops[i] = len(paths[i]) + 1
                prob = prob / hops[i]
                idx = candidates.index(min(candidates))
                candidates[idx] = prob
            for i in range(len(probs)):
                probs[i] = probs[i] / hops[i]
            maxValueIdx = probs.index(max(probs))
            if len(topkidx) < self.upgraph.TOPK_SIZE:
                topkidx.append(cnodes[maxValueIdx])
            for x in paths[maxValueIdx]:
                if len(topkidx) < self.upgraph.TOPK_SIZE:
                    topkidx.append(x)
        topk = [self.upgraph.mapper[idx] for idx in topkidx]
        return topkidx,topk
    def get_all_path(self, pgraph, topkidx,cnodes):
        '''
        Get the community to the other nodes' shortest paths
        '''
        g = nx.Graph()
        seed = topkidx[0]
        for u, v in pgraph.edges:
            if (u in topkidx) and (v in topkidx):
                continue
            else:
                if (u in topkidx): u = seed
                if (v in topkidx): v = seed
                g.add_edge(u, v)
        p1 = nx.shortest_path(g, source=seed)
        paths = [[]] * len(pgraph.nodes)
        for item in p1.keys():
            idx=cnodes.index(item)
            path = p1[item][1:]
            path.reverse()
            paths[idx] = path[1:]
        return paths
