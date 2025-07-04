import torch
import random
import numpy as np
from tqdm import trange
from src.layers import StackedGCN
from sklearn import metrics


class ClusterGCNTrainer(object):
    def __init__(self, args, clustering_machine):
        self.args = args
        self.clustering_machine = clustering_machine
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.create_model()

    def create_model(self):
        """
        Creating a StackedGCN and transferring to CPU/GPU.
        """
        self.model = StackedGCN(self.args, self.clustering_machine.feature_count, self.clustering_machine.class_count)
        self.model = self.model.to(self.device)

    def do_forward_pass(self, cluster):
        """
        Making a forward pass with data from a given partition.
        :param cluster: Cluster index.
        :return average_loss: Average loss on the cluster.
        :return node_count: Number of nodes.
        """
        edges = self.clustering_machine.sg_edges[cluster].to(self.device)
        macro_nodes = self.clustering_machine.sg_nodes[cluster].to(self.device)
        train_nodes = self.clustering_machine.sg_train_nodes[cluster].to(self.device)
        features = self.clustering_machine.sg_features[cluster].to(self.device)
        target = self.clustering_machine.sg_targets[cluster].to(self.device).squeeze()
        predictions = self.model(edges, features)
        #print(self.model.embeddings.size())
        average_loss = torch.nn.functional.nll_loss(predictions[train_nodes], target[train_nodes])
        #average_loss = torch.nn.functional.cross_entropy(predictions[train_nodes], target[train_nodes])
        if self.clustering_machine.rankloss ==1:
            softmax_prediction = torch.nn.functional.softmax(predictions, dim=1)
            pos = softmax_prediction[self.clustering_machine.posforrank][:,1]
            neg = softmax_prediction[self.clustering_machine.negforrank][:,1]
            target = [1] * len(self.clustering_machine.posforrank)
            target = torch.LongTensor(target).to(self.device)
            rank_loss = torch.nn.functional.margin_ranking_loss(pos, neg, target)
            average_loss = average_loss + rank_loss
        node_count = train_nodes.shape[0]
        return average_loss, node_count

    def update_average_loss(self, batch_average_loss, node_count):
        """
        Updating the average loss in the epoch.
        :param batch_average_loss: Loss of the cluster. 
        :param node_count: Number of nodes in currently processed cluster.
        :return average_loss: Average loss in the epoch.
        """
        self.accumulated_training_loss = self.accumulated_training_loss + batch_average_loss.item()*node_count
        self.node_count_seen = self.node_count_seen + node_count
        average_loss = self.accumulated_training_loss/self.node_count_seen
        return average_loss

    def do_prediction(self, cluster):
        """
        Scoring a cluster.
        :param cluster: Cluster index.
        :return prediction: Prediction matrix with probabilities.
        :return target: Target vector.
        """
        edges = self.clustering_machine.sg_edges[cluster].to(self.device)
        macro_nodes = self.clustering_machine.sg_nodes[cluster].to(self.device)
        test_nodes = self.clustering_machine.sg_test_nodes[cluster].to(self.device)
        features = self.clustering_machine.sg_features[cluster].to(self.device)
        target = self.clustering_machine.sg_targets[cluster].to(self.device).squeeze()
        target = target[test_nodes]
        prediction = self.model(edges, features)
        prediction = prediction[test_nodes,:]
        return prediction, target





    def train_test_community(self):
        """
        Training a model.
        """
        ### Train Phrase

        print("Training started")
        epochs = trange(self.args.epochs, desc="Train Loss") #使用 tqdm.trange 显示训练进度条（例如 100 epochs）。
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.model.train()
        #for _ in range(self.args.epochs):
        for _ in epochs:
            random.shuffle(self.clustering_machine.clusters) #随机打乱 cluster 顺序，增加模型泛化能力。
            self.node_count_seen = 0
            self.accumulated_training_loss = 0
            for cluster in self.clustering_machine.clusters:
                self.optimizer.zero_grad()
                batch_average_loss, node_count = self.do_forward_pass(cluster)
                batch_average_loss.backward()
                self.optimizer.step()
                average_loss = self.update_average_loss(batch_average_loss, node_count)
                # print("average_loss is "+str(average_loss))
            epochs.set_description("Train Loss: %g" % round(average_loss, 4))

        ### Test Phrase
        self.model.eval()
        self.predictions = [] #Tensor (394,2)
        self.targets = []  #target是Tensor"{394,)，全是0，1，，
        for cluster in self.clustering_machine.clusters:
            prediction, target = self.do_prediction(cluster)
            self.predictions.append(prediction.cpu().detach().numpy())
            self.targets.append(target.cpu().detach().numpy())
        self.targets = np.concatenate(self.targets) #392
        self.predictions = np.concatenate(self.predictions).argmax(1) #392

        score = metrics.f1_score(self.targets, self.predictions, average="micro")

        softmax_prediction=torch.nn.functional.softmax(prediction, dim =1)
        softmax_prediction = softmax_prediction.data[:,1]


        return softmax_prediction, self.predictions, score






