
import os
import torch
import networkx as nx

def relabel_graph(graph):
    mapping = {old_id: new_id for new_id, old_id in enumerate(graph.nodes())}
    graph_new = nx.relabel_nodes(graph, mapping)
    return graph_new, mapping  # mapping: old_id -> new_id

def reorder_features(nodes_feats, mapping):
    n_nodes = len(mapping)
    dim = nodes_feats.shape[1]
    reordered = torch.zeros((n_nodes, dim), dtype=nodes_feats.dtype)
    for old_id, new_id in mapping.items():
        reordered[new_id] = nodes_feats[old_id]
    return reordered

def remap_query_data(train, val, test, mapping):
    def remap_list(data):
        new_data = []
        for item in data:
            if len(item) == 3:
                q, pos, comm = item
                q = mapping[q]
                pos = [mapping[p] for p in pos]
                comm = [mapping[c] for c in comm]
                new_data.append((q, pos, comm))
            elif len(item) == 2:
                q, comm = item
                q = mapping[q]
                comm = [mapping[c] for c in comm]
                new_data.append((q, comm))
        return new_data
    return remap_list(train), remap_list(val), remap_list(test)

def remap_cluster_membership(old_cluster_membership, mapping):
    return {mapping[old_id]: cluster_id for old_id, cluster_id in old_cluster_membership.items()}

def preprocess_graph_dataset(graph, nodes_feats, train, val, test, cluster_membership):
    # 1. 重编号图节点
    graph, mapping = relabel_graph(graph)

    # 2. 特征按编号重排
    nodes_feats = reorder_features(nodes_feats, mapping)

    # 3. 查询数据重排
    train, val, test = remap_query_data(train, val, test, mapping)

    # 4. 聚类结果同步编号
    cluster_membership = remap_cluster_membership(cluster_membership, mapping)

    return graph, nodes_feats, train, val, test, cluster_membership