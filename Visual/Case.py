
import networkx as nx
import matplotlib.pyplot as plt

def draw_community(G, query, community_nodes, title, ax):
    # 创建子图
    subgraph = G.subgraph(community_nodes | {query})

    # 设置节点颜色：query节点为白色，其余为黑色
    node_colors = ['white' if node == query else 'black' for node in subgraph.nodes()]
    node_border_colors = ['black'] * len(subgraph.nodes())

    # 节点位置可重复使用
    pos = nx.spring_layout(subgraph, seed=42)

    nx.draw_networkx_nodes(subgraph, pos, node_color=node_colors, edgecolors=node_border_colors, node_size=200, ax=ax)
    nx.draw_networkx_edges(subgraph, pos, ax=ax, width=0.8)
    ax.set_title(title)
    ax.axis('off')

# 示例：创建一个图并调用上述函数画出六张图
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.flatten()
# 示例加载图（实际应替换为你的 Amazon 图）
G = nx.karate_club_graph()
query = 0

# 示例：使用邻居模拟搜索结果（实际替换成你运行模型的结果）
kcore_result = set(nx.k_core(G, k=3).nodes())
ktruss_result = set(G.neighbors(query))  # 示例
kecc_result = set(list(G.neighbors(query))[:3])
ctc_result = set(list(G.neighbors(query))[:4])
coclep_result = set(nx.ego_graph(G, query, radius=2).nodes())
ground_truth_result = set(G.neighbors(query)) | {1, 2, 3}  # 假设 ground-truth

# 绘图
draw_community(G, query, kcore_result, 'K-Core', axes[0])
draw_community(G, query, ktruss_result, 'K-Truss', axes[1])
draw_community(G, query, kecc_result, 'K-ECC', axes[2])
draw_community(G, query, ctc_result, 'CTC', axes[3])
draw_community(G, query, coclep_result, 'COCLEP', axes[4])
draw_community(G, query, ground_truth_result, 'Ground-truth', axes[5])

plt.tight_layout()
plt.show()
