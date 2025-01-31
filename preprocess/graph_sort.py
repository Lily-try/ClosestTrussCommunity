

def process_graph_file(input_file, output_file):
    # 使用集合存储无向边，确保不重复
    edges = set()
    with open(input_file, 'r') as f:
        for line in f:
            # 提取两个节点，并按顺序存储
            node1, node2 = map(int, line.strip().split())
            # 确保无向边用 (较小节点, 较大节点) 的形式存储
            edge = tuple(sorted((node1, node2)))
            edges.add(edge)

    # 将边排序：先按第一个节点，再按第二个节点
    sorted_edges = sorted(edges)

    # 写入结果到新文件
    with open(output_file, 'w') as f:
        for edge in sorted_edges:
            f.write(f"{edge[0]} {edge[1]}\n")


# 使用示例
# input_file = '../data/football/football.txt'  # 原始文件路径
# output_file = '../data/football/processed_football.txt'  # 输出文件路径


input_file = '../data/fb107/fb107.edges'  # 原始文件路径
output_file = '../data/fb107/processed_fb107.txt'  # 输出文件路径
process_graph_file(input_file, output_file)
