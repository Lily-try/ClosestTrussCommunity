'''
处理coclep中的football和facebook_full数据集
'''
import os

from collections import defaultdict, Counter

def truth2comms(input_file,output_file):
    communities = {}    # 存储所有社区编号和节点信息
    with open(input_file, 'r') as f:    # 读取原文件并提取信息
        for line in f:
            parts = line.strip().split()
            community_id = parts[0][4:]  # 社区编号
            nodes = parts[1:]  # 节点列表
            communities[community_id] = nodes
    # 写入新文件
    with open(output_file, 'w') as f:
        # 写入所有社区编号
        f.write(" ".join(communities.keys()) + "\n")
        # 依次写入每个社区的所有节点
        for community_id in communities:
            f.write(" ".join(communities[community_id]) + "\n")
    print(f"文件已成功创建为 {output_file}")

# 读取facebook_full_truth.txt并重新组织内容写入facebook_full_comms.txt
# input_file = '../data/facebook_all/facebook_all_truth.txt'
# output_file = '../data/facebook_all/facebook_all_comms.txt'
# truth2comms(input_file,output_file)

#将邻接矩阵从大小到达排序
# def facebook_all_sort(input_file,output_file):
#     # 文件路径
#     # 读取文件并将每一行存储为(源点, 目的顶点)的元组
#     edges = []
#     with open(input_file, 'r') as f:
#         for line in f:
#             src, dest = map(int, line.strip().split())  # 转换为整数方便排序
#             edges.append((src, dest))
#     # 按照源点从大到小排序
#     edges.sort(key=lambda x: x[0])
#     # 将排序后的边写入新的文件
#     with open(output_file, 'w') as f:
#         for src, dest in edges:
#             f.write(f"{src} {dest}\n")
#     print(f"文件已成功按照源点从大到小排序并保存为 {output_file}")
#
# input_file = '../data/facebook_all/facebook_all.txt'
# output_file = '../data/facebook_all/facebook_all_sorted.txt'
# facebook_all_sort(input_file,output_file)


if __name__ == '__main__':
    input_file = '../data/facebook_all/facebook_all.txt'

    degree_count = defaultdict(int)     # 初始化一个字典来存储每个顶点的度数

    # 读取文件并计算每个顶点的度数
    with open(input_file, 'r') as f:
        for line in f:
            src, dest = map(int, line.strip().split())
            degree_count[src] += 1
            degree_count[dest] += 1

    # 计算平均度数
    total_degrees = sum(degree_count.values())
    num_vertices = len(degree_count)
    average_degree = total_degrees / num_vertices

    top_5_max = sorted(degree_count.items(), key=lambda x: x[1], reverse=True)[:10]
    top_5_min = sorted(degree_count.items(), key=lambda x: x[1])[:10]

    print(f"平均度数: {average_degree:.2f}")
    print("\n度数最大的前10个顶点及其度数:")
    for vertex, degree in top_5_max:
        print(f"顶点: {vertex}, 度数: {degree}")
    print("\n度数最小的前10个顶点及其度数:")
    for vertex, degree in top_5_min:
        print(f"顶点: {vertex}, 度数: {degree}")

    # 统计每种度数出现的节点数量
    degree_frequency = Counter(degree_count.values())
    most_common_degree, max_nodes_count = degree_frequency.most_common(1)[0]  # 找出节点数目最多的度数
    print(f"节点数目最多的度数是: {most_common_degree}, 具有该度数的节点数量为: {max_nodes_count}")
