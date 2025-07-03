import argparse
import os


def load_kclique_community_results(result_dir):
    """
    加载 k-kclique 方法生成的结果。

    每个查询节点 q 对应一个 output_clique_q.txt 文件：
    - 第一行存储 q 所在社区的节点个数
    - 第二行以空格隔开存储 q 所在社区的所有节点集合

    :param result_dir: k-kclique 方法生成结果的目录
    :return: 一个字典，键为查询节点，值为其所在社区的节点集合
    """
    kclique_results = {}
    oknum=0 #测试集中能够正确找到clique的查询节点数量。
    for filename in os.listdir(result_dir):
        if filename.startswith('output_clique_') and filename.endswith('.txt'):
            #根据文件名得到查询节点
            query_node = filename.replace('output_clique_', '').replace('.txt', '')
            with open(os.path.join(result_dir, filename), 'r') as f:
                lines = f.readlines()
                if len(lines) < 2:
                    # raise ValueError(f"File {filename} does not have the expected format.")
                    continue #跳过这个查询节点。
                # 第二行是社区节点集合
                community_nodes = set(lines[1].strip().split())
                kclique_results[query_node] = community_nodes
                oknum += 1

    return kclique_results,oknum

def load_ground_truth(file_path):
    '''
    每行是查询节点，以及查询节点所在的社区
    :param file_path:
    :return:
    '''
    ground_truth = {}
    with open(file_path, 'r') as f:
        for line in f:
            query, community = line.strip().split(',')
            ground_truth[query] = set(community.split())
    return ground_truth
def calculate_metrics(ground_truth, kcore_results):
    precision_list, recall_list, f1_list = [], [], []
    for query_node, true_community in ground_truth.items():
        predicted_community = kcore_results.get(query_node, set())

        # Calculate precision, recall, and f1-score
        tp = len(true_community & predicted_community)
        fp = len(predicted_community - true_community)
        fn = len(true_community - predicted_community)

        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    # Return the average metrics
    avg_precision = sum(precision_list) / len(precision_list)
    avg_recall = sum(recall_list) / len(recall_list)
    avg_f1 = sum(f1_list) / len(f1_list)

    return avg_precision, avg_recall, avg_f1
def save_metrics_to_file(file_path, precision, recall, f1,total_query,oknum=None,coverage=None):
    if oknum is None:
        with open(file_path, 'w') as f:
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1-Score: {f1:.4f}\n")
            f.write(f"Total: {total_query}\n")
    else:
        with open(file_path, 'w') as f:
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1-Score: {f1:.4f}\n")

            f.write(f"Total: {total_query}\n")
            f.write(f"Oknum: {oknum}\n")
            f.write(f"Coverage: {coverage:.4f}\n")
    print(f"Metrics saved to {file_path}")

def evaluate_kclique_result(dataset='cora', attack='random_add', ptb_rate=0.20, k=3, root='./Dataset'):
    """
    加载 k-clique 结果并计算评估指标。

    :param dataset: 数据集名称
    :param attack: 扰动类型，如 'none', 'random_add' 等
    :param ptb_rate: 扰动率
    :param k: k 值（保留接口以兼容其他方法）
    :param root: 数据根目录
    :return: dict 包含 precision、recall、f1、coverage 等评估指标
    """
    if attack != 'none':
        target_dataset = f'{dataset}_{attack}_{ptb_rate}'
    else:
        target_dataset = dataset

    result_dir = f"{root}/{target_dataset}"

    # 加载 ground-truth 数据
    ground_truth_path = f'{result_dir}/comms.txt'
    if not os.path.exists(ground_truth_path):
        raise FileNotFoundError(f"Ground-truth file not found: {ground_truth_path}")
    ground_truth = load_ground_truth(ground_truth_path)

    # 加载 k-clique 结果
    result_path = f'{result_dir}/clique'
    if not os.path.exists(result_path):
        raise FileNotFoundError(f"Clique result directory not found: {result_path}")
    results, oknum = load_kclique_community_results(result_path)

    # 计算指标
    precision, recall, f1 = calculate_metrics(ground_truth, results)

    total_queries = len(ground_truth)
    coverage = oknum / total_queries if total_queries > 0 else 0.0

    # 打印结果
    print(f"Evaluating on: {target_dataset}")
    print(f'oknum: {oknum}, total_queries: {total_queries}, coverage: {coverage:.4f}')
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    # 保存指标
    metric_file = f"{result_dir}/metrics.txt"
    save_metrics_to_file(metric_file, precision, recall, f1, total_queries, oknum, coverage)

    return {
        'dataset': target_dataset,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'coverage': coverage,
        'oknum': oknum,
        'total_queries': total_queries
    }


if __name__ == "__main__":
    # Input files and directory
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=15, help='Random seed.')
    parser.add_argument('--type', type=str, default='kclique', help='method type',
                        choices=['kclique', 'ktruss'])
    #choices=['football', 'facebook_all', 'cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'],
    parser.add_argument('--dataset', type=str, default='cora',help='dataset')
    # choices=['none','add','random_add','gaddm','del','gdelm','gflipm']
    parser.add_argument('--attack', type=str, default='random_add', help='attack type')
    parser.add_argument('--ptb_rate', type=float, default=0.20, help='pertubation rate')
    parser.add_argument('--k', type=int, default=3, help='k value in ktruss')
    parser.add_argument('--root', type=str, default='./Dataset', help='data store root')
    # 配置
    args = parser.parse_args()

    if args.attack != 'none':
        target_dataset = f'{args.dataset}_{args.attack}_{args.ptb_rate}'
    else:
        target_dataset = args.dataset

    result_dir = f"{args.root}/{target_dataset}"

    datasets = ['cora']
    attack = ['cdelm']
    rates = [0.2, 0.4, 0.6, 0.8]

    for ds in datasets:
        for at in attack:
            for rate in rates:
                evaluate_kclique_result(dataset=ds, attack=at, ptb_rate=rate)
    evaluate_kclique_result(dataset='cora', attack='none', ptb_rate=0)


    # #load ground-truth数据
    # ground_truth = load_ground_truth(f'{result_dir}/comms.txt')
    # #加载查询结果
    # results, oknum = load_kclique_community_results(f'{result_dir}/clique')
    #
    # # 计算指标
    # precision, recall, f1 = calculate_metrics(ground_truth, results)
    #
    # total_queries = len(ground_truth)  # 总的查询
    # coverage = oknum / total_queries if total_queries > 0 else 0.0  # 查询的成功率
    #
    # # Print results
    # print(f'oknum: {oknum}, total_queries: {total_queries}, coverage: {coverage:.4f}')
    # print(f"Precision: {precision:.4f}")
    # print(f"Recall: {recall:.4f}")
    # print(f"F1-Score: {f1:.4f}")
    #
    # metric_file = f"{result_dir}/metrics.txt"''
    # save_metrics_to_file(metric_file, precision, recall, f1,total_queries,oknum,coverage)

