import argparse
import os
from collections import defaultdict
import datetime


def load_ground_truth(file_path):
    '''
    每行是查询节点，以及查询节点所在的社区
    :param file_path:
    :return: 字典，每个查询节点所在的社区。
    # '''
    # ground_truth = {}
    # with open(file_path, 'r') as f:
    #     for line in f:
    #         query, community = line.strip().split(',')
    #         ground_truth[query] = set(community.split())
    # return ground_truth
    ground_truth = []
    with open(file_path, 'r') as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line or ',' not in line:
                print(f"[Warning] 第 {idx} 行格式错误: {line}")
                continue
            try:
                query, community = line.split(',', 1)
                # nodes = community.split()
                nodes = list(map(int, community.split()))
                ground_truth.append(set(nodes))
            except Exception as e:
                print(f"[Error] 第 {idx} 行解析失败: {line}, 错误: {e}")
    return ground_truth

def load_kcore_community_results(result_dir):
    kcore_results = {}
    for filename in os.listdir(result_dir):
        if filename.endswith('.txt'):
            query_node = filename.split('.')[0]
            edges = set()
            with open(os.path.join(result_dir, filename), 'r') as f:
                for line in f:
                    if line.strip() == '-1':
                        break
                    u, v = line.strip().split()
                    edges.add(u)
                    edges.add(v)
            kcore_results[query_node] = edges
    return kcore_results


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

def compute_precision(pred, true):
    if not pred: return 0
    return len(set(pred) & set(true)) / len(pred)

def compute_recall(pred, true):
    if not true: return 0
    return len(set(pred) & set(true)) / len(true)

def compute_f1(p, r):
    if p + r == 0: return 0
    return 2 * p * r / (p + r)

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

def save_metrics_to_file(file_path, precision, recall, f1,oknum,total_query,coverage):
    with open(file_path, 'w') as f:
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-Score: {f1:.4f}\n")
        f.write(f"Oknum: {oknum}\n")
        f.write(f"Total: {total_query}\n")
        f.write(f"Coverage: {coverage:.4f}\n")
    print(f"Metrics saved to {file_path}")

def val_CTC(dataset,attack,ptb_rate,num=0):
    # load ground-truth数据
    if attack !='none':
        dataset = f'{dataset}_{attack}_{ptb_rate}'
    else:
        dataset =f'{dataset}'

    ground_truth_file = f"./Dataset/{dataset}/comms.txt"
    ground_truth = load_ground_truth(ground_truth_file)

    print(f"len gt_truth {len(ground_truth)}")

    #加载查到的结果数据
    if num==0:
        filepath =f"./Dataset/{dataset}/output.txt"
    else:
        filepath =f"./Dataset/{dataset}/output-{num}.txt"
    query_results = []
    times = []
    fail_count = 0
    success_count = 0
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    i = 0
    while i < len(lines):
        if lines[i] == '-1':
            fail_count += 1
            query_results.append(None)  # 标记失败查询
            times.append(None)
            i += 1
        else:
            success_count += 1
            nodes = list(map(int, lines[i].split()))
            # ajusted_nodes = [u-1 for u in nodes[1:]] #去除第一个节点数目，并将节点编号减1
            ajusted_nodes = nodes[1:]  # 去除首个节点个数，保留原始编号
            query_results.append(ajusted_nodes)  # 去除首个节点个数
            times.append(float(lines[i + 1]))
            i += 2
    print(f"Fail: {fail_count}, Success: {success_count}, Total: {fail_count + success_count}")
    print(f"len query_results {len(query_results)}")
    assert len(ground_truth) == len(query_results), "预测结果数与ground-truth不一致"
    precision_list = []
    recall_list = []
    f1_list = []
    for pred, true in zip(query_results, ground_truth):
        if pred is None:
            precision_list.append(0)
            recall_list.append(0)
            f1_list.append(0)
        else:
            tp = len(set(pred) & set(true))
            p = compute_precision(pred, true)
            r = compute_recall(pred, true)
            f1 = compute_f1(p, r)
            precision_list.append(p)
            recall_list.append(r)
            f1_list.append(f1)

    avg_precision = sum(precision_list) / len(precision_list)
    avg_recall = sum(recall_list) / len(recall_list)
    avg_f1 = sum(f1_list) / len(f1_list)

    # 只对成功的时间求平均
    valid_times = [t for t in times if t is not None]
    avg_time = sum(valid_times) / len(valid_times) if valid_times else 0

    res_dir = f"./Dataset/{dataset}/{dataset}_val_res.txt"
    with open(res_dir, 'a+', encoding='utf-8') as fh:  # 记录的是 count 次的各个平均结果
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = (
            f"dataset: {dataset}\n"
            f'Precision: {avg_precision:.4f}\n'
            f'Recall: {avg_recall:.4f}\n'
            f'F1-Score: {avg_f1:.4f}\n'
            f'Avg_time: {avg_time:.4f}\n'
            f"current_time: {current_time}\n"
            "----------------------------------------\n"
        )
        fh.write(line)
        fh.close()
    print(f"Precision: {avg_precision:.4f}")
    print(f"Recall:    {avg_recall:.4f}")
    print(f"F1-Score:  {avg_f1:.4f}")
    print(f"Avg_time:{avg_time:.2f}")



# if __name__ == "__main__":
#     # Input files and directory
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--seed', type=int, default=15, help='Random seed.')
#     parser.add_argument('--root', type=str, default='./Dataset', help='data store root')
#     parser.add_argument('--dataset', type=str, default='citeseer',
#                         choices=['football', 'facebook_all', 'cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'],
#                         help='dataset')
#     parser.add_argument('--attack', type=str, default='add', help='attack type', choices=['none','add','random_add','random_remove','random_flip','gaddm','del','gdelm','gflipm'])
#     parser.add_argument('--ptb_rate', type=float, default=0.30, help='pertubation rate')
#     # 配置
#     args = parser.parse_args()
#
#
#     type = args.type #要计算指标的方法类型
#     dataset = f'{args.dataset}_{args.attack}_{args.ptb_rate}'
#     attack = args.attack #none,add,random_add,gaddm,random_remove,del,None
#     ptb_rate = args.ptb_rate
#     k = args.k
#
#     #load ground-truth数据
#     ground_truth_file = f"./data/{dataset}/comms.txt"
#     ground_truth = load_ground_truth(ground_truth_file)
#
#     #load 方法的results数据
#     if type in('ktruss','kcore'):
#         if attack != 'none':
#             result_dir = f"./data/{dataset}/{type}/{k}-{type[1:]}com-{attack}_{ptb_rate}/"
#         else:
#             result_dir = f"./data/{dataset}/{type}/{k}-{type[1:]}com/"
#         results = load_kcore_community_results(result_dir)
#     elif type == 'kclique':#需要用的
#         if attack != 'none':
#             result_dir = f"./data/{dataset}/kclique/clique-{attack}_{ptb_rate}"
#         else:
#             result_dir = f"./data/{dataset}/kclique/clique"
#         results,oknum = load_kclique_community_results(result_dir)
#
#     # 计算指标
#     precision, recall, f1 = calculate_metrics(ground_truth, results)
#
#     total_queries = len(ground_truth) #总的查询
#     coverage = oknum / total_queries if total_queries > 0 else 0.0 #查询的成功率
#
#     # Print results
#     print(f'oknum: {oknum}, total_queries: {total_queries}, coverage: {coverage:.4f}')
#     print(f"Precision: {precision:.4f}")
#     print(f"Recall: {recall:.4f}")
#     print(f"F1-Score: {f1:.4f}")
#
#     #将评价指标存入文件
#     if type in ('ktruss', 'kcore'):
#         if attack != None:
#             output_file = f'./data/{dataset}/val/{type}-{k}-{attack}_{ptb_rate}_res.txt'
#         else:
#             output_file = f'./data/{dataset}/val/{type}-{k}_res.txt'
#     elif type == 'kclique': #需要用的
#         if attack != None:
#             output_file = f'./data/{dataset}/val/{type}-{attack}_{ptb_rate}_res.txt'
#         else:
#             output_file = f'./data/{dataset}/val/{type}_res.txt'
#     save_metrics_to_file(output_file, precision, recall, f1,oknum,total_queries,coverage)
if __name__ == "__main__":
    # Input files and directory
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=15, help='Random seed.')
    parser.add_argument('--root', type=str, default='./Dataset', help='data store root')
    #choices=['football', 'facebook_all', 'cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed']
    parser.add_argument('--dataset', type=str, default='fb107',
                        help='dataset')
    parser.add_argument('--attack', type=str, default='random_add', help='attack type', choices=['none','add','random_add','random_remove','random_flip','gaddm','del','gdelm','gflipm'])
    parser.add_argument('--ptb_rate', type=float, default=0.40, help='pertubation rate')
    parser.add_argument('--num',type=int,default=0,help='query num')
    # 配置
    args = parser.parse_args()
    num = args.num

    # val_CTC(args.dataset,args.attack,args.ptb_rate,10)
    # val_CTC(args.dataset,args.attack,args.ptb_rate,20)

    # val_CTC(args.dataset,args.attack,args.ptb_rate,5)

    # #cora数据集
    # print('###########cora none评估结果是：#################')
    # val_CTC('cora','none',args.ptb_rate,5)
    # print('###########cora random_add 评估结果是：#################')
    # val_CTC('cora','random_add','0.4',5)
    # print('###########cora flipm 评估结果是：#################')
    # val_CTC('cora','flipm','0.4',5)
    # print('###########cora cdelm 评估结果是：#################')
    # val_CTC('cora','cdelm','0.4',5)


    #cora_stb
    # print('cora_stb none评估结果是：')
    # val_CTC('cora_stb','none',args.ptb_rate,5)
    # print('cora_stb random_add 评估结果是：')
    # val_CTC('cora_stb','random_add','0.4',5)
    # print('cora_stb cdelm评估结果是：')
    # val_CTC('cora_stb','cdelm','0.4',5)
    # print('cora_stb flipm评估结果是：')
    # val_CTC('cora_stb','flipm','0.4',5)


    #facebook数据集
    # print('###########fb107 none评估结果是：#################')
    # val_CTC('fb107','none',args.ptb_rate,1)
    # print('###########fb107 random_add：#################')
    # val_CTC('fb107','random_add','0.4',1)
    # print('###########fb107 flipm：#################')
    # val_CTC('fb107','flipm','0.4',1)
    # print('###########fb107 cdelm：#################')
    # val_CTC('fb107', 'cdelm', '0.4', 1)


    # val_CTC('fb107_stb','none',args.ptb_rate,1)
    # val_CTC('fb107_stb','random_add','0.4',1)
    # print('评估结果是：')
    # val_CTC('fb107_stb','cdelm','0.4',1)
    # print('评估结果是：')
    # val_CTC('fb107_stb','flipm','0.4',1)

    #cora数据集
    print('###########cocs none评估结果是：#################')
    val_CTC('cocs','none',args.ptb_rate,5)
    # print('###########cocs random_add 评估结果是：#################')
    # val_CTC('cocs','random_add','0.4',5)
    # print('###########cocs flipm 评估结果是：#################')
    # val_CTC('cocs','flipm','0.4',5)
    # print('###########cocs cdelm 评估结果是：#################')
    # val_CTC('cocs','cdelm','0.4',5)




