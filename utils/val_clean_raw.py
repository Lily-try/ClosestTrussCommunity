import argparse

from load_utils import load_graph




if __name__ == "__main__":
    # Input files and directory
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=15, help='Random seed.')
    parser.add_argument('--root', type=str, default='../data', help='data store root')
    parser.add_argument('--dataset', type=str, default='cora',
                        choices=['football', 'facebook_all', 'cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'],
                        help='dataset')
    parser.add_argument('--method', type=str, default='gsr',choices=['stb','gsr','our'])
    parser.add_argument('--attack', type=str, default='random_add', help='attack type', choices=['none','add','random_add','random_remove','random_flip','gaddm','del','gdelm','gflipm'])
    parser.add_argument('--ptb_rate', type=float, default=0.40, help='pertubation rate')
    parser.add_argument('--num',type=int,default=0,help='query num')
    # 配置
    args = parser.parse_args()

    graphx_raw, raw_nodes = load_graph(args.root, args.dataset,'none', args.ptb_rate)
    graphx_cleaned, cleand_nodes = load_graph(args.root, f'{args.dataset}_{args.method}', args.attack, args.ptb_rate)

    # 边集合（无向图时自动处理对称性）
    edges_raw = set(graphx_raw.edges())
    edges_cleaned = set(graphx_cleaned.edges())

    # 求交集、差集
    common_edges = edges_raw & edges_cleaned
    removed_edges_1 = edges_raw - edges_cleaned
    removed_edges_2 = edges_cleaned - edges_raw
    added_edges = edges_cleaned - edges_raw

    print(f"原始图边数: {len(edges_raw)}")
    print(f"清理后图边数: {len(edges_cleaned)}")
    print(f"共同边数: {len(common_edges)}")   #stb:2754,5278
    print(f"被移除的边数: {len(removed_edges_1)}")
    print(f"新增边数: {len(added_edges)}")


    #交集/raw，交集/clean，值越小说明橡胶越小
    #raw\clean, clean\raw,
    #目前不用这个