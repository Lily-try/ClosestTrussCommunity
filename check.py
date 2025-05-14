class UnionFind:
    def __init__(self):
        self.parent = {}
        self.rank = {}
    
    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # 路径压缩
        return self.parent[x]
    
    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            # 按秩合并
            if self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
            else:
                self.parent[root_x] = root_y
                if self.rank[root_x] == self.rank[root_y]:
                    self.rank[root_y] += 1

def check_connectivity(input_file, node_a, node_b):
    """ 判断两个节点是否连通 """
    uf = UnionFind()
    existing_nodes = set()
    
    with open(input_file, 'r') as f:
        # 跳过第一行（顶点数和边数）
        f.readline()
        
        # 处理所有边
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                continue  # 忽略格式错误的行
            
            u, v = map(int, parts)
            existing_nodes.add(u)
            existing_nodes.add(v)
            uf.union(u, v)
    
    # 检查节点是否存在
    if node_a not in existing_nodes or node_b not in existing_nodes:
        return False
    
    # 检查连通性
    return uf.find(node_a) == uf.find(node_b)

# 使用示例
result = check_connectivity(".\Dataset\cora\cora.txt", 33, 2695)
print("节点33和2695连通" if result else "节点33和2695不连通")