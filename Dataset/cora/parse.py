with open('graph.txt', 'r') as f_in:
    edges = []
    max_node = 0
    for line in f_in:
        u, v = map(int, line.strip().split())
        u_new = u + 1
        v_new = v + 1
        edges.append((u_new, v_new))
        current_max = max(u_new, v_new)
        max_node = max(max_node, current_max)
    num_nodes = max_node
    num_edges = len(edges)

with open('cora.txt', 'w') as f_out:
    f_out.write(f"{num_nodes} {num_edges}\n")
    for u, v in edges:
        f_out.write(f"{u} {v}\n")