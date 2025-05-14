#ifndef __TRUSS_HPP__
#define __TRUSS_HPP__

#include <bits/stdc++.h>
#include "utils.hpp"
using namespace std;

struct edge
{
    int u, v;
    int sup;
    int trussness;
    edge() : u(0), v(0), sup(0), trussness(0) {}
};

vector<edge> E(maxm);
unordered_set<int> trussness_class[max_trussness];
int vertices_in_degree_decending_order[maxn];

void truss_decomposition()
{
    for (int i = 1; i <= n; i++)
        vertices_in_degree_decending_order[i] = i;

    auto cmp = [](const int &v1, const int &v2) -> bool
    {
        return D[v1] != D[v2] ? D[v1] > D[v2] : v1 < v2;
    };
    sort(vertices_in_degree_decending_order + 1, vertices_in_degree_decending_order + n + 1, cmp);

    vector<unordered_set<int>> A(maxn);

    for (int i = 1; i <= n; i++)
    {
        if(i % 50000 == 0) cerr << "Computing support " << fixed << setprecision(4) << 100.0 * i / n << "%.\n";
        int vertex = vertices_in_degree_decending_order[i];
        for (const int eid : G[vertex])
        {
            int v = E[eid].v ^ E[eid].u ^ vertex;
            if (cmp(v, vertex))
                continue;
            for (const int w : A[vertex])
            {
                // cerr << vertex << " " << v << " " << w << endl;
                if (A[v].count(w))
                {
                    E[eid].sup++;
                    E[hash_table_1[1ll * v * n + w]].sup++;
                    E[hash_table_1[1ll * vertex * n + w]].sup++;
                }
            }
            A[v].insert(vertex);
        }
    }
    for(int i = 1;i <= n;i++) A[i].clear();

    vector<bool> removed_edges(maxm, false);
    cerr << "Support computed!\n";
    int cur_minimum_trussness = max_trussness, sub_minimum_trussness = max_trussness;
    for (int i = 1; i <= m; i++)
    {
        trussness_class[E[i].sup + 2].insert(i);
        if (E[i].sup < cur_minimum_trussness - 2)
        {
            sub_minimum_trussness = cur_minimum_trussness;
            cur_minimum_trussness = E[i].sup + 2;
        }
    }
    queue<int> q;
    for (int i = 1; i <= m; i++)
    {
        if (E[i].sup == cur_minimum_trussness - 2)
        {
            q.push(i);
            removed_edges[i] = true;
            E[i].trussness = cur_minimum_trussness;
        }
    }
    cur_minimum_trussness = sub_minimum_trussness;
    
    cerr << "Computing trussness!\n";
    int num = 0;
    while (!q.empty())
    {
        int u = E[q.front()].u, v = E[q.front()].v;
        int eid_uv = hash_table_1[1ll * u * n + v];
        hash_table_1.erase(1ll * v * n + u);
        hash_table_1.erase(1ll * u * n + v);
        // cerr << ((eid_uv - 1) << 1) << endl;
        // ++num;
        // if(num % 10000 == 0) cerr << num << ".\n";
        e_link.erase((eid_uv - 1) << 1);
        e_link.erase((eid_uv - 1) << 1 | 1);

        q.pop();
        for (int i = e_link.p[u]; ~i; i = e_link.e[i].next)
        {
            int eid = e_link.e[i].index_in_e;
            if (removed_edges[eid])
                continue;

            int w = e_link.e[i].v;
            if (w == v)
                continue;

            if (hash_table_1.count(1ll * v * n + w))
            {
                trussness_class[E[eid].sup + 2].erase(eid);
                E[eid].sup--;
                trussness_class[E[eid].sup + 2].insert(eid);

                if (E[eid].sup <= E[eid_uv].trussness - 2 && !removed_edges[eid])
                {
                    removed_edges[eid] = true;
                    E[eid].trussness = E[eid_uv].trussness;
                    q.push(eid);
                }
            }
        }

        for (int i = e_link.p[v]; ~i; i = e_link.e[i].next)
        {
            int eid = e_link.e[i].index_in_e;
            if (removed_edges[eid])
                continue;

            int w = e_link.e[i].v;
            if (w == u)
                continue;

            if (hash_table_1.count(1ll * u * n + w))
            {
                trussness_class[E[eid].sup + 2].erase(eid);
                E[eid].sup--;
                trussness_class[E[eid].sup + 2].insert(eid);

                if (E[eid].sup <= E[eid_uv].trussness - 2 && !removed_edges[eid])
                {
                    removed_edges[eid] = true;
                    E[eid].trussness = E[eid_uv].trussness;
                    q.push(eid);
                }
            }
        }
        if (q.empty())
        {
            for (int i = E[eid_uv].trussness + 1; i < max_trussness; i++)
            {
                if ((int) trussness_class[i].size() != 0)
                {
                    cerr << "Current trussness : " << i << ".\n";
                    for (const int &eid : trussness_class[i])
                    {
                        q.push(eid);
                        removed_edges[eid] = true;
                        E[eid].trussness = i;
                    }
                    break;
                }
            }
        }
    }

    int max_trussness = 0;
    for (int u = 1;u <= n;u++) {
        for(const int & eid : G[u]) {
            int v = E[eid].u ^ E[eid].v ^ u;
            tau[u] = max(tau[u], E[eid].trussness);
            max_trussness = max(max_trussness, tau[u]);
        }
    }
}

#endif