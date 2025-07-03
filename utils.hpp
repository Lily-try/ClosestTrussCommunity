#ifndef __UTILS_HPP__
#define __UTILS_HPP__
#include <bits/stdc++.h>
using namespace std;

const int max_trussness = 1005;
const int maxn = 40505;
const int maxm = 120000;
struct HashTable
{
    const int mod = maxm << 2;
    struct data
    {
        long long u;
        int v, next;
    };
    vector<data> e;
    vector<int> h;
    int cnt;

    int hash(long long u) { return u < mod ? u : u % mod; }

    int &operator[](long long u)
    {
        int hu = hash(u);
        for (int i = h[hu]; i; i = e[i].next)
        {
            if (e[i].u == u)
                return e[i].v;
        }
        return e[++cnt] = (data){u, -1, h[hu]}, h[hu] = cnt, e[cnt].v;
    }

    bool count(long long u)
    {
        int hu = hash(u);
        for (int i = h[hu]; i; i = e[i].next)
        {
            if (e[i].u == u)
                return true;
        }
        return false;
    }

    void erase(long long u)
    {
        int hu = hash(u), pre;
        for (int i = h[hu]; i; i = e[i].next)
        {
            if (e[i].u == u)
            {
                if (i == h[hu])
                    h[hu] = e[i].next;
                else
                    e[pre].next = e[i].next;
            }
            pre = i;
        }
    }

    HashTable()
    {
        cnt = 0;
        e.resize(maxm << 2);
        h.resize(maxm << 2, 0);
    }
};


struct edge_link
{

    struct data
    {
        int v, pre, next, index_in_e;
    };
    
    vector<data> e;
    vector<int> p;
    int eid;

    void insert(int u, int v, int index_in_e)
    {
        if (eid >= (maxm << 1)) {
            assert(0);
        }
        e[eid].v = v;
        e[eid].next = p[u];
        e[eid].pre = -1;
        e[eid].index_in_e = index_in_e;
        if (p[u] != -1)
            e[p[u]].pre = eid;
        p[u] = eid++;
    }

    void erase(int cur_id)
    {
        assert(cur_id >= 0);
        if (~e[cur_id].next)
            e[e[cur_id].next].pre = e[cur_id].pre;
        if (~e[cur_id].pre)
            e[e[cur_id].pre].next = e[cur_id].next;
    }
    
    edge_link()
    {
        eid = 0;
        p.resize(maxn, -1);
        e.resize(maxm << 1);
    }
};


struct pair_hash{
    template<class T1, class T2>
    size_t operator () (const pair<T1, T2> & p) const{
        auto h1 = hash<T1>()(p.first);
        auto h2 = hash<T2>()(p.second);
        return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
    }
};

struct Graph {
    vector<int> V;
    unordered_set<int> E;
    unordered_map<int, unordered_set<int>> G; // 存点u连出去的点，这里与最开始的原图存的不一样，原图存的是边的编号
    unordered_map<int, int> trussness; // eid -> tau
    unordered_map<int, unordered_set<pair<int, int>, pair_hash> > triangles;
    int query_dis = 0;
};

vector<int> G[maxn]; // 存原图每个点连出的边的编号
int D[maxn]; // 存原图每个点的点度
int n, m, theta;
set<int> q;
HashTable hash_table_1, hash_table_2;
// map<pair<int, int>, int> mp1, mp2;
edge_link e_link;
int tau[maxn]; // 存每个节点的tau值
unordered_set<int> attr_set;

bool vis[maxn];
bool _vis[maxn];
int dis[maxn];
int _dis[maxn];

map<string, pair<int, int> > data_info = {{"facebook_combined", {4039, 88234}}, {"DBLP", {500000, 1049866}}, {"com-lj.ungraph", {4040000, 34681189}}};

int compute_diam(Graph &C) {
    cerr << "Computing diameter..." << endl;
    int diam = 0;
    for(const int & w : C.V) {
        // cerr << w << " ";
        queue<int> que;
        que.push(w);
        memset(dis, 0x3f, sizeof(dis));
        dis[w] = 0;
        while(!que.empty()) { // 计算出每个点到查询点的距离
            int u = que.front();
            que.pop();
            for(const int & v : C.G[u]) {
                if(dis[v] == 0x3f3f3f3f) {
                    dis[v] = dis[u] + 1;
                    que.push(v);
                }
            }
        }
        for(const int & t : C.V) diam = max(diam, dis[t]);
    }

    // for(const int & t : C.V) cout << t << " ";
    // cout << endl;
    return diam;
}

#endif