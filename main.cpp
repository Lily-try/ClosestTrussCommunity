#pragma GCC optimize(2,"Ofast")
#pragma GCC optimize("Ofast,no-stack-protector,unroll-loops,fast-math")
// #pragma GCC target("sse,sse2,sse3,ssse3,sse4.1,sse4.2,avx,avx2,popcnt,tune=native")

#include <bits/stdc++.h>
#include "utils.hpp"
#include "truss.hpp"
#define debug(a) cout << (#a) << " : " << (a) << endl
using namespace std;
bool in_que[maxm];
Graph ans, C, tmp;
queue<int> que;
unordered_set<int> A[maxn];

void CTC() {
    cerr << "CTC running...\n";
    clock_t start = clock();

    ans.query_dis = 0x3f3f3f3f;
    int lk = 2, rk = tau[*q.begin()] + 1, k;
    while(lk < rk) {
        k = (lk + rk) >> 1;
        tmp.V.clear();
        memset(vis, false, sizeof(vis));

        while (!que.empty()) que.pop();
        
        que.push(*q.begin());
        vis[*q.begin()] = true;
        tmp.V.push_back(*q.begin());

        while(!que.empty()) {
            int u = que.front();
            que.pop();
            for(const int & eid : G[u]) {
                int v = E[eid].u ^ E[eid].v ^ u;
                if(!vis[v] && E[eid].trussness >= k) {
                    vis[v] = true;
                    tmp.V.push_back(v);
                    que.push(v);
                }
            }
        }
        memset(_vis, 0, sizeof(_vis));
        for(const int v : tmp.V) {
            _vis[v] = 1;
        }
        bool qflag = true;
        for(const int cur_q : q) {
            if(!_vis[cur_q]) {
                qflag = false;
                break;
            }
        }
        cerr << "lk : " << lk << " , rk : " << rk << '\n';
        if(qflag) {
            lk = k + 1;
            C.V.clear();
            for(const int x : tmp.V) C.V.push_back(x);
        } else {
            rk = k;
        }
        // cerr << "fairness:" << theta * (int) attr_set.size() << '\n';
    }
    k = lk - 1;
    // debug(k);
    if(k < 2) {
        cerr << "No solotion!\n";
        cout << -1 << endl;
        assert(0);
    }

    memset(vis, false, sizeof(vis));
    
    while (!que.empty()) que.pop();
 // 把边加进去
    for(auto cur_q : q) que.push(cur_q), vis[cur_q] = true;

    while(!que.empty()) {
        int u = que.front();
        que.pop();
        for(const int & eid : G[u]) {
            int v = E[eid].u ^ E[eid].v ^ u;
            if(!vis[v] && E[eid].trussness >= k) {
                vis[v] = true;
                C.E.insert(hash_table_2[1ll * v * n + u]);
                C.G[u].insert(v);
                C.G[v].insert(u);
                que.push(v);
            } else if(vis[v] && E[eid].trussness >= k && !C.G[u].count(v)) {
                C.E.insert(hash_table_2[1ll * v * n + u]);
                C.G[u].insert(v);
                C.G[v].insert(u);
            }
        }
    }


    // ----------------------------------------------------------------------------------

    auto cmp = [](const int &v1, const int &v2) -> bool
    {
        return C.G[v1].size() != C.G[v2].size() ? C.G[v1].size() > C.G[v2].size() : v1 < v2;
    };
    sort(C.V.begin(), C.V.end(), cmp);
    cerr << "C.V size: " << C.V.size() << '\n';
    // if(C.V.size() >= 50000) {
    //     assert(0);
    // }
    

    for (int i = 0; i < C.V.size(); i++)
    {
        int vertex = C.V[i];
        for (const int v : C.G[vertex])
        {
            if (cmp(v, vertex))
                continue;
            for (const int &w : A[vertex])
            {
                if (A[v].count(w))
                {
                    int eid1 = hash_table_2[1ll * v * n + vertex], eid2 = hash_table_2[1ll * v * n + w], eid3 = hash_table_2[1ll * vertex * n + w];
                    C.trussness[eid1]++;
                    C.trussness[eid2]++;
                    C.trussness[eid3]++;
                    C.triangles[eid1].insert({min(eid2, eid3), max(eid2, eid3)});
                    C.triangles[eid2].insert({min(eid1, eid3), max(eid1, eid3)});
                    C.triangles[eid3].insert({min(eid1, eid2), max(eid1, eid2)});
                }
            }
            A[v].insert(vertex);
        }
    }

    // for(const int & eid : C.E) {
    //     cerr << E[eid].u << ", " << E[eid].v << " triangle size:" << C.triangles[eid].size() << '\n';
    // }

    // // -----------------------------------------------------------------------------
    
    // clock_t start = clock();
    clock_t end = clock();
    cerr << "Binary search running time: " << double(end - start) / CLOCKS_PER_SEC << "s." << endl;
    clock_t t_dist = 0, t_maintain = 0;
    int tot = 0;
    while(1) {
        clock_t t1 = clock();

        
        memset(dis, 0x3f, sizeof(dis));
        for(auto v : C.V) dis[v] = 0;
        
        for(auto cur_q : q) {
            while (!que.empty()) que.pop();
            memset(_dis, 0x3f, sizeof(_dis));
            que.push(cur_q), _dis[cur_q] = 0;
            while(!que.empty()) { // 计算出每个点到查询点的距离
                int u = que.front();
                que.pop();
                for(const int v : C.G[u]) {
                    if(_dis[v] == 0x3f3f3f3f) {
                        _dis[v] = _dis[u] + 1;
                        // cerr << v << ": " << dis[v] << endl;
                        que.push(v);
                    }
                }
            }
            for(auto v : C.V) dis[v] = max(dis[v], _dis[v]);
        }
        // if(tot == 9) {
        //     cout << "\n\n\n";
        //     cout << "C.V.size: " << C.V.size() << endl;
        //     for(auto u : C.V) cout << u << " " << phi[u] << '\n'; 
        //     cout << endl;
        //     for(auto u : C.V) {
        //         for(auto v : C.G[u]) {
        //             if(u < v) cout << u << " " << v << endl;
        //         }
        //     }
        // }


        // for(const int & v : C.V) {
        //     if(attr_sum2[phi[v]] == theta) {
        //         dis[v] = 0;
        //     }
        // }

        sort(C.V.begin(), C.V.end(), [] (const int &v1, const int &v2) -> bool {
            if(dis[v1] != dis[v2]) return dis[v1] > dis[v2];
            int cnt1 = 0, cnt2 = 0;
            for(auto tq : q) {
                if(C.G[tq].count(v1)) cnt1++;
                if(C.G[tq].count(v2)) cnt2++;
            }
            if(cnt1 != cnt2) return cnt1 < cnt2;
            return v1 < v2;
        });

        clock_t t2 = clock();
        t_dist += t2 - t1;
        // ofstream fout("./case.txt");
        // cerr <<(int) C.V.size() << endl;
        // sort(C.V.begin(), C.V.end());
        // for(const int u : C.V) {
        //     cerr << u << " ";
        // }
        // cerr << endl;

        assert(que.empty());

        C.query_dis = dis[C.V[0]];
        tot++;
        if(tot % 10 == 0) {
            cerr << "iter : " << tot << ", current query_dis: " << ans.query_dis << ", C.V.size:" << C.V.size() << '\n';
        }
        // if(C.V.size() <= 2 * _gamma) _gamma = 1;
        if(C.query_dis < ans.query_dis) {
            ans.V.clear();
            ans.G.clear();
            for(const int x : C.V) {
                ans.V.emplace_back(x);
                for(const int y : C.G[x]) {
                    ans.G[x].insert(y);
                }
            }
            ans.query_dis = C.query_dis;

            cerr << "iter : " << tot << ", current query_dis: " << ans.query_dis << ", C.V.size: " << C.V.size() << '\n';
        }

        memset(in_que, false, sizeof(in_que));

        // for(int i = 0;i < C.V.size();i++) {
        for(int i = 0;i < C.V.size() && i < 1;i++) {
        // for(int i = 0, po = int(ceil(_gamma * pow(0.990, tot))) + 1;i < C.V.size() && i < po;i++) {
            int u_star = C.V[i];
            // if(attr_sum2[phi[u_star]] <= theta)  {
            //     lim++;
            //     continue;
            // }// if(i == 0) cerr << "u* : " << u_star << " " << C.V.size() << '\n';
            // if(dis[u_star] != dis[C.V[0]]) break;
            for(const int u : C.G[u_star]) {
                int eid = hash_table_2[1ll * u_star * n + u];
                if(!in_que[eid]) {
                    in_que[eid] = true;
                    que.push(eid);
                }
            }
        }
        
        
        bool qflag = false;
        clock_t t3 = clock();
        while(!que.empty()) {
            int eid1 = que.front();
            // if(C.V.size() == ) debug(E[eid1].u);
            // debug(E[eid1].v);
            que.pop();
            int u = E[eid1].u, v = E[eid1].v;
            C.G[u].erase(v), C.G[v].erase(u);
            C.E.erase(eid1);
            

            for(auto ele : C.triangles[eid1]) {
                int eid2 = ele.first, eid3 = ele.second;
                // debug(E[eid2].u);
                // debug(E[eid2].v);

                C.trussness[eid2]--;
                C.trussness[eid3]--;
                
                C.triangles[eid2].erase({min(eid1, eid3), max(eid1, eid3)});
                if(C.trussness[eid2] < k - 2 && !in_que[eid2]) {
                    in_que[eid2] = true;
                    que.push(eid2);
                }
                C.triangles[eid3].erase({min(eid1, eid2), max(eid1, eid2)});
                if(C.trussness[eid3] < k - 2 && !in_que[eid3]) {
                    in_que[eid3] = true;
                    que.push(eid3);
                }
            }
            C.triangles[eid1].clear();
        }
        clock_t t4 = clock();
        t_maintain += t4 - t3;
        sort(C.V.begin(), C.V.end(), [] (const int &v1, const int &v2) -> bool {
            return C.G[v1].size() > C.G[v2].size();
        });

        while((int) C.V.size() > 0 && C.G[C.V[C.V.size() - 1]].size() == 0) {
            if(q.count(C.V[C.V.size() - 1])) {
                qflag = true;
                break;
            }
            C.V.pop_back();
        }


        // for(const int u : C.V) {
        //     cerr << u << " ";
        //     assert((int) C.G[u].size() <= k - 2);
        // }
        // cerr << endl;

        if(qflag) break;
    }

    cerr << "Distance time: " << double(t_dist) / CLOCKS_PER_SEC << "s.\n";
    cerr << "Maintain time: " << double(t_maintain) / CLOCKS_PER_SEC << "s.\n";

    clock_t endend = clock();
    cerr << "Delete running time: " << double(endend - end) / CLOCKS_PER_SEC << "s.\n";
    // cerr << "CTC completed." << endl;
    // cout << "Overall running time: " << double(endend - start) / CLOCKS_PER_SEC << "s.\n";
    cerr << "CTC completed." << endl;
    // assert(0);
}

mt19937 rnd(time(0));
int get_rand(int l, int r) {
    return l + rnd() % (r - l + 1);
}

int main(int argc, char * argv[])
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);

    if(argc < 3 || argc != atoi(argv[2]) + 3) {
        cerr << "Usage: ./main <dataset name> <q number> <q1> ... <qn>\n";
        return -1;
    }

    string dataset(argv[1]);
    // string dataset = "butterfly";
    cerr << "Loading ./Dataset/"+ dataset + "/" + dataset + ".txt" << endl;
    FILE *result = freopen(("./Dataset/"+ dataset + "/" + dataset + ".txt").c_str(), "r", stdin);
    if (result == NULL) {
        cerr << "[ERROR] Dataset Error!\n";
        return -1;
    }
    
    // theta = 3;
    // q = 170;
    // int attr_range = 10;
    // _gamma = 1;
    int q_cnt = stoi(argv[2]);

    for(int i = 3;i <= q_cnt + 2;i++) {
        q.insert(stoi(argv[i]));
    }
    cerr << "q: ";
    for(auto tq : q) cerr << tq << " ";
    cerr << endl;
    
    string output_name = "./output/" + dataset + "_q";
    // for(auto tq : q) output_name += "_" + to_string(tq);
    // output_name += ".txt";
    output_name = "./output/cora_summary.txt";
    freopen(output_name.c_str(), "a", stdout);
    
    cin >> n >> m;
    for (int i = 1; i <= m; i++)
    {
        cin >> E[i].u >> E[i].v;
        if(i % 200000 == 0) cerr << "Reading edges " << fixed << setprecision(4) << 100.0 * i / m << "%.\n";
        if(E[i].u == E[i].v)
            cerr << "[ERROR] Self Loop!\n";
        assert(E[i].u != E[i].v);
        if(hash_table_1.count(1ll * E[i].u * n + E[i].v)) continue;
        if(hash_table_1.count(1ll * E[i].v * n + E[i].u)) continue;
        hash_table_1[1ll * E[i].u * n + E[i].v] = i;
        hash_table_1[1ll * E[i].v * n + E[i].u] = i;
        hash_table_2[1ll * E[i].u * n + E[i].v] = i;
        hash_table_2[1ll * E[i].v * n + E[i].u] = i;
        D[E[i].u]++;
        D[E[i].v]++;
        G[E[i].u].push_back(i);
        G[E[i].v].push_back(i);

        e_link.insert(E[i].u, E[i].v, i);
        e_link.insert(E[i].v, E[i].u, i);
    }

    cerr << "Edges reading completed." << endl;

    
    cerr << "Truss decomposition is running...\n";
    truss_decomposition();
    cerr << "Truss decomposition completed." << endl;
    CTC();
    // cout << "diameter : " << compute_diam(ans) << '\n';
    cerr << "Diameter calculation completed." << endl;

    // cout << "\n\n\n";
    cout << ans.V.size() << " ";
    for(auto u : ans.V) cout << u << " ";
    cout << endl;
    // for(auto u : ans.V) {
    //     for(auto v : ans.G[u]) {
    //         if(u < v) cout << u << " " << v << endl;
    //     }
    // }
    return 0;
}