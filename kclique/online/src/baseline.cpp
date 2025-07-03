/*
 * baseline.cpp
 *
 *  Created on: 21Jun.,2017
 *      Author: longyuan
 */

#include "baseline.h"
#include <vector>
#include <iostream>
#include <set>
#include <queue>
#include "cliques.h"
#include "util.h"
#include "cpm.h"
using namespace std;

vector<set<int> > result;
vector<vector<int> > graph;
CLIQUESET cliques;
NODE2CLIQUE node2clique;


vector<vector<int> > dgraph;
vector<vector<int> > weight;
vector<int> vcomputed;
vector<int> vq;
set<int> kcpc;

void pCliques()
{
//	cout<<"total cliques: "<<cliques.size()<<endl;
	for (int i = 0; i < cliques.size(); ++i)
	{
		if (cliques[i].size() > 0)
		{
			cout << i << ": ";
			for (int j = 0; j < cliques[i].size(); ++j)
				cout << cliques[i][j] << ", ";
			cout << endl;
		}
	}

}

void pnode2clique()
{
//	cout<<"total cliques: "<<cliques.size()<<endl;
	for (int i = 0; i < node2clique.size(); ++i)
	{
		if (node2clique[i].size() > 0)
		{
			cout << i << ": ";
			for (int j = 0; j < node2clique[i].size(); ++j)
				cout << node2clique[i][j] << ", ";
			cout << endl;
		}
	}

}

void pGraph()
{
	for (int i = 0; i < graph.size(); ++i)
	{
		cout << "node " << i << " : ";
		for (int j = 0; j < graph[i].size(); ++j)
		{
			cout << graph[i][j] << ", ";
		}
		cout << endl;
	}

}

void deleteEdge(MSTEDGE &graph, int a, int b, int *degree)
{
	for (int i = 0; i < degree[a]; ++i)
	{
		if (graph[a][i].first == b)
		{
			swap(graph[a][i], graph[a][degree[a]]);
			degree[a]--;
		}
	}
	for (int i = 0; i < degree[b]; ++i)
	{
		if (graph[b][i].first == a)
		{
			swap(graph[b][i], graph[b][degree[b]]);
			degree[b]--;
		}
	}
}

void printResult()
{
	for (int i = 0; i < result.size(); ++i)
	{
		cout << "i: ";
		for (auto it = result[i].begin(); it != result[i].end(); ++it)
		{
			cout << *it << ", ";
		}
		cout << endl;
	}
}

void naiveOATTree(MSTEDGE &edges, int n)
{


	cout << "running naive" << endl;

#ifdef __linux__
	struct timeval start, end;
	gettimeofday(&start, NULL);
#else
	int start, end;
	start = clock();
#endif

	MSTEDGE mst;
	mst.resize(n);
	int *degree = new int[n];
	fill_n(degree, n, 0);
	for (int i = 0; i < edges.size(); ++i)
	{
		for (int j = 0; j < edges[i].size(); ++j)
		{
			int a = edges[i][j].first;
			int b = edges[i][j].second;

			mst[a].push_back(make_pair(b, i));
			mst[b].push_back(make_pair(a, i));
			degree[a]++;
			degree[b]++;
		}
	}

	queue<int> q;
	int minweight = 100000;
	int *computed = new int[n];
	fill_n(computed, n, 0);
	int *qq = new int[n];
	q.push(0);
	while (!q.empty())
	{
		int t = q.front();
		q.pop();
		int a, b;
		int q_c = 1;
		qq[0] = t;
		for (int j = 0; j < q_c; ++j)
		{
			int node = qq[j];
			for (int k = 0; k < degree[node]; ++k)
			{
				int nnode = mst[node][k].first;
				if (computed[nnode] != 1)
				{
					qq[q_c++] = nnode;
					computed[nnode] = 1;
					if (mst[node][k].second < minweight)
					{
						minweight = mst[node][k].second;
						a = node;
						b = nnode;
					}

				}
			}
		}

		deleteEdge(mst, a, b, degree);
		if (degree[a] > 0)
			q.push(a);
		if (degree[b] > 0)
			q.push(b);
		fill_n(computed, n, 0);
		minweight = 100000;
	}

#ifdef __linux__
	gettimeofday(&end, NULL);
	long long mtime, seconds, useconds;
	seconds = end.tv_sec - start.tv_sec;
	useconds = end.tv_usec - start.tv_usec;
	mtime = seconds * 1000000 + useconds;

	printf("processing time, %lld us\n", mtime);
#else

	end = clock();
	printf("processing time, %f second\n", (float)(end-start)/CLOCKS_PER_SEC);
#endif
	cout << "running naive oat" << endl;



}

void detectKCPC(CLIQUESET& cliques, int k)
{
    cout << "[DEBUG] Enter detectKCPC, k = " << k << ", cliques = " << cliques.size() << endl;
	result.clear();
	graph.clear();
	//cout << "k: " << k << endl;

	int cliqueNum = (int) cliques.size();
	graph.resize(cliqueNum);
//	for (int i = 0; i < cliqueNum; ++i)
//		for (int j = i + 1; j < cliqueNum; ++j)
//		{
//			int n = intersection(cliques[i], cliques[j]);
//			if (n >= k)
//			{
//				graph[i].push_back(j);
//				graph[j].push_back(i);
//			}
//		}

	set<int> tmp;
	for (int i = 0; i < cliques.size(); ++i)
	{
		tmp.clear();
		for (int j = 0; j < cliques[i].size(); ++j)
		{
			int node = cliques[i][j];
			for (int k1 = 0; k1 < node2clique[node].size(); ++k1)
			{
				int c2 = node2clique[node][k1];

				if (c2 > i)
				{
					if (tmp.count(c2) == 0)
					{
						tmp.insert(c2);
						int n = intersection(cliques[i], cliques[c2]);
						//cout << "i: " << i << ", c2: " << c2 <<", n: "<<n<< endl;
						if (n >= k)
						{
							graph[i].push_back(c2);
							graph[c2].push_back(i);
						}

					}
				}
			}
		}
	}
	int edge_cnt = 0;
    for (auto &v : graph) edge_cnt += v.size();
    cout << "[DEBUG] Clique graph edges: " << edge_cnt / 2 << endl;


	//pGraph();

	int *computed = new int[cliqueNum];
	int *q = new int[cliqueNum];
	fill_n(computed, cliqueNum, 0);

	for (int i = 0; i < cliqueNum; ++i)
	{
		if (computed[i] == 1)
			continue;
		set<int> r;
		if (graph[i].size() > 0
				|| (graph[i].size() == 0 && cliques[i].size() > k))
		{
			for (int f = 0; f < cliques[i].size(); ++f)
			{
				r.insert(cliques[i][f]);
			}
			int q_c = 1;
			q[0] = i;
			for (int j = 0; j < q_c && q_c <= 2000; ++j)  //限制最大规模
			{
				int node = q[j];
				int size = graph[node].size();
				for (int k = 0; k < size; ++k)
				{
					int nnode = graph[node][k];
					if (computed[nnode] != 1)
					{
						q[q_c++] = nnode;
						computed[nnode] = 1;
						for (int f = 0; f < cliques[nnode].size(); ++f)
						{
							r.insert(cliques[nnode][f]);
						}
					}
				}
			}
			if (q_c > 2000) {
                cout << "[WARN] BFS queue too large at clique " << i << ", aborted early." << endl;
            }
			result.emplace_back(r);
		}
	}

	//printResult();

}

int cntCPC(std::set<int> &query) {
	int cnt = 0;
	for (int i = 0; i < result.size(); ++i)
	{
		auto it = query.begin();
		for (; it != query.end(); ++it)
			if (result[i].count(*it) == 0)
				break;
		if (it == query.end()) ++cnt;
	}
	return cnt;
}

int isInKCPC(std::set<int> &query)
{
	for (int i = 0; i < result.size(); ++i)
	{
		auto it = query.begin();
		for (; it != query.end(); ++it)
		{
			if (result[i].count(*it) == 0)
				break;
		}
		if (it == query.end())
		{
			return i+1;
		}

	}
	return 0;
}

void baseline(std::set<int> &query, std::string dir)
{

	size_t maximumCliqueSize = 0;
	cliques.clear();

	int n = Cliques::getMaxCliques(dir, cliques, maximumCliqueSize);
	if (cliques.size() > 50000) {
    cout << "[WARN] Too many cliques: " << cliques.size() << ", truncating to 50000" << endl;
    cliques.resize(50000);  // 只保留前 50000 个最大团
    }

	node2clique.clear();
	node2clique.resize(n);
	for (int i = 0; i < cliques.size(); ++i)
	{
		size_t size = cliques[i].size();
		for (int j = 0; j < size; ++j)
		{
			node2clique[cliques[i][j]].push_back(i);
		}
		cliques[i].shrink_to_fit();
	}

	for (int i = 0; i < node2clique.size(); ++i)
	{
		node2clique[i].shrink_to_fit();
	}

	//pCliques();
	//pnode2clique();
	int start = 1;
	int end = maximumCliqueSize;
	int t;
	while (start <= end)
	{
		int mid = (start + end) / 2;
		detectKCPC(cliques, mid);
		if (isInKCPC(query))
		{
			start = mid + 1;
			t = mid;
			//cout << "in t = " << t << endl;
		}
		else
		{
			end = mid - 1;
		}
	}

	detectKCPC(cliques, t);
	int p = isInKCPC(query)-1;
	printf( "output to %s\n", CPM::output_name.c_str() );
	FILE *fout = fopen(CPM::output_name.c_str(), "w");
	fprintf( fout, "%d\n", (int)result[p].size() );
	for( auto it = result[p].begin(); it != result[p].end(); it++)
		fprintf( fout, "%d ", *it );
	fclose(fout);
	cout << "K=" << t << ",Number of Communities=" << cntCPC(query) << endl;
	cout << "[DEBUG] node2clique[687].size() = " << node2clique[687].size() << endl;
}

bool isInKCPCCAG(int node, int connectivity, set<int> &query)
{

//	cerr<<"in isInKCPCCAG"<<endl;
	int cliqueNum = dgraph.size();

	vcomputed.resize(cliqueNum);
	fill_n(vcomputed.begin(), cliqueNum, 0);

	vq.resize(cliqueNum);


	for(int i = 0; i < node2clique[node].size(); ++i)
	{
		kcpc.clear();
		int c1 = node2clique[node][i];
		for(int f = 0; f < cliques[c1].size(); ++f)
		{
			kcpc.insert(cliques[c1][f]);
		}
		if(cliques[c1].size() > connectivity)
		{

			int q_c = 1;
			vq[0] = c1;
			vcomputed[c1] = 1;
			for(int j = 0; j < q_c; ++j)
			{
				int node = vq[j];
				int size = dgraph[node].size();
				for(int k = 0; k < size; ++k)
				{
					int nnode = dgraph[node][k];
					if(vcomputed[nnode] != 1 && weight[node][k] >= connectivity)
					{
						vq[q_c++] = nnode;
						vcomputed[nnode] = 1;
						for(int f = 0; f < cliques[nnode].size(); ++f)
						{
							kcpc.insert(cliques[nnode][f]);
						}

					}
				}
			}

		}

		fill_n(vcomputed.begin(), cliqueNum, 0);

		auto it = query.begin();
		for(; it != query.end(); ++it)
		{
			if(kcpc.find(*it) == kcpc.end())
				break;

		}
		if(it == query.end())
			return true;
	}

	return false;

}

void baseline2(vector<vector<set<int> > > &queryNodeSet, std::string dir)
{
	size_t maximumCliqueSize = 0;
	cliques.clear();


	cerr<<"computing maximal cliques......"<<endl;
	int n = Cliques::getMaxCliques(dir, cliques, maximumCliqueSize);
	node2clique.clear();
	node2clique.resize(n);
	for (int i = 0; i < cliques.size(); ++i)
	{
		size_t size = cliques[i].size();
		for (int j = 0; j < size; ++j)
		{
			node2clique[cliques[i][j]].push_back(i);
		}
		cliques[i].shrink_to_fit();
	}

	for (int i = 0; i < node2clique.size(); ++i)
	{
		node2clique[i].shrink_to_fit();
	}

	cerr<<"building density graph"<<endl;

	int cliqueNum = (int) cliques.size();
	dgraph.resize(cliqueNum);
	weight.resize(cliqueNum);

	set<int> tmp;
        for (int i = 0; i < cliques.size(); ++i)
        {
                tmp.clear();
                for (int j = 0; j < cliques[i].size(); ++j)
                {
                        int node = cliques[i][j];
                        for (int k1 = 0; k1 < node2clique[node].size(); ++k1)
                        {
                                int c2 = node2clique[node][k1];

                                if (c2 > i)
                                {
                                        if (tmp.count(c2) == 0)
                                        {
                                                tmp.insert(c2);
                                                int n = intersection(cliques[i], cliques[c2]);
                                                dgraph[i].push_back(c2);
                                                weight[i].push_back(n);
                                                dgraph[c2].push_back(i);
                                                weight[c2].push_back(n);
                                        }
                                }
                        }
                }
        }


	long long graphSize = 0;
	for(int i = 0; i < cliques.size(); ++i)
	{
		graphSize += cliques[i].size()*4;
	}
	for(int i = 0; i < node2clique.size(); ++i)
	{
		graphSize += node2clique[i].size()*4;
	}

	for(int i = 0; i < dgraph.size(); ++i)
	{
		graphSize += dgraph[i].size()*8;

	}
	cerr<<"graph size: "<<graphSize<<endl;
	cerr<<"building end"<<endl;

	long long mmtime[QUERYNUM2];
	for(int i = 2; i < 3; ++i)
	//for(int i = 0; i < queryNodeSet.size(); ++i)
	{
	for(int j = 0; j < queryNodeSet[i].size(); ++j)
	 {
		//cout<<j<<endl;

#ifdef __linux__
	struct timeval start, end;
	gettimeofday(&start, NULL);
#else
	int start, end;
	start = clock();
#endif

		set<int> &query = queryNodeSet[i][j];
		int nodeWithMinClique = 0;
		int minClique = 10000000;
		for(auto it = query.begin(); it != query.end(); ++it)
		{
			int u = *it;
			if(node2clique[u].size() < minClique)
			{
				nodeWithMinClique = u;
				minClique = node2clique[u].size();
			}

		}

		int start1 = 1;
		int end1 = maximumCliqueSize;
		int t;
		while (start1 <= end1)
		{
			int mid = (start1 + end1) / 2;
			if (isInKCPCCAG(nodeWithMinClique, mid, query))
			{
				start1 = mid + 1;
				t = mid;
					//cout << "in t = " << t << endl;
			}
			else
			{
				end1 = mid - 1;
			}
		}

#ifdef __linux__
	gettimeofday(&end, NULL);
	long long mtime, seconds, useconds;
	seconds = end.tv_sec - start.tv_sec;
	useconds = end.tv_usec - start.tv_usec;
	mtime = seconds*1000000 + useconds;
	mmtime[j] = mtime;
	//printf("processsing time, %lld us\n", mtime/(queryNodeSet[i].size()));
#else
	end = clock();
	printf("processing time, %f second\n", (float)(end-start)/(queryNodeSet[i].size()*CLOCKS_PER_SEC));
#endif
	}
	long long avg = 0;
	long long var = 0;

	for(int j = 0; j < queryNodeSet[i].size(); ++j)
		avg += mmtime[j];
	avg = avg/queryNodeSet[i].size();
	for(int j = 0; j < queryNodeSet[i].size(); ++j)
		var += (mmtime[j]-avg)*(mmtime[j]-avg);
	var = var / queryNodeSet[i].size();
	printf("avg: %lld, var: %lld\n", avg, var);


	}

}


