/*
 * densitygraph.cpp
 *
 *  Created on: 7May,2017
 *      Author: longyuan
 */
#include <cstddef>
#include <utility>
#include <iostream>
//#include <omp.h>
#include <algorithm>
#include <set>
#include <unordered_set>
#include "densitygraph.h"
#include "util.h"
#include "unionfind.h"
using namespace std;

extern int Overlap;

void DensityGraph::buildGraph(string dir)
{

	cout << "building graph" << endl;

	graphEdges.resize(maxCliqueSize);
	cout << "maxCliqueSize: " << maxCliqueSize << endl;

//	size_t i = 0;
//	size_t j = 0;
	int size = (int) cliques.size();

//	for (i = 0; i < size; ++i)
//		for (j = i + 1; j < size; ++j)
//		{
//			int n = intersection(cliques[i], cliques[j]);
//			if (n > 0)
//				graphEdges[n].push_back(Edge(i, j));
//		}

#ifdef __linux__
	struct timeval start, end;
	gettimeofday(&start, NULL);
#else
	int start, end;
	start = clock();
#endif

////#pragma omp parallel for schedule (guided)
//	for (int i = 0; i < size; ++i)
//	{
//		for (int j = 0; j < cliques[i].size(); ++j)
//		{
//			int node = cliques[i][j];
//			for (int k = 0; k < nq[node].size(); ++k)
//			{
//				int c2 = nq[node][k];
//				if (c2 > i)
//				{
//					int n = intersection(cliques[i], cliques[c2]);
////#pragma omp critical
//					{
//						if (n > Overlap)
//							graphEdges[n].push_back(Edge(i, c2));
//					}
//				}
//			}
//		}
//		if (i % 100000 == 0)
//			cout << i << ", " << i * 1.0 / size << endl;
//	}
//
//	cout<<"processing time: "<<time(NULL)-begin<<endl;
//	for (int i = 0; i < maxCliqueSize; ++i)
//	{
//		graphEdges[i].shrink_to_fit();
//	}

//#pragma omp parallel for schedule (guided)
//	for (int i = 0; i < size; ++i)
//	{
//
//		int *overlap = new int[size];
//		fill_n(overlap, size, 0);
//		vector<int> overlappingClique;
//		for (int j = 0; j < cliques[i].size(); ++j)
//		{
//			int node = cliques[i][j];
//			for (int k = 0; k < nq[node].size(); ++k)
//			{
//				int c2 = nq[node][k];
//				if (c2 > i)
//				{
//					overlap[c2]++;
//					overlappingClique.push_back(c2);
//
//				}
//			}
//		}
//		sort(overlappingClique.begin(), overlappingClique.end());
//		overlappingClique.erase(
//				unique(overlappingClique.begin(), overlappingClique.end()),
//				overlappingClique.end());
//		for (int j = 0; j < overlappingClique.size(); ++j)
//		{
//			int node = overlappingClique[j];
//			int coverlap = overlap[node];
//
//			if (coverlap > Overlap)
//			{
//#pragma omp critical
//				{
//					graphEdges[coverlap].push_back(Edge(i, node));
//				}
//			}
//			overlap[node] = 0;
//		}
//
//		overlappingClique.clear();
//		if (i % 100000 == 0)
//				cout << i << ", " << i * 1.0 / size << endl;
//	}
//

	int *overlap = new int[size];
	fill_n(overlap, size, 0);
	//unordered_set<int> overlappingClique;
	int *nq_size = new int[nq.size()];
	fill_n(nq_size, nq.size(), 1);
	int *computed = new int[size];
	int computed_index = 0;

	for (int i = 0; i < size; ++i)
	{

		for (int j = 0; j < cliques[i].size(); ++j)
		{
			int node = cliques[i][j];
			for (int k = nq_size[node]; k < nq[node].size(); ++k)
			{
				int c2 = nq[node][k];
				//if (c2 != i)
				//{
				if (overlap[c2]++ == 0)
					computed[computed_index++] = c2;
				//overlappingClique.insert(c2);
				//cout<<"i: "<<i<<", c2: "<<c2<<endl;

				//}
			}
			nq_size[node]++;
		}

		for (int j = 0; j < computed_index; j++)
		{
			int node = computed[j];
			int coverlap = overlap[node];

			if (coverlap > Overlap)
			{

				graphEdges[coverlap].push_back(Edge(i, node));
			}
			overlap[node] = 0;
		}

		computed_index = 0;
		if (i % 100000 == 0)
			cout << i << ", " << i * 1.0 / size << endl;
	}

#ifdef __linux__
	gettimeofday(&end, NULL);
	long long mtime, seconds, useconds;
	seconds = end.tv_sec - start.tv_sec;
	useconds = end.tv_usec - start.tv_usec;
	mtime = seconds * 1000000 + useconds;
	printf("cag processing time, %lld us\n", mtime);
#else
	end = clock();
	printf("cag processing time, %f second\n", (float)(end-start)/CLOCKS_PER_SEC);
#endif

	size_t  gsize = 0;
	for(int i = 0; i < graphEdges.size(); ++i)
	{
		gsize += (12*graphEdges[i].size());

	}
	cout<<"graph size: "<<gsize/1000000.0<<" MB"<<endl;

	/*string graphFile = dir + "phi.bin";
	FILE *graphF = fopen(graphFile.c_str(), "wb");
	for(int i = 0; i < graphEdges.size(); ++i)
	{
		for(int j = 0; j < graphEdges[i].size(); ++j)
		{
			int u = graphEdges[i][j].a;
			int v = graphEdges[i][j].b;
			int w = i;
			fwrite(&u, sizeof(int), 1, graphF);
			fwrite(&v, sizeof(int), 1, graphF);
			fwrite(&w, sizeof(int), 1, graphF);
		}

	}
	fclose(graphF);*/

	delete[] overlap;
	delete[] nq_size;
	delete[] computed;

	//cout << "processing time: " << time(NULL) - begin << endl;

	cout << "finshing building " << endl;
#ifdef DEBUG
	cout<<"maximum clique size: "<<maxCliqueSize<<endl;
	for(int i = 0; i < maxCliqueSize; i++)
	{
		cout<<"weight "<<i<<": ";
		for(int j = 0; j < graphEdges[i].size(); j++)
		cout<<"("<<graphEdges[i][j].a<<", "<<graphEdges[i][j].b<<") ";
		cout<<endl;

	}
	cout<<endl;

#endif
}

void DensityGraph::maxSpanningTree()
{

	cerr << "Building max spanning tree" << endl;

#ifdef __linux__
        struct timeval start, end;
        gettimeofday(&start, NULL);
#else
        int start, end;
        start = clock();
#endif


	UF uf(cliques.size());
//	mstEdges.resize(cliques.size());
//	for (int i = maxCliqueSize - 1; i > 0; --i)
//	{
//		auto size = graphEdges[i].size();
//		for (int j = 0; j < size; ++j)
//		{
//			int a = graphEdges[i][j].a;
//			int b = graphEdges[i][j].b;
//			if (!uf.connected(a, b))
//			{
//				mstEdges[a].push_back(make_pair(b, i));
//				mstEdges[b].push_back(make_pair(a, i));
//				uf.uni(a, b);
//			}
//		}
//
//	}

	mstEdges.resize(maxCliqueSize);
	for (int i = maxCliqueSize - 1; i > 0; --i)
	{
		auto size = graphEdges[i].size();
		for (int j = 0; j < size; ++j)
		{
			int a = graphEdges[i][j].a;
			int b = graphEdges[i][j].b;
			if (!uf.connected(a, b))
			{
				mstEdges[i].push_back(make_pair(a, b));
				//mstEdges[a].push_back(make_pair(b, i));
				uf.uni(a, b);
			}
		}
	}


#ifdef __linux__
        gettimeofday(&end, NULL);

        long long mtime, seconds, useconds;
        seconds = end.tv_sec - start.tv_sec;
        useconds = end.tv_usec - start.tv_usec;
        mtime = seconds * 1000000 + useconds;

        printf("processing  time, %lld us\n", mtime);

#else
        end = clock();

        printf("processing time, %f second\n", (float)(end-start)/CLOCKS_PER_SEC);
#endif


	cerr << "finish buiding" << endl;
#ifdef DEBUG
	for(int i = 0; i < mstEdges.size(); ++i)
	{
		cout<<"clique "<<i<<": ";
		for(int j = 0; j < mstEdges[i].size(); ++j)
		cout<<"("<<mstEdges[i][j].first<<", "<<mstEdges[i][j].second<<") ";
		cout<<endl;
	}
	cout<<endl;
#endif
}

void DensityGraph::getMST(string dir, size_t ms, CLIQUESET &c, MSTEDGE &m, NODE2CLIQUE &n)
{
	DensityGraph dg(ms, c, m, n);
	dg.buildGraph(dir);
	dg.maxSpanningTree();
}
