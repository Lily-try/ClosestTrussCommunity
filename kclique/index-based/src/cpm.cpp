/*
 * cpm.cpp
 *
 *  Created on: 8May,2017
 *      Author: longyuan
 */

#include "cpm.h"
#include <iostream>
#include <algorithm>
#include <cstdio>
#include <string>
#include <cstdlib>
#include "cliques.h"
#include "querytree.h"
#include "tree.h"
#include "baseline.h"
using namespace std;

string CPM::output_name = "";

void cycle(vector<int> &nodes, int size, int x, vector<int> &result,
		NODE2CLIQUE &nq, CLIQUESET &cs, int &minweight, QueryTree &t,
		vector<int> &cliqueBound);

void printVector(vector<int> &r)
{
	for (int i = 0; i < r.size(); ++i)
		cout << r[i] << ", ";
}
void printCliques(CLIQUESET &cliques)
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

void CPM::runBaseline(string dir)
{
	loadQuery(dir);
	long long mmtime[QUERYNUM2];

	//for(int i = 2; i < 3; ++i)
	for (int i = 0; i < queryNodeSet.size(); ++i)
	{

		cerr<<QUERYNUM2<<", "<<queryNodeSet[i].size()<<endl;
		for(int j = 0; j < queryNodeSet[i].size(); ++j)
		{
			char st[16];
			sprintf( st, "%d", j+1 );
			CPM::output_name = dir + "output_clique_" + string(st) + ".txt";

#ifdef __linux__
		struct timeval start, end;
		gettimeofday(&start, NULL);
#else
		int start, end;
		start = clock();
#endif

		//for (int j = 0; j < queryNodeSet[i].size(); ++j)
		//{
			baseline(queryNodeSet[i][j], dir);
		//}

#ifdef __linux__
		gettimeofday(&end, NULL);

		long long mtime, seconds, useconds;
		seconds = end.tv_sec - start.tv_sec;
		useconds = end.tv_usec - start.tv_usec;
		mtime = seconds * 1000000 + useconds;

		//printf("processing  time, %lld us\n", mtime / (queryNodeSet[i].size()));
		mmtime[j] = mtime;

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
		var = var/ queryNodeSet[i].size();
		printf("agv: %lld, var: %lld \n", avg, var);
	}


}

void CPM::runBaseline2(string dir)
{
	loadQuery(dir);
	baseline2(queryNodeSet, dir);

/*	for(int i = 2; i < 3; ++i)
	//for (int i = 0; i < queryNodeSet.size(); ++i)
	{

#ifdef __linux__
		struct timeval start, end;
		gettimeofday(&start, NULL);
#else
		int start, end;
		start = clock();
#endif

		for (int j = 0; j < queryNodeSet[i].size(); ++j)
		{
			baseline(queryNodeSet[i][j], dir);
		}

#ifdef __linux__
		gettimeofday(&end, NULL);

		long long mtime, seconds, useconds;
		seconds = end.tv_sec - start.tv_sec;
		useconds = end.tv_usec - start.tv_usec;
		mtime = seconds * 1000000 + useconds;

		printf("processing  time, %lld us\n", mtime / (queryNodeSet[i].size()));

#else
		end = clock();

		printf("processing time, %f second\n", (float)(end-start)/(queryNodeSet[i].size()*CLOCKS_PER_SEC));
#endif
	}*/
}

void CPM::runOATIndex(string dir)
{
	CLIQUESET cliques;
	NODE2CLIQUE node2clique;
	MSTEDGE *mst = new MSTEDGE;
	size_t maximumCliqueSize = 0;

	loadQuery(dir);
	//queryNodeSet = {{{12, 15}, {6, 7}}};

	OATTreeArray tree(dir, cliques.size(), *mst, cliques, node2clique);
	tree.load_index();
	long long mmtime[QUERYNUM2];

	//for(int i = 2; i < 3; ++i)
	for (int i = 0; i < queryNodeSet.size(); ++i)
	{
		cerr<<QUERYNUM2<<", "<<queryNodeSet[i].size()<<endl;
		for(int j = 0; j < queryNodeSet[i].size(); ++j)
		{

#ifdef __linux__
		struct timeval start, end;
		gettimeofday(&start, NULL);
#else
		int start, end;
		start = clock();
#endif
		printf( "processing query %d\n", j + 1 );
		//for (int j = 0; j < queryNodeSet[i].size(); ++j)
		//{
			tree.queryNaiveProcess(queryNodeSet[i][j]);
		//}
#ifdef __linux__
		gettimeofday(&end, NULL);

		long long mtime, seconds, useconds;
		seconds = end.tv_sec - start.tv_sec;
		useconds = end.tv_usec - start.tv_usec;
		mtime = seconds * 1000000 + useconds;
		mmtime[j] = mtime;

		//printf("processing  time, %lld us\n", mtime / (queryNodeSet[i].size()));

#else
		end = clock();

		printf("processing time, %f second\n", (float)(end-start)/(queryNodeSet[i].size()*CLOCKS_PER_SEC));
#endif
		}
	//}

	long long avg = 0;
	long long var = 0;
	for(int j = 0; j < queryNodeSet[i].size(); ++j)
		avg += mmtime[j];

	double tavg = avg*1.0 / queryNodeSet[i].size();
	avg = avg/queryNodeSet[i].size();
	for(int j = 0; j < queryNodeSet[i].size(); ++j)
		var += (mmtime[j] - avg)*(mmtime[j] - avg);
	var = var/queryNodeSet[i].size();
	//printf("avg: %lld, var is: %lld \n", avg, var);
	printf( "average processing time/query = %0.3lf microseconds\n", tavg);

	}

	delete mst;
}

void CPM::buildOAT(string dir)
{
	CLIQUESET cliques;
	NODE2CLIQUE node2clique;
	MSTEDGE *mst = new MSTEDGE;
	size_t maximumCliqueSize = 0;
#ifdef __linux__
	struct timeval start, end;
	gettimeofday(&start, NULL);
#else
	int start, end;
	start = clock();
#endif

	cerr << "start computing cliques......" << endl;
	int n = Cliques::getMaxCliques(dir, cliques, maximumCliqueSize);
	cerr << "finish computing cliques......" << endl;
	cerr << "clique size: " << cliques.size() << endl;

	cerr << "node2cliques......" << endl;
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
		sort(node2clique[i].begin(), node2clique[i].end());
	}

#ifdef DEBUG
	cout<<"total cliques: "<<cliques.size()<<endl;
	cout<<"cliques"<<endl;
	printCliques(cliques);
	cout<<endl;
	cout<<"node2index"<<endl;
	printCliques(node2clique);
	cout<<endl;
#endif
	DensityGraph::getMST(dir, maximumCliqueSize, cliques, *mst, node2clique);
	OATTreeArray tree(dir, cliques.size(), *mst, cliques, node2clique);
	tree.buildOATTree();
	vector<int> cliqueBound;
	cliqueBound.resize(cliques.size());
	for (int i = 0; i < cliqueBound.size(); ++i)
	{
		cliqueBound[i] = cliques[i].size() - 1;
	}

	cerr << "sorting" << endl;

	for (int i = 0; i < node2clique.size(); ++i)
	{
		sort(node2clique[i].begin(), node2clique[i].end(),
				[&cliqueBound](int i, int j)
				{	return cliqueBound[j] < cliqueBound[i];});
	}

	//delete mst;
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


	//naiveOATTree(*mst, cliques.size());

#ifdef __linux__
        gettimeofday(&start, NULL);

        seconds = start.tv_sec - end.tv_sec;
        useconds = start.tv_usec - end.tv_usec;
        mtime = seconds * 1000000 + useconds;

        printf("processing  time, %lld us\n", mtime);

#else
        start = clock();

        printf("processing time, %f second\n", (float)(start-end)/CLOCKS_PER_SEC);
#endif


}

void CPM::loadOAT(string dir)
{
	MSTEDGE mst;
	CLIQUESET cliques1;
	NODE2CLIQUE node2clique1;
	OATTreeArray tree(dir, 0, mst, cliques1, node2clique1);
	tree.load_index();

}

void CPM::runOAT(string dir)
{
	CLIQUESET cliques;
	NODE2CLIQUE node2clique;
	MSTEDGE *mst = new MSTEDGE;
	size_t maximumCliqueSize = 0;

	loadQuery(dir);

//	queryNodeSet =
//	{
//		{
//			{	3,6}},
//		{
//			{	3, 6, 11}},
//		{
//			{	3, 6, 9}}};

//	cerr << "start computing cliques......" << endl;
//	int n = Cliques::getMaxCliques(dir, cliques, maximumCliqueSize);
//	cerr << "finish computing cliques......" << endl;
//	cerr << "clique size: " << cliques.size() << endl;
//
//	cerr << "node2cliques......" << endl;
//	node2clique.resize(n);
//	for (int i = 0; i < cliques.size(); ++i)
//	{
//		size_t size = cliques[i].size();
//		for (int j = 0; j < size; ++j)
//		{
//			node2clique[cliques[i][j]].push_back(i);
//		}
//		cliques[i].shrink_to_fit();
//	}
//
//	for (int i = 0; i < node2clique.size(); ++i)
//	{
//		node2clique[i].shrink_to_fit();
//	}
//
//#ifdef DEBUG
//	cout<<"total cliques: "<<cliques.size()<<endl;
//	cout<<"cliques"<<endl;
//	printCliques(cliques);
//	cout<<endl;
//	cout<<"node2index"<<endl;
//	printCliques(node2clique);
//	cout<<endl;
//#endif
//	DensityGraph::getMST(maximumCliqueSize, cliques, *mst, node2clique);
	OATTreeArray tree(dir, cliques.size(), *mst, cliques, node2clique);
	tree.load_index();
	long long mmtime[QUERYNUM2];

	for (int i = 0; i < queryNodeSet.size(); ++i)
	{
	//for(int ii = 0; ii < 3; ++ii)
	//{
		cerr<< QUERYNUM2 <<", "<<queryNodeSet[i].size()<<endl;

		for(int j = 0; j < queryNodeSet[i].size(); ++j)
		{
#ifdef __linux__
		struct timeval start, end;
		gettimeofday(&start, NULL);
#else
		int start, end;
		start = clock();
#endif

		//for (int j = 0; j < queryNodeSet[i].size(); ++j)
		//{
			tree.queryProcess(queryNodeSet[i][j]);
		//}
#ifdef __linux__
		gettimeofday(&end, NULL);

		long long mtime, seconds, useconds;
		seconds = end.tv_sec - start.tv_sec;
		useconds = end.tv_usec - start.tv_usec;
		mtime = seconds * 1000000 + useconds;

		//printf("processing  time, %lld us\n", mtime / (queryNodeSet[i].size()));
		//printf("processing time: %lld us\n", mtime);
		mmtime[j] = mtime;

#else
		end = clock();

		printf("processing time, %f second\n", (float)(end-start)/(queryNodeSet[i].size()*CLOCKS_PER_SEC));
#endif
		}
	//}
	long long avg = 0;
	long long var = 0;
	for(int j = 0; j < queryNodeSet[i].size(); ++j)
		avg += mmtime[j];
	avg = avg/queryNodeSet[i].size();
	for(int j = 0; j < queryNodeSet[i].size(); ++j)
		var += (mmtime[j] - avg) * (mmtime[j] -avg);
	var = var / queryNodeSet[i].size();
	printf("avg: %lld, var is : %lld \n", avg, var);

	}



	delete mst;

}

void CPM::loadQuery(std::string dir)
{
	queryNodeSet.resize(QUERYSIZE2);

	int size[QUERYSIZE2] = {1};
	//{ 2, 5, 10, 20, 30 };

	for (int i = 0; i < QUERYSIZE2; ++i)
	{
		string queryFile(dir);
		queryFile += "query";
		queryFile += to_string(size[i]);
		queryFile += ".txt";
		FILE *queryF = fopen(queryFile.c_str(), "r");
		if (queryF == NULL)
		{
			cout << "generate query node set please" << endl;
			exit(1);
		}
		set<int> query;
		for (int j = 0; j < QUERYNUM2; ++j)
		{
			int node;
			for (int k = 0; k < size[i]; ++k)
			{
				fscanf(queryF, "%d ", &node);
				query.insert(node);
			}
			queryNodeSet[i].push_back(query);
			query.clear();
		}
		fclose(queryF);
	}

}

void CPM::printQueryNodeSet()
{
	int size[QUERYSIZE] =
	{ 2, 5, 10, 20, 30 };
	for (int i = 0; i < queryNodeSet.size(); ++i)
	{
		cout << "SIZE: " << size[i] << endl;
		for (int j = 0; j < QUERYNUM; ++j)
		{
			for (auto it = queryNodeSet[i][j].begin();
					it != queryNodeSet[i][j].end(); ++it)
				cout << *it << ", ";
			cout << endl;
		}
		cout << endl;
	}
}

void CPM::generateQuery(std::string dir)
{
	CLIQUESET cliques;
	NODE2CLIQUE node2clique;
	MSTEDGE *mst = new MSTEDGE;
	size_t maximumCliqueSize = 0;

	/*cerr << "start computing cliques......" << endl;
	 int n = Cliques::getMaxCliques(dir, cliques, maximumCliqueSize);
	 cerr << "finish computing cliques......" << endl;
	 cerr << "clique size: " << cliques.size() << endl;

	 cerr << "node2cliques......" << endl;
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
	 }*/

	OATTreeArray tree(dir, cliques.size(), *mst, cliques, node2clique);
	tree.load_index();

	int n = node2clique.size();
	cout<<n<<endl;

	int size[QUERYSIZE] =
	{ 2, 5, 10, 20, 30 };

	srand(unsigned(time(0)));

	for (int i = 0; i < QUERYSIZE; ++i)
	{
		string queryFile(dir);
		queryFile += "query_100_";
		queryFile += to_string(size[i]);
		queryFile += ".txt";
		FILE *queryF = fopen(queryFile.c_str(), "w");
		vector<int> query;
		query.resize(size[i]);
		for (int j = 0; j < QUERYNUM; ++j)
		{

			vector<int> nodes;
			nodes.resize(n);
			for (int k = 0; k < n; ++k)
			{
				nodes[k] = k;
			}

			random_shuffle(nodes.begin(), nodes.end());

			int t = 0;
			for (int k = 0; k < n && t < size[i]; ++k)
			{
				if (node2clique[nodes[k]].size() > 0)
				{
					query[t] = nodes[k];
					t++;
				}
			}

			for (int k = 0; k < size[i]; ++k)
			{
				fprintf(queryF, "%d ", query[k]);
			}
			fprintf(queryF, "\n");
			query.clear();

		}
		fclose(queryF);

	}

}

/*void CPM::runNaiveOAT(string dir)
{
	CLIQUESET cliques;
	NODE2CLIQUE node2clique;
	MSTEDGE *mst = new MSTEDGE;
	size_t maximumCliqueSize = 0;

	cerr << "start computing cliques......" << endl;
	int n = Cliques::getMaxCliques(dir, cliques, maximumCliqueSize);
	cerr << "finish computing cliques......" << endl;
	cerr << "clique size: " << cliques.size() << endl;

	cerr << "node2cliques......" << endl;
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

#ifdef DEBUG
	cout<<"total cliques: "<<cliques.size()<<endl;
	cout<<"cliques"<<endl;
	printCliques(cliques);
	cout<<endl;
	cout<<"node2index"<<endl;
	printCliques(node2clique);
	cout<<endl;
#endif
	DensityGraph::getMST(dir, maximumCliqueSize, cliques, *mst, node2clique);
	naiveOATTree(*mst, cliques.size());

}*/

void CPM::run(string dir)
{
	CLIQUESET cliques;
	NODE2CLIQUE node2clique;
	MSTEDGE * mst = new MSTEDGE;
	size_t maximumCliqueSize = 0;

	loadQuery(dir);

	cerr << "start computing cliques......" << endl;
	int n = Cliques::getMaxCliques(dir, cliques, maximumCliqueSize);
	cerr << "finish computing cliques......" << endl;
	cerr << "clique size: " << cliques.size() << endl;

	cerr << "node2cliques......" << endl;
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

#ifdef DEBUG
	cout<<"total cliques: "<<cliques.size()<<endl;
	cout<<"cliques"<<endl;
	printCliques(cliques);
	cout<<endl;
	cout<<"node2index"<<endl;
	printCliques(node2clique);
	cout<<endl;
#endif
	DensityGraph::getMST(dir, maximumCliqueSize, cliques, *mst, node2clique);

	vector<int> cliqueBound;
	cliqueBound.resize(cliques.size());
	for (int i = 0; i < cliqueBound.size(); ++i)
	{
		int maxBound = 0;
		for (int j = 0; j < (*mst)[i].size(); ++j)
		{
			maxBound = max(maxBound, (*mst)[i][j].second);
		}
		cliqueBound[i] = maxBound;
	}

	cerr << "sorting" << endl;

	for (int i = 0; i < n; ++i)
	{
		sort(node2clique[i].begin(), node2clique[i].end(),
				[&cliqueBound](int i, int j)
				{	return cliqueBound[j] < cliqueBound[i];});
	}
	cerr << "finish sorting" << endl;

	QueryTree t(cliques.size(), *mst);
	t.buildTree();
	delete mst;

	for (size_t i = 0; i < queryNodeSet.size(); ++i)
		for (size_t j = 0; j < queryNodeSet[i].size(); ++j)
		{
			{
				int minweight = -1;
				vector<int> result;
				vector<int> query;
				for (auto it = queryNodeSet[i][j].begin();
						it != queryNodeSet[i][j].end(); ++it)
				{
					query.push_back(*it);
				}
				size_t qsize = query.size();
				cycle(query, qsize, qsize, result, node2clique, cliques,
						minweight, t, cliqueBound);
				cout << i << ": " << minweight << endl;
				query.clear();
			}
		}
}

void cycle(vector<int> &nodes, int size, int x, vector<int> &result,
		NODE2CLIQUE &nq, CLIQUESET &cs, int &minweight, QueryTree &t,
		vector<int>& cliqueBound)
{
	if (x == 0)
	{

#ifdef DEBUG
		cout<<"result: ";
		for(int i = 0; i < result.size(); ++i)
		cout<<result[i]<<", ";
		cout<<endl;
#endif
		vector<int> result1(result);
		sort(result1.begin(), result1.end());
		result1.erase(unique(result1.begin(), result1.end()), result1.end());
#ifdef DEBUG
		cout<<"result1: ";
		for(int i = 0; i < result1.size(); ++i)
		cout<<result1[i]<<", ";
		cout<<endl;
#endif
		if (result1.size() == 1)
			minweight = max(minweight, (int) cs[result1[0]].size()-1);
		else
			/*for (int i = 0; i < result1.size(); ++i)
			{
				if (cliqueBound[result1[i]] <= minweight)
					return;
			}*/
		minweight = max(minweight, t.query(result1));
		return;
#ifdef DEBUG
		cout<<"minweight: "<<minweight<<endl;
#endif
	}
	int n = nodes[size - x];
	for (int i = 0; i < nq[n].size(); ++i)
	{
		result.push_back(nq[n][i]);
		cycle(nodes, size, --x, result, nq, cs, minweight, t, cliqueBound);
		result.pop_back();
		x++;
	}

}
