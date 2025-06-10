/*
 * querytree.cpp
 *
 *  Created on: 7May,2017
 *      Author: longyuan
 */
#include "querytree.h"
#include <map>
#include <utility>
#include <queue>
#include <algorithm>
#include <iostream>
#include "unionfind.h"

using namespace std;


void QueryTree::buildOAT()
{

}

void QueryTree::buildTree()
{
	id2index.resize(num + 1);
	index2id.resize(num + 1);

	id2index[num] = num;
	index2id[num] = num;

	int *computed = new int[num];
	int *q = new int[num];
	fill_n(computed, num, 0);

	for (int i = 0; i < num; ++i)
	{
		if (computed[i] == 1)
			continue;

		int index = t.newNode();
		id2index[i] = index;
		index2id[index] = i;
		t.insertNode(num, index);
		computed[i] = 1;
		int q_c = 1;
		q[0] = i;

		for (int j = 0; j < q_c; ++j)
		{
			int node = q[j];
			int size = edges[node].size();
			for (int k = 0; k < size; ++k)
			{
				int nnode = edges[node][k].first;
				if (computed[nnode] != 1)
				{
					int w = edges[node][k].second;
					q[q_c++] = nnode;
					computed[nnode] = 1;
					index = t.newNode(w);
					id2index[nnode] = index;
					index2id[index] = nnode;
					t.insertNode(id2index[node], index);
				}
			}
		}
	}
	delete[]q;
	delete []computed;
#ifdef DEBUG
	for(int i = 0; i < num; ++i)
	{
		cout<<i<<"->"<<id2index[i]<<endl;
	}

	t.print();
#endif
}

int QueryTree::query(std::vector<int> &cliqueIDs)
{

	int minw = 1000000;
	int lca = id2index[cliqueIDs[0]];
	for (int i = 1; i < cliqueIDs.size(); ++i)
	{
		lca = t.lca(lca, id2index[cliqueIDs[i]], minw);
		if (lca == num)
		{

			break;
		}
	}

//return index2id[lca];
	return minw;
}



