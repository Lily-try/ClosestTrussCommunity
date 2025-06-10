/*
 * unionfind.cpp

 *
 *  Created on: 7May,2017
 *      Author: longyuan
 */
#include <algorithm>
#include "unionfind.h"
using namespace std;

UF::UF(int n)
{
	num = n;
	parent = new int[num];
	rank = new int[num];
	fill_n(rank, num, 0);
	for (int i = 0; i < num; ++i)
		parent[i] = i;
	p1 = -1;
	p2 = -1;
}

UF::~UF()
{
	if (parent != nullptr)
		delete[] parent;
	if (rank != nullptr)
		delete[] rank;
}

int UF::find(int e)
{
	int root = e;
	while (parent[root] != root)
	{
		root = parent[root];
	}

	while (parent[e] != root)
	{
		int tmp = parent[e];
		parent[e] = root;
		e = tmp;
	}

	return root;
}

bool UF::connected(int e1, int e2)
{
	p1 = find(e1);
	p2 = find(e2);
	return p1 == p2;
}

void UF::uni(int e1, int e2)
{
	if (rank[p1] < rank[p2])
		parent[p1] = p2;
	else if (rank[p1] > rank[p2])
		parent[p2] = p1;
	else
	{
		parent[p1] = p2;
		++rank[p2];
	}
}

UFOAT::UFOAT(int n)
{
	num = n;
	parent = new int[num];
	for (int i = 0; i < num; ++i)
		parent[i] = i;
}

UFOAT::~UFOAT()
{
	if (parent != nullptr)
		delete[] parent;
}

int UFOAT::find(int e)
{
	int root = e;
	while (parent[root] != root)
	{
		root = parent[root];
	}

	while (parent[e] != root)
	{
		int tmp = parent[e];
		parent[e] = root;
		e = tmp;
	}

	return root;
}

void UFOAT::setParent(int p, int c)
{
	parent[c] = p;

}
