/*
 * tree.cpp
 *
 *  Created on: 7May,2017
 *      Author: longyuan
 */

#include "tree.h"
#include <iostream>
#include <algorithm>
#include <queue>
#include "unionfind.h"
#include "config.h"

using namespace std;

void Tree::adjustDepth()
{
	adjustDepthHelper(root, 0);

}

void Tree::adjustDepthHelper(TreeNode*tn, int d)
{
	if (tn != nullptr)
	{
		tn->setDepth(d);
		for (size_t i = 0; i < tn->children.size(); ++i)
		{
			adjustDepthHelper(tn->children[i], d + 1);
		}
	}
}

void Tree::insertNode(TreeNode *p, TreeNode *c)
{
	c->setParent(p);
	p->insertChild(c);
}

TreeNode *Tree::newNode(int weight)
{
	TreeNode * nNode = new TreeNode();
	nNode->setWeight(weight);
	if (root == nullptr)
		root = nNode;
	return nNode;
}

TreeNode *Tree::lca(TreeNode *tn1, TreeNode *tn2)
{
	while (tn1 != tn2)
	{
		if (tn1->depth < tn2->depth)
		{
			tn2 = tn2->parent;
		}
		else
		{
			tn1 = tn1->parent;
		}

	}
	return tn1;
}

Tree::~Tree()
{
	deleteNode(root);
}

void Tree::deleteNode(TreeNode *tn)
{
	if (tn != nullptr)
	{
		for (size_t i = 0; i < tn->children.size(); ++i)
		{
			deleteNode(tn->children[i]);
		}
		delete tn;
	}
}

int TreeArray::lca(int a, int b, int&minw)
{

	while (a != b)
	{
		if (nodes[a].getDepth() < nodes[b].getDepth())
		{
			minw = min(minw, nodes[b].getWeight());
			b = nodes[b].getParent();
		}
		else
		{
			minw = min(minw, nodes[a].getWeight());
			a = nodes[a].getParent();
		}

	}
	return a;
}

void TreeArray::insertNode(int parent, int child)
{
	nodes[child].setParent(parent);
	nodes[child].setDepth(nodes[parent].getDepth() + 1);
}

int TreeArray::newNode(int weight)
{
	nodes[index].setWeight(weight);
	nodes[index].setDepth(0);
	return index++;
}

void TreeArray::print()
{
	cout << "num: " << num << ", index: " << index << endl;
	for (int i = 0; i < num; ++i)
	{
		cout << "node " << i << ": " << endl;
		cout << "\tParent:" << nodes[i].getParent() << endl;
		cout << "\tWeight:" << nodes[i].getWeight() << endl;
		cout << "\tDepth:" << nodes[i].getDepth() << endl;
	}
}

void OATTreeArray::print()
{
	cout << "num: " << num << ", index: " << index << endl;
	printMC();
	printIMCI();
	printOAT();
}

void OATTreeArray::printOAT()
{
	for (int i = 0; i < index; ++i)
	{
		cout << "node " << i << ": " << endl;
		cout << "\tParent:" << nodes[i].getParent() << endl;
		cout << "\tWeight:" << nodes[i].getWeight() << endl;
		cout << "\tDepth:" << nodes[i].getDepth() << endl;
		cout << "\tlChild:" << nodes[i].getlChild() << endl;
		cout << "\trChild:" << nodes[i].getrChild() << endl;
	}
}

void OATTreeArray::printMC()
{
	cout << "total cliques: " << mc.size() << endl;
	cout << "cliques" << endl;
	for (int i = 0; i < mc.size(); ++i)
	{
		if (mc[i].size() > 0)
		{
			cout << i << ": ";
			for (int j = 0; j < mc[i].size(); ++j)
				cout << mc[i][j] << ", ";
			cout << endl;
		}
	}

}

void OATTreeArray::printIMCI()
{
	cout << "total nodes: " << imci.size() << endl;
	cout << "cliques contained" << endl;
	for (int i = 0; i < imci.size(); ++i)
	{
		if (imci[i].size() > 0)
		{
			cout << i << ": ";
			for (int j = 0; j < imci[i].size(); ++j)
				cout << imci[i][j] << ", ";
			cout << endl;
		}
	}
}

void OATTreeArray::buildOATTree()
{

	cout << "Building OAT Tree" << endl;

#ifdef __linux__
	struct timeval start, end;
	gettimeofday(&start, NULL);
#else
	int start, end;
	start = clock();
#endif
	UFOAT uf(num * 2);
	initLeafNode();

	std::set<int> root;

	int ind = 0;

	for (int i = edges.size() - 1; i > 0; --i)
	{
		for (int j = 0; j < edges[i].size(); ++j)
		{
			int c = i;
			int a = edges[i][j].first;
			int b = edges[i][j].second;
			int pa = uf.find(a);
			int pb = uf.find(b);
			ind = newNode(c);
			nodes[pa].setParent(ind);
			nodes[pb].setParent(ind);
			nodes[ind].setChild(pa);
			nodes[ind].setChild(pb);
			nodes[ind].setWeight(c);
			uf.setParent(ind, pa);
			uf.setParent(ind, pb);
		}
	}

	//print();
	nodes.resize(index + 1);
	nodes[index].setParent(-1);
	nodes[index].setWeight(0);

//	for (int i = 0; i < num; ++i)
//	{
//		int t = i;
//		while (nodes[t].getParent() != -1)
//		{
//			t = nodes[t].getParent();
//		}
//		root.insert(t);
//	}

	for (int i = 0; i < index; ++i)
	{
		if (nodes[i].getParent() == -1)
			root.insert(i);
	}

	for (auto it = root.begin(); it != root.end(); ++it)
	{
		int t = *it;
		nodes[t].setDepth(0);
		nodes[t].setParent(index);
		queue<int> q;
		if (nodes[t].getlChild() != -1)
			q.push(nodes[t].getlChild());
		if (nodes[t].getrChild() != -1)
			q.push(nodes[t].getrChild());
		while (!q.empty())
		{
			int h = q.front();
			q.pop();
			int p = nodes[h].getParent();
			nodes[h].setDepth(nodes[p].getDepth() + 1);
			if (nodes[h].getlChild() != -1)
				q.push(nodes[h].getlChild());
			if (nodes[h].getrChild() != -1)
				q.push(nodes[h].getrChild());

		}
	}
	//print();
	cerr << "finish building" << endl;

#ifdef __linux__
	gettimeofday(&end, NULL);
	long long mtime, seconds, useconds;
	seconds = end.tv_sec - start.tv_sec;
	useconds = end.tv_usec - start.tv_usec;
	mtime = seconds*1000000 + useconds;

	printf("oat processing time, %lld us\n", mtime);
#else
	end = clock();
	printf("oat processsing time, %f second\n", (float)(end-start)/CLOCKS_PER_SEC);
#endif

	size_t memory = 0;
	for(int i = 0; i < mc.size(); ++i)
		memory += (mc[i].size()*4);
	for(int i = 0; i < imci.size(); ++i)
		memory += (imci[i].size()*4);
	memory += (index*20);
	cout<<"memory size: "<<memory/1000000.0 <<" M"<<endl;
	output_index();
}

void OATTreeArray::queryProcess(std::set<int> &query)
{
	char *inPQ = new char[num * 2];
	fill_n(inPQ, num * 2, 0);
	std::set<int> result;
	vector<OATBit> bitmap;
	bitmap.resize(num * 2);
	priority_queue<OATSearchNode> pq;
	int ind = 0;
	for (auto it = query.begin(); it != query.end(); ++it)
	{
		int n = *it;
		for (int i = 0; i < imci[n].size(); ++i)
		{
			int c = imci[n][i];
			bitmap[c].setbit(ind);
			if (inPQ[c] == 0)
			{
				pq.push(OATSearchNode(c, nodes[c].getDepth()));
				inPQ[c] = 1;
			}
		}
		ind++;
	}

	while (!pq.empty())
	{
		OATSearchNode sn = pq.top();
		pq.pop();
		int n = sn.id;
		int pn = nodes[n].getParent();
		if (bitmap[n].allSet(ind)
				&& nodes[n].getWeight() > nodes[pn].getWeight())
		{
			result.insert(n);
		}
		else
		{
			bitmap[pn].merge(bitmap[n]);
			if (pn != index)
			{
				if (inPQ[pn] == 0)
				{
					pq.push(OATSearchNode(pn, nodes[pn].getDepth()));
					inPQ[pn] = 1;
				}
			}
		}

	}

	if (!result.empty())
	{
		auto it = result.begin();
		int t = *it;
		int w = nodes[t].getWeight();
		while (++it != result.end())
		{
			int t1 = *it;
			int w1 = nodes[t1].getWeight();
			if (w < w1)
			{
				w = w1;
				t = t1;
			}
		}
		//cout << "t: " << t << ",w: " << w << endl;
	}
	//else
		//cout << "no dcpc" << endl;
	delete[] inPQ;
}


void OATTreeArray::queryNaiveProcess(std::set<int> &query)
{

	vector<int> query1;
	for (auto it = query.begin(); it != query.end(); ++it)
	{
		query1.push_back(*it);
	}
	size_t qsize = query1.size();
	int minweight = -1;
	vector<int> result;
	cycle(query1, qsize, qsize, result, minweight);
	//cout << "minweight: " << minweight << endl;
}

void OATTreeArray::cycle(std::vector<int> &nodes, int size, int x,
		std::vector<int> &result, int &minweight)
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
		{
			minweight = max(minweight, (int) mc[result1[0]].size());
			return;
		}
		else
			for (int i = 0; i < result1.size(); ++i)
			{
				if ((int)(mc[result1[i]].size()-1) <= minweight)
				{
					return;
				}
			}
		minweight = max(minweight, (int)queryHelper(result1));
		return;
#ifdef DEBUG
		cout<<"minweight: "<<minweight<<endl;
#endif
	}
	int n = nodes[size - x];
	for (int i = 0; i < imci[n].size(); ++i)
	{
		result.push_back(imci[n][i]);
		cycle(nodes, size, --x, result, minweight);
		result.pop_back();
		x++;
	}

}

int OATTreeArray::queryHelper(std::vector<int> &query)
{
	//cout<<"query help"<<endl;
	char *inPQ = new char[num * 2];
	fill_n(inPQ, num * 2, 0);
	std::set<int> result;
	priority_queue<OATSearchNode> pq;
	for (auto it = query.begin(); it != query.end(); ++it)
	{
		int c = *it;

		if (inPQ[c] == 0)
		{
			pq.push(OATSearchNode(c, nodes[c].getDepth()));
			inPQ[c] = 1;
		}
	}

	while (!pq.empty())
	{
		OATSearchNode sn = pq.top();
		pq.pop();
		int n = sn.id;
		int pn = nodes[n].getParent();
		if (nodes[n].getWeight() > nodes[pn].getWeight())
		{
			result.insert(n);
		}
		else
		{
			if (pn != index)
			{
				if (inPQ[pn] == 0)
				{
					pq.push(OATSearchNode(pn, nodes[pn].getDepth()));
					inPQ[pn] = 1;
				}
			}
		}

	}

	int t = 0;
	int w = -1;
	if (!result.empty())
	{
		auto it = result.begin();
		int t = *it;
		w = nodes[t].getWeight();
		while (++it != result.end())
		{
			int t1 = *it;
			int w1 = nodes[t1].getWeight();
			if (w < w1)
			{
				w = w1;
				t = t1;
			}
		}
		//cout << "t: " << t << ",w: " << w << endl;
	}
	//else
		//cout << "no dcpc" << endl;
	delete[] inPQ;
	return w;

}

void OATTreeArray::output_index()
{
	string output = dir;
	output += "index.bin";
	FILE *outf = fopen(output.c_str(), "wb");
	int size = nodes.size();
	fwrite(&size, sizeof(int), 1, outf);
	for (int i = 0; i < nodes.size(); ++i)
	{
		fwrite(&nodes[i], sizeof(OATNode), 1, outf);
	}
	fclose(outf);

	output = dir;
	output += "meta.bin";
	outf = fopen(output.c_str(), "wb");
	fwrite(&num, sizeof(int), 1, outf);
	fwrite(&index, sizeof(int), 1, outf);
	fclose(outf);

	output = dir;
	output += "mc.bin";
	outf = fopen(output.c_str(), "wb");
	size = mc.size();
	fwrite(&size, sizeof(int), 1, outf);
	for (int i = 0; i < mc.size(); ++i)
	{
		size = mc[i].size();
		fwrite(&size, sizeof(int), 1, outf);
		for (int j = 0; j < size; ++j)
		{
			fwrite(&mc[i][j], sizeof(int), 1, outf);
		}
	}
	fclose(outf);

	output = dir;
	output += "imci.bin";
	outf = fopen(output.c_str(), "wb");
	size = imci.size();
	fwrite(&size, sizeof(int), 1, outf);
	for (int i = 0; i < imci.size(); ++i)
	{
		size = imci[i].size();
		fwrite(&size, sizeof(int), 1, outf);
		for (int j = 0; j < size; ++j)
		{
			fwrite(&imci[i][j], sizeof(int), 1, outf);
		}
	}
	fclose(outf);

	//print();
}

void OATTreeArray::load_index()
{
	string input = dir;
	input += "index.bin";
	FILE *inf = fopen(input.c_str(), "rb");
	int size;
	fread(&size, sizeof(int), 1, inf);
	nodes.resize(size);
	for (int i = 0; i < size; ++i)
	{
		fread(&nodes[i], sizeof(OATNode), 1, inf);
	}
	fclose(inf);

	input = dir;
	input += "meta.bin";
	inf = fopen(input.c_str(), "rb");
	fread(&num, sizeof(int), 1, inf);
	fread(&index, sizeof(int), 1, inf);
	fclose(inf);

	input = dir;
	input += "mc.bin";
	inf = fopen(input.c_str(), "rb");
	fread(&size, sizeof(int), 1, inf);
	mc.resize(size);
	for (int i = 0; i < mc.size(); ++i)
	{
		fread(&size, sizeof(int), 1, inf);
		mc[i].resize(size);
		for (int j = 0; j < size; ++j)
		{
			fread(&mc[i][j], sizeof(int), 1, inf);
		}
	}
	fclose(inf);

	input = dir;
	input += "imci.bin";
	inf = fopen(input.c_str(), "rb");
	fread(&size, sizeof(int), 1, inf);
	imci.resize(size);
	for (int i = 0; i < imci.size(); ++i)
	{
		fread(&size, sizeof(int), 1, inf);
		imci[i].resize(size);
		for (int j = 0; j < size; ++j)
		{
			fread(&imci[i][j], sizeof(int), 1, inf);
		}
	}
	fclose(inf);

//print();

}

void OATTreeArray::insertNode(int parent, int child)
{
	nodes[child].setParent(parent);
	nodes[parent].setChild(child);
}

int OATTreeArray::newNode(int weight)
{
	nodes[index].setWeight(weight);
	nodes[index].setDepth(0);
	return index++;
}

void OATTreeArray::initLeafNode()
{
	for (int i = 0; i < num; ++i)
		newNode();
	for (int i = 0; i < num; ++i)
		nodes[i].setWeight(mc[i].size() - 1);
}

