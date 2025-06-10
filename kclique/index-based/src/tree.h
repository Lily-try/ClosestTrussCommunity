/*
 * tree.h
 *
 *  Created on: 7May,2017
 *      Author: longyuan
 */

#ifndef TREE_H_
#define TREE_H_
#include <vector>
#include <map>
#include <set>
#include <string>
#include "config.h"

//  for the performance consideration, polymorphism is not considered here!!

class TreeNodeHierarchy
{
public:
	TreeNodeHierarchy(int l, std::vector<int> &cs) :
			level(l), cliques(cs), children(std::vector<TreeNodeHierarchy *>())
	{
	}
	void setLevel(int l)
	{
		level = l;
	}
	int getLevel()
	{
		return level;
	}

private:
	int level;
	std::vector<int>& cliques;
	std::vector<TreeNodeHierarchy *> children;

};

class TreeHierarchy
{
public:
	void insertNode(int l, std::vector<int> &cs);
	TreeNodeHierarchy *lca(TreeNodeHierarchy *t1, TreeNodeHierarchy *t2);
private:
	TreeNodeHierarchy *root;
};

class TreeNode
{
public:
	TreeNode() :
			parent(nullptr), weight(0), depth(0)
	{
	}
	void setParent(TreeNode *p)
	{
		parent = p;
	}
	void setWeight(int w)
	{
		weight = w;
	}
	void setDepth(int d)
	{
		depth = d;
	}
	void insertChild(TreeNode *c)
	{
		children.push_back(c);
	}

public:
	std::vector<TreeNode *> children;
	TreeNode * parent;
	int weight;
	int depth;
};

class Tree
{
public:
	Tree() :
			root(nullptr)
	{
	}
	~Tree();

	void insertNode(TreeNode *p, TreeNode*c);
	TreeNode *newNode(int weight = 0);
	void adjustDepth();
	TreeNode *lca(TreeNode *tn1, TreeNode *tn2);

private:
	void deleteNode(TreeNode *tn);
	void adjustDepthHelper(TreeNode *tn, int d);
private:
	TreeNode *root;
};

class TreeNodeArray
{
public:
	TreeNodeArray() :
			parent(-1), weight(-1), depth(-1)
	{
	}
	void setParent(int p)
	{
		parent = p;
	}
	void setWeight(int w)
	{
		weight = w;
	}
	void setDepth(int d)
	{
		depth = d;
	}
	int getParent()
	{
		return parent;
	}
	int getWeight()
	{
		return weight;
	}
	int getDepth()
	{
		return depth;
	}

private:
	int parent;
	int weight;
	int depth;
};

class TreeArray
{
public:
	TreeArray(int n)
	{
		num = n;
		index = 0;
		nodes.resize(num + 1);
	}
	~TreeArray()
	{
	}
	//void setRoot(int n) {root = n;}
	void insertNode(int parent, int child);
	int newNode(int weight = -1);
	int lca(int a, int b, int &minw);
	void print();

private:
	int num;
	int index;
	std::vector<TreeNodeArray> nodes;
};

class OATBit
{
public:
	void setbit(int i)
	{
		bits |= 1 << i;
	}
	void clearbit(int i)
	{
		bits &= ~(1 << i);
	}
	void merge(OATBit &other)
	{
		bits |= other.bits;
	}
	bool isSet(int i)
	{
		return (bits & (1 << i)) > 0;
	}
	bool allSet(int size)
	{
		for (int i = 0; i < size; ++i)
			if (!isSet(i))
			{
				return false;
			}
		return true;
	}
	OATBit()
	{
		bits = 0;
	}
	unsigned int bits;
};

class OATSearchNode
{
public:
	OATSearchNode(int id_, int depth_) :
			id(id_), depth(depth_)
	{
	}
	int id;
	int depth;
	const bool operator<(const OATSearchNode & other) const
	{
		return this->depth < other.depth;
	}
};

class OATNode
{
public:
	OATNode() :
			parent(-1), weight(-1), depth(-1), lchild(-1), rchild(-1)
	{
	}
	void setParent(int p)
	{
		parent = p;
	}
	void setWeight(int w)
	{
		weight = w;
	}
	void setDepth(int d)
	{
		depth = d;
	}
	int getParent()
	{
		return parent;
	}
	int getWeight()
	{
		return weight;
	}
	int getDepth()
	{
		return depth;
	}
	void setChild(int c)
	{
		if (lchild == -1)
			lchild = c;
		else
			rchild = c;
	}
	int getlChild()
	{
		return lchild;
	}
	int getrChild()
	{
		return rchild;
	}
	bool operator<(OATNode & other) const
	{
		return depth < other.getWeight();
	}

private:
	int parent;
	int weight;
	int depth;
	int lchild;
	int rchild;
};

class OATTreeArray
{
public:
	OATTreeArray(std::string d, int n, MSTEDGE &es, CLIQUESET &mc_,
			NODE2CLIQUE &imc) :
			dir(d), num(n), index(0), edges(es), mc(mc_), imci(imc), nodes()
	{
		nodes.resize(2 * n);
	}
	~OATTreeArray()
	{
	}
	void queryProcess(std::set<int> &query);
	void queryNaiveProcess(std::set<int> &query);
	int queryHelper(std::vector<int> &query);
	void cycle(std::vector<int> &nodes, int size, int x,
			std::vector<int> &result, int &minweight);

	void buildOATTree();

	void output_index();
	void load_index();

	void print();
	void printOAT();
	void printMC();
	void printIMCI();
public:
	void insertNode(int parent, int child);
	void initLeafNode();
	int newNode(int weight = -1);

private:
	std::string dir;
	int num;
	int index;
	MSTEDGE &edges;
	CLIQUESET &mc;
	NODE2CLIQUE &imci;
	std::vector<OATNode> nodes;

};

#endif /* TREE_H_ */
