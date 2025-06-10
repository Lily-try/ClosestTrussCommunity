/*
 * querytree.h
 *
 *  Created on: 7May,2017
 *      Author: longyuan
 */

#ifndef QUERYTREE_H_
#define QUERYTREE_H_


#include <vector>
#include "config.h"
#include "tree.h"
#include "densitygraph.h"


class QueryTree
{
public:
	QueryTree(size_t cs, MSTEDGE &e): num(cs), t(TreeArray(cs)), edges(e), id2index(std::vector<int>()), index2id(std::vector<int>()) {}
	void buildTree();
	void buildOAT();
	int query(std::vector<int> &cliqueIDs);

private:
	size_t num;
	TreeArray t;
	MSTEDGE &edges;
	std::vector<int> id2index;
	std::vector<int> index2id;
};


#endif /* QUERYTREE_H_ */
