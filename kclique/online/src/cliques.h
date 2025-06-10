/*
 * cliques.h
 *
 *  Created on: 7May,2017
 *      Author: longyuan
 */

#ifndef CLIQUES_H_
#define CLIQUES_H_

#include <string>
#include "config.h"
#include "list.h"

struct NeighborList
{
	int vertex;
	LinkedList* earlier;
	LinkedList* later;
	int orderNumber;
};

struct NeighborListArray
{
	int vertex;
	int* earlier;
	int earlierDegree;
	int* later;
	int laterDegree;
	int orderNumber;
};

class Cliques
{
public:
	Cliques(std::string str, CLIQUESET &cq, size_t &maximumSize_) :
			dir(str), cliques(cq), maximumSize(maximumSize_)
	{
	}
	static int getMaxCliques(std::string str, CLIQUESET &cq, size_t &maximumSize_);
	int computeMaxliques(std::string str, CLIQUESET &cq);

private:

	//for computing degeneracy
	int computeDegeneracy(LinkedList** list, int size);
	NeighborList** computeDegeneracyOrderList(LinkedList** list, int size);
	NeighborListArray** computeDegeneracyOrderArray(LinkedList** list,
			int size);
	int neighborListComparator(void* nl1, void* nl2);

	//for computing maximal clqiues;
	void listAllMaximalCliquesDegeneracyRecursive(long* cliqueCount,
			CLIQUESET& cliques, LinkedList* partialClique, int* vertexSets,
			int* vertexLookup, int** neighborsInP, int* numNeighbors,
			int beginX, int beginP, int beginR);

	long listAllMaximalCliquesDegeneracy(LinkedList** adjList,
			CLIQUESET& cliques,  int size);

private:
	std::string dir;
	CLIQUESET &cliques;
	size_t &maximumSize;

};

#endif /* CLIQUES_H_ */
