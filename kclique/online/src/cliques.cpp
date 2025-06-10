/*
 * cliques.cpp
 *
 *  Created on: 7May,2017
 *      Author: longyuan
 */

//#pragma GCC diagnostic ignored "-Wint-to-void-pointer-cast"
#pragma GCC diagnostic ignored "-Wint-to-pointer-cast"

#include "config.h"
#include "list.h"
#include "cliques.h"


#include <algorithm>
#include <vector>
#include <string>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <iostream>

using namespace std;
extern int Overlap;

void printList(LinkedList *list)
{
	cout << "length: " << length(list) << ", ";

	Link* curr = list->head->next;

	while (!isTail(curr))
	{
		cout << *(int *) &curr->data;
		if (!isTail(curr->next))
		{
			printf(" ");
		}
		curr = curr->next;
	}

	printf("\n");

}

void printAdjList(LinkedList **adjList, int n)
{
	for (int i = 0; i < n; ++i)
	{
		printList(adjList[i]);
	}
	printf("\n");

}

LinkedList** readInGraphAdjList(size_t* n, size_t* m, string dir)
{

	string metaFile = dir + "graph.meta";
	string edgeFile = dir + "graph.e";

#ifdef DEBUG
	cout <<metaFile<<endl;
	cout<< edgeFile<<endl;
#endif

	FILE *metaGraph = fopen(metaFile.c_str(), "r");
	FILE *edgeGraph = fopen(edgeFile.c_str(), "r");
	int u, v; // endvertices, to read edges.

	fscanf(metaGraph, "%zu", n);
	fscanf(metaGraph, "%zu", m);
#ifdef DEBUG
	cout<<*n<<", "<<*m<<endl;
#endif

	LinkedList** adjList = (LinkedList**) calloc(*n, sizeof(LinkedList*));

	int i = 0;
	while (i < *n)
	{
		adjList[i] = createLinkedList();
		i++;
	}

	i = 0;

	while (fscanf(edgeGraph, "%d %d", &u, &v) == 2)
	{

		addLast(adjList[u], (void*) v);
		addLast(adjList[v], (void*)u);
		i++;
	}
	fclose(edgeGraph);
	fclose(metaGraph);
#ifdef DEBUG
	cerr << "reading finish" << endl;
#endif
	return adjList;
}

int Cliques::getMaxCliques(string str, CLIQUESET &cq, size_t &maximumSize_)
{
	Cliques c(str, cq, maximumSize_);
	return c.computeMaxliques(str, cq);
}

int Cliques::computeMaxliques(std::string str, CLIQUESET &cq)
{
	size_t n;
	size_t m;

	LinkedList** adjacencyList = readInGraphAdjList(&n, &m, str);

#ifdef DEBUG
	//printAdjList(adjacencyList, n);
#endif

	listAllMaximalCliquesDegeneracy(adjacencyList, cliques, n);

	int i = 0;
	while (i < n)
	{
		destroyLinkedList(adjacencyList[i]);
		i++;
	}

	free(adjacencyList);

	return n;

}

int Cliques::computeDegeneracy(LinkedList** list, int size)
{
	int i = 0;

	int degeneracy = 0;

	// array of lists of vertices, indexed by degree
	LinkedList** verticesByDegree = (LinkedList**) calloc(size,
			sizeof(LinkedList*));

	// array of lists of vertices, indexed by degree
	Link** vertexLocator = (Link**) calloc(size, sizeof(Link*));

	int* degree = (int*) calloc(size, sizeof(int));

	for (i = 0; i < size; i++)
	{
		verticesByDegree[i] = createLinkedList();
	}

	for (i = 0; i < size; i++)
	{
		degree[i] = length(list[i]);
		vertexLocator[i] = addFirst(verticesByDegree[degree[i]], ((void*) i));
	}

	int currentDegree = 0;

	int numVerticesRemoved = 0;

	while (numVerticesRemoved < size)
	{
		if (!isEmpty(verticesByDegree[currentDegree]))
		{
			degeneracy = max(degeneracy, currentDegree);

			void *pv = getFirst(verticesByDegree[currentDegree]);

			int vertex = *(int *) pv;
			deleteNode(vertexLocator[vertex]);

			degree[vertex] = -1;

			LinkedList* neighborList = list[vertex];

			Link* neighborLink = neighborList->head->next;

			while (!isTail(neighborLink))
			{
				int neighbor = *(int *) &neighborLink->data;

				if (degree[neighbor] != -1)
				{
					deleteNode(vertexLocator[neighbor]);

					degree[neighbor]--;

					if (degree[neighbor] != -1)
					{
						vertexLocator[neighbor] = addFirst(
								verticesByDegree[degree[neighbor]],
								(void*) neighbor);
					}
				}

				neighborLink = neighborLink->next;
			}

			numVerticesRemoved++;
			currentDegree = 0;
		}
		else
		{
			currentDegree++;
		}

	}

	for (i = 0; i < size; i++)
	{
		destroyLinkedList(verticesByDegree[i]);
	}

	free(vertexLocator);
	free(verticesByDegree);
	free(degree);

	return degeneracy;
}

NeighborList** computeDegeneracyOrderList(LinkedList** list, int size)
{

	NeighborList** ordering = (NeighborList**) calloc(size,
			sizeof(NeighborList*));

	int i = 0;

	int degeneracy = 0;

	// array of lists of vertices, indexed by degree
	LinkedList** verticesByDegree = (LinkedList**) calloc(size,
			sizeof(LinkedList*));

	// array of lists of vertices, indexed by degree
	Link** vertexLocator = (Link**) calloc(size, sizeof(Link*));

	int* degree = (int*) calloc(size, sizeof(int));

	for (i = 0; i < size; i++)
	{
		verticesByDegree[i] = createLinkedList();
		ordering[i] = (NeighborList *) malloc(sizeof(NeighborList));
		ordering[i]->earlier = createLinkedList();
		ordering[i]->later = createLinkedList();
	}

	for (i = 0; i < size; i++)
	{
		degree[i] = length(list[i]);
		//printf("degree[%d] = %d\n", i, degree[i]);
		vertexLocator[i] = addFirst(verticesByDegree[degree[i]], ((void*) i));
	}

	int currentDegree = 0;

	int numVerticesRemoved = 0;

	while (numVerticesRemoved < size)
	{
		if (!isEmpty(verticesByDegree[currentDegree]))
		{
			degeneracy = max(degeneracy, currentDegree);

			void *pv = getFirst(verticesByDegree[currentDegree]);
			int vertex = *(int *) &pv;

			deleteNode(vertexLocator[vertex]);

			ordering[vertex]->vertex = vertex;
			ordering[vertex]->orderNumber = numVerticesRemoved;

			degree[vertex] = -1;

			LinkedList* neighborList = list[vertex];

			Link* neighborLink = neighborList->head->next;

			while (!isTail(neighborLink))
			{
				int neighbor = *(int *) &neighborLink->data;
				//printf("Neighbor: %d\n", neighbor);

				if (degree[neighbor] != -1)
				{
					deleteNode(vertexLocator[neighbor]);
					addLast(ordering[vertex]->later, (void*) neighbor);

					degree[neighbor]--;

					if (degree[neighbor] != -1)
					{
						vertexLocator[neighbor] = addFirst(
								verticesByDegree[degree[neighbor]],
								(void*) neighbor);
					}
				}
				else
				{
					addLast(ordering[vertex]->earlier, (void*) neighbor);
				}

				neighborLink = neighborLink->next;
			}

			numVerticesRemoved++;
			currentDegree = 0;
		}
		else
		{
			currentDegree++;
		}

	}

	for (i = 0; i < size; i++)
	{
		destroyLinkedList(verticesByDegree[i]);
	}

	free(vertexLocator);
	free(verticesByDegree);
	free(degree);

	return ordering;
}

NeighborListArray** Cliques::computeDegeneracyOrderArray(LinkedList** list,
		int size)
{

	NeighborList** ordering = (NeighborList**) calloc(size,
			sizeof(NeighborList*));

	int i = 0;

	int degeneracy = 0;

	// array of lists of vertices, indexed by degree
	LinkedList** verticesByDegree = (LinkedList**) calloc(size,
			sizeof(LinkedList*));

	// array of lists of vertices, indexed by degree
	Link** vertexLocator = (Link**) calloc(size, sizeof(Link*));

	int* degree = (int*) calloc(size, sizeof(int));

	for (i = 0; i < size; i++)
	{
		verticesByDegree[i] = createLinkedList();
		ordering[i] = (NeighborList *) malloc(sizeof(NeighborList));
		ordering[i]->earlier = createLinkedList();
		ordering[i]->later = createLinkedList();
	}

	// fill each cell of degree lookup table
	// then use that degree to populate the
	// lists of vertices indexed by degree

	for (i = 0; i < size; i++)
	{
		degree[i] = length(list[i]);
		vertexLocator[i] = addFirst(verticesByDegree[degree[i]], ((void*) i));
	}

	int currentDegree = 0;

	int numVerticesRemoved = 0;

	while (numVerticesRemoved < size)
	{
		if (!isEmpty(verticesByDegree[currentDegree]))
		{
			degeneracy = max(degeneracy, currentDegree);

			void *pv = getFirst(verticesByDegree[currentDegree]);
			int vertex = *(int *) &pv;

			deleteNode(vertexLocator[vertex]);

			ordering[vertex]->vertex = vertex;
			ordering[vertex]->orderNumber = numVerticesRemoved;

			degree[vertex] = -1;

			LinkedList* neighborList = list[vertex];

			Link* neighborLink = neighborList->head->next;

			while (!isTail(neighborLink))
			{
				int neighbor = *(int*) &neighborLink->data;

				if (degree[neighbor] != -1)
				{
					deleteNode(vertexLocator[neighbor]);
					addLast(ordering[vertex]->later, (void*) neighbor);

					degree[neighbor]--;

					if (degree[neighbor] != -1)
					{
						vertexLocator[neighbor] = addFirst(
								verticesByDegree[degree[neighbor]],
								(void*) neighbor);
					}
				}
				else
				{
					addLast(ordering[vertex]->earlier, (void*) neighbor);
				}

				neighborLink = neighborLink->next;
			}

			numVerticesRemoved++;
			currentDegree = 0;
		}
		else
		{
			currentDegree++;
		}

	}

	NeighborListArray** orderingArray = (NeighborListArray**) calloc(size,
			sizeof(NeighborListArray*));

	for (i = 0; i < size; i++)
	{
		orderingArray[i] = (NeighborListArray*) malloc(
				sizeof(NeighborListArray));
		orderingArray[i]->vertex = ordering[i]->vertex;
		orderingArray[i]->orderNumber = ordering[i]->orderNumber;

		orderingArray[i]->laterDegree = length(ordering[i]->later);
		orderingArray[i]->later = (int *) calloc(orderingArray[i]->laterDegree,
				sizeof(int));

		int j = 0;
		Link* curr = ordering[i]->later->head->next;
		while (!isTail(curr))
		{
			orderingArray[i]->later[j++] = *(int*) &(curr->data);
			curr = curr->next;
		}

		orderingArray[i]->earlierDegree = length(ordering[i]->earlier);
		orderingArray[i]->earlier = (int *) calloc(
				orderingArray[i]->earlierDegree, sizeof(int));

		j = 0;
		curr = ordering[i]->earlier->head->next;
		while (!isTail(curr))
		{
			orderingArray[i]->earlier[j++] = *(int*) &(curr->data);
			curr = curr->next;
		}
	}

	for (i = 0; i < size; i++)
	{
		destroyLinkedList(ordering[i]->earlier);
		destroyLinkedList(ordering[i]->later);
		free(ordering[i]);
		destroyLinkedList(verticesByDegree[i]);
	}

	free(ordering);

	free(vertexLocator);
	free(verticesByDegree);
	free(degree);

	return orderingArray;
}

inline int findBestPivotNonNeighborsDegeneracy(int** pivotNonNeighbors,
		int* numNonNeighbors, int* vertexSets, int* vertexLookup,
		int** neighborsInP, int* numNeighbors, int beginX, int beginP,
		int beginR)
{
	int pivot = -1;
	int maxIntersectionSize = -1;

	// iterate over each vertex in P union X
	// to find the vertex with the most neighbors in P.
	int j = beginX;
	while (j < beginR)
	{
		int vertex = vertexSets[j];
		int numPotentialNeighbors = min(beginR - beginP, numNeighbors[vertex]);

		int numNeighborsInP = 0;

		int k = 0;
		while (k < numPotentialNeighbors)
		{
			int neighbor = neighborsInP[vertex][k];
			int neighborLocation = vertexLookup[neighbor];

			if (neighborLocation >= beginP && neighborLocation < beginR)
			{
				numNeighborsInP++;
			}
			else
			{
				break;
			}

			k++;
		}

		if (numNeighborsInP > maxIntersectionSize)
		{
			pivot = vertex;
			maxIntersectionSize = numNeighborsInP;
		}

		j++;
	}

	// compute non neighbors of pivot by marking its neighbors
	// and moving non-marked vertices into pivotNonNeighbors.
	// we must do this because this is an efficient way
	// to compute non-neighbors of a vertex in
	// an adjacency list.

	// we initialize enough space for all of P; this is
	// slightly space inefficient, but it results in faster
	// computation of non-neighbors.
	*pivotNonNeighbors = (int *) calloc(beginR - beginP, sizeof(int));
	memcpy(*pivotNonNeighbors, &vertexSets[beginP],
			(beginR - beginP) * sizeof(int));

	// we will decrement numNonNeighbors as we find neighbors
	*numNonNeighbors = beginR - beginP;

	int numPivotNeighbors = min(beginR - beginP, numNeighbors[pivot]);

	// mark the neighbors of pivot that are in P.
	j = 0;
	while (j < numPivotNeighbors)
	{
		int neighbor = neighborsInP[pivot][j];
		int neighborLocation = vertexLookup[neighbor];

		if (neighborLocation >= beginP && neighborLocation < beginR)
		{
			(*pivotNonNeighbors)[neighborLocation - beginP] = -1;
		}
		else
		{
			break;
		}

		j++;
	}

	// move non-neighbors of pivot in P to the beginning of
	// pivotNonNeighbors and set numNonNeighbors appriopriately.

	// if a vertex is marked as a neighbor, the we move it
	// to the end of pivotNonNeighbors and decrement numNonNeighbors.
	j = 0;
	while (j < *numNonNeighbors)
	{
		int vertex = (*pivotNonNeighbors)[j];

		if (vertex == -1)
		{
			(*numNonNeighbors)--;
			(*pivotNonNeighbors)[j] = (*pivotNonNeighbors)[*numNonNeighbors];
			continue;
		}

		j++;
	}

	return pivot;
}

inline void fillInPandXForRecursiveCallDegeneracy(int vertex, int orderNumber,
		int* vertexSets, int* vertexLookup, NeighborListArray** orderingArray,
		int** neighborsInP, int* numNeighbors, int* pBeginX, int *pBeginP,
		int *pBeginR, int* pNewBeginX, int* pNewBeginP, int *pNewBeginR)
{
	int vertexLocation = vertexLookup[vertex];

	(*pBeginR)--;
	vertexSets[vertexLocation] = vertexSets[*pBeginR];
	vertexLookup[vertexSets[*pBeginR]] = vertexLocation;
	vertexSets[*pBeginR] = vertex;
	vertexLookup[vertex] = *pBeginR;

	*pNewBeginR = *pBeginR;
	*pNewBeginP = *pBeginR;

	// swap later neighbors of vertex into P section of vertexSets
	int j = 0;
	while (j < orderingArray[orderNumber]->laterDegree)
	{
		int neighbor = orderingArray[orderNumber]->later[j];
		int neighborLocation = vertexLookup[neighbor];

		(*pNewBeginP)--;

		vertexSets[neighborLocation] = vertexSets[*pNewBeginP];
		vertexLookup[vertexSets[*pNewBeginP]] = neighborLocation;
		vertexSets[*pNewBeginP] = neighbor;
		vertexLookup[neighbor] = *pNewBeginP;

		j++;
	}

	*pNewBeginX = *pNewBeginP;

	// swap earlier neighbors of vertex into X section of vertexSets
	j = 0;
	while (j < orderingArray[orderNumber]->earlierDegree)
	{
		int neighbor = orderingArray[orderNumber]->earlier[j];
		int neighborLocation = vertexLookup[neighbor];

		(*pNewBeginX)--;
		vertexSets[neighborLocation] = vertexSets[*pNewBeginX];
		vertexLookup[vertexSets[*pNewBeginX]] = neighborLocation;
		vertexSets[*pNewBeginX] = neighbor;
		vertexLookup[neighbor] = *pNewBeginX;

		free(neighborsInP[neighbor]);
		neighborsInP[neighbor] = (int *) calloc(
				min(*pNewBeginR - *pNewBeginP,
						orderingArray[neighbor]->laterDegree), sizeof(int));
		numNeighbors[neighbor] = 0;

		// fill in NeighborsInP
		int k = 0;
		while (k < orderingArray[neighbor]->laterDegree)
		{
			int laterNeighbor = orderingArray[neighbor]->later[k];
			int laterNeighborLocation = vertexLookup[laterNeighbor];
			if (laterNeighborLocation >= *pNewBeginP
					&& laterNeighborLocation < *pNewBeginR)
			{
				neighborsInP[neighbor][numNeighbors[neighbor]] = laterNeighbor;
				numNeighbors[neighbor]++;
			}

			k++;
		}

		j++;

	}

	// reset numNeighbors and neighborsInP for this vertex
	j = *pNewBeginP;
	while (j < *pNewBeginR)
	{
		int vertexInP = vertexSets[j];
		numNeighbors[vertexInP] = 0;
		free(neighborsInP[vertexInP]);
		neighborsInP[vertexInP] = (int *) calloc(
				min(*pNewBeginR - *pNewBeginP,
						orderingArray[vertexInP]->laterDegree
								+ orderingArray[vertexInP]->earlierDegree),
				sizeof(int));

		j++;
	}

	// count neighbors in P, and fill in array of neighbors
	// in P
	j = *pNewBeginP;
	while (j < *pNewBeginR)
	{
		int vertexInP = vertexSets[j];

		int k = 0;
		while (k < orderingArray[vertexInP]->laterDegree)
		{
			int laterNeighbor = orderingArray[vertexInP]->later[k];
			int laterNeighborLocation = vertexLookup[laterNeighbor];

			if (laterNeighborLocation >= *pNewBeginP
					&& laterNeighborLocation < *pNewBeginR)
			{
				neighborsInP[vertexInP][numNeighbors[vertexInP]] =
						laterNeighbor;
				numNeighbors[vertexInP]++;
				neighborsInP[laterNeighbor][numNeighbors[laterNeighbor]] =
						vertexInP;
				numNeighbors[laterNeighbor]++;
			}

			k++;
		}

		j++;
	}
}

long Cliques::listAllMaximalCliquesDegeneracy(LinkedList** adjList,
		CLIQUESET& cliques, int size)
{
	// vertex sets are stored in an array like this:
	// |--X--|--P--|
	int* vertexSets = (int *) calloc(size, sizeof(int));

	// vertex i is stored in vertexSets[vertexLookup[i]]
	int* vertexLookup = (int *) calloc(size, sizeof(int));

	int** neighborsInP = (int **) calloc(size, sizeof(int*));
	int* numNeighbors = (int *) calloc(size, sizeof(int));

	// compute the degeneracy order
	NeighborListArray** orderingArray = computeDegeneracyOrderArray(adjList,
			size);

	int i = 0;

	while (i < size)
	{
		vertexLookup[i] = i;
		vertexSets[i] = i;
		neighborsInP[i] = (int *) calloc(1, sizeof(int));
		numNeighbors[i] = 1;
		i++;
	}

	int beginX = 0;
	int beginP = 0;
	int beginR = size;

	long cliqueCount = 0;

	LinkedList* partialClique = createLinkedList();

	// for each vertex
	for (i = 0; i < size; i++)
	{
		int vertex = (int) orderingArray[i]->vertex;

		// add vertex to partial clique R
		addLast(partialClique, (void*) vertex);

		int newBeginX, newBeginP, newBeginR;

		// set P to be later neighbors and X to be be earlier neighbors
		// of vertex
		fillInPandXForRecursiveCallDegeneracy(i, vertex, vertexSets,
				vertexLookup, orderingArray, neighborsInP, numNeighbors,
				&beginX, &beginP, &beginR, &newBeginX, &newBeginP, &newBeginR);

		// recursively compute maximal cliques containing vertex, some of its
		// later neighbors, and avoiding earlier neighbors
		listAllMaximalCliquesDegeneracyRecursive(&cliqueCount, cliques,
				partialClique, vertexSets, vertexLookup, neighborsInP,
				numNeighbors, newBeginX, newBeginP, newBeginR);

		beginR = beginR + 1;

		deleteLast(partialClique);
	}

	destroyLinkedList(partialClique);

	free(vertexSets);
	free(vertexLookup);

	for (i = 0; i < size; i++)
	{
		free(neighborsInP[i]);
		free(orderingArray[i]->later);
		free(orderingArray[i]->earlier);
		free(orderingArray[i]);
	}

	free(orderingArray);
	free(neighborsInP);
	free(numNeighbors);

	return cliqueCount;
}

inline void moveToRDegeneracy(int vertex, int* vertexSets, int* vertexLookup,
		int** neighborsInP, int* numNeighbors, int* pBeginX, int *pBeginP,
		int *pBeginR, int* pNewBeginX, int* pNewBeginP, int *pNewBeginR)
{

	int vertexLocation = vertexLookup[vertex];

	(*pBeginR)--;
	vertexSets[vertexLocation] = vertexSets[*pBeginR];
	vertexLookup[vertexSets[*pBeginR]] = vertexLocation;
	vertexSets[*pBeginR] = vertex;
	vertexLookup[vertex] = *pBeginR;

	// this is not a typo, initially newX is empty
	*pNewBeginX = *pBeginP;
	*pNewBeginP = *pBeginP;
	*pNewBeginR = *pBeginP;

	int sizeOfP = *pBeginR - *pBeginP;

	int j = *pBeginX;
	while (j < *pNewBeginX)
	{
		int neighbor = vertexSets[j];
		int neighborLocation = j;

		int incrementJ = 1;

		int numPotentialNeighbors = min(sizeOfP, numNeighbors[neighbor]);

		int k = 0;
		while (k < numPotentialNeighbors)
		{
			if (neighborsInP[neighbor][k] == vertex)
			{
				(*pNewBeginX)--;
				vertexSets[neighborLocation] = vertexSets[(*pNewBeginX)];
				vertexLookup[vertexSets[(*pNewBeginX)]] = neighborLocation;
				vertexSets[(*pNewBeginX)] = neighbor;
				vertexLookup[neighbor] = (*pNewBeginX);
				incrementJ = 0;
			}

			k++;
		}

		if (incrementJ)
			j++;
	}

	j = (*pBeginP);
	while (j < (*pBeginR))
	{
		int neighbor = vertexSets[j];
		int neighborLocation = j;

		int numPotentialNeighbors = min(sizeOfP, numNeighbors[neighbor]);

		int k = 0;
		while (k < numPotentialNeighbors)
		{
			if (neighborsInP[neighbor][k] == vertex)
			{
				vertexSets[neighborLocation] = vertexSets[(*pNewBeginR)];
				vertexLookup[vertexSets[(*pNewBeginR)]] = neighborLocation;
				vertexSets[(*pNewBeginR)] = neighbor;
				vertexLookup[neighbor] = (*pNewBeginR);
				(*pNewBeginR)++;
			}

			k++;
		}

		j++;
	}

	j = (*pNewBeginX);

	while (j < *pNewBeginR)
	{
		int thisVertex = vertexSets[j];

		int numPotentialNeighbors = min(sizeOfP, numNeighbors[thisVertex]);

		int numNeighborsInP = 0;

		int k = 0;
		while (k < numPotentialNeighbors)
		{
			int neighbor = neighborsInP[thisVertex][k];
			int neighborLocation = vertexLookup[neighbor];
			if (neighborLocation >= *pNewBeginP
					&& neighborLocation < *pNewBeginR)
			{
				neighborsInP[thisVertex][k] =
						neighborsInP[thisVertex][numNeighborsInP];
				neighborsInP[thisVertex][numNeighborsInP] = neighbor;
				numNeighborsInP++;
			}
			k++;
		}

		j++;
	}
}

inline void moveFromRToXDegeneracy(int vertex, int* vertexSets,
		int* vertexLookup, int* pBeginX, int* pBeginP, int* pBeginR)
{
	int vertexLocation = vertexLookup[vertex];

	//swap vertex into X and increment beginP and beginR
	vertexSets[vertexLocation] = vertexSets[*pBeginP];
	vertexLookup[vertexSets[*pBeginP]] = vertexLocation;
	vertexSets[*pBeginP] = vertex;
	vertexLookup[vertex] = *pBeginP;

	*pBeginP = *pBeginP + 1;
	*pBeginR = *pBeginR + 1;
}

void Cliques::listAllMaximalCliquesDegeneracyRecursive(long* cliqueCount,
		CLIQUESET& cliques, LinkedList* partialClique, int* vertexSets,
		int* vertexLookup, int** neighborsInP, int* numNeighbors, int beginX,
		int beginP, int beginR)
{

	// if X is empty and P is empty, process partial clique as maximal
	if (beginX >= beginP && beginP >= beginR)
	{

		int l = length(partialClique);
		if (l < Overlap+1)
			return;
		(*cliqueCount)++;
		if (l > maximumSize)
			maximumSize = l;
		vector<int> c;
		c.resize(l);
		int i = 0;
		Link*curr = partialClique->head->next;
		while (!isTail(curr))
		{
			c[i] = *(int *) &curr->data;
			i++;
			curr = curr->next;
		}

		sort(c.begin(), c.end());
		cliques.emplace_back(c);

		return;
	}

	// avoid work if P is empty.
	if (beginP >= beginR)
		return;

	int* myCandidatesToIterateThrough;
	int numCandidatesToIterateThrough;

	// get the candidates to add to R to make a maximal clique
	findBestPivotNonNeighborsDegeneracy(&myCandidatesToIterateThrough,
			&numCandidatesToIterateThrough, vertexSets, vertexLookup,
			neighborsInP, numNeighbors, beginX, beginP, beginR);

	// add candiate vertices to the partial clique one at a time and
	// search for maximal cliques
	if (numCandidatesToIterateThrough != 0)
	{
		int iterator = 0;
		while (iterator < numCandidatesToIterateThrough)
		{
			// vertex to be added to the partial clique
			int vertex = myCandidatesToIterateThrough[iterator];

			int newBeginX, newBeginP, newBeginR;

			// add vertex into partialClique, representing R.
			Link* vertexLink = addLast(partialClique, (void*) vertex);

			// swap vertex into R and update all data structures
			moveToRDegeneracy(vertex, vertexSets, vertexLookup, neighborsInP,
					numNeighbors, &beginX, &beginP, &beginR, &newBeginX,
					&newBeginP, &newBeginR);

			// recursively compute maximal cliques with new sets R, P and X
			listAllMaximalCliquesDegeneracyRecursive(cliqueCount, cliques,
					partialClique, vertexSets, vertexLookup, neighborsInP,
					numNeighbors, newBeginX, newBeginP, newBeginR);

			// remove vertex from partialCliques
			deleteNode(vertexLink);

			moveFromRToXDegeneracy(vertex, vertexSets, vertexLookup, &beginX,
					&beginP, &beginR);

			iterator++;
		}

		// swap vertices that were moved to X back into P, for higher recursive calls.
		iterator = 0;
		while (iterator < numCandidatesToIterateThrough)
		{
			int vertex = myCandidatesToIterateThrough[iterator];
			int vertexLocation = vertexLookup[vertex];

			beginP--;
			vertexSets[vertexLocation] = vertexSets[beginP];
			vertexSets[beginP] = vertex;
			vertexLookup[vertex] = beginP;
			vertexLookup[vertexSets[vertexLocation]] = vertexLocation;

			iterator++;
		}
	}

	free(myCandidatesToIterateThrough);
}

