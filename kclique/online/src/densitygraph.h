#include <vector>
#include <utility>
#include <string>
#include "config.h"



struct wEdge
{
	int a;
	int b;
	int weight;
	wEdge(int a_, int b_, int weight_):a(a_), b(b_), weight(weight_){}

	bool operator<(const wEdge &b) const
	{
		return weight < b.weight;

	}
};



struct Edge
{
	int a;
	int b;
	Edge(int a_, int b_):a(a_), b(b_){}
};



class DensityGraph
{
public:
	DensityGraph(size_t ms, CLIQUESET &c, MSTEDGE &m, NODE2CLIQUE &n):maxCliqueSize(ms),cliques(c),mstEdges(m), graphEdges(GRAPHEDGE()), nq(n){}
	void buildGraph(std::string dir);
	void maxSpanningTree();

	static void getMST(std::string dir, size_t ms, CLIQUESET &c, MSTEDGE &m, NODE2CLIQUE &n);

private:
	size_t maxCliqueSize;
	CLIQUESET &cliques;
	MSTEDGE &mstEdges;
	GRAPHEDGE graphEdges;
	NODE2CLIQUE &nq;
};



