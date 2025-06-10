/*
 * config.h
 *
 *  Created on: 8May,2017
 *      Author: longyuan
 */

#ifndef CONFIG_H_
#define CONFIG_H_

#include <vector>
#include <set>
#include <utility>
#include <sys/time.h>

struct Edge;

typedef std::vector<std::vector<int> > CLIQUESET;
typedef std::vector<std::vector<std::pair<int, int> > > MSTEDGE;
typedef std::vector<std::vector<Edge> > GRAPHEDGE;
typedef std::vector<std::vector<int> > QUERYNODESET;
typedef std::vector<std::set<int> > QUERYNODESETOAT;
typedef std::vector<std::vector<int> > NODE2CLIQUE;
typedef std::vector<std::pair<std::pair<int, int>, int> > WEDGES;

//extern int Overlap;
//extern int CliqueSize = Overlap+1;



#endif /* CONFIG_H_ */
