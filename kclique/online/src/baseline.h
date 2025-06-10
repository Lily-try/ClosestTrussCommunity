/*
 * baseline.h
 *
 *  Created on: 21Jun.,2017
 *      Author: longyuan
 */

#ifndef BASELINE_H_
#define BASELINE_H_

#include <set>
#include <string>
#include <vector>
#include "config.h"

void baseline(std::set<int> &query, std::string dir);
void naiveOATTree(MSTEDGE &edges, int n);

void baseline2(std::vector<std::vector<std::set<int> > > &queryNodeSet, std::string dir);


#endif /* BASELINE_H_ */
