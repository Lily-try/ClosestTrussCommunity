/*
 * cpm.h
 *
 *  Created on: 8May,2017
 *      Author: longyuan
 */

#ifndef CPM_H_
#define CPM_H_
#include <string>
#include <vector>
#include <set>

#define QUERYSIZE 5
#define QUERYNUM 20

#define QUERYSIZE2 1
#define QUERYNUM2 500 //查询集的大小

using namespace std;

class CPM
{
public:
	static string output_name;
public:
	void run(std::string dir);
	void runOAT(std::string dir);
	void runBaseline(std::string dir);
	void runBaseline2(std::string dir);
	void runOATIndex(std::string dir);

	void generateQuery(std::string dir);
	void loadQuery(std::string dir);
	void printQueryNodeSet();


	void buildOAT(std::string dir);
	void loadOAT(std::string dir);

private:
	std::vector<std::vector<std::set<int> > > queryNodeSet;
};




#endif /* CPM_H_ */
