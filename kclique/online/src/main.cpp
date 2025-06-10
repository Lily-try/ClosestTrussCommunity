/*
 * main.cpp
 *
 *  Created on: 7May,2017
 *      Author: longyuan
 */

#include "cpm.h"
#include <string>
#include <iostream>
#include <cstring>
#include "config.h"

using namespace std;

int Overlap = 0;

void build_oat(string dir) {
	CPM cpm;
	cpm.buildOAT(dir);
	cpm.loadOAT(dir);
}

void run_baseline(string dir) {
	CPM cpm;
	cpm.runBaseline(dir);
}

void run_oatindex(string dir) {
	CPM cpm;
	cpm.runOATIndex(dir);
}

void run_oat(string dir) {
	CPM cpm;
	cpm.runOAT(dir);
}

int main(int argc, char **argv)
{
	if( argc == 1 ) {
		string dir = "/Users/lqin/Documents/workspace/data/Twitter/";
		//string dir = "/Users/lqin/Documents/workspace/data/test_cp/";
		//build_oat(dir);
        cout<<'in argc==1'<<endl;
		run_baseline(dir);
		//run_oat(dir);
		return 0;
	}
	if (!(argc == 3 ||argc == 4))
	{
		cout << "cpm dir  o(opt) or i(index-based) or b(baseline) or g(generate query) or t(building tree) or b2(baseline2)"
				<< endl;
		return -1;

	}
	CPM cpm;
	if (strcmp(argv[2], "o") == 0)
	{
		cpm.runOAT(std::string(argv[1]));
	}
	else if(strcmp(argv[2], "i") == 0)
	{
		cpm.runOATIndex(std::string(argv[1]));
	}
	else if (strcmp(argv[2], "b") == 0)
	{
		cout<<"in argv[2],b"<<endl;
		cout<<"argv[1]"<<std::string(argv[1])<<endl;
		cpm.runBaseline(std::string(argv[1]));
	}
	else if (strcmp(argv[2], "g") == 0)
	{
		cpm.generateQuery(std::string(argv[1]));
		cpm.loadQuery(string(argv[1]));
		cpm.printQueryNodeSet();

	}
	else if (strcmp(argv[2], "t") == 0)
	{
		//Overlap = atoi(argv[3]);
		cpm.buildOAT(string(argv[1]));
		cpm.loadOAT(string(argv[1]));

	}
	else if(strcmp(argv[2], "b2") == 0)
	{
		cpm.runBaseline2(std::string(argv[1]));
	}
	else
	{
		cout << "cannot recognize the option" << endl;
		return -1;
	}
}
