/*
 * unionfind.h
 *
 *  Created on: 7May,2017
 *      Author: longyuan
 */

#ifndef UNIONFIND_H_
#define UNIONFIND_H_


// A good example for smart pointer

class UF
{
public:
	UF(int n);
	~UF();
	int find(int e);
	bool connected(int e1, int e2);
	void uni(int e1, int e2);
private:
	int num;
	int *parent;
	int *rank;
	int p1;
	int p2;
};

class UFOAT
{
public:
	UFOAT(int n);
	~UFOAT();
	int find(int e);
	void setParent(int parent, int child);

private:
	int num;
	int *parent;
};


#endif /* UNIONFIND_H_ */
