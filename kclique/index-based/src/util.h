/*
 * utility.h
 *
 *  Created on: 7May,2017
 *      Author: longyuan
 */

#ifndef UTIL_H_
#define UTIL_H_
#include <vector>

inline int intersection(std::vector<int> &v1, std::vector<int> &v2)
{
	auto size1 = v1.size();
	auto size2 = v2.size();
	if(v1[0] > v2[size2-1] || v1[size1-1] < v2[0])
		return 0;
	else if( v1[0] == v2[size2-1] || v1[size1-1] == v2[0])
		return 1;
	else
	{
		int i1 = 0;
		int i2 = 0;
		int n = 0;
		while(i1 < size1 && i2 < size2)
		{
			if(v1[i1] < v2[i2]) ++i1;
			else if(v2[i2] < v1[i1] ) ++i2;
			else
			{
				++n;
				++i1;
				++i2;
			}
		}
		return n;
	}
}




#endif /* UTIL_H_ */
