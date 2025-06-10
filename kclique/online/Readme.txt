online kclique
****** 1.准备项目需要的数据文件****************
#1.graph file: graph.txt
#2.querynode file: query1.txt |format each line: querynode |example: 201
需要注意攻击后的图和query要匹配，不然query中会包含不在graph中的点导致报错。
输出数据：查询节点q.txt，里面是查询节点所在的社区的节点列表。控制台打印每个查询的时间
#The program will output the query results of each querynode to a file with the same order as the querynode id (i.e. 1.txt), and the output is the node list of the community
#The query time of each query will also output in the console.

*************2.将项目编译***************
g++ -std=c++14 -O3 src/*.cpp -o kclique

***********3.运行项目*****************

需要1个输入参数：即上述的数据文件夹，graph.txt文件的格式需要通过如下命令进行预处理：
./run graph-cp ../../data/Astroph/
/lib64/ld-linux-x86-64.so.2 ./run graph-cp ../../data/Astroph/
/lib64/ld-linux-x86-64.so.2 ./run graph-cp ../../data/cora/
/lib64/ld-linux-x86-64.so.2 ./run graph-cp ../../data/citeseer/
3个参数控制项目运行
#The program can then be run with three parameters: 1.search method, 2.data folder, 3. output file prefix
./kclique ../../data/Astroph/ b
./kclique ../../data/cora/ b
./kclique ../../data/citeseer/ b
4个参数进一步将社区节点列表转换成边列表
#To further process the community nodelist to edgelist with 4 parameters: 1.gen-graph method name, 2. data folder, 3.community nodelist file prefix, 4.number of querynodes:
./run gen-graph ../../dataAstroph/ output_clique_ 5 
