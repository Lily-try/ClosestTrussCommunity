#This program is the index-based approach for kclique community search
#The input files should be contain in a folder including:
#1.graph file: graph.txt
#2.querynode file: query1.txt |format each line: querynode |example: 201
#The query time of each query will also output in the console.

#To complie the project:
g++ -std=c++14 -O3 src/*.cpp -o kclique

#The program input includes one parameter: The data folder described above.
#The format of the graph.txt should be processed using the following command:
./run graph-cp ../../data/Astroph/

#The program should first build the index using the following command:
./kclique ../../data/Astroph/ t

#The query step can then be run with three parameters: 
./kclique ../../data/Astroph/ i
