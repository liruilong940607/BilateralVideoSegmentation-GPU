all: main

main: main.o graph.o maxflow.o
	g++ -o main main.o graph.o maxflow.o
main.o: main.cpp graph.h
	g++ -c main.cpp
maxflow.o: maxflow.cpp graph.h block.h instances.inc
	g++ -c maxflow.cpp
graph.o: graph.cpp graph.h block.h instances.inc
	g++ -c graph.cpp

	
clean: 
	rm *.o main