main: Image.cpp Image.h main.cpp macros.h myfilter.o graph.o maxflow.o
	/usr/local/cuda/bin/nvcc -O3 main.cpp Image.cpp myfilter.o graph.o maxflow.o -o main -lpng -ljpeg -L. -lcutil -I./Graphcut

myfilter.o: myfilter.cu graph.h
	/usr/local/cuda/bin/nvcc -arch=sm_50  -c myfilter.cu -o myfilter.o 

maxflow.o: maxflow.cpp graph.h block.h instances.inc
	g++ -c maxflow.cpp
	
graph.o: graph.cpp graph.h block.h instances.inc
	g++ -c graph.cpp

clean: 
	rm main *.o
