MATLAB_PATH = /usr/local/MATLAB/R2016a

######  lib error:
# $ export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/MATLAB/R2016a/bin/glnxa64/
######

INCS = -I$(MATLAB_PATH)/extern/include `pkg-config --cflags opencv` -I/usr/local/cuda/include
LIBS = -L$(MATLAB_PATH)/bin/glnxa64/ -lmat -lmx -lmex -leng `pkg-config --libs opencv`
SRCS = main.cpp arrayptr2mat.cpp
OBJS = $(SRCS:.cpp=.o)

all: main.cpp  arrayptr2mat.o  bilateralSpace.o
	/usr/local/cuda/bin/nvcc -O3 main.cpp -o BilateralVideoSegmentation arrayptr2mat.o bilateralSpace.o $(INCS) $(LIBS)

arrayptr2mat.o: arrayptr2mat.cpp
	g++ -o arrayptr2mat.o -c arrayptr2mat.cpp -I$(MATLAB_PATH)/extern/include

bilateralSpace.o: bilateralSpace.cu bilateralSpace.h
	/usr/local/cuda/bin/nvcc -arch=sm_50  -c bilateralSpace.cu -o bilateralSpace.o 

clean:
	rm *.o **/*.o
