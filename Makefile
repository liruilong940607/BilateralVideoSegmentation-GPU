MATLAB_PATH = /usr/local/MATLAB/R2016a

######  lib error:
# $ export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/MATLAB/R2016a/bin/glnxa64/
######

INCS = -I$(MATLAB_PATH)/extern/include `pkg-config --cflags opencv`
LIBS = -L$(MATLAB_PATH)/bin/glnxa64/ -lmat -lmx -lmex -leng `pkg-config --libs opencv`
SRCS = main.cpp arrayptr2mat.cpp
OBJS = $(SRCS:.cpp=.o)

all: $(OBJS)
	g++ -o BilateralVideoSegmentation $(OBJS) $(LIBS)

%.o: %.cpp
	g++ -o $@ -c $< $(INCS)

clean:
	rm *.o **/*.o
