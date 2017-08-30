INCS = `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`
SRCS = main.cpp 
OBJS = $(SRCS:.cpp=.o)

all: $(OBJS)
	g++ -o BilateralVideoSegmentation $(OBJS) $(LIBS)

%.o: %.cpp
	g++ -o $@ -c $< $(INCS)

clean:
	rm *.o **/*.o
