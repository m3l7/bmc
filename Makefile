# CC=g++
CC=mpiCC
CFLAGS=-fopenmp -O3 -lm
DEPS = bmc.hpp hermite_rule.hpp
OBJ = bmc.o hermite_rule.o

%.o: %.cpp $(DEPS)
	$(CC) $(CFLAGS) -c -o $@ $<

bmc: bmc.o hermite_rule.o
	$(CC) $(CFLAGS) -o bmc $(OBJ)

clean:
	rm -f *.o