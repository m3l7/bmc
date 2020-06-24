# CC=g++
CC=mpiCC
CFLAGS=-fopenmp -O3 -lm
DEPS = bmc.hpp

%.o: %.cpp $(DEPS)
	$(CC) $(CFLAGS) -c -o $@ $<

bmc: bmc.o
	$(CC) $(CFLAGS) -o bmc bmc.o
# bmc: bmc.o
# 	$(CC) -lm -o bmc bmc.o
# bmc: bmc.o hermite_rule.o
# 	$(CC) -o bmc bmc.o hermite_rule.o

clean:
	rm -f *.o