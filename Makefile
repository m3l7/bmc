# CC=g++
CC=mpiCC
CFLAGS=-fopenmp -O3 -lm
DEPS = bmc.hpp hermite_rule.hpp
OBJ = bmc.o hermite_rule.o

ifeq ($(MPI),1)
	DEFINES+=-DENABLEMPI
	CC=mpiCC
else
	CC=g++
endif

%.o: %.cpp $(DEPS)
	$(CC) $(CFLAGS) $(DEFINES) -c -o $@ $<

bmc: bmc.o hermite_rule.o
	$(info Using compiler ${CC})
	$(CC) $(CFLAGS) -o bmc $(OBJ)
bmc5d: bmc5d.o hermite_rule.o
	$(info Using compiler ${CC})
	$(CC) $(CFLAGS) -o bmc5d bmc5d.o hermite_rule.o

clean:
	rm -f *.o