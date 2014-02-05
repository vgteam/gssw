CC=			gcc
CFLAGS=-Wall -O3 -msse2 -g #-pg 
LOBJS=		ssw.o	
PROG=		ssw_example
all:$(PROG)

.PHONY:all clean cleanlocal
#ssw_test:$(LOBJS) main.c 
#		$(CC) $(CFLAGS) main.c -o $@ $(LOBJS) -lm -lz
ssw_example:$(LOBJS) example.c
	$(CC) $(CFLAGS) example.c -o $@ $(LOBJS) -lm -lz
ssw.o:ssw.h
cleanlocal:
		rm -fr *.o $(PROG) *~ 

clean:cleanlocal


