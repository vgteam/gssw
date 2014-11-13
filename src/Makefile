CC=			gcc
CFLAGS=-Wall -O3 -msse2 # -g #-pg 
LOBJS=		gssw.o	
PROG=		gssw_example
all:$(PROG)

.PHONY:all clean cleanlocal
#ssw_test:$(LOBJS) main.c 
#		$(CC) $(CFLAGS) main.c -o $@ $(LOBJS) -lm -lz
gssw_example:$(LOBJS) example.c
	$(CC) $(CFLAGS) example.c -o $@ $(LOBJS) -lm -lz
gssw.o:gssw.h
libgssw.a:gssw.o
	ar rvs libgssw.a gssw.o
cleanlocal:
		rm -fr *.o $(PROG) *~ libgssw.a

clean:cleanlocal


