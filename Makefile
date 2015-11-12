CC:=gcc
CFLAGS:=-Wall -O3 -msse4
OBJ_DIR:=obj
BIN_DIR:=bin
SRC_DIR:=src
LIB_DIR:=lib

OBJ=gssw.o
EXE=gssw_example

.PHONY:all clean cleanlocal pre

all:$(BIN_DIR)/$(EXE) $(LIB_DIR)/libgssw.a

pre:
	mkdir -p bin
	mkdir -p obj
	mkdir -p lib

$(BIN_DIR)/$(EXE):$(OBJ_DIR)/$(OBJ) $(SRC_DIR)/example.c pre
	$(CC) $(CFLAGS) $(SRC_DIR)/example.c -o $@ $< -lm -lz

$(OBJ_DIR)/$(OBJ):$(SRC_DIR)/gssw.h pre
	$(CC) $(CFLAGS) -c -o $@ $(SRC_DIR)/gssw.c

$(LIB_DIR)/libgssw.a:$(OBJ_DIR)/$(OBJ) pre
	ar rvs $@ $<

cleanlocal:
	$(RM) -r lib/
	$(RM) -r bin/
	$(RM) -r obj/

clean:cleanlocal


