CC:=gcc
CFLAGS:=-Wall -O3 -msse4 -g
OBJ_DIR:=obj
BIN_DIR:=bin
SRC_DIR:=src
LIB_DIR:=lib

OBJ=gssw.o
EXE=gssw_example
EXEADJ=gssw_example_adj
EXETEST=gssw_test

.PHONY:all clean cleanlocal

all:$(BIN_DIR)/$(EXE) $(BIN_DIR)/$(EXEADJ) $(BIN_DIR)/$(EXETEST) $(LIB_DIR)/libgssw.a

$(BIN_DIR)/$(EXE):$(OBJ_DIR)/$(OBJ) $(SRC_DIR)/example.c
	# Make dest directory
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $(SRC_DIR)/example.c -o $@ $< -lm -lz

$(BIN_DIR)/$(EXEADJ):$(OBJ_DIR)/$(OBJ) $(SRC_DIR)/example_adj.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $(SRC_DIR)/example_adj.c -o $@ $< -lm -lz

$(BIN_DIR)/$(EXETEST):$(OBJ_DIR)/$(OBJ) $(SRC_DIR)/gssw_test.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $(SRC_DIR)/gssw_test.c -o $@ $< -lm -lz

$(OBJ_DIR)/$(OBJ):$(SRC_DIR)/gssw.h
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -c -o $@ $(SRC_DIR)/gssw.c

$(LIB_DIR)/libgssw.a:$(OBJ_DIR)/$(OBJ)
	@mkdir -p $(@D)
	ar rvs $@ $<

cleanlocal:
	$(RM) -r lib/
	$(RM) -r bin/
	$(RM) -r obj/

clean:cleanlocal




