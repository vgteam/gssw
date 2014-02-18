/*	example.c
 *	This is a simple example to show you how to use the SSW C library.
 *	To run this example:
 *	1) gcc -Wall -lz ssw.c example.c
 *	2) ./a.out
 *	Created by Mengyao Zhao on 07/31/12.
 */

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include "ssw.h"


//	Align a pair of genome sequences.
int main (int argc, char * const argv[]) {
    // default parameters for genome sequence alignment
	int32_t match = 2, mismatch = 2, gap_open = 3, gap_extension = 1;
    // from Mengyao's example about the importance of using all three matrices in traceback.
    // int32_t l, m, k, match = 2, mismatch = 1, gap_open = 2, gap_extension = 1;

    char *ref_seq_1 = argv[1];
    char *ref_seq_2 = argv[2];
    char *ref_seq_3 = argv[3];
    char *ref_seq_4 = argv[4];
    char *read_seq = argv[5];

	/* This table is used to transform nucleotide letters into numbers. */
    int8_t* nt_table = create_nt_table();
    
	// initialize scoring matrix for genome sequences
	//  A  C  G  T	N (or other ambiguous code)
	//  2 -2 -2 -2 	0	A
	// -2  2 -2 -2 	0	C
	// -2 -2  2 -2 	0	G
	// -2 -2 -2  2 	0	T
	//	0  0  0  0  0	N (or other ambiguous code)
	int8_t* mat = create_score_matrix(match, mismatch);

    node* nodes[4];
    nodes[0] = (node*)node_create("A", 1, ref_seq_1, nt_table, mat);
    nodes[1] = (node*)node_create("B", 2, ref_seq_2, nt_table, mat);
    nodes[2] = (node*)node_create("C", 2, ref_seq_3, nt_table, mat);
    nodes[3] = (node*)node_create("D", 2, ref_seq_4, nt_table, mat);

    // makes a diamond
    nodes_add_edge(nodes[0], nodes[1]);
    //nodes_add_edge(nodes[1], nodes[3]);
    nodes_add_edge(nodes[0], nodes[2]);
    nodes_add_edge(nodes[1], nodes[2]);

    graph* graph = graph_create(4);
    memcpy((void*)graph->nodes, (void*)nodes, 4*sizeof(node*));
    graph->size = 4;
    //graph_add_node(graph, nodes[0]);
    //graph_add_node(graph, nodes[1]);

    graph_fill(graph, read_seq, nt_table, mat, gap_open, gap_extension, 15, 2);
    graph_print_score_matrices(graph, read_seq, strlen(read_seq));
    graph_cigar* gc = graph_trace_back (graph,
                                        read_seq,
                                        strlen(read_seq),
                                        match,
                                        mismatch,
                                        gap_open,
                                        gap_extension);

    print_graph_cigar(gc);
    graph_cigar_destroy(gc);
    graph_destroy(graph);
    node_destroy(nodes[0]);
    node_destroy(nodes[1]);
    node_destroy(nodes[2]);
    node_destroy(nodes[3]);

    free(nt_table);
	free(mat);

	return(0);
}
