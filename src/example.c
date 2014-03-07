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
#include "gssw.h"


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
    int8_t* nt_table = gssw_create_nt_table();
    
	// initialize scoring matrix for genome sequences
	//  A  C  G  T	N (or other ambiguous code)
	//  2 -2 -2 -2 	0	A
	// -2  2 -2 -2 	0	C
	// -2 -2  2 -2 	0	G
	// -2 -2 -2  2 	0	T
	//	0  0  0  0  0	N (or other ambiguous code)
	int8_t* mat = gssw_create_score_matrix(match, mismatch);

    gssw_node* nodes[4];
    nodes[0] = (gssw_node*)gssw_node_create("A", 1, ref_seq_1, nt_table, mat);
    nodes[1] = (gssw_node*)gssw_node_create("B", 2, ref_seq_2, nt_table, mat);
    nodes[2] = (gssw_node*)gssw_node_create("C", 3, ref_seq_3, nt_table, mat);
    nodes[3] = (gssw_node*)gssw_node_create("D", 4, ref_seq_4, nt_table, mat);

    // makes a diamond
    gssw_nodes_add_edge(nodes[0], nodes[1]);
    gssw_nodes_add_edge(nodes[0], nodes[2]);
    gssw_nodes_add_edge(nodes[1], nodes[3]);
    gssw_nodes_add_edge(nodes[2], nodes[3]);


    gssw_graph* graph = gssw_graph_create(4);
    //memcpy((void*)graph->nodes, (void*)nodes, 4*sizeof(gssw_node*));
    //graph->size = 4;
    gssw_graph_add_node(graph, nodes[0]);
    gssw_graph_add_node(graph, nodes[1]);
    gssw_graph_add_node(graph, nodes[2]);
    gssw_graph_add_node(graph, nodes[3]);

    gssw_graph_fill(graph, read_seq, nt_table, mat, gap_open, gap_extension, 15, 2);
    gssw_graph_print_score_matrices(graph, read_seq, strlen(read_seq));
    gssw_graph_mapping* gm = gssw_graph_trace_back (graph,
                                                    read_seq,
                                                    strlen(read_seq),
                                                    match,
                                                    mismatch,
                                                    gap_open,
                                                    gap_extension);

    gssw_print_graph_mapping(gm);
    gssw_graph_mapping_destroy(gm);
    // note that nodes which are referred to in this graph are destroyed as well
    gssw_graph_destroy(graph);

    free(nt_table);
	free(mat);

	return(0);
}
