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
#include <string.h>
#include "gssw.h"

void remove_phred_offset(char* qual_str) {
    int len = strlen(qual_str);
    int i;
    for (i = 0; i < len; i++) {
        qual_str[i] -= '!';
    }
}

//	Align a pair of genome sequences.
int main (int argc, char * const argv[]) {
    // default parameters for genome sequence alignment
    int8_t match = 1, mismatch = 4, gap_open = 6, gap_extension = 1;
    // from Mengyao's example about the importance of using all three matrices in traceback.
    // int32_t l, m, k, match = 2, mismatch = 1, gap_open = 2, gap_extension = 1;
    
    
    char *ref_seq_1 = argv[1];
    char *ref_seq_2 = argv[2];
    char *ref_seq_3 = argv[3];
    char *ref_seq_4 = argv[4];
    char *read_seq = argv[5];
    char *read_qual = argv[6];
    
    remove_phred_offset(read_qual);

	/* This table is used to transform nucleotide letters into numbers. */
    int8_t* nt_table = gssw_create_nt_table();
    
	// initialize adjusted scoring matrices for genome sequences
    
    // with full base quality, the matrix will be proportional to
	//  A  C  G  T	N (or other ambiguous code)
	//  2 -2 -2 -2 	0	A
	// -2  2 -2 -2 	0	C
	// -2 -2  2 -2 	0	G
	// -2 -2 -2  2 	0	T
	//  0  0  0  0  0	N (or other ambiguous code)
    // scores shrink toward 0 with lower base quality
    
    int8_t max_score = 32; // scores can be scaled to larger values automatically for improved sensitivity
    uint8_t max_qual = 40; // maximum base quality value to compute adjusted scores for
    double gc_content = 0.6;
    double tol = 1e-12;
    int8_t* adj_mat = gssw_dna_scaled_adjusted_qual_matrix(max_score, max_qual, &gap_open,
                                                           &gap_extension, match, mismatch,
                                                           gc_content, tol);
    
    gssw_node* nodes[4];
    nodes[0] = (gssw_node*)gssw_node_create("A", 1, ref_seq_1, nt_table, NULL);
    nodes[1] = (gssw_node*)gssw_node_create("B", 2, ref_seq_2, nt_table, NULL);
    nodes[2] = (gssw_node*)gssw_node_create("C", 3, ref_seq_3, nt_table, NULL);
    nodes[3] = (gssw_node*)gssw_node_create("D", 4, ref_seq_4, nt_table, NULL);
    
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
    
    gssw_graph_fill_qual_adj(graph, read_seq, read_qual, nt_table, adj_mat, gap_open, gap_extension, 0, 0, 15, 2, true);
    gssw_graph_print_score_matrices(graph, read_seq, strlen(read_seq), stdout);
    gssw_graph_mapping* gm = gssw_graph_trace_back_qual_adj (graph,
                                                             read_seq,
                                                             read_qual,
                                                             strlen(read_seq),
                                                             nt_table,
                                                             adj_mat,
                                                             gap_open,
                                                             gap_extension,
                                                             0, 0);
    
    printf("Optimal local mapping:\n");
    gssw_print_graph_mapping(gm, stdout);
    gssw_graph_mapping_destroy(gm);
    
    gssw_graph_mapping* gmp = gssw_graph_trace_back_pinned_qual_adj (graph,
                                                                     read_seq,
                                                                     read_qual,
                                                                     strlen(read_seq),
                                                                     nt_table,
                                                                     adj_mat,
                                                                     gap_open,
                                                                     gap_extension,
                                                                     0, 0);
    
    printf("Optimal pinned mapping:\n");
    gssw_print_graph_mapping(gmp, stdout);
    gssw_graph_mapping_destroy(gmp);
    
    int num_alts = 30;
    gssw_graph_mapping** gmps = gssw_graph_trace_back_pinned_qual_adj_multi (graph,
                                                                             num_alts,
                                                                             1,
                                                                             read_seq,
                                                                             read_qual,
                                                                             strlen(read_seq),
                                                                             nt_table,
                                                                             adj_mat,
                                                                             gap_open,
                                                                             gap_extension,
                                                                             0, 0);
    
    printf("Best %d pinned mappings:\n", num_alts);
    int j;
    for (j = 0; j < num_alts; j++) {
        gssw_print_graph_mapping(gmps[j], stdout);
        gssw_graph_mapping_destroy(gmps[j]);
    }
    
    free(gmps);
    
    // note that nodes which are referred to in this graph are destroyed as well
    gssw_graph_destroy(graph);

    free(nt_table);
	free(adj_mat);

	return(0);
}
