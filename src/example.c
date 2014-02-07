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


	// reference sequence
    /*
	char ref_seq[40] = {'C', 'A', 'G', 'C', 'C', 'T', 'T', 'T', 'C', 'T', 'G', 'A', 'C', 'C', 'C', 'G', 'G', 'A', 'A', 'A', 'T',
						'C', 'A', 'A', 'A', 'A', 'T', 'A', 'G', 'G', 'C', 'A', 'C', 'A', 'A', 'C', 'A', 'A', 'A', '\0'};
	char read_seq[16] = {'C', 'T', 'G', 'A', 'G', 'C', 'C', 'G', 'G', 'T', 'A', 'A', 'A', 'T', 'C', '\0'};	// read sequence
    */

    char *ref_seq_1 = argv[1];
    char *ref_seq_2 = argv[2];
    char *read_seq = argv[3];

	s_profile* profile;

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

	int8_t* num = create_num(read_seq, strlen(read_seq), nt_table);
	int8_t* ref_num_1 = create_num(ref_seq_1, strlen(ref_seq_1), nt_table);
	int8_t* ref_num_2 = create_num(ref_seq_2, strlen(ref_seq_2), nt_table);

	profile = ssw_init(num, strlen(read_seq), mat, 5, 2);

	// Only the 8 bit of the flag is setted. ssw_align will always return the best alignment beginning position and cigar.
	//result = ssw_align (profile, ref_num, strlen(ref_seq), gap_open, gap_extension, 1, 0, 0, 15);
	//ssw_write(result, ref_seq, read_seq, nt_table);

	s_align* result1 = ssw_fill (profile, ref_num_1, strlen(ref_seq_1), gap_open, gap_extension, 1, 0, 0, 15, 0, NULL);
    print_score_matrix(ref_seq_1, strlen(ref_seq_1), read_seq, strlen(read_seq), result1);
    cigar* path = trace_back (result1, result1->ref_end1, result1->read_end1, ref_seq_1, strlen(ref_seq_1), read_seq, strlen(read_seq), match, mismatch, gap_open, gap_extension);
    print_cigar(path); printf("\n");
    cigar_destroy(path);

	s_align* result2 = ssw_fill (profile, ref_num_2, strlen(ref_seq_2), gap_open, gap_extension, 1, 0, 0, 15, 1, result1);
    print_score_matrix(ref_seq_2, strlen(ref_seq_2), read_seq, strlen(read_seq), result2);
    path = trace_back (result2, result2->ref_end1, result2->read_end1, ref_seq_2, strlen(ref_seq_2), read_seq, strlen(read_seq), match, mismatch, gap_open, gap_extension);
    print_cigar(path); printf("\n");
    cigar_destroy(path);

    align_destroy(result1);
    align_destroy(result2);
    init_destroy(profile);

    free(nt_table);
	free(mat);
	free(ref_num_1);
	free(ref_num_2);
	free(num);
	return(0);
}
