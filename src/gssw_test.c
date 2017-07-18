/**
 * Unit tests for GSSW.
 * Not using any real unit testing framework to avoid dependencies.
 */
 
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include "gssw.h"

////////////////////////////////////////////////////////////////////////////////
// Test tools
////////////////////////////////////////////////////////////////////////////////

/**
 * Note that a test is starting. Give the name of the thing being tested.
 * Say something like "MyModule".
 */
void start_case(char* const name) {
    printf("%s...\n", name);
}

/**
 * Note that a tert is starting. Say the property being tested.
 * Say something like "should do the thing".
 */
void start_test(char* const property) {
    printf("\t%s.\n", property);
}

/**
 * Make sure the condition is true.
 */
void check_condition(int condition, char* const message) {
    if(condition) {
        // Test succeeded
        printf("\t\t[OK] ");
    } else {
        // Test failed
        printf("\t\t[FAIL] ");
    }
    
    printf("%s\n", message);
    
    if (!condition) {
        // Abort the tests at the first failure.
        exit(1);
    }
}

////////////////////////////////////////////////////////////////////////////////
// GSSW wrappers for easy alignment
////////////////////////////////////////////////////////////////////////////////

/**
 * Do a GSSW alignment between the two given strings. Returns a newly allocated
 * gssw_graph that the caller must destroy with gssw_graph_destroy.
 */
gssw_graph* align_strings(char* const ref, char* const read) {

    // default parameters for genome sequence alignment
    int8_t match = 1, mismatch = 4;
    uint8_t gap_open = 6, gap_extension = 1;

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

    gssw_node* node;
    node = gssw_node_create("Node", 1, ref, nt_table, mat);
    
    gssw_graph* graph = gssw_graph_create(1);
    gssw_graph_add_node(graph, node);
    
    gssw_graph_fill(graph, read, nt_table, mat, gap_open, gap_extension, 0, 0, 15, 2);

    // Free the translation table
    free(nt_table);
    
    return graph;
}

/**
 * Compare score matrices for two filled nodes. Return 1 if they are equal and 0
 * otherwise.
 */
int gssw_node_score_matrices_equal(gssw_node* n1, gssw_node* n2, int32_t readLen) {
    // Nodes must be the same length
    if (n1->len != n2->len) {
        return 0;
    }

    // Pull out the alignment objects
    gssw_align* a1 = n1->alignment;
    gssw_align* a2 = n2->alignment;
    
    if (gssw_is_byte(a1) != gssw_is_byte(a2)) {
        // Both alignments should be the same score size
        return 0;
    }
    
    if (gssw_is_byte(a1)) {
        // Compare byte scores
        // Grab all the matrices
        uint8_t* mH1 = a1->mH;
        uint8_t* mE1 = a1->mE;
        uint8_t* mF1 = a1->mF;
        
        uint8_t* mH2 = a2->mH;
        uint8_t* mE2 = a2->mE;
        uint8_t* mF2 = a2->mF;
        
        int32_t j;
        for (j = 0; j < readLen; ++j) {
            // For each row (read base)
            int32_t i;
            for (i = 0; i < n1->len; ++i) {
                // For each column (ref base)
                
                if (mH1[i * readLen + j] != mH2[i * readLen + j]) {
                    // Main (H) matrices don't match
                    return 0;
                }
                if (mE1[i * readLen + j] != mE2[i * readLen + j]) {
                    // Gap in read (E) matrices don't match
                    return 0;
                }
                if (mF1[i * readLen + j] != mF2[i * readLen + j]) {
                    // Gap in ref (F) matrices don't match
                    return 0;
                }
            }
        }
    } else {
        // Compare word scores
        // Grab all the matrices
        uint16_t* mH1 = a1->mH;
        uint16_t* mE1 = a1->mE;
        uint16_t* mF1 = a1->mF;
        
        uint16_t* mH2 = a2->mH;
        uint16_t* mE2 = a2->mE;
        uint16_t* mF2 = a2->mF;
        
        int32_t j;
        for (j = 0; j < readLen; ++j) {
            // For each row (read base)
            int32_t i;
            for (i = 0; i < n1->len; ++i) {
                // For each column (ref base)
                
                if (mH1[i * readLen + j] != mH2[i * readLen + j]) {
                    // Main (H) matrices don't match
                    return 0;
                }
                if (mE1[i * readLen + j] != mE2[i * readLen + j]) {
                    // Gap in read (E) matrices don't match
                    return 0;
                }
                if (mF1[i * readLen + j] != mF2[i * readLen + j]) {
                    // Gap in ref (F) matrices don't match
                    return 0;
                }
            }
        }
    }
    
    // If we can't find a mismatch, we say they match
    return 1;
    
}

/**
 * Compare the score matrices for two filled graphs. Return 1 if they are equal
 * and 0 otherwise.
 */
int gssw_graph_score_matrices_equal(gssw_graph* g1, gssw_graph* g2, int32_t readLen) {
    if (g1->size != g2->size) {
        // Different graph sizes/shapes
        return 0;
    }
    
    uint32_t i = 0;
    for (i = 0; i < g1->size; i++) {
        gssw_node* n1 = g1->nodes[i];
        gssw_node* n2 = g2->nodes[i];
        
        if (!gssw_node_score_matrices_equal(n1, n2, readLen)) {
            // We had a mismatch between these nodes
            return 0;
        }
    }
    
    // If we get here, everything matched.
    return 1;
}

/**
 * Test helper for aligning two strings and making sure they match.
 */
void check_alignments_match(char* const reference, char* const read) {
    // Do the alignment in SSE2 mode
    gssw_sse2_enable();
    gssw_graph* sse2_aligned = align_strings(reference, read);
    // And in software mode
    gssw_sse2_disable();
    gssw_graph* software_aligned = align_strings(reference, read);

    // Now make sure the matrices match
    check_condition(gssw_graph_score_matrices_equal(sse2_aligned, software_aligned, strlen(read)),
        "score matrices are identical");
        
    // Clean up graphs
    gssw_graph_destroy(sse2_aligned);
    gssw_graph_destroy(software_aligned);
}
 


////////////////////////////////////////////////////////////////////////////////
// Test cases
////////////////////////////////////////////////////////////////////////////////

/**
 * Test case to make sure that we get the same results from the hardware and
 * software fillers.
 */
void test_gssw_software_fill() {
    {start_case("The GSSW matrix filler");
        {start_test("should produce identical results for a small single-node graph");
            check_alignments_match("GATTACA", "GATTTACA");
        }
        
        {start_test("should produce identical results on short identical strings");
            char* const reference = "GATTACA";
            check_alignments_match(reference, reference);
        }
        
        {start_test("should produce identical results on large inserts");
            check_alignments_match("GATTACA", "GATTTTTTTTTTTTTTACA");
        }
        
        {start_test("should produce identical results on large deletions");
            check_alignments_match("GATTTTTTTTTTTTTTACA", "GATTACA");
        }
        
        {start_test("should produce identical results on empty strings");
            check_alignments_match("", "");
        }
        
        {start_test("should produce identical results on 15 As");
            check_alignments_match("AAAAAAAAAAAAAAA", "AAAAAAAAAAAAAAA");
        }
        
        {start_test("should produce identical results on 16 As");
            check_alignments_match("AAAAAAAAAAAAAAAA", "AAAAAAAAAAAAAAAA");
        }
        
        {start_test("should produce identical results on all suffixes of a longer string");
            char* const reference = "GTGTTCCAGTTCTTATCCTATATCGGAAGTTCAATTATACATCGCACCAGCATATTCATG";
            int32_t i;
            for (i = strlen(reference); i >= 0; i--) {
                printf("Checking: %s, %s\n", &reference[i], &reference[i]);
                check_alignments_match(&reference[i], &reference[i]);
            }
        }
        
        {start_test("should produce identical results for a larger single-node graph with more differences");
            char* const reference = "GTGTTCCAGTTCTTATCCTATATCGGAAGTTCAATTATACATCGCACCAGCATATTCATG";
            char* const read = "GTGTTCAAGTTCATCGGAAGTTCAATTCTACATCGCACCAGCATATAAGATAAATTTCTTG";
            check_alignments_match(reference, read);
        }
    }
        
    
    
}

int main (int argc, char * const argv[]) {
    
    // Run all the tests
    test_gssw_software_fill();

    // Success!
    return 0;
}
 
