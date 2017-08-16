/**
 * Unit tests for GSSW.
 * Not using any real unit testing framework to avoid dependencies.
 */
 
#include <assert.h>
#include <stdarg.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include "gssw.h"

////////////////////////////////////////////////////////////////////////////////
// Test tools (basically all printf-like)
////////////////////////////////////////////////////////////////////////////////

// Did any tests fail? If so, how many?
int tests_failed = 0;
// How many tests have run?
int tests_run = 0;

/**
 * Print the result report an dprovide the return code for main.
 */
int test_results() {
    if (tests_failed) {
        printf("!!!FAILURE!!!\n");
    } else {
        printf("+++SUCCESS+++\n");
    }
    printf("Ran %d tests with %d failures.\n", tests_run, tests_failed);

    if (tests_failed) {
        return 1;
    }
    return 0;
}

// Has the current test failed?
int test_current_failed = 0;

/**
 * Note that a test is starting. Give the name of the thing being tested.
 * Say something like "MyModule".
 */
void start_case(char* const fmt, ...) {
    if (tests_failed) {
        // Stop!
        exit(1);
    }
    // Print name, forwarding args
    va_list argp;
	va_start(argp, fmt);
	vprintf(fmt, argp);
	printf("...\n");
    va_end(argp);
}

/**
 * Note that a tert is starting. Say the property being tested.
 * Say something like "should do the thing".
 */
void start_test(char* const fmt, ...) {
    if (tests_failed) {
        // Stop!
        exit(1);
    }
    // Print property, forwarding args
    va_list argp;
	va_start(argp, fmt);
	printf("\t");
	vprintf(fmt, argp);
	printf(".\n");
    va_end(argp);
    
    // Say we're runn ing a test
    tests_run++;
    
    // Reset the test failure flag.
    test_current_failed = 0;
}

/**
 * Fail a test with the given message.
 */
void vcheck_fail(char* const fmt, va_list argp) {
    // Report the message
    printf("\t\t[FAIL] ");
    vprintf(fmt, argp);
    printf("\n");
    
    if (!test_current_failed) {
        // This made the test fail
        test_current_failed = 1;
        tests_failed++;
    }
}

/**
 * Fail a test with the given message.
 */
void check_fail(char* const fmt, ...) {
    // Delegate to the version that takes forwarded arguments
    va_list argp;
	va_start(argp, fmt);
    vcheck_fail(fmt, argp);
    va_end(argp);
}

/**
 * Make sure the condition is true.
 * Returns whether it is true.
 */
int check_condition(int condition, char* const fmt, ...) {
    
    va_list argp;
    va_start(argp, fmt);
    if (!condition) {
        // Report failure
        vcheck_fail(fmt, argp);
    }
    va_end(argp);
    return condition;
}

/**
 * Make sure the two integers are equal.
 * Returns whether they are equal.
 */
int check_equal(int a, int b, char* const fmt, ...) {
    va_list argp;
    va_start(argp, fmt);
    int condition = (a == b);
    if (!condition) {
        // Report failure
        vcheck_fail(fmt, argp);
        
        // Report difference
        printf("\t\t\t%d != %d\n", a, b);
    }
    va_end(argp);
    return condition;
}

////////////////////////////////////////////////////////////////////////////////
// GSSW wrappers for easy alignment
////////////////////////////////////////////////////////////////////////////////

/**
 * Do a GSSW alignment between the two given strings. Returns a newly allocated
 * gssw_graph that the caller must destroy with gssw_graph_destroy.
 */
gssw_graph* align_strings(char* const ref, char* const read, uint8_t scoreSize) {

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
    
    gssw_graph_fill(graph, read, nt_table, mat, gap_open, gap_extension, 0, 0, 15, scoreSize, true);

    // Free the translation table
    free(nt_table);
    
    return graph;
}

/**
 * Do a GSSW alignment between the given diamond graph and the given read.
 * Returns a newly allocated gssw_graph that the caller must destroy with
 * gssw_graph_destroy.
 */
gssw_graph* align_diamond(char* const start, char* const alt1, char* const alt2, char* const end, char* const read,
    uint8_t scoreSize) {

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

    gssw_node* nodes[4];
    nodes[0] = (gssw_node*)gssw_node_create("A", 1, start, nt_table, mat);
    nodes[1] = (gssw_node*)gssw_node_create("B", 2, alt1, nt_table, mat);
    nodes[2] = (gssw_node*)gssw_node_create("C", 3, alt2, nt_table, mat);
    nodes[3] = (gssw_node*)gssw_node_create("D", 4, end, nt_table, mat);
    
    // makes a diamond
    gssw_nodes_add_edge(nodes[0], nodes[1]);
    gssw_nodes_add_edge(nodes[0], nodes[2]);
    gssw_nodes_add_edge(nodes[1], nodes[3]);
    gssw_nodes_add_edge(nodes[2], nodes[3]);
    
    gssw_graph* graph = gssw_graph_create(4);
    gssw_graph_add_node(graph, nodes[0]);
    gssw_graph_add_node(graph, nodes[1]);
    gssw_graph_add_node(graph, nodes[2]);
    gssw_graph_add_node(graph, nodes[3]);
    
    gssw_graph_fill(graph, read, nt_table, mat, gap_open, gap_extension, 0, 0, 15, scoreSize, true);

    // Free the translation table
    free(nt_table);
    
    return graph;
}

/**
 * Compare score matrices for two filled nodes. Return 1 if they are equal and 0
 * otherwise.
 */
int check_gssw_node_score_matrices_equal(gssw_node* n1, gssw_node* n2, int32_t readLen) {
    // Nodes must be the same length
    if (!check_equal(n1->len, n2->len, "node lengths equal")) {
        return 0;
    }

    // Pull out the alignment objects
    gssw_align* a1 = n1->alignment;
    gssw_align* a2 = n2->alignment;
    
    if (!check_equal(gssw_is_byte(a1), gssw_is_byte(a2), "matrix types equal")) {
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
                
                if (!check_equal(mE1[i * readLen + j], mE2[i * readLen + j],
                    "gap in read E entries (%d,%d) match", i, j)) {
                    
                    // Gap in read (E) matrices don't match
                    return 0;
                }
                if (!check_equal(mF1[i * readLen + j], mF2[i * readLen + j],
                    "gap in ref F entries (%d,%d) match", i, j)) {
                    
                    // Gap in ref (F) matrices don't match
                    return 0;
                }
                if (!check_equal(mH1[i * readLen + j], mH2[i * readLen + j],
                    "best overall H entries (%d,%d) match", i, j)) {\
                    
                    // Main (H) matrices don't match
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
                
                if (!check_equal(mE1[i * readLen + j], mE2[i * readLen + j],
                    "gap in read E entries (%d,%d) match", i, j)) {
                    
                    // Gap in read (E) matrices don't match
                    return 0;
                }
                if (!check_equal(mF1[i * readLen + j], mF2[i * readLen + j],
                    "gap in ref F entries (%d,%d) match", i, j)) {
                    
                    // Gap in ref (F) matrices don't match
                    return 0;
                }
                if (!check_equal(mH1[i * readLen + j], mH2[i * readLen + j],
                    "best overall H entries (%d,%d) match", i, j)) {\
                    
                    // Main (H) matrices don't match
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
int check_gssw_graph_score_matrices_equal(gssw_graph* g1, gssw_graph* g2, int32_t readLen) {
    if (!check_equal(g1->size, g2->size, "graph sizes match")) {
        // Different graph sizes/shapes
        return 0;
    }
    
    uint32_t i = 0;
    for (i = 0; i < g1->size; i++) {
        gssw_node* n1 = g1->nodes[i];
        gssw_node* n2 = g2->nodes[i];
        
        if (!check_gssw_node_score_matrices_equal(n1, n2, readLen)) {
            // We had a mismatch between these nodes
            check_fail("matrices on node %d match", i+1);
            return 0;
        }
    }
    
    // If we get here, everything matched.
    return 1;
}

/**
 * Test helper for aligning two strings and making sure it's the same in SSE2 and software modes.
 */
void check_string_alignments_match(char* const reference, char* const read) {
    uint8_t scoreSize;
    for (scoreSize = 2; scoreSize > 0; scoreSize--) {
        // For score sizes 2 (byte with word fallback) and 1 (word only)

        // Do the alignment in SSE2 mode
        gssw_sse2_enable();
        gssw_graph* sse2_aligned = align_strings(reference, read, scoreSize);
        // And in software mode
        gssw_sse2_disable();
        gssw_graph* software_aligned = align_strings(reference, read, scoreSize);

        // Do they match?
        int match = check_gssw_graph_score_matrices_equal(sse2_aligned, software_aligned, strlen(read));
        
        // Now make sure the matrices match (and fail if they didn't)
        check_condition(match, "score matrices for size mode %hhd are identical", scoreSize);
        
        if (!match) {
            // No match, so dump matrices
            printf("SSE2 Filler:\n");
            gssw_graph_print_score_matrices(sse2_aligned, read, strlen(read), stdout);
            printf("Software Filler:\n");
            gssw_graph_print_score_matrices(software_aligned, read, strlen(read), stdout);
        }

        // Clean up graphs
        gssw_graph_destroy(sse2_aligned);
        gssw_graph_destroy(software_aligned);
    }
}

/**
 * Test helper for aligning a string to a small graph and making sure it's the
 * same in SSE2 and software modes across all score sizes.
 */
void check_diamond_alignments_match(char* const start, char* const alt1, char* const alt2, char* const end, char* const read) {
    uint8_t scoreSize;
    for (scoreSize = 2; scoreSize > 0; scoreSize--) {
        // For score sizes 2 (byte with word fallback) and 1 (word only)
    
        // Do the alignment in SSE2 mode
        gssw_sse2_enable();
        gssw_graph* sse2_aligned = align_diamond(start, alt1, alt2, end, read, scoreSize);
        // And in software mode
        gssw_sse2_disable();
        gssw_graph* software_aligned = align_diamond(start, alt1, alt2, end, read, scoreSize);

        // Do they match?
        int match = check_gssw_graph_score_matrices_equal(sse2_aligned, software_aligned, strlen(read));
        
        // Now make sure the matrices match (and fail if they didn't)
        check_condition(match, "score matrices for size mode %hhd are identical", scoreSize);
        
        if (!match) {
            // No match, so dump matrices
            printf("SSE2 Filler:\n");
            gssw_graph_print_score_matrices(sse2_aligned, read, strlen(read), stdout);
            printf("Software Filler:\n");
            gssw_graph_print_score_matrices(software_aligned, read, strlen(read), stdout);
        }

        // Clean up graphs
        gssw_graph_destroy(sse2_aligned);
        gssw_graph_destroy(software_aligned);
        
    }
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
            check_string_alignments_match("GATTACA", "GATTTACA");
        }
        
        {start_test("should produce identical results on short identical strings");
            char* const reference = "GATTACA";
            check_string_alignments_match(reference, reference);
        }
        
        {start_test("should produce identical results on large inserts");
            check_string_alignments_match("GATTACA", "GATTTTTTTTTTTTTTACA");
        }
        
        {start_test("should produce identical results on large deletions");
            check_string_alignments_match("GATTTTTTTTTTTTTTACA", "GATTACA");
        }
        
        {start_test("should produce identical results on empty strings");
            check_string_alignments_match("", "");
        }
        
        {start_test("should produce identical results on 15 As");
            check_string_alignments_match("AAAAAAAAAAAAAAA", "AAAAAAAAAAAAAAA");
        }
        
        {start_test("should produce identical results on 17 As");
            check_string_alignments_match("AAAAAAAAAAAAAAAAA", "AAAAAAAAAAAAAAAAA");
        }
        
        {start_test("should produce identical results on 16 As");
            check_string_alignments_match("AAAAAAAAAAAAAAAA", "AAAAAAAAAAAAAAAA");
        }
        
        {
            char* const reference = "GTGTTCCAGTTCTTATCCTATATCGGAAGTTCAATTATACATCGCACCAGCATATTCATG";
            int32_t i;
            for (i = strlen(reference); i >= 0; i--) {
                {start_test("should produce identical results on a %d bp string", strlen(reference) - i);
                    check_string_alignments_match(&reference[i], &reference[i]);
                }
            }
        }
        
        {start_test("should produce identical results on a small diamond");
            check_diamond_alignments_match("GA", "TTA", "TA", "CA", "GATTACA");
        }
        
        {start_test("should produce identical results for a larger single-node graph with more differences");
            char* const reference = "GTGTTCCAGTTCTTATCCTATATCGGAAGTTCAATTATACATCGCACCAGCATATTCATG";
            char* const read = "GTGTTCAAGTTCATCGGAAGTTCAATTCTACATCGCACCAGCATATAAGATAAATTTCTTG";
            check_string_alignments_match(reference, read);
        }
        
        {start_test("should produce identical results for Jordan's smaller case");
            char* const ref = "CCCCCCCCCTCCCCCCCCCCT";
            char* const read = "CCCCCCCTCCCCCCCCCCTCC";
            
            check_string_alignments_match(ref, read);
        }
        
        {start_test("should produce identical results for Jordan's edge case");
            char* const reference = "CCCCCCCCCTCCCCCCCCCCTCCCCCCCCCCGACCCCCCCCCCCCCCCCCCCCCACCCCCCCCCCACCCCCCCCCCTCCCACCCCCCCCCCCCGCCCCCCCCCCGCCCCCCCCC";
            char* const read = "CCCCCCCTCCCCCCCCCCTCCCCCCCCCCGACCCCCCCCCCCCCCCCCCCCCACCCCCCCCCCACCCCCCCCCCTCCCACCCCCCCCCCCCGCCCCCCCCCCGCCCCCCCCC";
            check_string_alignments_match(reference, read);
        }
        
        {start_test("should produce identical results for Jordan's other edge case");
            char* const reference = "CCCCCCCCCTCCCCCCCCCCTCCCCCCCCCCGACCCCCCCCCCCCCCCCACCCCCCCCGTCCCCCCCCCCCACCCCCCCCCCCCGCCCCCCCCCCGCCCCCCCCC";
            char* const read = "CCCCCCCTCCCCCCCCCCTCCCCCCCCCCGACCCCCCCCCCCCCCCCCCCCCACCCCCCCCCCACCCCCCCCCCTCCCACCCCCCCCCCCCGCCCCCCCCCCGCCCCCCCCC";
            check_string_alignments_match(reference, read);
        }
        
        {start_test("should produce identical results for Jordan's diamond");
            char* const start = "CCCCCCCCCTCCCCCCCCCCTCCCCCCCCCCGACCCCCCCCCCC";
            char* const alt1 = "CCCCCCCCCCACCCCCCCCCCACCCCCCCCCCTCCCA";
            char* const alt2 = "CCCCCACCCCCCCCGTCCCCCCCCCCCA";
            char* const end = "CCCCCCCCCCCCGCCCCCCCCCCGCCCCCCCCC";
            char* const read = "CCCCCCCTCCCCCCCCCCTCCCCCCCCCCGACCCCCCCCCCCCCCCCCCCCCACCCCCCCCCCACCCCCCCCCCTCCCACCCCCCCCCCCCGCCCCCCCCCCGCCCCCCCCC";
            
            check_diamond_alignments_match(start, alt1, alt2, end, read);
        }
        
    }
        
    
    
}

int main (int argc, char * const argv[]) {
    
    // Run all the tests
    test_gssw_software_fill();

    // Report result
    return test_results();
}
 
