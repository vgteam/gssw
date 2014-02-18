
# *GSSW (a string to graph aligner)*
## An *Generalized* SIMD Smith-Waterman C/C++ Library for Use in Genomic Applications

### Authors: Erik Garrison, Mengyao Zhao, Wan-Ping Lee

### Erik Garrison <erik.garrison@bc.edu>
### Mengyao Zhao <zhangmp@bc.edu>

Last revision: 18/02/2014

## Overview

SSW is a fast implementation of the Smith-Waterman algorithm, which uses the
Single-Instruction Multiple-Data (SIMD) instructions to parallelize the
algorithm at the instruction level. SSW library provides an API that can be
flexibly used by programs written in C, C++ and other languages. We also
provide a software that can do protein and genome alignment directly. Current
version of our implementation is ~50 times faster than an ordinary
Smith-Waterman. It can return the Smith-Waterman score, alignment location and
traceback path (cigar) of the optimal alignment accurately; and return the
sub-optimal alignment score and location heuristically.

This is an extension of the main 
[SSW implementation](https://github.com/mengyao/Complete-Striped-Smith-Waterman-Library/tree/master/src)
which allows access to the alignment score matrix, H, and E vectors.  Using
these features, we can extend the recurrence relation that defines the scores
in the score matrix across junctions in the graph.  Affine gaps and gaps
overlapping junctions in the graph are also supported.  Traceback is driven by
storing the entire score matrix of each node (called the *H* matrix in the parlance
of [Farrar's original paper on the striped variant of the Smith-Waterman
algorithm](http://bioinformatics.oxfordjournals.org/content/23/2/156.short)).
Due to the memory requirements of this approach, it is currently only feasible
to use GSSW in a local context (e.g. reference and reads in the tens of
thousands of bases).

By constructing a graph using basic methods described in gssw.h, you can align
reads against the graph and obtain a graph mapping comprised of a graph "cigar"
describing the optimal alignment of the read across the nodes in the graph and
the starting position of the alignment on the first node.  Nodes contain
`(char*)`s pointers which could be used to link to arbitrary metadata.

## Citation

If you use this method the course of your work, please cite our article in PLOS One:
[SSW Library: An SIMD Smith-Waterman C/C++ Library for Use in Genomic
Applications](http://www.plosone.org/article/info%3Adoi%2F10.1371%2Fjournal.pone.0082138)

This article does not cover the generalization of the method to work on partial
order graphs, but rather the underlying structure and performance of the API
used to score alignments against the graph.

## How to use the API

The API files include gssw.h and gssw.c, which can be directly used by any C or
C++ program. For the C++ users who are more comfortable to use a C++ style
interface, an additional C++ wrapper is provided with the file ssw\_cpp.cpp and
ssw\_cpp.h. 

To use the C style API, please: 

1. Download gssw.h and gssw.c, and put them in the same folder of your own
program files.
2. Write `#include "gssw.h"` into your file that will call the API functions.
3. The API files are ready to be compiled together with your own C/C++ files.

The API function descriptions are in the file ssw.h. One simple example of the
API usage is example.c. The Smith-Waterman penalties need to be integers. Small
penalty numbers such as: match: 2, mismatch: -1, gap open: -3, gap extension:
-1 are recommended, which will lead to shorter running time.  

### License: MIT

Copyright (c) 2012-2015 Boston College

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


<!--
To use the C++ style API, please: 

1. Download ssw.h, ssw.c, ssw\_cpp.cpp and ssw\_cpp.h and put them in the same
folder of your own program files.
2. Write #include "ssw\_cpp.h" into your file that will call the API functions.
3. The API files are ready to be compiled together with your own C/C++ files.

The API function descriptions are in the file ssw\_cpp.h. A simple example of
using the C++ API is example.cpp.

## Speed and memory usage of the API

Test data set: 
Target sequence: reference genome of E. coli strain 536 (4,938,920 nucleotides) from NCBI
Query sequences: 1000 reads of Ion Torrent sequenced E. coli strain DH10B (C23-140, 318 PGM Run, 11/2011), read length: ~25-540 bp, most reads are ~200 bp

CPU time:
AMD CPU: default penalties: ~880 seconds; -m1 -x3 -o5 -e2: ~460 seconds
Intel CPU: default penalties: ~960 seconds; -m1 -x3 -o5 -e2: ~500 seconds 

Memory usage: ~40MB
 
## Install the software

1. Download the software from https://github.com/mengyao/Complete-Striped-Smith-Waterman-Library.
2. cd src
3. make
4. the executable file will be ssw\_test

## Run the software

    Usage: ssw\_test [options] ... <target.fasta> <query.fasta>(or <query.fastq>)
    Options:
        -m N	N is a positive integer for weight match in genome sequence alignment. [default: 2]
        -x N	N is a positive integer. -N will be used as weight mismatch in genome sequence alignment. [default: 2]
        -o N	N is a positive integer. -N will be used as the weight for the gap opening. [default: 3]
        -e N	N is a positive integer. -N will be used as the weight for the gap extension. [default: 1]
        -p	Do protein sequence alignment. Without this option, the ssw_test will do genome sequence alignment.
        -a FILE	FILE is either the Blosum or Pam weight matrix. [default: Blosum50]
        -c	Return the alignment path.
        -f N	N is a positive integer. Only output the alignments with the Smith-Waterman score >= N.
        -r	The best alignment will be picked between the original read alignment and the reverse complement read alignment.
        -s	Output in SAM format. [default: no header]
        -h	If -s is used, include header in SAM output.


-->
