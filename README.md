
# *GSSW (a string to graph aligner)*
## An *Graph* SIMD Smith-Waterman C/C++ Library for Use in Genomic Applications

[![Build Status](https://www.travis-ci.org/vgteam/gssw.svg?branch=master)](https://www.travis-ci.org/vgteam/gssw)

### Authors: Erik Garrison, Adam Novak, Jordan Eizenga, Mengyao Zhao

## Overview

GSSW is an extended generalization of [Farrar's striped Smith-Waterman algorithm](http://bioinformatics.oxfordjournals.org/content/23/2/156.short)
to graphs. This repository extends the main 
[SSW implementation](https://github.com/mengyao/Complete-Striped-Smith-Waterman-Library/tree/master/src)
in a manner that provides access to the alignment score matrix, H, and E vectors.
Using these features, we extend the recurrence relation that defines the scores
in the score matrix across junctions in the graph.
In addition to the standard scoring features of SSW, GSSW supports affine gaps, the pinning of the traceback to an optimal start or end node, 
and a bonus for full-length alignments that discourages the genertaion of soft clips by small numbers of mismatches near the ends of a read.

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

Copyright (c) 2012-2016 Boston College and Wellcome Trust Sanger Institute

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
