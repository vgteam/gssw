/* The MIT License

   Copyright (c) 2012-2015 Boston College, 2014-2018 Wellcome Sanger Institute

   Permission is hereby granted, free of charge, to any person obtaining
   a copy of this software and associated documentation files (the
   "Software"), to deal in the Software without restriction, including
   without limitation the rights to use, copy, modify, merge, publish,
   distribute, sublicense, and/or sell copies of the Software, and to
   permit persons to whom the Software is furnished to do so, subject to
   the following conditions:

   The above copyright notice and this permission notice shall be
   included in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
   EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
   MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
   NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
   BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
   ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
   CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE.
*/

/* Contact: Erik Garrison <erik.garrison@gmail.com> */

/*
 *  Created by Mengyao Zhao on 6/22/10.
 *  Generalized to operate on graphs by Erik Garrison and renamed gssw.c
 */
#define SIMDE_ENABLE_NATIVE_ALIASES
#include "simde/x86/sse2.h"
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <inttypes.h>
#include <assert.h>
#include "gssw.h"

//#define DEBUG_TRACEBACK

#ifdef __GNUC__
#define LIKELY(x) __builtin_expect((x),1)
#define UNLIKELY(x) __builtin_expect((x),0)
#else
#define LIKELY(x) (x)
#define UNLIKELY(x) (x)
#endif

/* Convert the coordinate in the scoring matrix into the coordinate in one line of the band. */
#define set_u(u, w, i, j) { int x=(i)-(w); x=x>0?x:0; (u)=(j)-x+1; }

/* Convert the coordinate in the direction matrix into the coordinate in one line of the band. */
#define set_d(u, w, i, j, p) { int x=(i)-(w); x=x>0?x:0; x=(j)-x; (u)=x*3+p; }

/*! @function
  @abstract  Round an integer to the next closest power-2 integer.
  @param  x  integer to be rounded (in place)
  @discussion x will be modified.
 */
#define kroundup32(x) (--(x), (x)|=(x)>>1, (x)|=(x)>>2, (x)|=(x)>>4, (x)|=(x)>>8, (x)|=(x)>>16, ++(x))

/**
 * We can turn SSE2 on and off globally, for testing purposes. When SSE2 is off
 * we use a non-SIMD pure software matrix filler, which is easier to believe is
 * correct by inspection.
 */
int gssw_sse2_enabled = 1;

/**
 * Disable SSE2 matrix filler and use the pure software matrix filler.
 */
void gssw_sse2_disable() {
    gssw_sse2_enabled = 0;
}

/**
 * Enable SSE2 matrix filler.
 */
void gssw_sse2_enable() {
    gssw_sse2_enabled = 1;
}


/* Generate query profile rearrange query sequence & calculate the weight of match/mismatch. */
__m128i* gssw_qP_byte (const int8_t* read_num,
                       const int8_t* mat,
                       const int32_t readLen,
                       const int32_t n,    /* the edge length of the squre matrix mat */
                       uint8_t bias,
                       int8_t start_full_length_bonus,
                       int8_t end_full_length_bonus) {

    int32_t segLen = (readLen + 15) / 16; /* Split the 128 bit register into 16 pieces.
                                     Each piece is 8 bit. Split the read into 16 segments.
                                     Calculate 16 segments in parallel.
                                     This holds the number of segments needed to fit the read.
                                   */
    __m128i* vProfile = (__m128i*)malloc(n * segLen * sizeof(__m128i));
    int8_t* t = (int8_t*)vProfile; // This points to each byte in the profile vector, one at a time
    // nt tracks the nucleotide we're computing the profile for. We do each possible character.
    // i tracks which swizzled register we're working on
    // j tracks the character in the read we're working on
    // segNum counts which of the 16 read segments we're working on, each of which gets its own byte in each swizzled register
    int32_t nt, i, j, segNum; 
    
    /* Generate query profile rearrange query sequence & calculate the weight of match/mismatch */
    for (nt = 0; LIKELY(nt < n); nt ++) {
        
        // special logic for first vector to add bonus for full length left alignment
        if (segLen > 0) {
            j = 0;
            // add bonus to first position in first register (corresponds to first position in read)
            // also account for the start potentially being the end
            *t = j>= readLen ? bias : mat[nt * n + read_num[j]] + bias +
                start_full_length_bonus + (j == readLen - 1 ? end_full_length_bonus : 0);
            t++;
            j += segLen;
            // use normal score for the rest of the vector
            // except for the last base which also gets a bonus
            for (segNum = 1; LIKELY(segNum < 16) ; segNum ++) {
                *t = j>= readLen ? bias : mat[nt * n + read_num[j]] + bias + (j == readLen - 1 ? end_full_length_bonus : 0);
                t++;
                j += segLen;
            }
        }
        
        for (i = 1; i < segLen; i ++) {
            j = i;
            for (segNum = 0; LIKELY(segNum < 16) ; segNum ++) {
                // Now handle all the vectors after the first
                *t = j>= readLen ? bias : mat[nt * n + read_num[j]] + bias + (j == readLen - 1 ? end_full_length_bonus : 0);
                t++;
                j += segLen;
            }
        }
    }
    return vProfile;
}

__m128i* gssw_adj_qP_byte (const int8_t* read_num,
                           const int8_t* qual,
                           const int8_t* adj_mat,
                           const int32_t readLen,
                           const int32_t n,    /* the edge length of the squre matrix mat */
                           uint8_t bias,
                           int8_t start_full_length_bonus,
                           int8_t end_full_length_bonus) {
    
    int32_t segLen = (readLen + 15) / 16; /* Split the 128 bit register into 16 pieces.
                                           Each piece is 8 bit. Split the read into 16 segments.
                                           Calculat 16 segments in parallel.
                                           */
    __m128i* vProfile = (__m128i*)malloc(n * segLen * sizeof(__m128i));
    int8_t* t = (int8_t*)vProfile;
    int32_t nt, i, j, segNum;
    
    
    int32_t matSize = n * n;
    
    /* Generate query profile rearrange query sequence & calculate the weight of match/mismatch */
    for (nt = 0; LIKELY(nt < n); nt ++) {
        
        // special logic for first vector to add bonus for full length pinned alignment
        if (segLen > 0) {
            j = 0;
            // add bonus to first position in first register (corresponds to first position in read)
            // also account for the start potentially being the end
            *t = j>= readLen ? bias : adj_mat[qual[j] * matSize + nt * n + read_num[j]] + bias +
                start_full_length_bonus + (j == readLen - 1 ? end_full_length_bonus : 0);
            t++;
            j += segLen;
            // use normal score for the rest of the vector
            for (segNum = 1; LIKELY(segNum < 16) ; segNum ++) {
                *t = j>= readLen ? bias : adj_mat[qual[j] * matSize + nt * n + read_num[j]] + bias +
                    (j == readLen - 1 ? end_full_length_bonus : 0);
                t++;
                j += segLen;
            }
        }
        
        for (i = 1; i < segLen; i ++) {
            j = i;
            for (segNum = 0; LIKELY(segNum < 16) ; segNum ++) {
                *t = j>= readLen ? bias : 
                    adj_mat[qual[j] * matSize + nt * n + read_num[j]] + bias + (j == readLen - 1 ? end_full_length_bonus : 0);
                t++;
                j += segLen;
            }
        }
    }
    
    return vProfile;
}

/**
 * Look up the value in a profile matrix for the given base code observed at the given read index.
 * Useful for non-swizzled access to the the swizzled profile.
 */
uint8_t profile_get_byte(__m128i* vProfile, int32_t readLen, int32_t read_position, int32_t observed_base) {
    // Profile is stored by observed base (most significant), then by position in the segment, then by segment in the read (lwast significant).
    
    // How long is a segment? We have 16.
    int32_t segLen = (readLen + 15) / 16;
    // What segment are we in of the 16?
    int32_t segment = read_position / segLen;
    // And where are we in that segment?
    int32_t pos_in_segment = read_position % segLen;
    
    // Look at the profile as a byte array
    uint8_t* profile_bytes = (uint8_t*) vProfile;
    
    return profile_bytes[observed_base * (segLen * 16) + pos_in_segment * 16 + segment];
}

/**
 * Swizzle a vector of bytes into a "striped" vector, organized first by
 * position in segment and then by segment of 16. Size must be a multiple of 16.
 */
void swizzle_byte(uint8_t* to_swizzle, int32_t size) {
    if (size == 0) {
        // Nothing to do!
        return;
    }
    
    uint8_t* scratch = (uint8_t*) malloc(size * sizeof(uint8_t));
    if(scratch == NULL) {
        fprintf(stderr, "error:[gssw] Could not allocate swizzle buffer.\n");
        exit(1);
    }
    // Copy the data out of the way
    memcpy(scratch, to_swizzle, size);
    
    // How long is a segment? We have 16.
    int32_t segLen = (size + 15) / 16;
    
    // We'll walk this through the destination array.
    int32_t cursor = 0;
    
    int32_t pos_in_segment;
    for(pos_in_segment = 0; pos_in_segment < segLen; pos_in_segment++) {
        // For each position in a segment
    
        int32_t segNum;
        for (segNum = 0; segNum < 16; segNum++) {
            // For each segment
    
            // Grab the byte
            to_swizzle[cursor] = scratch[segNum * segLen + pos_in_segment];
            // Write the next byte at the next position
            cursor++;
        }
        
    }
   free(scratch);
}

/**
 * Unswizzle a swizzled vector of bytes into a normal start-to-end vector of bytes.
 * Size must be a multiple of 16.
 */
void unswizzle_byte(uint8_t* to_unswizzle, int32_t size) {
    if (size == 0) {
        // Nothing to do!
        return;
    }
    
    uint8_t* scratch = (uint8_t*) malloc(size * sizeof(uint8_t));
    if(scratch == NULL) {
        fprintf(stderr, "error:[gssw] Could not allocate unswizzle buffer.\n");
        exit(1);
    }
    // Copy the data out of the way
    memcpy(scratch, to_unswizzle, size);
    
    // How long is a segment? We have 16.
    int32_t segLen = (size + 15) / 16;
    
    int32_t i;
    for (i = 0; i < size; i++) {
        // Swizzled vector is arranged first by position in segment, then by segment (of 16)
        // So go to the right position in the segment, and then to the right segment, and get the value
        // And save it to the right place in the unswizzled vector.
        to_unswizzle[i] = scratch[(i % segLen) * 16 + (i / segLen)];
    }
   free(scratch);
}

/**
 * Saturation arithmetic subtraction. (like the "subs" SSE2 intrinsics)
 * Compute a - b, returning 0 if it would be negative.
 */
uint8_t subs_byte(uint8_t a, uint8_t b) {
    if (b > a) {
        return 0;
    }
    return a - b;
}

/**
 * Saturation arithmetic addition. (like the "addss" SSE2 intrinsics)
 * Compute a + b, returning max
 */
uint8_t adds_byte(uint8_t a, uint8_t b) {
    uint16_t sum = (uint16_t) a + (uint16_t) b;
    if (sum > 255) {
        return 255;
    }
    return sum;
}

/**
 * We need a max for bytes.
 */
uint8_t max_byte(uint8_t a, uint8_t b) {
    if (a > b) {
        return a;
    }
    return b;
}
 

/**
 * Compute a byte-sized alignment in pure software, without any swizzling or SSE2.
 * Used for computing known good alingments, for testing.
 */
gssw_alignment_end* gssw_sw_software_byte (const int8_t* ref,
                                           int8_t ref_dir,    // 0: forward ref; 1: reverse ref
                                           int32_t refLen,
                                           int32_t readLen,
                                           const uint8_t weight_gapO, /* will be used as - */
                                           const uint8_t weight_gapE, /* will be used as - */
                                           __m128i* vProfile,
                                           uint8_t terminate,    /* the best alignment score: used to terminate
                                                                   the matrix calculation when locating the
                                                                   alignment beginning point. If this score
                                                                   is set to 0, it will not be used */
                                           uint8_t bias,  /* Shift 0 point to a positive value. */
                                           int32_t maskLen,
                                           gssw_align* alignment, /* to save seed and matrix */
                                           const gssw_seed* seed) {     /* to seed the alignment */
                                           
    
                                       
    uint8_t max = 0;                             /* the max alignment score */
    int32_t end_read = readLen - 1;
    int32_t end_ref = -1; /* 0_based best alignment ending point; Initialized as isn't aligned -1. */

    // We need to make sure out matrices are sized to the nearest 16, so pad the read length.
    int32_t padded_read_length = ((readLen + 15) / 16) * 16;

    // Allocate DP matrices (all stored unswizzled, even in the swizzled strategy)
    uint8_t* mH; // used to save matrices for external traceback: overall best score
    uint8_t* mE; // Gap in read best score
    uint8_t* mF; // Gap in ref best score
    // And buffers for matrix columns (unswizzled, but stored swizzled in the seeds)
    // Note that, like the SSE2 code, we calculate the E matrix one column ahead.
    uint8_t* pvHStore; // Current column of the main (H) matrix
    uint8_t* pvEStore; // *Next* column of the gap in read (E) matrix
    uint8_t* pvFStore; // Current column of the gap in reference (F) matrix
    uint8_t* pvHLoad; // Previous column of the main (H) matrix
    uint8_t* pvELoad; // *Current* column of the gap in read (E) matrix
    // No previous column needed for the gap in reference (F) martrix
    // But we do need a scratch pointer for swapping things
    uint8_t* pv;
    
    /* Note use of aligned memory.  Return value of 0 means success for posix_memalign. */
    if (!(!posix_memalign((void**)&pvHStore, sizeof(__m128i), padded_read_length) &&
          !posix_memalign((void**)&pvEStore, sizeof(__m128i), padded_read_length) &&
          !posix_memalign((void**)&pvFStore, sizeof(__m128i), padded_read_length) &&
          !posix_memalign((void**)&pvHLoad,  sizeof(__m128i), padded_read_length) &&
          !posix_memalign((void**)&pvELoad,  sizeof(__m128i), padded_read_length) &&
          !posix_memalign((void**)&alignment->seed.pvE,      sizeof(__m128i), padded_read_length) &&
          !posix_memalign((void**)&alignment->seed.pvHStore, sizeof(__m128i), padded_read_length) &&
          !posix_memalign((void**)&mH,           sizeof(__m128i), refLen*padded_read_length) &&
          !posix_memalign((void**)&mE,           sizeof(__m128i), refLen*padded_read_length) &&
          !posix_memalign((void**)&mF,           sizeof(__m128i), refLen*padded_read_length))) {
        fprintf(stderr, "error:[gssw] Could not allocate memory required for alignment buffers.\n");
        exit(1);
    }

    /* Workaround: zero memory ourselves because we don't have an aligned calloc */
    memset(pvHStore,                 0, padded_read_length);
    memset(pvEStore,                 0, padded_read_length);
    memset(pvFStore,                 0, padded_read_length);
    memset(pvHLoad,                  0, padded_read_length);
    memset(pvELoad,                  0, padded_read_length);
    memset(alignment->seed.pvE,      0, padded_read_length);
    memset(alignment->seed.pvHStore, 0, padded_read_length);
    memset(mH,                       0, refLen*padded_read_length);
    memset(mE,                       0, refLen*padded_read_length);
    memset(mF,                       0, refLen*padded_read_length);

    /* if we are running a seeded alignment, copy over the seeds */
    if (seed) {
        // Load the bufers with the seed contents
        memcpy(pvEStore, seed->pvE, padded_read_length);
        memcpy(pvHStore, seed->pvHStore, padded_read_length);
        
        // Unswizzle them so we can work on normal arrays.
        unswizzle_byte(pvEStore, padded_read_length);
        unswizzle_byte(pvHStore, padded_read_length);
        
    }

    /* Set external matrix pointers */
    alignment->mH = mH;
    alignment->mE = mE;
    alignment->mF = mF;

    /* Record that we have done a byte-order alignment */
    alignment->is_byte = 1;
    
    int32_t begin = 0, end = refLen, step = 1;

    /* outer loop to process the reference sequence */
    if (ref_dir == 1) {
        begin = refLen - 1;
        end = -1;
        step = -1;
    }
    int32_t i;
    for (i = begin; LIKELY(i != end); i += step) {
        // For each column i in the DP matrices (running in the appropriate direction)
        
        // We don't need the previous F column.
        // But we do need to push the current H to the previous H and the next E to the current E.
        // We use a double buffering settup so we don't need to malloc and free and copy stuff.
        // By working with a previous and next column, we don't need to do anything fancy to support the flipable ref_dir.
        pv = pvHLoad;
        pvHLoad = pvHStore;
        pvHStore = pv;
        pv = pvELoad;
        pvELoad = pvEStore;
        pvEStore = pv;
        
        int32_t j;
        for (j = 0; j < readLen; j++) {
            // For each row j in the DP matrices (running top to bottom)
            
            // Gap in read (E) matrix for this column is already calculated.
            
            uint8_t refGapOpenScore = 0;
            uint8_t refGapExtendScore = 0;
            if (j == 0) {
                // We can only end in a gap in the reference on the first read character with negative score.
                // But we saturate that out to 0.
                pvFStore[j] = subs_byte(0, weight_gapO);
            } else {
                // Set the gap-in-ref matrix (F) based on previous slot in current F and in current H
                refGapOpenScore = subs_byte(pvHStore[j-1], weight_gapO);
                refGapExtendScore = subs_byte(pvFStore[j-1], weight_gapE);
                pvFStore[j] = max_byte(refGapOpenScore, refGapExtendScore);
            }

            // What score are we adding to with a match?
            // If there's nowhere to come from it's 0.
            uint8_t matchFrom = 0;
            if (j > 0) {
                // Otherwise its the score where we came from
                matchFrom = pvHLoad[j-1];
            }
            // What's the profile say about a match/mismatch here?
            uint8_t profileScore = profile_get_byte(vProfile, readLen, j, ref[i]);
            
            // Compute the match/mismatch score with saturating addition.
            uint8_t matchScore = adds_byte(matchFrom, profileScore);
            // Subtract out the bias that the profile score had on it.
            matchScore = subs_byte(matchScore, bias);
            
            // Set the normal (H) matrix based on current E, current F, and match/mismatch score
            pvHStore[j] = max_byte(max_byte(pvELoad[j], pvFStore[j]), matchScore);
            
            // Calculate the next E column
            // Set the next gap-in-read matrix (E) based on current E and current H
            uint8_t readGapOpenScore = subs_byte(pvHStore[j], weight_gapO);
            uint8_t readGapExtendScore = subs_byte(pvELoad[j], weight_gapE);
            pvEStore[j] = max_byte(readGapOpenScore, readGapExtendScore);
        
            // Set the running max
            if (pvHStore[j] > max) {
                max = pvHStore[j];
                end_ref = i;
                end_read = j;
            }
            
            // Copy from columns to matrices
            mH[i * readLen + j] = pvHStore[j];
            mE[i * readLen + j] = pvELoad[j];
            mF[i * readLen + j] = pvFStore[j];
        }
    }
    
    // Reswizzle seeds
    swizzle_byte(pvEStore, padded_read_length);
    swizzle_byte(pvHStore, padded_read_length);
    
    // Save seeds.
    memcpy(alignment->seed.pvE,      pvEStore, padded_read_length);
    memcpy(alignment->seed.pvHStore, pvHStore, padded_read_length);

    // Clean up all the buffers
    free(pvHLoad);
    free(pvHStore);
    free(pvELoad);
    free(pvEStore);
    // No pvFLoad.
    free(pvFStore);

    /* Find the most possible 2nd best alignment. */
    // TODO: this only does the best alignment???
    gssw_alignment_end* bests = (gssw_alignment_end*) calloc(2, sizeof(gssw_alignment_end));
    bests[0].score = max + bias >= 255 ? 255 : max;
    bests[0].ref = end_ref;
    bests[0].read = end_read;

    return bests;
}

/* To determine the maximum values within each vector, rather than between vectors. */

#define m128i_max16(m, vm) \
    (vm) = _mm_max_epu8((vm), _mm_srli_si128((vm), 8)); \
    (vm) = _mm_max_epu8((vm), _mm_srli_si128((vm), 4)); \
    (vm) = _mm_max_epu8((vm), _mm_srli_si128((vm), 2)); \
    (vm) = _mm_max_epu8((vm), _mm_srli_si128((vm), 1)); \
    (m) = _mm_extract_epi16((vm), 0)

#define m128i_max8(m, vm) \
    (vm) = _mm_max_epi16((vm), _mm_srli_si128((vm), 8)); \
    (vm) = _mm_max_epi16((vm), _mm_srli_si128((vm), 4)); \
    (vm) = _mm_max_epi16((vm), _mm_srli_si128((vm), 2)); \
    (m) = _mm_extract_epi16((vm), 0)
    
// See https://stackoverflow.com/q/33824300 for this unsigned comparison macro
// for the missing unsigned comparison instruction _mm_cmpgt_epu8
#define m128i_cmpgt(v0, v1) \
         _mm_cmpgt_epi8(_mm_xor_si128(v0, _mm_set1_epi8(-128)), \
                        _mm_xor_si128(v1, _mm_set1_epi8(-128)))

/* Striped Smith-Waterman
   Record the highest score of each reference position.
   Return the alignment score and ending position of the best alignment, 2nd best alignment, etc.
   Gap begin and gap extension are different.
   wight_match > 0, all other weights < 0.
   The returned positions are 0-based.
 */
gssw_alignment_end* gssw_sw_sse2_byte (const int8_t* ref,
                                       int8_t ref_dir,    // 0: forward ref; 1: reverse ref
                                       int32_t refLen,
                                       int32_t readLen,
                                       const uint8_t weight_gapO, /* will be used as - */
                                       const uint8_t weight_gapE, /* will be used as - */
                                       __m128i* vProfile,
                                       uint8_t terminate,    /* the best alignment score: used to terminate
                                                               the matrix calculation when locating the
                                                               alignment beginning point. If this score
                                                               is set to 0, it will not be used */
                                       uint8_t bias,  /* Shift 0 point to a positive value. */
                                       int32_t maskLen,
                                       gssw_align* alignment, /* to save seed and matrix */
                                       bool save_matrixes,  /* don't save the H, E, and F matrixes */
                                       const gssw_seed* seed) {     /* to seed the alignment */

    uint8_t max = 0;                             /* the max alignment score */
    int32_t end_read = readLen - 1;
    int32_t end_ref = -1; /* 0_based best alignment ending point; Initialized as isn't aligned -1. */
    int32_t segLen = (readLen + 15) / 16; /* number of segment */

    /* Initialize buffers used in alignment */
    __m128i* pvHStore;
    __m128i* pvHLoad;
    __m128i* pvHmax;
    __m128i* pvE; // TODO: appears redundant with pvEStore
    // We have a couple extra arrays for logging columns
    __m128i* pvEStore;
    __m128i* pvFStore;
    uint8_t* mH = NULL; // used to save matrices for external traceback: overall best score
    uint8_t* mE = NULL; // Gap in read best score
    uint8_t* mF = NULL; // Gap in ref best score
    /* Note use of aligned memory.  Return value of 0 means success for posix_memalign. */
    if (!(!posix_memalign((void**)&pvHStore,     sizeof(__m128i), segLen*sizeof(__m128i)) &&
          !posix_memalign((void**)&pvHLoad,      sizeof(__m128i), segLen*sizeof(__m128i)) &&
          !posix_memalign((void**)&pvHmax,       sizeof(__m128i), segLen*sizeof(__m128i)) &&
          !posix_memalign((void**)&pvE,          sizeof(__m128i), segLen*sizeof(__m128i)) &&
          !posix_memalign((void**)&pvEStore,     sizeof(__m128i), segLen*sizeof(__m128i)) &&
          !posix_memalign((void**)&pvFStore,     sizeof(__m128i), segLen*sizeof(__m128i)) &&
          !posix_memalign((void**)&alignment->seed.pvE,      sizeof(__m128i), segLen*sizeof(__m128i)) &&
          !posix_memalign((void**)&alignment->seed.pvHStore, sizeof(__m128i), segLen*sizeof(__m128i)))) {
        fprintf(stderr, "error:[gssw] Could not allocate memory required for alignment buffers.\n");
        exit(1);
    }

    if (save_matrixes && !(!posix_memalign((void**)&mH,           sizeof(__m128i), segLen*refLen*sizeof(__m128i)) &&
                           !posix_memalign((void**)&mE,           sizeof(__m128i), segLen*refLen*sizeof(__m128i)) &&
                           !posix_memalign((void**)&mF,           sizeof(__m128i), segLen*refLen*sizeof(__m128i)))) {
        fprintf(stderr, "error:[gssw] Could not allocate memory required for alignment traceback matrixes.\n");
        exit(1);
    }

    /* Workaround: zero memory ourselves because we don't have an aligned calloc */
    memset(pvHStore,                 0, segLen*sizeof(__m128i));
    memset(pvHLoad,                  0, segLen*sizeof(__m128i));
    memset(pvHmax,                   0, segLen*sizeof(__m128i));
    memset(pvE,                      0, segLen*sizeof(__m128i));
    memset(pvEStore,                 0, segLen*sizeof(__m128i));
    memset(pvFStore,                 0, segLen*sizeof(__m128i));
    memset(alignment->seed.pvE,      0, segLen*sizeof(__m128i));
    memset(alignment->seed.pvHStore, 0, segLen*sizeof(__m128i));
    if (save_matrixes) {
        memset(mH,                       0, segLen*refLen*sizeof(__m128i));
        memset(mE,                       0, segLen*refLen*sizeof(__m128i));
        memset(mF,                       0, segLen*refLen*sizeof(__m128i));
    }

    /* if we are running a seeded alignment, copy over the seeds */
    if (seed) {
        memcpy(pvE, seed->pvE, segLen*sizeof(__m128i));
        memcpy(pvHStore, seed->pvHStore, segLen*sizeof(__m128i));
    }

    /* Set external matrix pointers */
    if (save_matrixes) {
        alignment->mH = mH;
        alignment->mE = mE;
        alignment->mF = mF;
    }

    /* Record that we have done a byte-order alignment */
    alignment->is_byte = 1;

    /* Define 16 byte 0 vector. */
    __m128i vZero = _mm_set1_epi32(0);

    /* Used for iteration */
    int32_t i, j;

    /* 16 byte insertion begin vector */
    __m128i vGapO = _mm_set1_epi8(weight_gapO);

    /* 16 byte insertion extension vector */
    __m128i vGapE = _mm_set1_epi8(weight_gapE);

    /* 16 byte bias vector */
    __m128i vBias = _mm_set1_epi8(bias);

    __m128i vMaxScore = vZero; /* Trace the highest score of the whole SW matrix. */
    __m128i vMaxMark = vZero; /* Trace the highest score till the previous column. */
    __m128i vTemp;
    int32_t begin = 0, end = refLen, step = 1;

    /* outer loop to process the reference sequence */
    if (ref_dir == 1) {
        begin = refLen - 1;
        end = -1;
        step = -1;
    }
    for (i = begin; LIKELY(i != end); i += step) {
        // For each column
    
        int32_t cmp;
        __m128i e = vZero, vF = vZero, vMaxColumn = vZero; /* Initialize F value to 0.
                               Any errors to vH values will be corrected in the Lazy_F loop.
                             */
        //max16(maxColumn[i], vMaxColumn);
        //fprintf(stderr, "middle[%d]: %d\n", i, maxColumn[i]);

        // Load the last column's last H value in each segment
        //__m128i vH = pvHStore[segLen - 1];
        __m128i vH = _mm_load_si128 (pvHStore + (segLen - 1));
        // Shift it over (TODO: why??? We only shift this initial read and not later reads.)
        vH = _mm_slli_si128 (vH, 1); /* Shift the 128-bit value in vH left by 1 byte. */
        // Find the profile entries for matching this column's ref base against each read base.
        __m128i* vP = vProfile + ref[i] * segLen; /* Right part of the vProfile */

        /* Swap the 2 H buffers. */
        __m128i* pv = pvHLoad;
        pvHLoad = pvHStore;
        pvHStore = pv;

        /* inner loop to process the query sequence */
        for (j = 0; LIKELY(j < segLen); ++j) {
            // For each vector of cursor positions within this column
            // at position j in each segment

            // Add the profile scores for matching against this ref base
            vH = _mm_adds_epu8(vH, _mm_load_si128(vP + j));
            // And subtract out the profile's bias (so profile scores can be <0)
            vH = _mm_subs_epu8(vH, vBias); /* vH will be always > 0 because of saturation arithmetic */
            //    max16(maxColumn[i], vH);
            //    fprintf(stderr, "H[%d]: %d\n", i, maxColumn[i]);
            /*
            int8_t* t;
            int32_t ti;
            fprintf(stdout, "%d\n", i);
            for (t = (int8_t*)&vH, ti = 0; ti < 16; ++ti) fprintf(stdout, "%d\t", *t++);
            fprintf(stdout, "\n");
            */
            
            // So now vH has the scores we would get if we did all matches/mismatches from the previous column.
            // Next we are going to replace entries if we have a better score from a gap matrix.

            /* Get max from vH, vE and vF. */
            e = _mm_load_si128(pvE + j);
            //_mm_store_si128(vE + j, e);
            
            // So e holds the *current* column's read gap open/extend scores,
            // which we computed on the *previous* column's pass.
            // vF stores the current column and *current* cursor position's ref
            // gap open/extend scores, which we computed on the *previous*
            // cursor position.

            vH = _mm_max_epu8(vH, e);
            vH = _mm_max_epu8(vH, vF);
            vMaxColumn = _mm_max_epu8(vMaxColumn, vH);
            
            // So now vH has the correct (modulo wrong F values) H matrix entries.

            // max16(maxColumn[i], vMaxColumn);
            //fprintf(stdout, "middle[%d]: %d\n", i, maxColumn[i]);
            //fprintf(stdout, "i=%d, j=%d\t", i, j);
            //for (t = (int8_t*)&vMaxColumn, ti = 0; ti < 16; ++ti) fprintf(stdout, "%d\t", *t++);
            //fprintf(stdout, "\n");

            /* Save vH values. */
            _mm_store_si128(pvHStore + j, vH);
            
            /* Save the vE and vF values they derived from */
            _mm_store_si128(pvEStore + j, e);
            _mm_store_si128(pvFStore + j, vF);

            // Now we need to compute the E values for the *next* column, based
            // on our non-F-loop-processed H values

            /* Update vE value. */
            vH = _mm_subs_epu8(vH, vGapO); /* saturation arithmetic, result >= 0 */
            e = _mm_subs_epu8(e, vGapE);
            e = _mm_max_epu8(e, vH);

            // And we compute the F values for the next cursor position.

            /* Compute new vF value, giving F matrix values at next cursor position */
            vF = _mm_subs_epu8(vF, vGapE);
            vF = _mm_max_epu8(vF, vH); // We already charged a gap open against vH

            /* Save the E values we computed for the next column */
            _mm_store_si128(pvE + j, e);

            /* Load the next vH. */
            vH = _mm_load_si128(pvHLoad + j);
        }


        /* reset pointers to the start of the saved data */
        j = 0;
        vH = _mm_load_si128 (pvHStore + j);

        /*  
         * Wrap vF around from the end of each segment to the start of the next.
         */
        vF = _mm_slli_si128 (vF, 1);
        
        // So now we're looking at the F value for every first position, after a
        // full pass. So the first F is guaranteed to be right, and other Fs
        // will be right if nothing had to propagate down more than 16 bases.

        // We're also looking at the H values that should be derived from those
        // F values.
        
        // Now we need to work out if we actually want to update anything. We
        // need to do an F loop if we would modify H, or if we would improve
        // over the old F.
        
        // If we beat the stored H
        vTemp = m128i_cmpgt (vF, vH);
        cmp = _mm_movemask_epi8 (vTemp);
        // Or we beat the stored F
        vTemp = _mm_load_si128 (pvFStore + j);
        vTemp = m128i_cmpgt (vF, vTemp);
        cmp |= _mm_movemask_epi8 (vTemp);
        while (cmp != 0x0000)
        {
            // Then we do the update
            
            // Update this stripe of the H matrix
            vH = _mm_max_epu8 (vH, vF);
            vMaxColumn = _mm_max_epu8(vMaxColumn, vH);
            _mm_store_si128 (pvHStore + j, vH);
            
            // Update the E matrix for the next column
            // Since we may have changed the H matrix
            // This is to allow a gap-to-gap transition in the alignment
            e = _mm_load_si128(pvE + j);
            // The H matrix can only get better, so the gap open scores can only
            // get better, so the E matrix can only get better too.
            vTemp = _mm_subs_epu8(vH, vGapO);
            e = _mm_max_epu8(e, vTemp);
            _mm_store_si128(pvE + j, e);
            // TODO: Instead of doing this, would it be smarter to just compute
            // the E matrix for each column when we're doing its H matrix? Or
            // would the extra buffer slow us down more than the extra compute?
            

            // Save the stripe of the F matrix
            // Only add in better F scores. Sometimes during this loop we'll
            // recompute worse ones.
            vTemp = _mm_load_si128 (pvFStore + j);
            vTemp = _mm_max_epu8 (vTemp, vF);
            _mm_store_si128(pvFStore + j, vTemp);

            // Then think about extending
            vF = _mm_subs_epu8 (vF, vGapE);
            // We never need to think about gap opens because nothing that came
            // from a gap open can ever change, because you won't close and then
            // immediately open a gap.

            j++;
            if (j >= segLen)
            {
                // Wrap around to the next segment again
                j = 0;
                vF = _mm_slli_si128 (vF, 1);
            }

            // Again compute if H or F needs updating based on this new set of F
            // values.
            vH = _mm_load_si128 (pvHStore + j);
            
            // See if we beat the stored H
            vTemp = m128i_cmpgt (vF, vH);
            cmp = _mm_movemask_epi8 (vTemp);
            // Or if we beat the stored F
            vTemp = _mm_load_si128 (pvFStore + j);
            vTemp = m128i_cmpgt (vF, vTemp);
            cmp |= _mm_movemask_epi8 (vTemp);
        }

        vMaxScore = _mm_max_epu8(vMaxScore, vMaxColumn);
        vTemp = _mm_cmpeq_epi8(vMaxMark, vMaxScore);
        cmp = _mm_movemask_epi8(vTemp);
        if (cmp != 0xffff) {
            uint8_t temp;
            vMaxMark = vMaxScore;
            m128i_max16(temp, vMaxScore);
            vMaxScore = vMaxMark;

            if (LIKELY(temp > max)) {
                max = temp;
                if (max + bias >= 255) break;    //overflow
                end_ref = i;

                /* Store the column with the highest alignment score in order to trace the alignment ending position on read. */
                for (j = 0; LIKELY(j < segLen); ++j) pvHmax[j] = pvHStore[j];

            }
        }

        // save the current column for traceback
        // Need to unswizzle all the stripes

        if (save_matrixes) {
            // Save H
            //fprintf(stdout, "%i %i\n", i, j);
            for (j = 0; LIKELY(j < segLen); ++j) {
                uint8_t* t;
                int32_t ti;
                vTemp = pvHStore[j];
                for (t = (uint8_t*)&vTemp, ti = 0; ti < 16; ++ti) {
                    //fprintf(stderr, "%d\t", *t);
                    ((uint8_t*)mH)[i*readLen + ti*segLen + j] = *t++;
                }
                //fprintf(stderr, "\n");
            }
        
            // Save E
            //fprintf(stdout, "%i %i\n", i, j);
            for (j = 0; LIKELY(j < segLen); ++j) {
                uint8_t* t;
                int32_t ti;
                vTemp = pvEStore[j];
                for (t = (uint8_t*)&vTemp, ti = 0; ti < 16; ++ti) {
                    //fprintf(stderr, "%d\t", *t);
                    ((uint8_t*)mE)[i*readLen + ti*segLen + j] = *t++;
                }
                //fprintf(stderr, "\n");
            }
        
            // Save F
            //fprintf(stdout, "%i %i\n", i, j);
            for (j = 0; LIKELY(j < segLen); ++j) {
                uint8_t* t;
                int32_t ti;
                vTemp = pvFStore[j];
                for (t = (uint8_t*)&vTemp, ti = 0; ti < 16; ++ti) {
                    //fprintf(stderr, "%d\t", *t);
                    ((uint8_t*)mF)[i*readLen + ti*segLen + j] = *t++;
                }
                //fprintf(stderr, "\n");
            }
        }

        /* Record the max score of current column. */
        //max16(maxColumn[i], vMaxColumn);
        //fprintf(stderr, "maxColumn[%d]: %d\n", i, maxColumn[i]);
        //if (maxColumn[i] == terminate) break;

    }
        
    //fprintf(stderr, "%p %p %p %p %p %p\n", *pmH, mH, pvHmax, pvE, pvHLoad, pvHStore);
    // save the last vH
    memcpy(alignment->seed.pvE,      pvE,      segLen*sizeof(__m128i));
    memcpy(alignment->seed.pvHStore, pvHStore, segLen*sizeof(__m128i));

    /* Trace the alignment ending position on read. */
    uint8_t *t = (uint8_t*)pvHmax;
    int32_t column_len = segLen * 16;
    for (i = 0; LIKELY(i < column_len); ++i, ++t) {
        int32_t temp;
        if (*t == max) {
            temp = i / 16 + i % 16 * segLen;
            if (temp < end_read) end_read = temp;
        }
    }

    //fprintf(stderr, "%p %p %p %p %p %p\n", *pmH, mH, pvHmax, pvE, pvHLoad, pvHStore);

    free(pvE);
    free(pvHmax);
    free(pvHLoad);
    free(pvHStore);
    free(pvEStore);
    free(pvFStore);

    /* Find the most possible 2nd best alignment. */
    gssw_alignment_end* bests = (gssw_alignment_end*) calloc(2, sizeof(gssw_alignment_end));
    bests[0].score = max + bias >= 255 ? 255 : max;
    bests[0].ref = end_ref;
    bests[0].read = end_read;


    return bests;
}

__m128i* gssw_qP_word (const int8_t* read_num,
                       const int8_t* mat,
                       const int32_t readLen,
                       const int32_t n,
                       int8_t start_full_length_bonus,
                       int8_t end_full_length_bonus) {

    int32_t segLen = (readLen + 7) / 8;
    __m128i* vProfile = (__m128i*)malloc(n * segLen * sizeof(__m128i));
    int16_t* t = (int16_t*)vProfile;
    int32_t nt, i, j;
    int32_t segNum;

    /* Generate query profile rearrange query sequence & calculate the weight of match/mismatch */
    for (nt = 0; LIKELY(nt < n); nt ++) {
        // special logic for first vector to add bonus for full length pinned alignment
        if (segLen > 0) {
            j = 0;
            // add bonus to first position in first register (corresponds to first position in read)
            // also account for the start potentially being the end
            *t = j>= readLen ? 0 : mat[nt * n + read_num[j]] +
                start_full_length_bonus + (j == readLen - 1 ? end_full_length_bonus : 0);
            t++;
            j += segLen;
            // use normal score for the rest of the vector
            for (segNum = 1; LIKELY(segNum < 8) ; segNum ++) {
                *t = j>= readLen ? 0 : mat[nt * n + read_num[j]] + (j == readLen - 1 ? end_full_length_bonus : 0);
                t++;
                j += segLen;
            }
        }
        
        for (i = 1; i < segLen; i ++) {
            j = i;
            for (segNum = 0; LIKELY(segNum < 8) ; segNum ++) {
                *t = j>= readLen ? 0 : mat[nt * n + read_num[j]] + (j == readLen - 1 ? end_full_length_bonus : 0);
                t++;
                j += segLen;
            }
        }
    }
    return vProfile;
}

__m128i* gssw_adj_qP_word (const int8_t* read_num,
                           const int8_t* qual,
                           const int8_t* adj_mat,
                           const int32_t readLen,
                           const int32_t n,
                           int8_t start_full_length_bonus,
                           int8_t end_full_length_bonus) {

    int32_t segLen = (readLen + 7) / 8;
    __m128i* vProfile = (__m128i*) malloc(n * segLen * sizeof(__m128i));
    int16_t* t = (int16_t*) vProfile;
    int32_t nt, i, j, segNum;

    int32_t matSize = n * n;
    
    /* Generate query profile rearrange query sequence & calculate the weight of match/mismatch */
    for (nt = 0; LIKELY(nt < n); nt++) {
        
        // special logic for first vector to add bonus for full length pinned alignment
        if (segLen > 0) {
            j = 0;
            // add bonus to first position in first register (corresponds to first position in read)
            // also account for the start potentially being the end
            *t = j>= readLen ? 0 : adj_mat[qual[j] * matSize + nt * n + read_num[j]] +
                start_full_length_bonus + (j == readLen - 1 ? end_full_length_bonus : 0);
            t++;
            j += segLen;
            // use normal score for the rest of the vector
            for (segNum = 1; LIKELY(segNum < 8) ; segNum ++) {
                *t = j>= readLen ? 0 : adj_mat[qual[j] * matSize + nt * n + read_num[j]] +
                    (j == readLen - 1 ? end_full_length_bonus : 0);
                t++;
                j += segLen;
            }
        }
        for (i = 1; i < segLen; i++) {
            j = i;
            for (segNum = 0; LIKELY(segNum < 8) ; segNum++) {
                *t = j>= readLen ? 0 :
                    adj_mat[qual[j] * matSize + nt * n + read_num[j]] + (j == readLen - 1 ? end_full_length_bonus : 0);
                t++;
                j += segLen;
            }
        }
    }
    
    return vProfile;
}

/**
 * Look up the value in a profile matrix for the given base code observed at the given read index.
 * Useful for non-swizzled access to the the swizzled profile.
 */
uint16_t profile_get_word(__m128i* vProfile, int32_t readLen, int32_t read_position, int32_t observed_base) {
    // Profile is stored by observed base (most significant), then by position in the segment, then by segment in the read (lwast significant).
    
    // How long is a segment? We have 8.
    int32_t segLen = (readLen + 7) / 8;
    // What segment are we in of the 8?
    int32_t segment = read_position / segLen;
    // And where are we in that segment?
    int32_t pos_in_segment = read_position % segLen;
    
    // Look at the profile as a byte array
    uint16_t* profile_words = (uint16_t*) vProfile;
    
    return profile_words[observed_base * (segLen * 8) + pos_in_segment * 8 + segment];
}

/**
 * Swizzle a vector of words into a "striped" vector, organized first by
 * position in segment and then by segment of 8. Size must be a multiple of 8.
 */
void swizzle_word(int16_t* to_swizzle, int32_t size) {
    if (size == 0) {
        // Nothing to do!
        return;
    }
    
    int16_t* scratch = (int16_t*) malloc(size * sizeof(int16_t));
    if(scratch == NULL) {
        fprintf(stderr, "error:[gssw] Could not allocate swizzle buffer.\n");
        exit(1);
    }
    // Copy the data out of the way
    memcpy(scratch, to_swizzle, size * sizeof(int16_t));
    
    // How long is a segment? We have 8.
    int32_t segLen = (size + 7) / 8;
    
    // We'll walk this through the destination array.
    int32_t cursor = 0;
    
    int32_t pos_in_segment;
    for(pos_in_segment = 0; pos_in_segment < segLen; pos_in_segment++) {
        // For each position in a segment
    
        int32_t segNum;
        for (segNum = 0; segNum < 8; segNum++) {
            // For each segment
    
            // Grab the byte
            to_swizzle[cursor] = scratch[segNum * segLen + pos_in_segment];
            // Write the next byte at the next position
            cursor++;
        }
        
    }
}

/**
 * Unswizzle a swizzled vector of words into a normal start-to-end vector of words.
 * Size must be a multiple of 8.
 */
void unswizzle_word(int16_t* to_unswizzle, int32_t size) {
    if (size == 0) {
        // Nothing to do!
        return;
    }
    
    int16_t* scratch = (int16_t*) malloc(size * sizeof(int16_t));
    if(scratch == NULL) {
        fprintf(stderr, "error:[gssw] Could not allocate unswizzle buffer.\n");
        exit(1);
    }
    // Copy the data out of the way
    memcpy(scratch, to_unswizzle, size * sizeof(int16_t));
    
    // How long is a segment? We have 8.
    int32_t segLen = (size + 7) / 8;
    
    int32_t i;
    for (i = 0; i < size; i++) {
        // Swizzled vector is arranged first by position in segment, then by segment (of 8)
        // So go to the right position in the segment, and then to the right segment, and get the value
        // And save it to the right place in the unswizzled vector.
        to_unswizzle[i] = scratch[(i % segLen) * 8 + (i / segLen)];
    }
}

/**
 * Saturation arithmetic subtraction. (like the "subs" SSE2 intrinsics)
 * Compute a - b, returning 0 if it would be negative.
 * Signed for 16 bit mode.
 */
int16_t subs_word(int16_t a, int16_t b) {
    int32_t diff = (int32_t) a - (int32_t) b;
    if (diff > 32767) {
        diff = 32767;
    }
    if (diff < -32768) {
        diff = -32768;
    }
    return diff;
}

/**
 * Saturation arithmetic addition. (like the "addss" SSE2 intrinsics)
 * Compute a + b, returning max.
 * Signed for 16 bit mode.
 */
int16_t adds_word(int16_t a, int16_t b) {
    int32_t sum = (int32_t) a + (int32_t) b;
    if (sum > 32767) {
        sum = 32767;
    }
    if (sum < -32768) {
        sum = -32768;
    }
    return sum;
}

/**
 * We need a max for words.
 * Signed for 16 bit mode.
 */
int16_t max_word(int16_t a, int16_t b) {
    if (a > b) {
        return a;
    }
    return b;
}

/**
 * Compute a word-sized alignment in pure software, without any swizzling or SSE2.
 * Used for computing known good alingments, for testing.
 */
gssw_alignment_end* gssw_sw_software_word (const int8_t* ref,
                                           int8_t ref_dir,    // 0: forward ref; 1: reverse ref
                                           int32_t refLen,
                                           int32_t readLen,
                                           const uint8_t weight_gapO, /* will be used as - */
                                           const uint8_t weight_gapE, /* will be used as - */
                                           __m128i* vProfile,
                                           uint16_t terminate,
                                           int32_t maskLen,
                                           gssw_align* alignment, /* to save seed and matrix */
                                           const gssw_seed* seed) {     /* to seed the alignment */
                                           
    
                                       
    int16_t max = 0;                             /* the max alignment score */
    int32_t end_read = readLen - 1;
    int32_t end_ref = -1; /* 0_based best alignment ending point; Initialized as isn't aligned -1. */

    // We need to make sure out matrices are sized to the nearest 8, so pad the read length.
    int32_t padded_read_length = ((readLen + 7) / 8) * 8;

    // Allocate DP matrices (all stored unswizzled, even in the swizzled strategy)
    int16_t* mH; // used to save matrices for external traceback: overall best score
    int16_t* mE; // Gap in read best score
    int16_t* mF; // Gap in ref best score
    // And buffers for matrix columns (unswizzled, but stored swizzled in the seeds)
    // Note that, like the SSE2 code, we calculate the E matrix one column ahead.
    int16_t* pvHStore; // Current column of the main (H) matrix
    int16_t* pvEStore; // *Next* column of the gap in read (E) matrix
    int16_t* pvFStore; // Current column of the gap in reference (F) matrix
    int16_t* pvHLoad; // Previous column of the main (H) matrix
    int16_t* pvELoad; // *Current* column of the gap in read (E) matrix
    // No previous column needed for the gap in reference (F) martrix
    // But we do need a scratch pointer for swapping things
    int16_t* pv;
    
    /* Note use of aligned memory.  Return value of 0 means success for posix_memalign. */
    if (!(!posix_memalign((void**)&pvHStore, sizeof(__m128i), padded_read_length * sizeof(int16_t)) &&
          !posix_memalign((void**)&pvEStore, sizeof(__m128i), padded_read_length * sizeof(int16_t)) &&
          !posix_memalign((void**)&pvFStore, sizeof(__m128i), padded_read_length * sizeof(int16_t)) &&
          !posix_memalign((void**)&pvHLoad,  sizeof(__m128i), padded_read_length * sizeof(int16_t)) &&
          !posix_memalign((void**)&pvELoad,  sizeof(__m128i), padded_read_length * sizeof(int16_t)) &&
          !posix_memalign((void**)&alignment->seed.pvE,      sizeof(__m128i), padded_read_length * sizeof(int16_t)) &&
          !posix_memalign((void**)&alignment->seed.pvHStore, sizeof(__m128i), padded_read_length * sizeof(int16_t)) &&
          !posix_memalign((void**)&mH,           sizeof(__m128i), refLen * padded_read_length * sizeof(int16_t)) &&
          !posix_memalign((void**)&mE,           sizeof(__m128i), refLen * padded_read_length * sizeof(int16_t)) &&
          !posix_memalign((void**)&mF,           sizeof(__m128i), refLen * padded_read_length * sizeof(int16_t)))) {
        fprintf(stderr, "error:[gssw] Could not allocate memory required for alignment buffers.\n");
        exit(1);
    }

    /* Workaround: zero memory ourselves because we don't have an aligned calloc */
    memset(pvHStore,                 0, padded_read_length * sizeof(int16_t));
    memset(pvEStore,                 0, padded_read_length * sizeof(int16_t));
    memset(pvFStore,                 0, padded_read_length * sizeof(int16_t));
    memset(pvHLoad,                  0, padded_read_length * sizeof(int16_t));
    memset(pvELoad,                  0, padded_read_length * sizeof(int16_t));
    memset(alignment->seed.pvE,      0, padded_read_length * sizeof(int16_t));
    memset(alignment->seed.pvHStore, 0, padded_read_length * sizeof(int16_t));
    memset(mH,                       0, refLen * padded_read_length * sizeof(int16_t));
    memset(mE,                       0, refLen * padded_read_length * sizeof(int16_t));
    memset(mF,                       0, refLen * padded_read_length * sizeof(int16_t));

    /* if we are running a seeded alignment, copy over the seeds */
    if (seed) {
        // Load the bufers with the seed contents
        memcpy(pvEStore, seed->pvE, padded_read_length * sizeof(int16_t));
        memcpy(pvHStore, seed->pvHStore, padded_read_length * sizeof(int16_t));
        
        // Unswizzle them so we can work on normal arrays.
        unswizzle_word(pvEStore, padded_read_length);
        unswizzle_word(pvHStore, padded_read_length);
    }

    /* Set external matrix pointers */
    alignment->mH = mH;
    alignment->mE = mE;
    alignment->mF = mF;

    /* Record that we have done a word-order alignment */
    alignment->is_byte = 0;
    
    int32_t begin = 0, end = refLen, step = 1;

    /* outer loop to process the reference sequence */
    if (ref_dir == 1) {
        begin = refLen - 1;
        end = -1;
        step = -1;
    }
    int32_t i;
    for (i = begin; LIKELY(i != end); i += step) {
        // For each column i in the DP matrices (running in the appropriate direction)
        
        // We don't need the previous F column.
        // But we do need to push the current H to the previous H and the next E to the current E.
        // We use a double buffering settup so we don't need to malloc and free and copy stuff.
        // By working with a previous and next column, we don't need to do anything fancy to support the flipable ref_dir.
        pv = pvHLoad;
        pvHLoad = pvHStore;
        pvHStore = pv;
        pv = pvELoad;
        pvELoad = pvEStore;
        pvEStore = pv;
        
        int32_t j;
        for (j = 0; j < readLen; j++) {
            // For each row j in the DP matrices (running top to bottom)
            
            // Gap in read (E) matrix is already set for this column
            
            int16_t refGapOpenScore = 0;
            int16_t refGapExtendScore = 0;
            if (j == 0) {
                // We can only end in a gap in the reference on the first read character with negative score.
                pvFStore[j] = subs_word(0, weight_gapO);
            } else {
                // Set the gap-in-ref matrix (F) based on previous slot in current F and in current H
                refGapOpenScore = subs_word(pvHStore[j-1], weight_gapO);
                refGapExtendScore = subs_word(pvFStore[j-1], weight_gapE);
                pvFStore[j] = max_word(refGapOpenScore, refGapExtendScore);
            }
            // Nothing negative is allowed in score matrices
            pvFStore[j] = max_word(pvFStore[j], 0);

            // What score are we adding to with a match?
            // If there's nowhere to come from it's 0.
            int16_t matchFrom = 0;
            if (j > 0) {
                // Otherwise its the score where we came from
                matchFrom = pvHLoad[j-1];
            }
            // What's the profile say about a match/mismatch here?
            int16_t profileScore = profile_get_word(vProfile, readLen, j, ref[i]);
            
            // Compute the match/mismatch score with saturating addition.
            int16_t matchScore = adds_word(matchFrom, profileScore);
            // No bias
            // We're working 16 bit with signed profile words)
            
            // Set the normal (H) matrix based on current E, current F, and match/mismatch score
            pvHStore[j] = max_word(max_word(pvELoad[j], pvFStore[j]), matchScore);
            // Nothing negative is allowed in score matrices
            pvHStore[j] = max_word(pvHStore[j], 0);

            // Set the next gap-in-read matrix (E) based on current E and current H
            int16_t readGapOpenScore = subs_word(pvHStore[j], weight_gapO);
            int16_t readGapExtendScore = subs_word(pvELoad[j], weight_gapE);
            pvEStore[j] = max_word(readGapOpenScore, readGapExtendScore);
            // Nothing negative is allowed in score matrices
            pvEStore[j] = max_word(pvEStore[j], 0);
        
            // Set the running max
            if (pvHStore[j] > max) {
                max = pvHStore[j];
                end_ref = i;
                end_read = j;
            }
            
            // Copy from columns to matrices
            mH[i * readLen + j] = pvHStore[j];
            mE[i * readLen + j] = pvELoad[j];
            mF[i * readLen + j] = pvFStore[j];
        }
    }
    
    // Reswizzle seeds
    swizzle_word(pvEStore, padded_read_length);
    swizzle_word(pvHStore, padded_read_length);
    
    // Save seeds.
    memcpy(alignment->seed.pvE,      pvEStore, padded_read_length * sizeof(int16_t));
    memcpy(alignment->seed.pvHStore, pvHStore, padded_read_length * sizeof(int16_t));

    // Clean up all the buffers
    free(pvHLoad);
    free(pvHStore);
    free(pvELoad);
    free(pvEStore);
    // No pvFLoad.
    free(pvFStore);

    /* Find the most possible 2nd best alignment. */
    // TODO: this only does the best alignment???
    gssw_alignment_end* bests = (gssw_alignment_end*) calloc(2, sizeof(gssw_alignment_end));
    bests[0].score = max;
    bests[0].ref = end_ref;
    bests[0].read = end_read;

    return bests;
}


gssw_alignment_end* gssw_sw_sse2_word (const int8_t* ref,
                                       int8_t ref_dir,    // 0: forward ref; 1: reverse ref
                                       int32_t refLen,
                                       int32_t readLen,
                                       const uint8_t weight_gapO, /* will be used as - */
                                       const uint8_t weight_gapE, /* will be used as - */
                                       __m128i* vProfile,
                                       uint16_t terminate,
                                       int32_t maskLen,
                                       gssw_align* alignment, /* to save seed and matrix */
                                       bool save_matrixes,  /* don't save the H, E, and F matrixes */
                                       const gssw_seed* seed) {     /* to seed the alignment */
    

    uint16_t max = 0;                             /* the max alignment score */
    int32_t end_read = readLen - 1;
    int32_t end_ref = 0; /* 1_based best alignment ending point; Initialized as isn't aligned - 0. */
    int32_t segLen = (readLen + 7) / 8; /* number of segment */

    /* Initialize buffers used in alignment */
    __m128i* pvHStore;
    __m128i* pvHLoad;
    __m128i* pvHmax;
    __m128i* pvE;
    // We have a couple extra arrays for logging columns
    __m128i* pvEStore;
    __m128i* pvFStore;
    uint16_t* mH = NULL; // used to save matrices for external traceback: overall best
    uint16_t* mE = NULL; // Read gap
    uint16_t* mF = NULL; // Ref gap
    /* Note use of aligned memory */

    if (!(!posix_memalign((void**)&pvHStore,     sizeof(__m128i), segLen*sizeof(__m128i)) &&
          !posix_memalign((void**)&pvHLoad,      sizeof(__m128i), segLen*sizeof(__m128i)) &&
          !posix_memalign((void**)&pvHmax,       sizeof(__m128i), segLen*sizeof(__m128i)) &&
          !posix_memalign((void**)&pvE,          sizeof(__m128i), segLen*sizeof(__m128i)) &&
          !posix_memalign((void**)&pvEStore,     sizeof(__m128i), segLen*sizeof(__m128i)) &&
          !posix_memalign((void**)&pvFStore,     sizeof(__m128i), segLen*sizeof(__m128i)) &&
          !posix_memalign((void**)&alignment->seed.pvE,      sizeof(__m128i), segLen*sizeof(__m128i)) &&
          !posix_memalign((void**)&alignment->seed.pvHStore, sizeof(__m128i), segLen*sizeof(__m128i)))) {
        fprintf(stderr, "error:[gssw] Could not allocate memory required for alignment buffers.\n");
        exit(1);
    }

    if (save_matrixes && !(!posix_memalign((void**)&mH,           sizeof(__m128i), segLen*refLen*sizeof(__m128i)) &&
                           !posix_memalign((void**)&mE,           sizeof(__m128i), segLen*refLen*sizeof(__m128i)) &&
                           !posix_memalign((void**)&mF,           sizeof(__m128i), segLen*refLen*sizeof(__m128i)))) {
        fprintf(stderr, "error:[gssw] Could not allocate memory required for alignment traceback matrixes.\n");
        exit(1);
    }

    /* Workaround: zero ourselves because we don't have an aligned calloc */
    memset(pvHStore,                 0, segLen*sizeof(__m128i));
    memset(pvHLoad,                  0, segLen*sizeof(__m128i));
    memset(pvHmax,                   0, segLen*sizeof(__m128i));
    memset(pvE,                      0, segLen*sizeof(__m128i));
    memset(pvEStore,                 0, segLen*sizeof(__m128i));
    memset(pvFStore,                 0, segLen*sizeof(__m128i));
    memset(alignment->seed.pvE,      0, segLen*sizeof(__m128i));
    memset(alignment->seed.pvHStore, 0, segLen*sizeof(__m128i));
    if (save_matrixes) {
        memset(mH,                       0, segLen*refLen*sizeof(__m128i));
        memset(mE,                       0, segLen*refLen*sizeof(__m128i));
        memset(mF,                       0, segLen*refLen*sizeof(__m128i));
    }

    /* if we are running a seeded alignment, copy over the seeds */
    if (seed) {
        memcpy(pvE, seed->pvE, segLen*sizeof(__m128i));
        memcpy(pvHStore, seed->pvHStore, segLen*sizeof(__m128i));
    }

    /* Set external matrix pointers */
    if (save_matrixes) {
        alignment->mH = mH;
        alignment->mE = mE;
        alignment->mF = mF;
    }

    /* Record that we have done a word-order alignment */
    alignment->is_byte = 0;

    /* Define 16 byte 0 vector. */
    __m128i vZero = _mm_set1_epi32(0);

    /* Used for iteration */
    int32_t i, j;

    /* 16 byte insertion begin vector */
    __m128i vGapO = _mm_set1_epi16(weight_gapO);

    /* 16 byte insertion extension vector */
    __m128i vGapE = _mm_set1_epi16(weight_gapE);

    __m128i vMaxScore = vZero; /* Trace the highest score of the whole SW matrix. */
    __m128i vMaxMark = vZero; /* Trace the highest score till the previous column. */
    __m128i vTemp;
    int32_t begin = 0, end = refLen, step = 1;

    /* outer loop to process the reference sequence */
    if (ref_dir == 1) {
        begin = refLen - 1;
        end = -1;
        step = -1;
    }
    for (i = begin; LIKELY(i != end); i += step) {
        int32_t cmp;
        __m128i e = vZero, vF = vZero; /* Initialize F value to 0.
                               Any errors to vH values will be corrected in the Lazy_F loop.
                             */
        __m128i vH = pvHStore[segLen - 1];
        vH = _mm_slli_si128 (vH, 2); /* Shift the 128-bit value in vH left by 2 byte. */

        __m128i vMaxColumn = vZero; /* vMaxColumn is used to record the max values of column i. */

        __m128i* vP = vProfile + ref[i] * segLen; /* Right part of the vProfile */
        
        /* Swap the 2 H buffers. */
        __m128i* pv = pvHLoad;
        pvHLoad = pvHStore;
        pvHStore = pv;

        /* inner loop to process the query sequence */
        for (j = 0; LIKELY(j < segLen); j ++) {
            vH = _mm_adds_epi16(vH, _mm_load_si128(vP + j));

            /* Get max from vH, vE and vF. */
            e = _mm_load_si128(pvE + j);
            vH = _mm_max_epi16(vH, e);
            vH = _mm_max_epi16(vH, vF);
            vMaxColumn = _mm_max_epi16(vMaxColumn, vH);

            /* Save vH values. */
            _mm_store_si128(pvHStore + j, vH);
            
            /* Save the vE and vF values they derived from */
            _mm_store_si128(pvEStore + j, e);
            _mm_store_si128(pvFStore + j, vF);

            /* Update vE value. */
            vH = _mm_subs_epu16(vH, vGapO); /* saturation arithmetic, result >= 0 */
            e = _mm_subs_epu16(e, vGapE);
            e = _mm_max_epi16(e, vH);
            _mm_store_si128(pvE + j, e);

            /* Update vF value. */
            vF = _mm_subs_epu16(vF, vGapE);
            vF = _mm_max_epi16(vF, vH);

            /* Load the next vH. */
            vH = _mm_load_si128(pvHLoad + j);
        }

        // Now we have the exact same lazy F loop as for bytes, but adapted.
        // No more using two algorithms.

        /* reset pointers to the start of the saved data */
        j = 0;
        vH = _mm_load_si128 (pvHStore + j);

        /*  
         * Wrap vF around from the end of each segment to the start of the next.
         */
        vF = _mm_slli_si128 (vF, 2);
        
        // Now we need to work out if we actually want to update anything. We
        // need to do an F loop if we would modify H, or if we would improve
        // over the old F.
        
        // If we beat the stored H
        vTemp = _mm_cmpgt_epi16 (vF, vH);
        cmp = _mm_movemask_epi8 (vTemp);
        // Or we beat the stored F
        vTemp = _mm_load_si128 (pvFStore + j);
        vTemp = _mm_cmpgt_epi16 (vF, vTemp);
        cmp |= _mm_movemask_epi8 (vTemp);
        while (cmp != 0x0000)
        {
            // Then we do the update
            
            // Update this stripe of the H matrix
            vH = _mm_max_epi16 (vH, vF);
            vMaxColumn = _mm_max_epi16(vMaxColumn, vH);
            _mm_store_si128 (pvHStore + j, vH);
            
            // Update the E matrix for the next column
            // Since we may have changed the H matrix
            // This is to allow a gap-to-gap transition in the alignment
            e = _mm_load_si128(pvE + j);
            // The H matrix can only get better, so the gap open scores can only
            // get better, so the E matrix can only get better too.
            vTemp = _mm_subs_epu16(vH, vGapO);
            e = _mm_max_epi16(e, vTemp);
            _mm_store_si128(pvE + j, e);
            // TODO: Instead of doing this, would it be smarter to just compute
            // the E matrix for each column when we're doing its H matrix? Or
            // would the extra buffer slow us down more than the extra compute?
            

            // Save the stripe of the F matrix
            // Only add in better F scores. Sometimes during this loop we'll
            // recompute worse ones.
            vTemp = _mm_load_si128 (pvFStore + j);
            vTemp = _mm_max_epi16 (vTemp, vF);
            _mm_store_si128(pvFStore + j, vTemp);

            // Then think about extending
            vF = _mm_subs_epu16 (vF, vGapE);
            // We never need to think about gap opens because nothing that came
            // from a gap open can ever change, because you won't close and then
            // immediately open a gap.

            j++;
            if (j >= segLen)
            {
                // Wrap around to the next segment again
                j = 0;
                vF = _mm_slli_si128 (vF, 2);
            }

            // Again compute if H or F needs updating based on this new set of F
            // values.
            vH = _mm_load_si128 (pvHStore + j);
            
            // See if we beat the stored H
            vTemp = _mm_cmpgt_epi16 (vF, vH);
            cmp = _mm_movemask_epi8 (vTemp);
            // Or if we beat the stored F
            vTemp = _mm_load_si128 (pvFStore + j);
            vTemp = _mm_cmpgt_epi16 (vF, vTemp);
            cmp |= _mm_movemask_epi8 (vTemp);
        }

        // Now H, E, and F are all up to date with downwards gap propagations.
        
        vMaxScore = _mm_max_epi16(vMaxScore, vMaxColumn);
        vTemp = _mm_cmpeq_epi16(vMaxMark, vMaxScore);
        cmp = _mm_movemask_epi8(vTemp);
        if (cmp != 0xffff) {
            uint16_t temp;
            vMaxMark = vMaxScore;
            m128i_max8(temp, vMaxScore);
            vMaxScore = vMaxMark;

            if (LIKELY(temp > max)) {
                max = temp;
                end_ref = i;
                for (j = 0; LIKELY(j < segLen); ++j) pvHmax[j] = pvHStore[j];
            }
        }

        /* save current column */
        if (save_matrixes) {
            // Do the un-swizzling of the stripes.
        
            // H matrix
            for (j = 0; LIKELY(j < segLen); ++j) {
                uint16_t* t;
                int32_t ti;
                vTemp = pvHStore[j];
                for (t = (uint16_t*)&vTemp, ti = 0; ti < 8; ++ti) {
                    //fprintf(stdout, "%d\t", *t++);
                    ((uint16_t*)mH)[i*readLen + ti*segLen + j] = *t++;
                }
                //fprintf(stdout, "\n");
            }
        
            // E matrix
            for (j = 0; LIKELY(j < segLen); ++j) {
                uint16_t* t;
                int32_t ti;
                vTemp = pvEStore[j];
                for (t = (uint16_t*)&vTemp, ti = 0; ti < 8; ++ti) {
                    //fprintf(stdout, "%d\t", *t++);
                    ((uint16_t*)mE)[i*readLen + ti*segLen + j] = *t++;
                }
                //fprintf(stdout, "\n");
            }
        
            // F matrix
            for (j = 0; LIKELY(j < segLen); ++j) {
                uint16_t* t;
                int32_t ti;
                vTemp = pvFStore[j];
                for (t = (uint16_t*)&vTemp, ti = 0; ti < 8; ++ti) {
                    //fprintf(stdout, "%d\t", *t++);
                    ((uint16_t*)mF)[i*readLen + ti*segLen + j] = *t++;
                }
                //fprintf(stdout, "\n");
            }
        }        

        /* Record the max score of current column. */
        //max8(maxColumn[i], vMaxColumn);
        //if (maxColumn[i] == terminate) break;

    }

    memcpy(alignment->seed.pvE,      pvE,      segLen*sizeof(__m128i));
    memcpy(alignment->seed.pvHStore, pvHStore, segLen*sizeof(__m128i));


    /* Trace the alignment ending position on read. */
    uint16_t *t = (uint16_t*)pvHmax;
    int32_t column_len = segLen * 8;
    for (i = 0; LIKELY(i < column_len); ++i, ++t) {
        int32_t temp;
        if (*t == max) {
            temp = i / 8 + i % 8 * segLen;
            if (temp < end_read) end_read = temp;
        }
    }

    free(pvE);
    free(pvHmax);
    free(pvHLoad);
    free(pvHStore);
    free(pvEStore);
    free(pvFStore);

    /* Find the most possible 2nd best alignment. */
    gssw_alignment_end* bests = (gssw_alignment_end*) calloc(2, sizeof(gssw_alignment_end));
    bests[0].score = max;
    bests[0].ref = end_ref;
    bests[0].read = end_read;

    return bests;
}

int8_t* gssw_seq_reverse(const int8_t* seq, int32_t end)    /* end is 0-based alignment ending position */
{
    int8_t* reverse = (int8_t*)calloc(end + 1, sizeof(int8_t));
    int32_t start = 0;
    while (LIKELY(start <= end)) {
        reverse[start] = seq[end];
        reverse[end] = seq[start];
        ++ start;
        -- end;
    }
    return reverse;
}

int8_t gssw_max_qual(const int8_t* qual, const int32_t len) {
    int8_t max_qual = -128;
    int32_t i;
    for (i = 0; i < len; i++) {
        if (qual[i] > max_qual) {
            max_qual = qual[i];
        }
    }
    return max_qual;
}

gssw_profile* gssw_init (const int8_t* read, const int32_t readLen, const int8_t* mat, const int32_t n,
                         int8_t start_full_length_bonus, int8_t end_full_length_bonus, const int8_t score_size) {
    gssw_profile* p = (gssw_profile*)calloc(1, sizeof(struct gssw_profile));
    p->profile_byte = 0;
    p->profile_word = 0;
    p->bias = 0;

    if (score_size == 0 || score_size == 2) {
        /* Find the bias to use in the substitution matrix */
        int32_t bias = 0, i;
        for (i = 0; i < n*n; i++) if (mat[i] < bias) bias = mat[i];
        bias = abs(bias);

        p->bias = bias;
        p->profile_byte = gssw_qP_byte (read, mat, readLen, n, bias, start_full_length_bonus, end_full_length_bonus);
    }
    if (score_size == 1 || score_size == 2) p->profile_word = gssw_qP_word (read, mat, readLen, n,
                                                                            start_full_length_bonus,
                                                                            end_full_length_bonus);
    p->read = read;
    p->mat = mat;
    p->readLen = readLen;
    p->n = n;
    return p;
}

/* Initiailize a profile with quality adjusted scores. */
gssw_profile* gssw_qual_adj_init (const int8_t* read, const int8_t* qual, const int32_t readLen, const int8_t* adj_mat,
                                  const int32_t n, int8_t start_full_length_bonus, int8_t end_full_length_bonus,
                                  const int8_t score_size) {
    
    gssw_profile* p = (gssw_profile*)calloc(1, sizeof(struct gssw_profile));
    p->profile_byte = 0;
    p->bias = 0;
    if (score_size == 0 || score_size == 2) {
        /* Find the bias to use in the substitution matrix */
        int32_t bias = 0, i;
        // only need to check highest quality matrix since scores shrink toward 0 with lower quality scores
        int32_t adj_mat_offset = n * n * gssw_max_qual(qual, readLen);
        for (i = 0; i < n*n; i++) if (adj_mat[adj_mat_offset + i] < bias) bias = adj_mat[adj_mat_offset + i];
        bias = abs(bias);
        
        p->bias = bias;
        p->profile_byte = gssw_adj_qP_byte (read, qual, adj_mat, readLen, n, bias, start_full_length_bonus,
                                            end_full_length_bonus);
    }
    if (score_size == 1 || score_size == 2) {
        p->profile_word = gssw_adj_qP_word(read, qual, adj_mat, readLen, n, start_full_length_bonus, end_full_length_bonus);
    }
    p->read = read;
    p->mat = adj_mat;
    p->readLen = readLen;
    p->n = n;
    return p;
}

void gssw_init_destroy (gssw_profile* p) {
    free(p->profile_byte);
    free(p->profile_word);
    free(p);
}

gssw_align* gssw_fill (const gssw_profile* prof,
                       const int8_t* ref,
                       const int32_t refLen,
                       const uint8_t weight_gapO,
                       const uint8_t weight_gapE,
                       const int32_t maskLen,
                       bool save_matrixes,
                       gssw_seed* seed) {

    gssw_alignment_end* bests = 0;
    int32_t readLen = prof->readLen;
    gssw_align* alignment = gssw_align_create();

    if (maskLen < 15) {
        fprintf(stderr, "When maskLen < 15, the function ssw_align doesn't return 2nd best alignment information.\n");
    }

    // Find the alignment scores and ending positions
    if (prof->profile_byte) {
        // Do a byte-sized fill
        
        if (gssw_sse2_enabled) {
            // Use SSE2
            bests = gssw_sw_sse2_byte(ref, 0, refLen, readLen, weight_gapO, weight_gapE,
                                      prof->profile_byte, -1, prof->bias, maskLen, alignment, save_matrixes, seed);
        } else {
            // Use software
            bests = gssw_sw_software_byte(ref, 0, refLen, readLen, weight_gapO, weight_gapE,
                                          prof->profile_byte, -1, prof->bias, maskLen, alignment, seed);
        }

        if (prof->profile_word && bests[0].score == 255) {
            free(bests);
            gssw_align_clear_matrix_and_seed(alignment);
            if (gssw_sse2_enabled) {
                // Use SSE2
                bests = gssw_sw_sse2_word(ref, 0, refLen, readLen, weight_gapO, weight_gapE, prof->profile_byte, -1, maskLen,
                                          alignment, save_matrixes, seed);
            } else {
                // Use software
                bests = gssw_sw_software_word(ref, 0, refLen, readLen, weight_gapO, weight_gapE, prof->profile_byte, -1, maskLen,
                                              alignment, seed);
            }
        } else if (bests[0].score == 255) {
            fprintf(stderr, "Please set 2 to the score_size parameter of the function ssw_init, otherwise the alignment results will be incorrect.\n");
            return 0;
        }
    } else if (prof->profile_word) {
        if (gssw_sse2_enabled) {
            // Use SSE2
            bests = gssw_sw_sse2_word(ref, 0, refLen, readLen, weight_gapO, weight_gapE, prof->profile_word, -1, maskLen,
                                      alignment, save_matrixes, seed);
        } else {
            // Use software
            bests = gssw_sw_software_word(ref, 0, refLen, readLen, weight_gapO, weight_gapE, prof->profile_word, -1, maskLen,
                                          alignment, seed);
        }
    } else {
        fprintf(stderr, "Please call the function ssw_init before ssw_align.\n");
        return 0;
    }
    
    
    
    alignment->score1 = bests[0].score;
    alignment->ref_end1 = bests[0].ref;
    alignment->read_end1 = bests[0].read;
    if (maskLen >= 15) {
        alignment->score2 = bests[1].score;
        alignment->ref_end2 = bests[1].ref;
    } else {
        alignment->score2 = 0;
        alignment->ref_end2 = -1;
    }
    free(bests);
    

    return alignment;
}

gssw_align* gssw_align_create (void) {
    gssw_align* a = (gssw_align*)calloc(1, sizeof(gssw_align));
    a->seed.pvHStore = NULL;
    a->seed.pvE = NULL;
    a->mH = NULL;
    a->mE = NULL;
    a->mF = NULL;
    a->ref_begin1 = -1;
    a->read_begin1 = -1;
    return a;
}

void gssw_align_destroy (gssw_align* a) {
    gssw_align_clear_matrix_and_seed(a);
    free(a);
}

void gssw_align_clear_matrix_and_seed (gssw_align* a) {
    free(a->mH);
    a->mH = NULL;
    free(a->mE);
    a->mE = NULL;
    free(a->mF);
    a->mF = NULL;
    free(a->seed.pvHStore);
    a->seed.pvHStore = NULL;
    free(a->seed.pvE);
    a->seed.pvE = NULL;
}

void gssw_print_score_matrix (const char* ref,
                              int32_t refLen,
                              const char* read,
                              int32_t readLen,
                              gssw_align* alignment,
                              FILE* out) {

    int32_t i, j;

    fprintf(out, "\t");
    for (i = 0; LIKELY(i < refLen); ++i) {
        fprintf(out, "%c\t\t", ref[i]);
    }
    fprintf(out, "\n");

    if (gssw_is_byte(alignment)) {
        uint8_t* mH = alignment->mH;
        uint8_t* mE = alignment->mE;
        uint8_t* mF = alignment->mF;
        for (j = 0; LIKELY(j < readLen); ++j) {
            fprintf(out, "%c\t", read[j]);
            for (i = 0; LIKELY(i < refLen); ++i) {
                fprintf(out, "(%u,%u) %u,%u,%u\t", i, j, mH[i*readLen + j],
                    mE[i*readLen + j], mF[i*readLen + j]);
            }
            fprintf(out, "\n");
        }
    } else {
        uint16_t* mH = alignment->mH;
        uint16_t* mE = alignment->mE;
        uint16_t* mF = alignment->mF;
        for (j = 0; LIKELY(j < readLen); ++j) {
            fprintf(out, "%c\t", read[j]);
            for (i = 0; LIKELY(i < refLen); ++i) {
                fprintf(out, "(%u,%u) %u,%u,%u\t", i, j, mH[i*readLen + j],
                    mE[i*readLen + j], mF[i*readLen + j]);
            }
            fprintf(out, "\n");
        }
    }

    fprintf(out, "\n");

}

void gssw_graph_print(gssw_graph* graph) {
    uint32_t i = 0, gs = graph->size;
    gssw_node** npp = graph->nodes;
    fprintf(stdout, "GRAPH digraph variants {\n");
    for (i=0; i<gs; ++i, ++npp) {
        gssw_node* n = *npp;
        fprintf(stdout, "GRAPH // node %llu %u %s\n", n->id, n->len, n->seq);
        uint32_t k;
        for (k=0; k<n->count_prev; ++k) {
            //fprintf(stdout, "GRAPH %u -> %u;\n", n->prev[k]->id, n->id);
            fprintf(stdout, "GRAPH \"%llu %s\" -> \"%llu %s\";\n", n->prev[k]->id, n->prev[k]->seq, n->id, n->seq);
        }
    }
    fprintf(stdout, "GRAPH }\n");
}

void gssw_graph_print_stderr(gssw_graph* graph) {
    uint32_t i = 0, gs = graph->size;
    gssw_node** npp = graph->nodes;
    fprintf(stderr, "GRAPH digraph variants {\n");
    for (i=0; i<gs; ++i, ++npp) {
        gssw_node* n = *npp;
        fprintf(stderr, "GRAPH // node %llu %u %s\n", n->id, n->len, n->seq);
        uint32_t k;
        for (k=0; k<n->count_prev; ++k) {
            //fprintf(stdout, "GRAPH %u -> %u;\n", n->prev[k]->id, n->id);
            fprintf(stderr, "GRAPH \"%llu %s\" -> \"%llu %s\";\n", n->prev[k]->id, n->prev[k]->seq, n->id, n->seq);
        }
    }
    fprintf(stderr, "GRAPH }\n");
}

void gssw_graph_print_score_matrices(gssw_graph* graph, const char* read, int32_t readLen, FILE* out) {
    uint32_t i = 0, gs = graph->size;
    gssw_node** npp = graph->nodes;
    for (i=0; i<gs; ++i, ++npp) {
        gssw_node* n = *npp;
        fprintf(out, "node %llu\n", n->id);
        gssw_print_score_matrix(n->seq, n->len, read, readLen, n->alignment, out);
    }
}

inline int gssw_is_byte (gssw_align* alignment) {
    if (alignment->is_byte) {
        return 1;
    } else {
        return 0;
    }
}

gssw_cigar* gssw_alignment_trace_back (gssw_node* node,
                                       gssw_multi_align_stack* alt_alignment_stack,
                                       gssw_alternate_alignment_ends* alignment_deflections,
                                       int32_t* deflection_idx,
                                       int32_t final_traceback,
                                       int32_t find_internal_node_alts,
                                       uint16_t* score,
                                       int32_t* refEnd,
                                       int32_t* readEnd,
                                       int32_t* refGapFlag,
                                       int32_t* readGapFlag,
                                       const char* ref,
                                       int32_t refLen,
                                       const char* read,
                                       int8_t* qual_num,
                                       int32_t readLen,
                                       int8_t* nt_table,
                                       int8_t* score_matrix,
                                       uint8_t gap_open,
                                       uint8_t gap_extension,
                                       int8_t start_full_length_bonus,
                                       int8_t end_full_length_bonus) {
    if (LIKELY(gssw_is_byte(node->alignment))) {
        return gssw_alignment_trace_back_byte(node,
                                              alt_alignment_stack,
                                              alignment_deflections,
                                              deflection_idx,
                                              final_traceback,
                                              find_internal_node_alts,
                                              score,
                                              refEnd,
                                              readEnd,
                                              refGapFlag,
                                              readGapFlag,
                                              ref,
                                              refLen,
                                              read,
                                              qual_num,
                                              readLen,
                                              nt_table,
                                              score_matrix,
                                              gap_open,
                                              gap_extension,
                                              start_full_length_bonus,
                                              end_full_length_bonus);
    } else {
        return gssw_alignment_trace_back_word(node,
                                              alt_alignment_stack,
                                              alignment_deflections,
                                              deflection_idx,
                                              final_traceback,
                                              find_internal_node_alts,
                                              score,
                                              refEnd,
                                              readEnd,
                                              refGapFlag,
                                              readGapFlag,
                                              ref,
                                              refLen,
                                              read,
                                              qual_num,
                                              readLen,
                                              nt_table,
                                              score_matrix,
                                              gap_open,
                                              gap_extension,
                                              start_full_length_bonus,
                                              end_full_length_bonus);
    }
}

gssw_cigar* gssw_alignment_trace_back_byte (gssw_node* node,
                                            gssw_multi_align_stack* alt_alignment_stack,
                                            gssw_alternate_alignment_ends* alignment_deflections,
                                            int32_t* deflection_idx,
                                            int32_t final_traceback,
                                            int32_t find_internal_node_alts,
                                            uint16_t* score,
                                            int32_t* refEnd,
                                            int32_t* readEnd,
                                            int32_t* refGapFlag,
                                            int32_t* readGapFlag,
                                            const char* ref,
                                            int32_t refLen,
                                            const char* read,
                                            int8_t* qual_num,
                                            int32_t readLen,
                                            int8_t* nt_table,
                                            int8_t* score_matrix,
                                            uint8_t gap_open,
                                            uint8_t gap_extension,
                                            int8_t start_full_length_bonus,
                                            int8_t end_full_length_bonus) {

    gssw_align* alignment = node->alignment;
    
    // This is the alignment matrix.
    uint8_t* mH = (uint8_t*)alignment->mH;
    // And the two matrices for gaps
    uint8_t* mE = (uint8_t*)alignment->mE;
    uint8_t* mF = (uint8_t*)alignment->mF;
    
    // i, j are where we are currently in the reference and the read
    int32_t i = *refEnd;
    int32_t j = *readEnd;
    
    // These let us know if we're currently supposed to be in a gap, waiting for
    // the gap open that ought to open it.
    int32_t gRead = *readGapFlag; // If set we're in E actually
    int32_t gRef = *refGapFlag; // If set we're in F actually
    
    // This here variable holds the score that the current cell got from the DP
    // step. We'll look at the surrounding cells to work out which way we came
    // from to get this score here. We'll also update it in case we're the last
    // thing on this node and we take it along as the score to be at on the
    // other node (??)
    
#ifdef DEBUG_TRACEBACK
    fprintf(stderr, "traceback through byte matrices\n");
    int l, k;
    fprintf(stderr, "mH\n");
    fprintf(stderr, "\t");
    for (k = 0; k < readLen; k++) {
        fprintf(stderr, "%c\t", read[k]);
    }
    fprintf(stderr, "\n");
    for (l = 0; l < node->len; l++) {
        fprintf(stderr, "%c\t", ref[l]);
        for (k = 0; k < readLen; k++) {
            fprintf(stderr, "%d\t", mH[readLen*l + k]);
        }
        fprintf(stderr, "\n");
    }
    fprintf(stderr, "mE\n");
    fprintf(stderr, "\t");
    for (k = 0; k < readLen; k++) {
        fprintf(stderr, "%c\t", read[k]);
    }
    fprintf(stderr, "\n");
    for (l = 0; l < node->len; l++) {
        fprintf(stderr, "%c\t", ref[l]);
        for (k = 0; k < readLen; k++) {
            fprintf(stderr, "%d\t", mE[readLen*l + k]);
        }
        fprintf(stderr, "\n");
    }
    fprintf(stderr, "mF\n");
    fprintf(stderr, "\t");
    for (k = 0; k < readLen; k++) {
        fprintf(stderr, "%c\t", read[k]);
    }
    fprintf(stderr, "\n");
    for (l = 0; l < node->len; l++) {
        fprintf(stderr, "%c\t", ref[l]);
        for (k = 0; k < readLen; k++) {
            fprintf(stderr, "%d\t", mF[readLen*l + k]);
        }
        fprintf(stderr, "\n");
    }
#endif
    
    uint16_t scoreHere;
    if(gRead) {
        // we're in mE
        scoreHere = mE[readLen*i + j];
    } else if(gRef) {
        // We're in mF
        scoreHere = mF[readLen*i + j];
    } else {
        // We're in the main matrix mH
        scoreHere = mH[readLen*i + j];
    }
    
    // Start a CIGAR to hold the traceback.
    gssw_cigar* result = (gssw_cigar*)calloc(1, sizeof(gssw_cigar));
    result->length = 0;

    while (LIKELY(scoreHere > 0 && i >= 0 && j >= 0)) {
        // We're not out of score, and we're not off the left or top of the matrix
        
        // Are there more deflections left in this traceback?
        if (*deflection_idx < alignment_deflections->num_deflections) {
            gssw_trace_back_deflection* next_deflxn = &alignment_deflections->deflections[*deflection_idx];
            // Is the deflection here?
            gssw_matrix_t curr_matrix = gRead ? ReadGap : (gRef ? RefGap : Match);
            if (UNLIKELY(i == next_deflxn->ref_pos && j == next_deflxn->read_pos
                         && node == next_deflxn->from_node && node == next_deflxn->to_node
                         && curr_matrix == next_deflxn->from_matrix)) {
#ifdef DEBUG_TRACEBACK
                fprintf(stderr, "taking deflection from i = %d, j = %d in node %llu\n", i, j, node->id);
#endif
                // take the deflection instead of doing traceback this iteration
                if (gRead) {
                    i--;
                    gssw_cigar_push_back(result, 'D', 1);
                    switch (next_deflxn->to_matrix) {
                            
                        case Match:
#ifdef DEBUG_TRACEBACK
                            fprintf(stderr, "Deflection is read gap -> match\n");
#endif
                            scoreHere = mH[readLen*i + j];
                            gRead = 0;
                            break;
                            
                        case ReadGap:
#ifdef DEBUG_TRACEBACK
                            fprintf(stderr, "Deflection is read gap -> read gap\n");
#endif
                            scoreHere = mE[readLen*i + j];
                            break;
                            
                        default:
                            fprintf(stderr, "error:[gssw] Impossible alternate alignment deflection from read gap\n");
                            assert(0);
                            break;
                    }
                }
                else if (gRef) {
                    j--;
                    gssw_cigar_push_back(result, 'I', 1);
                    switch (next_deflxn->to_matrix) {
                        case Match:
#ifdef DEBUG_TRACEBACK
                            fprintf(stderr, "Deflection is ref gap -> match\n");
#endif
                            scoreHere = mH[readLen*i + j];
                            gRef = 0;
                            break;
                            
                        case RefGap:
#ifdef DEBUG_TRACEBACK
                            fprintf(stderr, "Deflection is ref gap -> ref gap\n");
#endif
                            scoreHere = mF[readLen*i + j];
                            break;
                            
                        default:
                            fprintf(stderr, "error:[gssw] Impossible alternate alignment deflection from reference gap\n");
                            assert(0);
                            break;
                    }
                }
                else {
                    if (next_deflxn->to_node != node) {
                        // This is a deflection from match to match, but it crosses a node boundary
                        // so we leave the deflection in place and exit to the POA traceback
                        break;
                    }
                    switch (next_deflxn->to_matrix) {
                        case Match:
#ifdef DEBUG_TRACEBACK
                            fprintf(stderr, "Deflection is match -> match\n");
#endif
                            if (ref[i] == 'N' || read[j] == 'N') {
                                gssw_cigar_push_back(result, 'N', 1);
                            }
                            else if(ref[i] == read[j]) {
                                gssw_cigar_push_back(result, 'M', 1);
                            }
                            else if (ref[i] != read[j]) {
                                gssw_cigar_push_back(result, 'X', 1);
                            }
                            i--;
                            j--;
                            scoreHere = mH[readLen*i + j];
                            break;
                            
                        case ReadGap:
#ifdef DEBUG_TRACEBACK
                            fprintf(stderr, "Deflection is match -> read gap\n");
#endif
                            scoreHere = mE[readLen*i + j];
                            gRead = 1;
                            break;
                            
                        case RefGap:
#ifdef DEBUG_TRACEBACK
                            fprintf(stderr, "Deflection is match -> ref gap\n");
#endif
                            scoreHere = mF[readLen*i + j];
                            gRef = 1;
                            break;
                            
                        default:
                            fprintf(stderr, "error:[gssw] Unrecognized matrix type\n");
                            assert(0);
                            break;
                    }
                }
                (*deflection_idx)++;
                continue;
            }
        }
    
#ifdef DEBUG_TRACEBACK
        fprintf(stderr, "score=%i at %i,%i with %c vs %c, gRef=%i gRead=%i\n", scoreHere, i, j, ref[i], read[j], gRef, gRead);
        fprintf(stderr, "mH[%d,%d] = %d, mE[%d,%d] = %d, mF[%d,%d] = %d\n", i, j, mH[readLen*i + j], i, j, mE[readLen*i + j], i, j, mF[readLen*i + j]);
#endif
        
        int32_t found_trace = 0;
        uint16_t source_score;
        uint16_t score_diff;
        int32_t next_i = i;
        int32_t next_j = j;
        int32_t next_g_read = gRead;
        int32_t next_g_ref = gRef;
        uint16_t next_score_here = scoreHere;
    
        if(gRead) {
            // We're in E
            
            // If we're in a gap matrix, see if we can leave the gap here (i.e.
            // gap open score is consistent). If so, do it. Otherwise, extend
            // the gap.
            if (i > 0) {
                source_score = mH[readLen*(i - 1) +  j];
                score_diff = scoreHere - (source_score - gap_open);
#ifdef DEBUG_TRACEBACK
                fprintf(stderr, "score diff from read gap to match: %d\n", score_diff);
#endif
                if (score_diff == 0 && !found_trace) {
                    found_trace = 1;
                    // We are consistent with a gap open, bringing us back to the
                    // main matrix. Take it.
                    
                    // D = gap in read
                    gssw_cigar_push_back(result, 'D', 1);
                    
                    // Fix score
                    next_score_here += gap_open;
                    
                    // Move in the reference
                    next_i--;
                    
                    // Go back to the main matrix
                    next_g_read = 0;
                    
#ifdef DEBUG_TRACEBACK
                    fprintf(stderr, "Read gap open\n");
#endif
                    // if this is the last alignment we do not need to look for suboptimal scores
                    if (final_traceback) {
                        i = next_i;
                        j = next_j;
                        gRef = next_g_ref;
                        gRead = next_g_read;
                        scoreHere = next_score_here;
                        continue;
                    }
 
                }
                else if (UNLIKELY(*deflection_idx == alignment_deflections->num_deflections &&
                                  score_diff < alignment_deflections->score &&
                                  source_score > 0 && find_internal_node_alts)) {
                    // score is suboptimal or we have already chosen an optimal trace and the alternate alignment
                    // does not involve any negative or 0 scores (these are not actually extensions of a local alignment)
                    uint16_t alt_score = alignment_deflections->score - score_diff;
#ifdef DEBUG_TRACEBACK
                    fprintf(stderr, "Considering alternate alignment read gap -> match with score %d\n", alt_score);
#endif
                    if (alt_score > gssw_min_alt_alignment_score(alt_alignment_stack)
                        || alt_alignment_stack->current_size < alt_alignment_stack->capacity) {
                        gssw_add_alignment(alt_alignment_stack, alignment_deflections, alt_score,
                                           j, i, node, node, ReadGap, Match);
                    }
                }
                
                source_score = mE[readLen*(i - 1) +  j];
                score_diff = scoreHere - (source_score - gap_extension);
#ifdef DEBUG_TRACEBACK
                fprintf(stderr, "score diff from read gap to read gap: %d\n", score_diff);
#endif
                if (score_diff == 0 && !found_trace) {
                    found_trace = 1;
                    // We are consistent with a gap extend
                    gssw_cigar_push_back(result, 'D', 1);
                    next_score_here += gap_extension;
                    next_i--;
#ifdef DEBUG_TRACEBACK
                    fprintf(stderr, "Read gap extend\n");
#endif
                    // if this is the last alignment we do not need to look for suboptimal scores
                    if (final_traceback) {
                        i = next_i;
                        j = next_j;
                        gRef = next_g_ref;
                        gRead = next_g_read;
                        scoreHere = next_score_here;
                        continue;
                    }
                }
                else if (UNLIKELY(*deflection_idx == alignment_deflections->num_deflections &&
                                  score_diff < alignment_deflections->score &&
                                  source_score > 0 && find_internal_node_alts)) {
                    // score is suboptimal or we have already chosen an optimal trace and the alternate alignment
                    // does not involve any negative or 0 scores (these are not actually extensions of a local alignment)
                    uint16_t alt_score = alignment_deflections->score - score_diff;
#ifdef DEBUG_TRACEBACK
                    fprintf(stderr, "Considering alternate alignment read gap -> read gap with score %d\n", alt_score);
#endif
                    if (alt_score > gssw_min_alt_alignment_score(alt_alignment_stack)
                        || alt_alignment_stack->current_size < alt_alignment_stack->capacity) {
                        gssw_add_alignment(alt_alignment_stack, alignment_deflections, alt_score,
                                           j, i, node, node, ReadGap, ReadGap);
                    }
                }
            }
            else if(i == 0) {
                // We are in a gap and at the very left edge. We need to look
                // left from here and trace into our previous node in order to
                // figure out if this is an open or an extend or what.
#ifdef DEBUG_TRACEBACK
                fprintf(stderr, "Read gap out left edge\n");
#endif
                break;
            }
            else {
                // Something has gone wrong. We're in this matrix but don't have
                // an open or an extend and can't leave left.
                fprintf(stderr, "error:[gssw] Stuck in read gap!\n");
                assert(0);
            }
        }
        else if(gRef) {
            // We're in F
            if(j > 0) {
                
                source_score = mH[readLen*i +  (j - 1)];
                score_diff = scoreHere - (source_score - gap_open);
#ifdef DEBUG_TRACEBACK
                fprintf(stderr, "score diff from ref gap to match: %d\n", score_diff);
#endif
                if (score_diff == 0 && !found_trace) {
                    found_trace = 1;
                    // We are consistent with a gap open, bringing us back to the
                    // main matrix. Take it.
                    
                    // I = gap in ref
                    gssw_cigar_push_back(result, 'I', 1);
                    
                    // Fix score
                    next_score_here += gap_open;
                    
                    // Move in the read
                    next_j--;
                    
                    // Go back to the main matrix
                    next_g_ref = 0;
#ifdef DEBUG_TRACEBACK
                    fprintf(stderr, "Ref gap open\n");
#endif
                    // if this is the last alignment we do not need to look for suboptimal scores
                    if (final_traceback) {
                        i = next_i;
                        j = next_j;
                        gRef = next_g_ref;
                        gRead = next_g_read;
                        scoreHere = next_score_here;
                        continue;
                    }
                }
                else if (UNLIKELY(*deflection_idx == alignment_deflections->num_deflections &&
                                  score_diff < alignment_deflections->score &&
                                  source_score > 0 && find_internal_node_alts)) {
                    // score is suboptimal or we have already chosen an optimal trace and the alternate alignment
                    // does not involve any negative or 0 scores (these are not actually extensions of a local alignment)
                    uint16_t alt_score = alignment_deflections->score - score_diff;
#ifdef DEBUG_TRACEBACK
                    fprintf(stderr, "Considering alternate alignment ref gap -> match with score %d\n", alt_score);
#endif
                    if (alt_score > gssw_min_alt_alignment_score(alt_alignment_stack)
                        || alt_alignment_stack->current_size < alt_alignment_stack->capacity) {
                        gssw_add_alignment(alt_alignment_stack, alignment_deflections, alt_score,
                                           j, i, node, node, RefGap, Match);
                    }
                }
                
                source_score = mF[readLen*i + (j - 1)];
                score_diff = scoreHere - (source_score - gap_extension);
#ifdef DEBUG_TRACEBACK
                fprintf(stderr, "score diff from ref gap to ref gap: %d\n", score_diff);
#endif
                if (score_diff == 0 && !found_trace) {
                    found_trace = 1;
                    // We are consistent with a gap extend
                    gssw_cigar_push_back(result, 'I', 1);
                    next_score_here += gap_extension;
                    next_j--;
#ifdef DEBUG_TRACEBACK
                    fprintf(stderr, "Ref gap extend\n");
#endif
                    if (final_traceback) {
                        i = next_i;
                        j = next_j;
                        gRef = next_g_ref;
                        gRead = next_g_read;
                        scoreHere = next_score_here;
                        continue;
                    }
                }
                else if (UNLIKELY(*deflection_idx == alignment_deflections->num_deflections &&
                                  score_diff < alignment_deflections->score &&
                                  source_score > 0 && find_internal_node_alts)) {
                    // score is suboptimal or we have already chosen an optimal trace and the alternate alignment
                    // does not involve any negative or 0 scores (these are not actually extensions of a local alignment)
                    uint16_t alt_score = alignment_deflections->score - score_diff;
#ifdef DEBUG_TRACEBACK
                    fprintf(stderr, "Considering alternate alignment ref gap -> ref gap with score %d from %d below traceback cell %d\n", alt_score, score_diff, scoreHere);
#endif
                    if (alt_score > gssw_min_alt_alignment_score(alt_alignment_stack)
                        || alt_alignment_stack->current_size < alt_alignment_stack->capacity) {
                        gssw_add_alignment(alt_alignment_stack, alignment_deflections, alt_score,
                                           j, i, node, node, RefGap, RefGap);
                    }
                }
            }
            else if (j == 0) {
                // We have hit the end of the read in a gap, which should be
                // impossible because there's nowhere to open from.
                fprintf(stderr, "error:[gssw] Ref gap hit edge!\n");
                assert(0);
            }
            else {
                // Something has gone wrong. We're in this matrix but don't have
                // an open or an extend and can't leave left.
                fprintf(stderr, "error:[gssw] Ref gap stuck!\n");
                assert(0);
            }
        }
        else {
            // We're in H
            
            // There may be alternate alignments to other nodes that we would take in the graph traceback function
            // If so, bail out here early to avoid taking the main alignment
            if (UNLIKELY(*deflection_idx < alignment_deflections->num_deflections && i == 0)) {
                gssw_trace_back_deflection* next_deflxn = &alignment_deflections->deflections[*deflection_idx];
                // Is the deflection here?
                if (UNLIKELY(i == next_deflxn->ref_pos && j == next_deflxn->read_pos
                             && node == next_deflxn->from_node && node != next_deflxn->to_node
                             && next_deflxn->from_matrix == Match)) {
#ifdef DEBUG_TRACEBACK
                    fprintf(stderr, "Breaking out of node traceback to take a graph traceback deflection\n");
#endif
                    break;
                }
            }
            
            // If we're in the main matrix, see if we can do a match, mismatch,
            // or N-match. If so, do it.
            
            int8_t align_score;
            if (qual_num) {
                align_score = score_matrix[qual_num[j] * 25 + nt_table[(uint8_t) ref[i]] * 5 + nt_table[(uint8_t) read[j]]];
            }
            else {
                align_score = score_matrix[nt_table[(uint8_t) ref[i]] * 5 + nt_table[(uint8_t) read[j]]];
            }
            
            // Full length left alignment bonus if we're matching the first position
            if (j == 0) {
                align_score += start_full_length_bonus;
            }
            // And full length right alignment bonus if we're matching the last position
            if (j == readLen - 1) {
                align_score += end_full_length_bonus;
            }
            
            if (i > 0 && j > 0) {
                
                source_score = mH[readLen*(i-1) + (j-1)];
                score_diff = scoreHere - (source_score + align_score);
#ifdef DEBUG_TRACEBACK
                fprintf(stderr, "score diff from match to match: %d\n", score_diff);
#endif
                if (score_diff == 0 && !found_trace) {
                    found_trace = 1;
                    
                    if (ref[i] == 'N' || read[j] == 'N') {
                        // This is an N-match.
                        gssw_cigar_push_back(result, 'N', 1);
                        // Leave score unchanged
#ifdef DEBUG_TRACEBACK
                        fprintf(stderr, "N-match\n");
#endif
                    }
                    else if(ref[i] == read[j]) {
                        // This is a match
                        gssw_cigar_push_back(result, 'M', 1);
#ifdef DEBUG_TRACEBACK
                        fprintf(stderr, "Match\n");
#endif
                    }
                    else if (ref[i] != read[j]) {
                        // This is a mismatch
                        gssw_cigar_push_back(result, 'X', 1);
#ifdef DEBUG_TRACEBACK
                        fprintf(stderr, "Mismatch\n");
#endif
                    }
                    
                    next_score_here -= align_score;
                    next_j--;
                    next_i--;
                    
                    // if this is the last alignment we do not need to look for suboptimal scores
                    if (final_traceback) {
                        i = next_i;
                        j = next_j;
                        gRef = next_g_ref;
                        gRead = next_g_read;
                        scoreHere = next_score_here;
                        continue;
                    }
                }
                else if (UNLIKELY(*deflection_idx == alignment_deflections->num_deflections &&
                                  score_diff < alignment_deflections->score &&
                                  source_score > 0 && find_internal_node_alts)) {
                    // score is suboptimal or we have already chosen an optimal trace and the alternate alignment
                    // does not involve any negative or 0 scores (these are not actually extensions of a local alignment)
                    uint16_t alt_score = alignment_deflections->score - score_diff;
#ifdef DEBUG_TRACEBACK
                    fprintf(stderr, "Considering alternate alignment match -> match with score %d\n", alt_score);
#endif
                    if (alt_score > gssw_min_alt_alignment_score(alt_alignment_stack)
                        || alt_alignment_stack->current_size < alt_alignment_stack->capacity) {
                        gssw_add_alignment(alt_alignment_stack, alignment_deflections, alt_score,
                                           j, i, node, node, Match, Match);
                    }
                }
                
            }
            else if (j == 0) {
                score_diff = scoreHere - align_score;
#ifdef DEBUG_TRACEBACK
                fprintf(stderr, "score diff from match to outside matrix: %d from score here %d and align score %d compared to alignment score %d\n", score_diff, scoreHere, align_score, alignment_deflections->score);
#endif
                if (score_diff == 0 && !found_trace) {
                    found_trace = 1;
                    if (ref[i] == 'N' || read[j] == 'N') {
                        // This is an N-match.
                        gssw_cigar_push_back(result, 'N', 1);
#ifdef DEBUG_TRACEBACK
                        fprintf(stderr, "Alignment start N-match, ref = %c, read = %c\n", ref[i], read[j]);
#endif
                    }
                    else if (ref[i] == read[j]) {
                        // This is a match
                        gssw_cigar_push_back(result, 'M', 1);
                        
#ifdef DEBUG_TRACEBACK
                        fprintf(stderr, "Alignment start match, ref = %c, read = %c\n", ref[i], read[j]);
#endif
                    }
                    else {
                        // This is a mismatch (possible with pinning bonus)
                        gssw_cigar_push_back(result, 'X', 1);
                        
#ifdef DEBUG_TRACEBACK
                        fprintf(stderr, "Alignment start mismatch, ref = %c, read = %c\n", ref[i], read[j]);
#endif
                    }
                    next_j--;
                    next_i--;
                    next_score_here -= align_score;
                    
                    // if this is the last alignment we do not need to look for suboptimal scores
                    if (final_traceback) {
                        i = next_i;
                        j = next_j;
                        gRef = next_g_ref;
                        gRead = next_g_read;
                        scoreHere = next_score_here;
                        continue;
                    }
                }
                else if (UNLIKELY(*deflection_idx == alignment_deflections->num_deflections &&
                                  score_diff < alignment_deflections->score &&
                                  score_diff == 0 && find_internal_node_alts)) {
                    // score is suboptimal or we have already chosen an optimal trace and the alternate alignment
                    // does not involve any negative or 0 scores (these are not actually extensions of a local alignment)
                    // also this alignment is only valid as a start if the implicit source score is 0
                    uint16_t alt_score = alignment_deflections->score - score_diff;
#ifdef DEBUG_TRACEBACK
                    fprintf(stderr, "Considering alternate alignment match -> start with score %d\n", alt_score);
#endif
                    if (alt_score > gssw_min_alt_alignment_score(alt_alignment_stack)
                        || alt_alignment_stack->current_size < alt_alignment_stack->capacity) {
                        gssw_add_alignment(alt_alignment_stack, alignment_deflections, alt_score,
                                           j, i, node, node, Match, Match);
                    }
                }
            }
            
            source_score = mF[readLen*i + j];
            score_diff = scoreHere - source_score;
            
#ifdef DEBUG_TRACEBACK
            fprintf(stderr, "score diff from match to ref gap: %d from score here %d and gap close %d\n", score_diff, scoreHere, mF[readLen*i + j]);
#endif
            if (score_diff == 0 && !found_trace) {
                found_trace = 1;
                // We can't do anything diagonal. But we can become a ref gap, because there is more read.
                next_g_ref = 1;
                //fprintf(stderr, "Ref gap close\n");
                if (final_traceback) {
                    i = next_i;
                    j = next_j;
                    gRef = next_g_ref;
                    gRead = next_g_read;
                    scoreHere = next_score_here;
                    continue;
                }
            }
            else if (UNLIKELY(*deflection_idx == alignment_deflections->num_deflections &&
                              score_diff < alignment_deflections->score &&
                              source_score > 0 && find_internal_node_alts)) {
                // score is suboptimal or we have already chosen an optimal trace and the alternate alignment
                // does not involve any negative or 0 scores (these are not actually extensions of a local alignment)
                uint16_t alt_score = alignment_deflections->score - score_diff;
#ifdef DEBUG_TRACEBACK
                fprintf(stderr, "Considering alternate alignment match -> ref gap: score diff %d from score here %d for alt score %d and compared to alignment score %d\n", score_diff, scoreHere, alt_score, alignment_deflections->score);
#endif
                if (alt_score > gssw_min_alt_alignment_score(alt_alignment_stack)
                    || alt_alignment_stack->current_size < alt_alignment_stack->capacity) {
                    gssw_add_alignment(alt_alignment_stack, alignment_deflections, alt_score,
                                       j, i, node, node, Match, RefGap);
                }
            }
            
            source_score = mE[readLen*i + j];
            score_diff = scoreHere - source_score;
#ifdef DEBUG_TRACEBACK
            fprintf(stderr, "score diff from match to read gap: %d\n", score_diff);
#endif
            if (score_diff == 0 && !found_trace) {
                found_trace = 1;
                // We assume there's always more ref off to the left
                next_g_read = 1;
#ifdef DEBUG_TRACEBACK
                fprintf(stderr, "Read gap close\n");
#endif
                if (final_traceback) {
                    i = next_i;
                    j = next_j;
                    gRef = next_g_ref;
                    gRead = next_g_read;
                    scoreHere = next_score_here;
                    continue;
                }
                else if (i == 0 && j > 0) {
#ifdef DEBUG_TRACEBACK
                    fprintf(stderr, "Also checking for matches across node boundary\n");
#endif
                    // We didn't check the alternate tracebacks that cross the node boundary
                    // from the match matrix, and now we're entering the read gap matrix, so we
                    // need to check these alternate tracebacks now because the POA function will
                    // not know that we were at the node boundary in the match matrix
                    
                    int k;
                    for (k = 0; k < node->count_prev; k++) {
                        gssw_node* prev_node = node->prev[k];
                        source_score = ((uint8_t*) prev_node->alignment->mH)[readLen * (prev_node->len - 1) + j - 1];
                        score_diff = scoreHere - (source_score + align_score);
                        
                        if (UNLIKELY(*deflection_idx == alignment_deflections->num_deflections &&
                                     score_diff < alignment_deflections->score &&
                                     source_score > 0)) {
                            // score is suboptimal or we have already chosen an optimal trace and the alternate alignment
                            // does not involve any negative or 0 scores (these are not actually extensions of a local alignment)
                            uint16_t alt_score = alignment_deflections->score - score_diff;
#ifdef DEBUG_TRACEBACK
                            fprintf(stderr, "Considering alternate alignment across node boundary match -> match with score %d vs current minimum %d and size %d of capacity %d\n", alt_score, gssw_min_alt_alignment_score(alt_alignment_stack), alt_alignment_stack->current_size, alt_alignment_stack->capacity);
#endif
                            if (alt_score > gssw_min_alt_alignment_score(alt_alignment_stack)
                                || alt_alignment_stack->current_size < alt_alignment_stack->capacity) {
                                gssw_add_alignment(alt_alignment_stack, alignment_deflections, alt_score,
                                                   j, i, node, prev_node, Match, Match);
                            }
                        }
                    }
                }
            }
            else {
                if (UNLIKELY(*deflection_idx == alignment_deflections->num_deflections &&
                             score_diff < alignment_deflections->score &&
                             source_score > 0 && find_internal_node_alts)) {
                    // score is suboptimal or we have already chosen an optimal trace and the alternate alignment
                    // does not involve any negative or 0 scores (these are not actually extensions of a local alignment)
                    uint16_t alt_score = alignment_deflections->score - score_diff;
#ifdef DEBUG_TRACEBACK
                    fprintf(stderr, "Considering alternate alignment match -> read gap with score %d vs current minimum %d and size %d of capacity %d\n", alt_score, gssw_min_alt_alignment_score(alt_alignment_stack), alt_alignment_stack->current_size, alt_alignment_stack->capacity);
#endif
                    if (alt_score > gssw_min_alt_alignment_score(alt_alignment_stack)
                        || alt_alignment_stack->current_size < alt_alignment_stack->capacity) {
                        gssw_add_alignment(alt_alignment_stack, alignment_deflections, alt_score,
                                           j, i, node, node, Match, ReadGap);
                    }
                }
            }
            
            if (i == 0 && !found_trace) {
                // We can't go anywhere, but we're at the left edge, so maybe we
                // can go to a previous node diagonally. Just let it slide.
#ifdef DEBUG_TRACEBACK
                fprintf(stderr, "Match/mismatch out left edge\n");
#endif
                break;
            }
            
            if (!found_trace) {
                // We're in H and can't go anywhere.
                fprintf(stderr, "error:[gssw] Stuck in main matrix!\n");
                assert(0);
            }
        }
        
        i = next_i;
        j = next_j;
        gRef = next_g_ref;
        gRead = next_g_read;
        scoreHere = next_score_here;
    }
    
    *score = scoreHere;
    *refEnd = i;
    *readEnd = j;
    *refGapFlag = gRef;
    *readGapFlag = gRead;
    gssw_reverse_cigar(result);

    return result;
}


// copy of the above but for 16 bit ints
// sometimes there are good reasons for C++'s templates... sigh

gssw_cigar* gssw_alignment_trace_back_word (gssw_node* node,
                                            gssw_multi_align_stack* alt_alignment_stack,
                                            gssw_alternate_alignment_ends* alignment_deflections,
                                            int32_t* deflection_idx,
                                            int32_t final_traceback,
                                            int32_t find_internal_node_alts,
                                            uint16_t* score,
                                            int32_t* refEnd,
                                            int32_t* readEnd,
                                            int32_t* refGapFlag,
                                            int32_t* readGapFlag,
                                            const char* ref,
                                            int32_t refLen,
                                            const char* read,
                                            int8_t* qual_num,
                                            int32_t readLen,
                                            int8_t* nt_table,
                                            int8_t* score_matrix,
                                            uint8_t gap_open,
                                            uint8_t gap_extension,
                                            int8_t start_full_length_bonus,
                                            int8_t end_full_length_bonus) {

    gssw_align* alignment = node->alignment;
    
    // This is the alignment matrix.
    uint16_t* mH = (uint16_t*)alignment->mH;
    // And the two matrices for gaps
    uint16_t* mE = (uint16_t*)alignment->mE;
    uint16_t* mF = (uint16_t*)alignment->mF;
    
    // i, j are where we are currently in the reference and the read
    int32_t i = *refEnd;
    int32_t j = *readEnd;
    
    // These let us know if we're currently supposed to be in a gap, waiting for
    // the gap open that ought to open it.
    int32_t gRead = *readGapFlag; // If set we're in E actually
    int32_t gRef = *refGapFlag; // If set we're in F actually
    
    // This here variable holds the score that the current cell got from the DP
    // step. We'll look at the surrounding cells to work out which way we came
    // from to get this score here. We'll also update it in case we're the last
    // thing on this node and we take it along as the score to be at on the
    // other node (??)
    
#ifdef DEBUG_TRACEBACK
    fprintf(stderr, "traceback through word matrices\n");
    int l, k;
    fprintf(stderr, "mH\n");
    fprintf(stderr, "\t");
    for (k = 0; k < readLen; k++) {
        fprintf(stderr, "%c\t", read[k]);
    }
    fprintf(stderr, "\n");
    for (l = 0; l < node->len; l++) {
        fprintf(stderr, "%c\t", ref[l]);
        for (k = 0; k < readLen; k++) {
            fprintf(stderr, "%d\t", mH[readLen*l + k]);
        }
        fprintf(stderr, "\n");
    }
    fprintf(stderr, "mE\n");
    fprintf(stderr, "\t");
    for (k = 0; k < readLen; k++) {
        fprintf(stderr, "%c\t", read[k]);
    }
    fprintf(stderr, "\n");
    for (l = 0; l < node->len; l++) {
        fprintf(stderr, "%c\t", ref[l]);
        for (k = 0; k < readLen; k++) {
            fprintf(stderr, "%d\t", mE[readLen*l + k]);
        }
        fprintf(stderr, "\n");
    }
    fprintf(stderr, "mF\n");
    fprintf(stderr, "\t");
    for (k = 0; k < readLen; k++) {
        fprintf(stderr, "%c\t", read[k]);
    }
    fprintf(stderr, "\n");
    for (l = 0; l < node->len; l++) {
        fprintf(stderr, "%c\t", ref[l]);
        for (k = 0; k < readLen; k++) {
            fprintf(stderr, "%d\t", mF[readLen*l + k]);
        }
        fprintf(stderr, "\n");
    }
#endif
    
    uint16_t scoreHere;
    if(gRead) {
        // we're in mE
        scoreHere = mE[readLen*i + j];
    } else if(gRef) {
        // We're in mF
        scoreHere = mF[readLen*i + j];
    } else {
        // We're in the main matrix mH
        scoreHere = mH[readLen*i + j];
    }
    
    // Start a CIGAR to hold the traceback.
    gssw_cigar* result = (gssw_cigar*)calloc(1, sizeof(gssw_cigar));
    result->length = 0;

    while (LIKELY(scoreHere > 0 && i >= 0 && j >= 0)) {
        // We're not out of score, and we're not off the left or top of the matrix
        
        // Are there more deflections left in this traceback?
        if (*deflection_idx < alignment_deflections->num_deflections) {
            gssw_trace_back_deflection* next_deflxn = &alignment_deflections->deflections[*deflection_idx];
            // Is the deflection here?
            gssw_matrix_t curr_matrix = gRead ? ReadGap : (gRef ? RefGap : Match);
            if (UNLIKELY(i == next_deflxn->ref_pos && j == next_deflxn->read_pos
                         && node == next_deflxn->from_node && node == next_deflxn->to_node
                         && curr_matrix == next_deflxn->from_matrix)) {
#ifdef DEBUG_TRACEBACK
                fprintf(stderr, "taking deflection from i = %d, j = %d in node %llu\n", i, j, node->id);
#endif
                // take the deflection instead of doing traceback this iteration
                if (gRead) {
                    i--;
                    gssw_cigar_push_back(result, 'D', 1);
                    switch (next_deflxn->to_matrix) {
                            
                        case Match:
#ifdef DEBUG_TRACEBACK
                            fprintf(stderr, "Deflection is read gap -> match\n");
#endif
                            scoreHere = mH[readLen*i + j];
                            gRead = 0;
                            break;
                            
                        case ReadGap:
#ifdef DEBUG_TRACEBACK
                            fprintf(stderr, "Deflection is read gap -> read gap\n");
#endif
                            scoreHere = mE[readLen*i + j];
                            break;
                            
                        default:
                            fprintf(stderr, "error:[gssw] Impossible alternate alignment deflection from read gap\n");
                            assert(0);
                            break;
                    }
                }
                else if (gRef) {
                    j--;
                    gssw_cigar_push_back(result, 'I', 1);
                    switch (next_deflxn->to_matrix) {
                        case Match:
#ifdef DEBUG_TRACEBACK
                            fprintf(stderr, "Deflection is ref gap -> match\n");
#endif
                            scoreHere = mH[readLen*i + j];
                            gRef = 0;
                            break;
                            
                        case RefGap:
#ifdef DEBUG_TRACEBACK
                            fprintf(stderr, "Deflection is ref gap -> ref gap\n");
#endif
                            scoreHere = mF[readLen*i + j];
                            break;
                            
                        default:
                            fprintf(stderr, "error:[gssw] Impossible alternate alignment deflection from reference gap\n");
                            assert(0);
                            break;
                    }
                }
                else {
                    if (next_deflxn->to_node != node) {
                        // This is a deflection from match to match, but it crosses a node boundary
                        // so we leave the deflection in place and exit to the POA traceback
                        break;
                    }
                    switch (next_deflxn->to_matrix) {
                        case Match:
#ifdef DEBUG_TRACEBACK
                            fprintf(stderr, "Deflection is match -> match\n");
#endif
                            if (ref[i] == 'N' || read[j] == 'N') {
                                gssw_cigar_push_back(result, 'N', 1);
                            }
                            else if(ref[i] == read[j]) {
                                gssw_cigar_push_back(result, 'M', 1);
                            }
                            else if (ref[i] != read[j]) {
                                gssw_cigar_push_back(result, 'X', 1);
                            }
                            i--;
                            j--;
                            scoreHere = mH[readLen*i + j];
                            break;
                            
                        case ReadGap:
#ifdef DEBUG_TRACEBACK
                            fprintf(stderr, "Deflection is match -> read gap\n");
#endif
                            scoreHere = mE[readLen*i + j];
                            gRead = 1;
                            break;
                            
                        case RefGap:
#ifdef DEBUG_TRACEBACK
                            fprintf(stderr, "Deflection is match -> ref gap\n");
#endif
                            scoreHere = mF[readLen*i + j];
                            gRef = 1;
                            break;
                            
                        default:
                            fprintf(stderr, "error:[gssw] Unrecognized matrix type\n");
                            assert(0);
                            break;
                    }
                }
                (*deflection_idx)++;
                continue;
            }
        }
        
#ifdef DEBUG_TRACEBACK
        fprintf(stderr, "score=%i at %i,%i with %c vs %c, gRef=%i gRead=%i\n", scoreHere, i, j, ref[i], read[j], gRef, gRead);
        fprintf(stderr, "mH[%d,%d] = %d, mE[%d,%d] = %d, mF[%d,%d] = %d\n", i, j, mH[readLen*i + j], i, j, mE[readLen*i + j], i, j, mF[readLen*i + j]);
#endif
        
        int32_t found_trace = 0;
        uint16_t source_score;
        uint16_t score_diff;
        int32_t next_i = i;
        int32_t next_j = j;
        int32_t next_g_read = gRead;
        int32_t next_g_ref = gRef;
        uint16_t next_score_here = scoreHere;
        
        if(gRead) {
            // We're in E
            
            // If we're in a gap matrix, see if we can leave the gap here (i.e.
            // gap open score is consistent). If so, do it. Otherwise, extend
            // the gap.
            if (i > 0) {
                source_score = mH[readLen*(i - 1) +  j];
                score_diff = scoreHere - (source_score - gap_open);
#ifdef DEBUG_TRACEBACK
                fprintf(stderr, "score diff from read gap to match: %d\n", score_diff);
#endif
                if (score_diff == 0 && !found_trace) {
                    found_trace = 1;
                    // We are consistent with a gap open, bringing us back to the
                    // main matrix. Take it.
                    
                    // D = gap in read
                    gssw_cigar_push_back(result, 'D', 1);
                    
                    // Fix score
                    next_score_here += gap_open;
                    
                    // Move in the reference
                    --next_i;
                    
                    // Go back to the main matrix
                    next_g_read = 0;
                    
#ifdef DEBUG_TRACEBACK
                    fprintf(stderr, "Read gap open\n");
#endif
                    // if this is the last alignment we do not need to look for suboptimal scores
                    if (final_traceback) {
                        i = next_i;
                        j = next_j;
                        gRef = next_g_ref;
                        gRead = next_g_read;
                        scoreHere = next_score_here;
                        continue;
                    }
                    
                }
                else if (UNLIKELY(*deflection_idx == alignment_deflections->num_deflections &&
                                  score_diff < alignment_deflections->score &&
                                  source_score > 0 && find_internal_node_alts)) {
                    // score is suboptimal or we have already chosen an optimal trace and the alternate alignment
                    // does not involve any negative or 0 scores (these are not actually extensions of a local alignment)
                    uint16_t alt_score = alignment_deflections->score - score_diff;
#ifdef DEBUG_TRACEBACK
                    fprintf(stderr, "Considering alternate alignment read gap -> match with score %d\n", alt_score);
#endif
                    if (alt_score > gssw_min_alt_alignment_score(alt_alignment_stack)
                        || alt_alignment_stack->current_size < alt_alignment_stack->capacity) {
                        gssw_add_alignment(alt_alignment_stack, alignment_deflections, alt_score,
                                           j, i, node, node, ReadGap, Match);
                    }
                }
                
                source_score = mE[readLen*(i - 1) +  j];
                score_diff = scoreHere - (source_score - gap_extension);
#ifdef DEBUG_TRACEBACK
                fprintf(stderr, "score diff from read gap to read gap: %d\n", score_diff);
#endif
                if (score_diff == 0 && !found_trace) {
                    found_trace = 1;
                    // We are consistent with a gap extend
                    gssw_cigar_push_back(result, 'D', 1);
                    next_score_here += gap_extension;
                    next_i--;
#ifdef DEBUG_TRACEBACK
                    fprintf(stderr, "Read gap extend\n");
#endif
                    // if this is the last alignment we do not need to look for suboptimal scores
                    if (final_traceback) {
                        i = next_i;
                        j = next_j;
                        gRef = next_g_ref;
                        gRead = next_g_read;
                        scoreHere = next_score_here;
                        continue;
                    }
                }
                else if (UNLIKELY(*deflection_idx == alignment_deflections->num_deflections &&
                                  score_diff < alignment_deflections->score &&
                                  source_score > 0 && find_internal_node_alts)) {
                    // score is suboptimal or we have already chosen an optimal trace and the alternate alignment
                    // does not involve any negative or 0 scores (these are not actually extensions of a local alignment)
                    uint16_t alt_score = alignment_deflections->score - score_diff;
#ifdef DEBUG_TRACEBACK
                    fprintf(stderr, "Considering alternate alignment read gap -> read gap with score %d\n", alt_score);
#endif
                    if (alt_score > gssw_min_alt_alignment_score(alt_alignment_stack)
                        || alt_alignment_stack->current_size < alt_alignment_stack->capacity) {
                        gssw_add_alignment(alt_alignment_stack, alignment_deflections, alt_score,
                                           j, i, node, node, ReadGap, ReadGap);
                    }
                }
            }
            else if(i == 0) {
                // We are in a gap and at the very left edge. We need to look
                // left from here and trace into our previous node in order to
                // figure out if this is an open or an extend or what.
#ifdef DEBUG_TRACEBACK
                fprintf(stderr, "Read gap out left edge\n");
#endif
                break;
            }
            else {
                // Something has gone wrong. We're in this matrix but don't have
                // an open or an extend and can't leave left.
                fprintf(stderr, "error:[gssw] Stuck in read gap!\n");
                assert(0);
            }
        }
        else if(gRef) {
            // We're in F
            if(j > 0) {
                
                source_score = mH[readLen*i +  (j - 1)];
                score_diff = scoreHere - (source_score - gap_open);
#ifdef DEBUG_TRACEBACK
                fprintf(stderr, "score diff from ref gap to match: %d\n", score_diff);
#endif
                if (score_diff == 0 && !found_trace) {
                    found_trace = 1;
                    // We are consistent with a gap open, bringing us back to the
                    // main matrix. Take it.
                    
                    // I = gap in ref
                    gssw_cigar_push_back(result, 'I', 1);
                    
                    // Fix score
                    next_score_here += gap_open;
                    
                    // Move in the read
                    next_j--;
                    
                    // Go back to the main matrix
                    next_g_ref = 0;
#ifdef DEBUG_TRACEBACK
                    fprintf(stderr, "Ref gap open\n");
#endif
                    // if this is the last alignment we do not need to look for suboptimal scores
                    if (final_traceback) {
                        i = next_i;
                        j = next_j;
                        gRef = next_g_ref;
                        gRead = next_g_read;
                        scoreHere = next_score_here;
                        continue;
                    }
                }
                else if (UNLIKELY(*deflection_idx == alignment_deflections->num_deflections &&
                                  score_diff < alignment_deflections->score &&
                                  source_score > 0 && find_internal_node_alts)) {
                    // score is suboptimal or we have already chosen an optimal trace and the alternate alignment
                    // does not involve any negative or 0 scores (these are not actually extensions of a local alignment)
                    uint16_t alt_score = alignment_deflections->score - score_diff;
#ifdef DEBUG_TRACEBACK
                    fprintf(stderr, "Considering alternate alignment ref gap -> match with score %d\n", alt_score);
#endif
                    if (alt_score > gssw_min_alt_alignment_score(alt_alignment_stack)
                        || alt_alignment_stack->current_size < alt_alignment_stack->capacity) {
                        gssw_add_alignment(alt_alignment_stack, alignment_deflections, alt_score,
                                           j, i, node, node, RefGap, Match);
                    }
                }
                
                source_score = mF[readLen*i + (j - 1)];
                score_diff = scoreHere - (source_score - gap_extension);
#ifdef DEBUG_TRACEBACK
                fprintf(stderr, "score diff from ref gap to ref gap: %d\n", score_diff);
#endif
                if (score_diff == 0 && !found_trace) {
                    found_trace = 1;
                    // We are consistent with a gap extend
                    gssw_cigar_push_back(result, 'I', 1);
                    next_score_here += gap_extension;
                    next_j--;
#ifdef DEBUG_TRACEBACK
                    fprintf(stderr, "Ref gap extend\n");
#endif
                    if (final_traceback) {
                        i = next_i;
                        j = next_j;
                        gRef = next_g_ref;
                        gRead = next_g_read;
                        scoreHere = next_score_here;
                        continue;
                    }
                }
                else if (UNLIKELY(*deflection_idx == alignment_deflections->num_deflections &&
                                  score_diff < alignment_deflections->score &&
                                  source_score > 0 && find_internal_node_alts)) {
                    // score is suboptimal or we have already chosen an optimal trace and the alternate alignment
                    // does not involve any negative or 0 scores (these are not actually extensions of a local alignment)
                    uint16_t alt_score = alignment_deflections->score - score_diff;
#ifdef DEBUG_TRACEBACK
                    fprintf(stderr, "Considering alternate alignment ref gap -> ref gap with score %d from %d below traceback cell %d\n", alt_score, score_diff, scoreHere);
#endif
                    if (alt_score > gssw_min_alt_alignment_score(alt_alignment_stack)
                        || alt_alignment_stack->current_size < alt_alignment_stack->capacity) {
                        gssw_add_alignment(alt_alignment_stack, alignment_deflections, alt_score,
                                           j, i, node, node, RefGap, RefGap);
                    }
                }
            }
            else if(j == 0) {
                // We have hit the end of the read in a gap, which should be
                // impossible because there's nowhere to open from.
                fprintf(stderr, "error:[gssw] Ref gap hit edge!\n");
                assert(0);
            }
            else {
                // Something has gone wrong. We're in this matrix but don't have
                // an open or an extend and can't leave left.
                fprintf(stderr, "error:[gssw] Ref gap stuck!\n");
                assert(0);
            }
        }
        else {
            // We're in H
            
            // There may be alternate alignments to other nodes that we would take in the graph traceback function
            // If so, bail out here early to avoid taking the main alignment
            if (UNLIKELY(*deflection_idx < alignment_deflections->num_deflections && i == 0)) {
                gssw_trace_back_deflection* next_deflxn = &alignment_deflections->deflections[*deflection_idx];
                // Is the deflection here?
                if (UNLIKELY(i == next_deflxn->ref_pos && j == next_deflxn->read_pos
                             && node == next_deflxn->from_node && node != next_deflxn->to_node
                             && next_deflxn->from_matrix == Match)) {
#ifdef DEBUG_TRACEBACK
                    fprintf(stderr, "Breaking out of node traceback to take a graph traceback deflection\n");
#endif
                    break;
                }
            }
            
            // If we're in the main matrix, see if we can do a match, mismatch,
            // or N-match. If so, do it.
            
            int8_t align_score;
            if (qual_num) {
                align_score = score_matrix[qual_num[j] * 25 + nt_table[(uint8_t) ref[i]] * 5 + nt_table[(uint8_t) read[j]]];
            }
            else {
                align_score = score_matrix[nt_table[(uint8_t) ref[i]] * 5 + nt_table[(uint8_t) read[j]]];
            }
            
            // Full length pinned alignment bonus if we're matching the first position
            if (j == 0) {
                align_score += start_full_length_bonus;
            }
            // And full length right alignment bonus if we're matching the last position
            if (j == readLen - 1) {
                align_score += end_full_length_bonus;
            }
            
            if (i > 0 && j > 0) {
                
                source_score = mH[readLen*(i-1) + (j-1)];
                score_diff = scoreHere - (source_score + align_score);
#ifdef DEBUG_TRACEBACK
                fprintf(stderr, "score diff from match to match: %d\n", score_diff);
#endif
                if (score_diff == 0 && !found_trace) {
                    found_trace = 1;
                    
                    if (ref[i] == 'N' || read[j] == 'N') {
                        // This is an N-match.
                        gssw_cigar_push_back(result, 'N', 1);
                        // Leave score unchanged
#ifdef DEBUG_TRACEBACK
                        fprintf(stderr, "N-match\n");
#endif
                    }
                    else if(ref[i] == read[j]) {
                        // This is a match
                        gssw_cigar_push_back(result, 'M', 1);
#ifdef DEBUG_TRACEBACK
                        fprintf(stderr, "Match\n");
#endif
                    }
                    else if (ref[i] != read[j]) {
                        // This is a mismatch
                        gssw_cigar_push_back(result, 'X', 1);
#ifdef DEBUG_TRACEBACK
                        fprintf(stderr, "Mismatch\n");
#endif
                    }
                    
                    next_score_here -= align_score;
                    next_j--;
                    next_i--;
                    
                    // if this is the last alignment we do not need to look for suboptimal scores
                    if (final_traceback) {
                        i = next_i;
                        j = next_j;
                        gRef = next_g_ref;
                        gRead = next_g_read;
                        scoreHere = next_score_here;
                        continue;
                    }
                }
                else if (UNLIKELY(*deflection_idx == alignment_deflections->num_deflections &&
                                  score_diff < alignment_deflections->score &&
                                  source_score > 0 && find_internal_node_alts)) {
                    // score is suboptimal or we have already chosen an optimal trace and the alternate alignment
                    // does not involve any negative or 0 scores (these are not actually extensions of a local alignment)
                    uint16_t alt_score = alignment_deflections->score - score_diff;
#ifdef DEBUG_TRACEBACK
                    fprintf(stderr, "Considering alternate alignment match -> match with score %d\n", alt_score);
#endif
                    if (alt_score > gssw_min_alt_alignment_score(alt_alignment_stack)
                        || alt_alignment_stack->current_size < alt_alignment_stack->capacity) {
                        gssw_add_alignment(alt_alignment_stack, alignment_deflections, alt_score,
                                           j, i, node, node, Match, Match);
                    }
                }
                
            }
            else if (j == 0) {
                score_diff = scoreHere - align_score;
#ifdef DEBUG_TRACEBACK
                fprintf(stderr, "score diff from match to outside matrix: %d from score here %d and align score %d compared to alignment score %d\n", score_diff, scoreHere, align_score, alignment_deflections->score);
#endif
                if (score_diff == 0 && !found_trace) {
                    found_trace = 1;
                    if (ref[i] == 'N' || read[j] == 'N') {
                        // This is an N-match.
                        gssw_cigar_push_back(result, 'N', 1);
#ifdef DEBUG_TRACEBACK
                        fprintf(stderr, "Alignment start N-match, ref = %c, read = %c\n", ref[i], read[j]);
#endif
                    }
                    else if (ref[i] == read[j]) {
                        // This is a match
                        gssw_cigar_push_back(result, 'M', 1);
                        
#ifdef DEBUG_TRACEBACK
                        fprintf(stderr, "Alignment start match, ref = %c, read = %c\n", ref[i], read[j]);
#endif
                    }
                    else {
                        // This is a mismatch (possible with pinning bonus)
                        gssw_cigar_push_back(result, 'X', 1);
                        
#ifdef DEBUG_TRACEBACK
                        fprintf(stderr, "Alignment start mismatch, ref = %c, read = %c\n", ref[i], read[j]);
#endif
                    }
                    next_j--;
                    next_i--;
                    next_score_here -= align_score;
                    
                    // if this is the last alignment we do not need to look for suboptimal scores
                    if (final_traceback) {
                        i = next_i;
                        j = next_j;
                        gRef = next_g_ref;
                        gRead = next_g_read;
                        scoreHere = next_score_here;
                        continue;
                    }
                }
                else if (UNLIKELY(*deflection_idx == alignment_deflections->num_deflections &&
                                  score_diff < alignment_deflections->score &&
                                  score_diff == 0 && find_internal_node_alts)) {
                    // score is suboptimal or we have already chosen an optimal trace and the alternate alignment
                    // does not involve any negative or 0 scores (these are not actually extensions of a local alignment)
                    // also this alignment is only valid as a start if the implicit source score is 0
                    uint16_t alt_score = alignment_deflections->score - score_diff;
#ifdef DEBUG_TRACEBACK
                    fprintf(stderr, "Considering alternate alignment match -> start with score %d\n", alt_score);
#endif
                    if (alt_score > gssw_min_alt_alignment_score(alt_alignment_stack)
                        || alt_alignment_stack->current_size < alt_alignment_stack->capacity) {
                        gssw_add_alignment(alt_alignment_stack, alignment_deflections, alt_score,
                                           j, i, node, node, Match, Match);
                    }
                }
            }
            
            source_score = mF[readLen*i + j];
            score_diff = scoreHere - source_score;
            
#ifdef DEBUG_TRACEBACK
            fprintf(stderr, "score diff from match to ref gap: %d from score here %d and gap close %d\n", score_diff, scoreHere, mF[readLen*i + j]);
#endif
            if (score_diff == 0 && !found_trace) {
                found_trace = 1;
                // We can't do anything diagonal. But we can become a ref gap, because there is more read.
                next_g_ref = 1;
                //fprintf(stderr, "Ref gap close\n");
                if (final_traceback) {
                    i = next_i;
                    j = next_j;
                    gRef = next_g_ref;
                    gRead = next_g_read;
                    scoreHere = next_score_here;
                    continue;
                }
            }
            else if (UNLIKELY(*deflection_idx == alignment_deflections->num_deflections &&
                              score_diff < alignment_deflections->score &&
                              source_score > 0 && find_internal_node_alts)) {
                // score is suboptimal or we have already chosen an optimal trace and the alternate alignment
                // does not involve any negative or 0 scores (these are not actually extensions of a local alignment)
                uint16_t alt_score = alignment_deflections->score - score_diff;
#ifdef DEBUG_TRACEBACK
                fprintf(stderr, "Considering alternate alignment match -> ref gap: score diff %d from score here %d for alt score %d and compared to alignment score %d\n", score_diff, scoreHere, alt_score, alignment_deflections->score);
#endif
                if (alt_score > gssw_min_alt_alignment_score(alt_alignment_stack)
                    || alt_alignment_stack->current_size < alt_alignment_stack->capacity) {
                    gssw_add_alignment(alt_alignment_stack, alignment_deflections, alt_score,
                                       j, i, node, node, Match, RefGap);
                }
            }
            
            source_score = mE[readLen*i + j];
            score_diff = scoreHere - source_score;
#ifdef DEBUG_TRACEBACK
            fprintf(stderr, "score diff from match to read gap: %d\n", score_diff);
#endif
            if (score_diff == 0 && !found_trace) {
                found_trace = 1;
                // We assume there's always more ref off to the left. We tried
                // everything else, so try a read gap.
                next_g_read = 1;
#ifdef DEBUG_TRACEBACK
                fprintf(stderr, "Read gap close\n");
#endif
                if (final_traceback) {
                    i = next_i;
                    j = next_j;
                    gRef = next_g_ref;
                    gRead = next_g_read;
                    scoreHere = next_score_here;
                    continue;
                }
                else if (i == 0 && j > 0) {
#ifdef DEBUG_TRACEBACK
                    fprintf(stderr, "Also checking for matches across node boundary\n");
#endif
                    // We didn't check the alternate tracebacks that cross the node boundary
                    // from the match matrix, and now we're entering the read gap matrix, so we
                    // need to check these alternate tracebacks now because the POA function will
                    // not know that we were at the node boundary in the match matrix
                    
                    int k;
                    for (k = 0; k < node->count_prev; k++) {
                        gssw_node* prev_node = node->prev[k];
                        source_score = ((uint16_t*) prev_node->alignment->mH)[readLen * (prev_node->len - 1) + j - 1];
                        score_diff = scoreHere - (source_score + align_score);
                        
                        if (UNLIKELY(*deflection_idx == alignment_deflections->num_deflections &&
                                     score_diff < alignment_deflections->score &&
                                     source_score > 0)) {
                            // score is suboptimal or we have already chosen an optimal trace and the alternate alignment
                            // does not involve any negative or 0 scores (these are not actually extensions of a local alignment)
                            uint16_t alt_score = alignment_deflections->score - score_diff;
#ifdef DEBUG_TRACEBACK
                            fprintf(stderr, "Considering alternate alignment across node boundary match -> match with score %d vs current minimum %d and size %d of capacity %d\n", alt_score, gssw_min_alt_alignment_score(alt_alignment_stack), alt_alignment_stack->current_size, alt_alignment_stack->capacity);
#endif
                            if (alt_score > gssw_min_alt_alignment_score(alt_alignment_stack)
                                || alt_alignment_stack->current_size < alt_alignment_stack->capacity) {
                                gssw_add_alignment(alt_alignment_stack, alignment_deflections, alt_score,
                                                   j, i, node, prev_node, Match, Match);
                            }
                        }
                    }
                }
            }
            else {
                if (UNLIKELY(*deflection_idx == alignment_deflections->num_deflections &&
                             score_diff < alignment_deflections->score &&
                             source_score > 0 && find_internal_node_alts)) {
                    // score is suboptimal or we have already chosen an optimal trace and the alternate alignment
                    // does not involve any negative or 0 scores (these are not actually extensions of a local alignment)
                    uint16_t alt_score = alignment_deflections->score - score_diff;
#ifdef DEBUG_TRACEBACK
                    fprintf(stderr, "Considering alternate alignment match -> read gap with score %d vs current minimum %d and size %d of capacity %d\n", alt_score, gssw_min_alt_alignment_score(alt_alignment_stack), alt_alignment_stack->current_size, alt_alignment_stack->capacity);
#endif
                    if (alt_score > gssw_min_alt_alignment_score(alt_alignment_stack)
                        || alt_alignment_stack->current_size < alt_alignment_stack->capacity) {
                        gssw_add_alignment(alt_alignment_stack, alignment_deflections, alt_score,
                                           j, i, node, node, Match, ReadGap);
                    }
                }
            }
            
            if (i == 0 && !found_trace) {
                // We can't go anywhere, but we're at the left edge, so maybe we
                // can go to a previous node diagonally. Just let it slide.
#ifdef DEBUG_TRACEBACK
                fprintf(stderr, "Match/mismatch out left edge\n");
#endif
                break;
            }
            
            if (!found_trace) {
                // We're in H and can't go anywhere.
                fprintf(stderr, "error:[gssw] Stuck in main matrix!\n");
                assert(0);
            }
        }
        
        i = next_i;
        j = next_j;
        gRef = next_g_ref;
        gRead = next_g_read;
        scoreHere = next_score_here;
    }

    *score = scoreHere;
    *refEnd = i;
    *readEnd = j;
    *refGapFlag = gRef;
    *readGapFlag = gRead;
    gssw_reverse_cigar(result);
    return result;
}

gssw_graph_mapping* gssw_graph_mapping_create(void) {
    gssw_graph_mapping* m = (gssw_graph_mapping*)calloc(1, sizeof(gssw_graph_mapping));
    return m;
}

void gssw_graph_mapping_destroy(gssw_graph_mapping* m) {
    int32_t i;
    // iterate through gssw_graph_cigar's gssw_node_cigars
    for (i = 0; i < m->cigar.length; ++i) {
        gssw_cigar_destroy(m->cigar.elements[i].cigar);
    }
    free(m->cigar.elements);
    free(m);
}

gssw_graph_cigar* gssw_graph_cigar_create(void) {
    return (gssw_graph_cigar*)calloc(1, sizeof(gssw_graph_cigar));
}

void gssw_graph_cigar_destroy(gssw_graph_cigar* g) {
    int32_t i;
    for (i = 0; i < g->length; ++i) {
        gssw_cigar_destroy(g->elements[i].cigar);
    }
    free(g->elements);
}

void gssw_print_graph_cigar(gssw_graph_cigar* g, FILE* out) {
    int32_t i;
    gssw_node_cigar* nc = g->elements;
    for (i = 0; i < g->length; ++i, ++nc) {
        fprintf(out, "%llu[", nc->node->id);
        gssw_print_cigar(nc->cigar, out);
        fprintf(out, "]");
    }
    fprintf(out, "\n");
}

void gssw_print_graph_mapping(gssw_graph_mapping* gm, FILE* out) {
    fprintf(out, "%i@%i:", gm->score, gm->position);
    gssw_print_graph_cigar(&gm->cigar, out);
}

/*
char* gssw_graph_cigar_to_string(gssw_graph_cigar* g) {
    int32_t bufsiz = g->length * 1024;
    char* s = calloc(bufsiz, sizeof(char));
    int32_t i;
    int32_t c = 0;
    gssw_node_cigar* nc = g->elements;
    for (i = 0; i < g->length; ++i, ++nc) {
        c = snprintf(s+c, bufsiz-c, "%u[", nc->node->id);
        int j;
        int l = c->length;
        gssw_cigar_element* e = c->elements;
        for (j=0; LIKELY(j < l); ++j, ++e) {
            c = snprintf(s+c, bufsiz-c, "%i%c", e->length, e->type);
        }
        c = snprintf(s+c, bufsiz-c, "]");
    }
    return s;
}

char* gssw_graph_mapping_to_string(gssw_graph_mapping* gm) {
}
*/

void gssw_reverse_graph_cigar(gssw_graph_cigar* c) {
    gssw_graph_cigar* reversed = (gssw_graph_cigar*)malloc(sizeof(gssw_graph_cigar));
    reversed->length = c->length;
    reversed->elements = (gssw_node_cigar*) malloc(c->length * sizeof(gssw_node_cigar));
    gssw_node_cigar* c1 = c->elements;
    gssw_node_cigar* c2 = reversed->elements;
    int32_t s = 0;
    int32_t e = c->length - 1;
    while (LIKELY(s <= e)) {
        c2[s] = c1[e];
        c2[e] = c1[s];
        ++ s;
        -- e;
    }
    free(c->elements);
    c->elements = reversed->elements;
    free(reversed);
}

// TODO: the suboptimal alignments this produces are all anchored to the end point of the optimal alignment
// this is valid for the pinned alignment, but if we want to expand this to produce alternate alignments
// for local alignment then we need to seed the multi-alignment stack with the top k end points before entering
// the inner loop
gssw_graph_mapping** gssw_graph_trace_back_internal (gssw_graph* graph,
                                                     int32_t doing_pinning,
                                                     int32_t num_tracebacks,
                                                     int32_t find_internal_node_alts,
                                                     const char* read,
                                                     const char* qual,
                                                     int32_t readLen,
                                                     gssw_node** pinning_nodes,
                                                     int32_t num_pinning_nodes,
                                                     int8_t* nt_table,
                                                     int8_t* score_matrix,
                                                     uint8_t gap_open,
                                                     uint8_t gap_extension,
                                                     int8_t start_full_length_bonus,
                                                     int8_t end_full_length_bonus) {

#ifdef DEBUG_TRACEBACK
    fprintf(stderr, "beginning traceback with params:\n");
    fprintf(stderr, "\tpinning? %d\n", doing_pinning);
    fprintf(stderr, "\tnum tracebacks %d\n", num_tracebacks);
    fprintf(stderr, "\tinternal alts? %d\n", find_internal_node_alts);
    gssw_graph_print_score_matrices(graph, read, readLen, stderr);
#endif
    
    // Get quality score as integers
    int8_t* qual_num = NULL;
    if (qual) {
        qual_num = gssw_create_qual_num(qual, readLen);
    }
    
    // Make the mappings
    gssw_graph_mapping** gms = (gssw_graph_mapping**) malloc(sizeof(gssw_graph_mapping*) * num_tracebacks);
    
    int i;
    for (i = 0; i < num_tracebacks; i++) {
        gms[i] = gssw_graph_mapping_create();
    }
    
    gssw_multi_align_stack* alt_alignment_stack = gssw_new_multi_align_stack(num_tracebacks);
    
    // Check that alignment has been run and find highest scoring node
    gssw_node* n = graph->max_node;;
    
    if (!n) {
        fprintf(stderr, "error:[gssw] Cannot trace back because graph alignment has not been run.\n");
        fprintf(stderr, "error:[gssw] You must call graph_fill(...) before tracing back.\n");
        exit(1);
    }
    
    // Are we using bytes, or did we have to do shorts?
    uint8_t score_is_byte = gssw_is_byte(n->alignment);
    
    // Create a null suffix to prefix the alignment starts with
    gssw_alternate_alignment_ends null_suffix;
    null_suffix.score = 0;
    null_suffix.num_deflections = 0;
    null_suffix.deflections = NULL;
    
    int32_t refEnd;
    int32_t readEnd;
    uint16_t score;
    if (num_pinning_nodes) {
        // Initialize the alternate alignment stack with the set of provided
        // pinning nodes

        int j;
        for (j = 0; j < num_pinning_nodes; j++) {
            gssw_node* n = pinning_nodes[j];
            // Get the coordinates of the score
            refEnd = n->len - 1;
            readEnd = readLen - 1;

            // Get the score in the bottom right corner
            if (score_is_byte) {
                uint8_t* mH = (uint8_t*) n->alignment->mH;
                score = mH[readLen * refEnd + readEnd];
            }
            else {
                uint16_t* mH = (uint16_t*) n->alignment->mH;
                score = mH[readLen * refEnd + readEnd];
            }

            // Add it to the alt alignment stack and let internal logic decide which ones
            // to keep
            gssw_add_alignment(alt_alignment_stack, &null_suffix, score, readEnd, refEnd, n, n, Match, Match);
        }
    }
    else if (doing_pinning) {
        // Initialize the alternate alignment stack with each sink node since
        // no other pinning node set was provided but we have indicated that we
        // are in fact doing pinned alignment anyway
        
        int j;
        for (j = 0; j < graph->size; j++) {
            n = graph->nodes[j];
            // Is it a sink?
            if (n->count_next == 0) {
                // Get the coordinates of the score
                refEnd = n->len - 1;
                readEnd = readLen - 1;
                
                // Get the score in the bottom right corner
                if (score_is_byte) {
                    uint8_t* mH = (uint8_t*) n->alignment->mH;
                    score = mH[readLen * refEnd + readEnd];
                }
                else {
                    uint16_t* mH = (uint16_t*) n->alignment->mH;
                    score = mH[readLen * refEnd + readEnd];
                }
                
                // Add it to the alt alignment stack and let internal logic decide which ones
                // to keep
                gssw_add_alignment(alt_alignment_stack, &null_suffix, score, readEnd, refEnd, n, n, Match, Match);
            }
        }
    }
    else {
        // Where in the reference and the read is the best place to trace back from
        // in this node?
        refEnd = n->alignment->ref_end1;
        readEnd = n->alignment->read_end1;
        
        // Get the best score at the node
        score = n->alignment->score1;
        
        gssw_add_alignment(alt_alignment_stack, &null_suffix, score, readEnd, refEnd, n, n, Match, Match);
    }
    
    // Iterate through alternate alignments in descending order of score
    int32_t traceback_idx;
    gssw_multi_align_stack_node* next_alt_align;
    for (traceback_idx = 0, next_alt_align = alt_alignment_stack->top_scoring;
         next_alt_align != NULL && traceback_idx < num_tracebacks;
         traceback_idx++, next_alt_align = next_alt_align->prev) {
        
        int32_t final_traceback = (traceback_idx == num_tracebacks - 1);
        // graph_mapping for return value
        gssw_graph_mapping* gm = gms[traceback_idx];
        
        // indices where this alignment deflects from optimal traceback
        gssw_alternate_alignment_ends* alt_alignment = next_alt_align->alt_alignment;
        
        // Store score of alignment in graph mapping
        gm->score = alt_alignment->score;
        
        // Get the ending position of this alignment
        refEnd = alt_alignment->deflections[0].ref_pos;
        readEnd = alt_alignment->deflections[0].read_pos;
        n = alt_alignment->deflections[0].to_node;
        
        // Index of next deflection
        int32_t deflection_idx = 1;
        
        // Get score that will be present in the matrix cell
        // note: can ignore the matrix field in the deflection because it is always safe to start in H matrix
        if (readEnd < 0 || refEnd < 0) {
            // edge case that occurs when then entire matrix is 0s (then there is no max score so the initial values for
            // refEnd and readEnd are retained)
            score = 0;
        }
        else if (score_is_byte) {
            uint8_t* mH = (uint8_t*) n->alignment->mH;
            score = mH[readLen * refEnd + readEnd];
        }
        else {
            uint16_t* mH = (uint16_t*) n->alignment->mH;
            score = mH[readLen * refEnd + readEnd];
        }
        
        // Get the cigar string
        gssw_graph_cigar* gc = &gm->cigar;
        
        // The cigar needs to double every so often so it is big enough. This is how
        // long is thould be to start.
        uint32_t graph_cigar_bufsiz = 16;
        gc->elements = NULL;
        gc->elements = realloc((void*) gc->elements, graph_cigar_bufsiz * sizeof(gssw_node_cigar));
        // And how much of it is used
        gc->length = 0;
        
        
        // We keep flags indicating if we're tracing along a gap in the reference or
        // a gap in the read. TODO: the gap in the reference flag doesn't really
        // need to be out here, because a gap in the reference by nature can't cross
        // nodes.
        int32_t gapInRef = 0;
        int32_t gapInRead = 0;
        //fprintf(stderr, "ref_end1 %i read_end1 %i\n", refEnd, readEnd);
        
        // Keep a cursor to the current CIGAR element in the buffer.
        // node cigar
        gssw_node_cigar* nc = gc->elements;
        
        // get terminal soft clipping
        int32_t end_soft_clip = 0;
        // -1 is as we are counting from the opposite side of the base
        if (readLen - readEnd - 1) {
            // We didn't end exactly at the end of the read.
            // Work out how much read is left.
            end_soft_clip = readLen - readEnd - 1;
        }
        
#ifdef DEBUG_TRACEBACK
        fprintf(stderr, "new trace back, starting node = %p %llu\n", n, n->id);
        fprintf(stderr, "score %d, deflections (%d, %p):\n", alt_alignment->score, alt_alignment->num_deflections, alt_alignment->deflections);
        int v;
        for (v = 0; v < alt_alignment->num_deflections; v++) {
            gssw_trace_back_deflection deflxn = alt_alignment->deflections[v];
            fprintf(stderr, "\t(n:%llu, %c[%d,%d]) -> (n:%llu, %c)\n", deflxn.from_node->id,
                    deflxn.from_matrix == Match ? 'H' : deflxn.from_matrix == ReadGap ? 'E' : 'F',
                    deflxn.ref_pos, deflxn.read_pos, deflxn.to_node->id,
                    deflxn.to_matrix == Match ? 'H' : deflxn.to_matrix == ReadGap ? 'E' : 'F');
        }
#endif
        while (score > 0) {
            // Until we've accounted for all the score
            
            if (gc->length == graph_cigar_bufsiz) {
                graph_cigar_bufsiz *= 2;
                gc->elements = realloc((void*) gc->elements, graph_cigar_bufsiz * sizeof(gssw_node_cigar));
            }
            
            // write the cigar to the current node
            nc = gc->elements + gc->length;
#ifdef DEBUG_TRACEBACK
            fprintf(stderr, "id=%llu\n", n->id);
#endif
            nc->cigar = gssw_alignment_trace_back (n,
                                                   alt_alignment_stack,
                                                   alt_alignment,
                                                   &deflection_idx,
                                                   final_traceback,
                                                   find_internal_node_alts,
                                                   &score,
                                                   &refEnd,
                                                   &readEnd,
                                                   &gapInRef,
                                                   &gapInRead,
                                                   n->seq,
                                                   n->len,
                                                   read,
                                                   qual_num,
                                                   readLen,
                                                   nt_table,
                                                   score_matrix,
                                                   gap_open,
                                                   gap_extension,
                                                   start_full_length_bonus,
                                                   end_full_length_bonus);
            
            //assert(0);
            
            
            if (end_soft_clip) {
                // This is the last node (the one we started the traceback from), so
                // stick the soft clip on its end. Note that the CIGAR is already
                // flipped around to forward order.
                gssw_cigar_push_back(nc->cigar, 'S', end_soft_clip);
                end_soft_clip = 0;
            }
            
            nc->node = n;
            ++gc->length;
#ifdef DEBUG_TRACEBACK
            fprintf(stderr, "score is %u as we end node %p %llu at position %i in read and %i in ref\n", score, n, n->id, readEnd, refEnd);
#endif
            if (score != 0 && refEnd > 0) {
                // We've stopped the traceback, possibly outside the matrix, but we
                // didn't account for all the score.
                // refEnd <0 would mean we made it to the very end of the reference node and used that character.
                // TODO: what to do with refEnd = 0?
                gm->score = -1;
                assert(false);
                break;
            }
            if (score == 0) {
#ifdef DEBUG_TRACEBACK
                fprintf(stderr, "soft clipping %i\n", readEnd+1);
#endif
                if (readEnd > -1) {
                    gssw_cigar_push_front(nc->cigar, 'S', readEnd+1);
                }
                break;
            }
            // the read did not complete here
            // check that we are at 0 in reference and > 0 in read
            /*
             if (readEnd == 0 || readEnd != 0) {
             fprintf(stderr, "graph traceback error, at end of read or ref but score not 0\n");
             exit(1);
             }
             */
            
            // so check its inbound nodes at the given read end position
            int32_t i;
            // We'll fill this in with the best node we find to go into
            gssw_node* best_prev = NULL;
            
            // determine direction across edge
            
            // Note that we need to push_front on our CIGAR here. TODO: this is
            // inefficient.
            
            // rationale: we have to check the left and diagonal directions
            // vertical would stay on this node even if we are in the last column
            
            // note that the loop is split depending on alignment score width...
            // this is done out of paranoia that optimization will not factor two loops into two if there
            // is an if statement with a consistent result inside of each iteration
            // TODO: is that a good reason?
            if (score_is_byte) {
                // Are there more deflections left in this traceback?
                if (deflection_idx < alt_alignment->num_deflections) {
#ifdef DEBUG_TRACEBACK
                    fprintf(stderr, "there are remaining deflections in this alternate alignment, checking whether to take one\n");
#endif
                    gssw_trace_back_deflection* next_deflxn = &alt_alignment->deflections[deflection_idx];
                    // Is there a deflection here?
                    gssw_matrix_t curr_matrix = gapInRead ? ReadGap : Match;
#ifdef DEBUG_TRACEBACK
                    fprintf(stderr, "next deflection deflection is from ref pos = %d, read pos = %d, matrix = %s in node %llu to matrix = %s in node %llu\n", next_deflxn->ref_pos, next_deflxn->read_pos, (next_deflxn->from_matrix == Match ? "Match" : (next_deflxn->from_matrix == ReadGap ? "ReadGap" : "RefGap")), next_deflxn->from_node->id, (next_deflxn->to_matrix == Match ? "Match" : (next_deflxn->to_matrix == ReadGap) ? "ReadGap" : "RefGap"), next_deflxn->to_node->id);
#endif
                    if (refEnd == next_deflxn->ref_pos && readEnd == next_deflxn->read_pos
                        && n == next_deflxn->from_node && curr_matrix == next_deflxn->from_matrix) {
                        // take the deflection instead of doing traceback this iteration
#ifdef DEBUG_TRACEBACK
                        fprintf(stderr, "taking new node deflection from i = %d, j = %d in node %llu to node %llu\n", refEnd, readEnd, n->id, next_deflxn->to_node->id);
#endif
                        // go to the node that deflection indicates
                        best_prev = next_deflxn->to_node;
                        
                        if (gapInRead) {
                            gssw_cigar_push_front(nc->cigar, 'D', 1);
                            switch (next_deflxn->to_matrix) {
                                    
                                case Match:
#ifdef DEBUG_TRACEBACK
                                    fprintf(stderr, "Deflection is read gap -> match\n");
#endif
                                    score = ((uint8_t*)best_prev->alignment->mH)[readLen*(best_prev->len-1) + readEnd];
                                    gapInRead = 0;
                                    break;
                                    
                                case ReadGap:
#ifdef DEBUG_TRACEBACK
                                    fprintf(stderr, "Deflection is read gap -> read gap\n");
#endif
                                    score = ((uint8_t*)best_prev->alignment->mE)[readLen*(best_prev->len-1) + readEnd];
                                    break;
                                    
                                default:
                                    fprintf(stderr, "error:[gssw] Impossible alternate alignment deflection from read gap\n");
                                    assert(0);
                                    break;
                            }
                        }
                        else {
                            char ref_char = n->seq[refEnd];
                            char read_char = read[readEnd];
                            if (ref_char == 'N' || read_char == 'N') {
                                gssw_cigar_push_front(nc->cigar, 'N', 1);
                            }
                            else if(ref_char == read_char) {
                                gssw_cigar_push_front(nc->cigar, 'M', 1);
                            }
                            else if (ref_char != read_char) {
                                gssw_cigar_push_front(nc->cigar, 'X', 1);
                            }
                            switch (next_deflxn->to_matrix) {
                                case Match:
#ifdef DEBUG_TRACEBACK
                                    fprintf(stderr, "Deflection is match -> match\n");
#endif
                                    readEnd--;
                                    refEnd--;
                                    score = ((uint8_t*)best_prev->alignment->mH)[readLen*(best_prev->len-1) + readEnd];
                                    break;
                                    
                                default:
                                    fprintf(stderr, "error:[gssw] Impossible alternate alignment deflection from match across node boundary\n");
                                    assert(0);
                                    break;
                            }
                        }
#ifdef DEBUG_TRACEBACK
                        fprintf(stderr, "Score at deflected position is %d\n", score);
#endif
                        deflection_idx++;
                    }
                }
                
                if (best_prev == NULL) {
                    // did not take a deflection, proceed to check POA backwards
       
                    // marks whether we've found the next cell in traceback
                    int32_t found_trace = 0;
                    
                    // store the next traceback in these so that we can avoid updating variables until end of loop
                    uint16_t next_score = score;
                    int32_t next_read_end = readEnd;
                    int32_t next_ref_end = refEnd;
                    int32_t next_gap_in_read = gapInRead;
                    int32_t next_gap_in_ref = gapInRef;
                    
                    
                    // If we were to match/mismatch, what characters are we comparing?
                    char refChar = n->seq[refEnd];
                    char readChar = read[readEnd];
                    
                    // And what is their score?
                    int8_t align_score;
                    if (qual_num) {
                        align_score = score_matrix[qual_num[readEnd] * 25 + nt_table[(uint8_t)refChar] * 5 + nt_table[(uint8_t)readChar]];
                    }
                    else {
                        align_score = score_matrix[nt_table[(uint8_t)refChar] * 5 + nt_table[(uint8_t)readChar]];
                    }
                    
                    // Full length right alignment bonus if we're matching the last position
                    if (readEnd == readLen - 1) {
                        align_score += end_full_length_bonus;
                    }
                    if (readEnd == 0) {
                        align_score += start_full_length_bonus;
                    }
                    
                    if (UNLIKELY(score == align_score)) {
                        // this is the last match in the alignment
                        if (refChar == 'N' || readChar == 'N') {
                            gssw_cigar_push_front(nc->cigar, 'N', 1);
                        }
                        else if (refChar == readChar) {
                            gssw_cigar_push_front(nc->cigar, 'M', 1);
                        }
                        else {
                            gssw_cigar_push_front(nc->cigar, 'X', 1);
                        }
                        refEnd--;
                        readEnd--;
                        if (readEnd >= 0) {
                            gssw_cigar_push_front(nc->cigar, 'S', readEnd + 1);
                        }
                        break;
                    }
                    else if (UNLIKELY(gapInRef && readEnd == 0 && score == start_full_length_bonus - gap_open)) {
                        // this is a weird event, but it could happen with some scoring regimes
                        // there's a penalized insertion taken to obtain the full length bonus
                        gssw_cigar_push_front(nc->cigar, 'I', 1);
                        readEnd--;
                        break;
                    }
                    
                    for (i = 0; i < n->count_prev; ++i) {
                        // Consider each node we could have come from
                        gssw_node* cn = n->prev[i];
                        
                        // What if we came diagonally on a match or mismatch? What score would we come from?
                        uint8_t diagonalSourceScore = ((uint8_t*)cn->alignment->mH)[readLen*(cn->len-1) + (readEnd-1)];
                        
                        // What if we came from the left, on a gap open in the read?
                        uint8_t gapOpenSourceScore = ((uint8_t*)cn->alignment->mH)[readLen*(cn->len-1) + readEnd];
                        
                        // And what if we came on a gap extend instead?
                        uint8_t gapExtendSourceScore = ((uint8_t*)cn->alignment->mE)[readLen*(cn->len-1) + readEnd];
                        
                        // If we could have entered a read gap before leaving our last
                        // node, we would have.
                        
#ifdef DEBUG_TRACEBACK
                        fprintf(stderr, "Consider prev node %d of %d with sequence %s: %p with score %i, %c vs %c, %i diagonal, %i open, %i extend\n", i + 1, n->count_prev, cn->seq, cn, score, refChar, readChar, diagonalSourceScore, gapOpenSourceScore, gapExtendSourceScore);
#endif
                        
                        if(!gapInRead) {
                            // If we're not in a gap...
                            
                            uint16_t score_diff = score - (diagonalSourceScore + align_score);
#ifdef DEBUG_TRACEBACK
                            fprintf(stderr, "Comparing match across nodes: score here %d, align score %d, source score %d, score diff %d\n", score, align_score, diagonalSourceScore, score_diff);
#endif
                            if(score_diff == 0 && !found_trace) {
                                // score is what we expect and we haven't chosen an optimum to trace
                                found_trace = 1;
                                
                                best_prev = cn;
                                next_read_end--;
                                next_score -= align_score;
                                if (refChar == 'N' || readChar == 'N') {
                                    gssw_cigar_push_front(nc->cigar, 'N', 1);
#ifdef DEBUG_TRACEBACK
                                    fprintf(stderr, "N-match across nodes to %p\n", cn);
#endif
                                }
                                else if (refChar == readChar) {
                                    gssw_cigar_push_front(nc->cigar, 'M', 1);
#ifdef DEBUG_TRACEBACK
                                    fprintf(stderr, "Match across nodes to %p\n", cn);
#endif
                                }
                                else {
                                    gssw_cigar_push_front(nc->cigar, 'X', 1);
#ifdef DEBUG_TRACEBACK
                                    fprintf(stderr, "Mismatch across nodes to %p\n", cn);
#endif
                                }
                                // is this the last alternate we will look for?
                                if (final_traceback) {
                                    // safe to stop looking for suboptimal scores
                                    score = next_score;
                                    readEnd = next_read_end;
                                    refEnd = next_ref_end;
                                    gapInRead = next_gap_in_read;
                                    gapInRef = next_gap_in_ref;
                                    break;
                                }
                                // A match starting the alignment should have been taken
                                // care of in the within-node function.
                                
                                // If none of those work, try the next option for the previous
                                // node.
                            }
                            else if (UNLIKELY(deflection_idx == alt_alignment->num_deflections &&
                                              score_diff < alt_alignment->score &&
                                              diagonalSourceScore > 0)) {
                                // score is suboptimal or we have already chosen an optimal trace
                                uint16_t alt_score = alt_alignment->score - score_diff;

                                if (alt_score > gssw_min_alt_alignment_score(alt_alignment_stack)
                                    || alt_alignment_stack->current_size < alt_alignment_stack->capacity) {
                                    gssw_add_alignment(alt_alignment_stack, alt_alignment, alt_score,
                                                       readEnd, refEnd, n, cn, Match, Match);
                                }
                            }
                        }
                        else {
#ifdef DEBUG_TRACEBACK
                            fprintf(stderr, "Comparing gap open across nodes: score here %d, penalty %d, source score %d\n", score, gap_open, gapOpenSourceScore);
#endif
                            
                            // If we are in a gap, it would have been a last resort in the node's traceback.
                            uint16_t score_diff = score - (gapOpenSourceScore - gap_open);
                            if (score_diff == 0 && !found_trace) {
                                found_trace = 1;
                                // This node is consistent with an open. Take it.
                                best_prev = cn;
                                gssw_cigar_push_front(nc->cigar, 'D', 1);
                                next_score += gap_open;
                                // Unset the gap flag
                                next_gap_in_read = 0;
#ifdef DEBUG_TRACEBACK
                                fprintf(stderr, "Gap open across nodes to %p\n", cn);
#endif
                                // is this the last alternate we will look for?
                                if (final_traceback) {
                                    // safe to stop looking for suboptimal scores
                                    score = next_score;
                                    readEnd = next_read_end;
                                    refEnd = next_ref_end;
                                    gapInRead = next_gap_in_read;
                                    gapInRef = next_gap_in_ref;
                                    break;
                                }
                            }
                            else if (UNLIKELY(deflection_idx == alt_alignment->num_deflections &&
                                              score_diff < alt_alignment->score &&
                                              gapOpenSourceScore > 0)) {
                                // score is suboptimal or we have already chosen an optimal trace
                                uint16_t alt_score = alt_alignment->score - score_diff;
                                if (alt_score > gssw_min_alt_alignment_score(alt_alignment_stack)
                                    || alt_alignment_stack->current_size < alt_alignment_stack->capacity) {
                                    gssw_add_alignment(alt_alignment_stack, alt_alignment, alt_score,
                                                       readEnd, refEnd, n, cn, ReadGap, Match);
                                }
                            }
                            
#ifdef DEBUG_TRACEBACK
                            fprintf(stderr, "Comparing gap extend across nodes: score here %d, penalty %d, source score %d\n", score, gap_extension, gapExtendSourceScore);
#endif
                            score_diff = score - (gapExtendSourceScore - gap_extension);
                            if (score_diff == 0 && !found_trace) {
                                found_trace = 1;
                                // This node is consistent with an extend. Take it.
                                best_prev = cn;
                                gssw_cigar_push_front(nc->cigar, 'D', 1);
                                next_score += gap_extension;
#ifdef DEBUG_TRACEBACK
                                fprintf(stderr, "Gap extend across nodes to %p\n", cn);
#endif
                                // is this the last alternate we will look for?
                                if (final_traceback) {
                                    // safe to stop looking for suboptimal scores
                                    score = next_score;
                                    readEnd = next_read_end;
                                    refEnd = next_ref_end;
                                    gapInRead = next_gap_in_read;
                                    gapInRef = next_gap_in_ref;
                                    break;
                                }
                            }
                            else if (UNLIKELY(deflection_idx == alt_alignment->num_deflections &&
                                              score_diff < alt_alignment->score &&
                                              gapExtendSourceScore > 0)) {
                                // score is suboptimal or we have already chosen an optimal trace
                                uint16_t alt_score = alt_alignment->score - score_diff;
                                if (alt_score > gssw_min_alt_alignment_score(alt_alignment_stack)
                                    || alt_alignment_stack->current_size < alt_alignment_stack->capacity) {
                                    gssw_add_alignment(alt_alignment_stack, alt_alignment, alt_score,
                                                       readEnd, refEnd, n, cn, ReadGap, ReadGap);
                                }
                            }
                        }
                    }
                    score = next_score;
                    readEnd = next_read_end;
                    refEnd = next_ref_end;
                    gapInRead = next_gap_in_read;
                    gapInRef = next_gap_in_ref;
                }
                
                // Once we go through all the possible previous nodes, we sure hope we found something consistent.
                if(best_prev == NULL) {
                    fprintf(stderr, "error:[gssw] Could not find a valid previous node\n");
                    assert(0);
                }
                
            }
            else {
                // Repeat the whole thing for shorts
                
                // Are there more deflections left in this traceback?
                if (deflection_idx < alt_alignment->num_deflections) {
#ifdef DEBUG_TRACEBACK
                    fprintf(stderr, "there are remaining deflections in this alternate alignment, checking whether to take one\n");
#endif
                    gssw_trace_back_deflection* next_deflxn = &alt_alignment->deflections[deflection_idx];
                    // Is there a deflection here?
                    gssw_matrix_t curr_matrix = gapInRead ? ReadGap : Match;
#ifdef DEBUG_TRACEBACK
                    fprintf(stderr, "next deflection deflection is from ref pos = %d, read pos = %d, matrix = %s in node %llu to matrix = %s in node %llu\n", next_deflxn->ref_pos, next_deflxn->read_pos, (next_deflxn->from_matrix == Match ? "Match" : (next_deflxn->from_matrix == ReadGap ? "ReadGap" : "RefGap")), next_deflxn->from_node->id, (next_deflxn->to_matrix == Match ? "Match" : (next_deflxn->to_matrix == ReadGap) ? "ReadGap" : "RefGap"), next_deflxn->to_node->id);
#endif
                    if (refEnd == next_deflxn->ref_pos && readEnd == next_deflxn->read_pos
                        && n == next_deflxn->from_node && curr_matrix == next_deflxn->from_matrix) {
                        // take the deflection instead of doing traceback this iteration
#ifdef DEBUG_TRACEBACK
                        fprintf(stderr, "taking new node deflection from i = %d, j = %d in node %llu to node %llu\n", refEnd, readEnd, n->id, next_deflxn->to_node->id);
#endif
                        // go to the node that deflection indicates
                        best_prev = next_deflxn->to_node;
                        
                        if (gapInRead) {
                            gssw_cigar_push_front(nc->cigar, 'D', 1);
                            switch (next_deflxn->to_matrix) {
                                    
                                case Match:
#ifdef DEBUG_TRACEBACK
                                    fprintf(stderr, "Deflection is read gap -> match\n");
#endif
                                    score = ((uint16_t*)best_prev->alignment->mH)[readLen*(best_prev->len-1) + readEnd];
                                    gapInRead = 0;
                                    break;
                                    
                                case ReadGap:
#ifdef DEBUG_TRACEBACK
                                    fprintf(stderr, "Deflection is read gap -> read gap\n");
#endif
                                    score = ((uint16_t*)best_prev->alignment->mE)[readLen*(best_prev->len-1) + readEnd];
                                    break;
                                    
                                default:
                                    fprintf(stderr, "error:[gssw] Impossible alternate alignment deflection from read gap\n");
                                    assert(0);
                                    break;
                            }
                        }
                        else {
                            char ref_char = n->seq[refEnd];
                            char read_char = read[readEnd];
                            if (ref_char == 'N' || read_char == 'N') {
                                gssw_cigar_push_front(nc->cigar, 'N', 1);
                            }
                            else if(ref_char == read_char) {
                                gssw_cigar_push_front(nc->cigar, 'M', 1);
                            }
                            else if (ref_char != read_char) {
                                gssw_cigar_push_front(nc->cigar, 'X', 1);
                            }
                            switch (next_deflxn->to_matrix) {
                                case Match:
#ifdef DEBUG_TRACEBACK
                                    fprintf(stderr, "Deflection is match -> match\n");
#endif
                                    readEnd--;
                                    refEnd--;
                                    score = ((uint16_t*)best_prev->alignment->mH)[readLen*(best_prev->len-1) + readEnd];
                                    break;
                                    
                                default:
                                    fprintf(stderr, "error:[gssw] Impossible alternate alignment deflection from match across node boundary\n");
                                    assert(0);
                                    break;
                            }
                        }
                        deflection_idx++;
                    }
                }
                
                if (best_prev == NULL) {
                    // did not take a deflection, proceed to check POA backwards
                    
                    // marks whether we've found the next cell in traceback
                    int32_t found_trace = 0;
                    
                    // store the next traceback in these so that we can avoid updating variables until end of loop
                    uint16_t next_score = score;
                    int32_t next_read_end = readEnd;
                    int32_t next_ref_end = refEnd;
                    int32_t next_gap_in_read = gapInRead;
                    int32_t next_gap_in_ref = gapInRef;
                    
                    // If we were to match/mismatch, what characters are we comparing?
                    char refChar = n->seq[refEnd];
                    char readChar = read[readEnd];
                    
                    // And what is their score?
                    int8_t align_score;
                    if (qual_num) {
                        align_score = score_matrix[qual_num[readEnd] * 25 + nt_table[(uint8_t)refChar] * 5 + nt_table[(uint8_t)readChar]];
                    }
                    else {
                        align_score = score_matrix[nt_table[(uint8_t)refChar] * 5 + nt_table[(uint8_t)readChar]];
                    }
                    
                    // Full length right alignment bonus if we're matching the last position
                    if (readEnd == readLen - 1) {
                        align_score += end_full_length_bonus;
                    }
                    if (readEnd == 0) {
                        align_score += start_full_length_bonus;
                    }
                    
                    if (UNLIKELY(score == align_score)) {
                        // this is the last match in the alignment
                        if (refChar == 'N' || readChar == 'N') {
                            gssw_cigar_push_front(nc->cigar, 'N', 1);
                        }
                        else if (refChar == readChar) {
                            gssw_cigar_push_front(nc->cigar, 'M', 1);
                        }
                        else {
                            gssw_cigar_push_front(nc->cigar, 'X', 1);
                        }
                        refEnd--;
                        readEnd--;
                        if (readEnd >= 0) {
                            gssw_cigar_push_front(nc->cigar, 'S', readEnd + 1);
                        }
                        break;
                    }
                    else if (UNLIKELY(gapInRef && readEnd == 0 && score == start_full_length_bonus - gap_open)) {
                        // this is a weird event, but it could happen with some scoring regimes
                        // there's a penalized insertion taken to obtain the full length bonus
                        gssw_cigar_push_front(nc->cigar, 'I', 1);
                        readEnd--;
                        break;
                    }
                    
                    for (i = 0; i < n->count_prev; ++i) {
                        // Consider each node we could have come from
                        gssw_node* cn = n->prev[i];
                        
                        // What if we came diagonally on a match or mismatch? What score would we come from?
                        uint16_t diagonalSourceScore = ((uint16_t*)cn->alignment->mH)[readLen*(cn->len-1) + (readEnd-1)];
                        
                        // What if we came from the left, on a gap open in the read?
                        uint16_t gapOpenSourceScore = ((uint16_t*)cn->alignment->mH)[readLen*(cn->len-1) + readEnd];
                        
                        // And what if we came on a gap extend instead?
                        uint16_t gapExtendSourceScore = ((uint16_t*)cn->alignment->mE)[readLen*(cn->len-1) + readEnd];
                        
                        // If we could have entered a read gap before leaving our last
                        // node, we would have.
                        
#ifdef DEBUG_TRACEBACK
                        fprintf(stderr, "Consider prev node %d of %d with sequence %s: %p with score %i, %c vs %c, %i diagonal, %i open, %i extend\n", i + 1, n->count_prev, cn->seq, cn, score, refChar, readChar, diagonalSourceScore, gapOpenSourceScore, gapExtendSourceScore);
#endif
                        
                        if(!gapInRead) {
                            // If we're not in a gap...
                            
                            uint16_t score_diff = score - (diagonalSourceScore + align_score);
#ifdef DEBUG_TRACEBACK
                            fprintf(stderr, "Comparing match across nodes: score here %d, align score %d, source score %d, score diff %d\n", score, align_score, diagonalSourceScore, score_diff);
#endif
                            if(score_diff == 0 && !found_trace) {
                                // score is what we expect and we haven't chosen an optimum to trace
                                found_trace = 1;
                                
                                best_prev = cn;
                                next_read_end--;
                                next_score -= align_score;
                                if (refChar == 'N' || readChar == 'N') {
                                    gssw_cigar_push_front(nc->cigar, 'N', 1);
#ifdef DEBUG_TRACEBACK
                                    fprintf(stderr, "N-match across nodes to %p\n", cn);
#endif
                                }
                                else if (refChar == readChar) {
                                    gssw_cigar_push_front(nc->cigar, 'M', 1);
#ifdef DEBUG_TRACEBACK
                                    fprintf(stderr, "Match across nodes to %p\n", cn);
#endif
                                }
                                else {
                                    gssw_cigar_push_front(nc->cigar, 'X', 1);
#ifdef DEBUG_TRACEBACK
                                    fprintf(stderr, "Mismatch across nodes to %p\n", cn);
#endif
                                }
                                // is this the last alternate we will look for?
                                if (final_traceback) {
                                    // safe to stop looking for suboptimal scores
                                    score = next_score;
                                    readEnd = next_read_end;
                                    refEnd = next_ref_end;
                                    gapInRead = next_gap_in_read;
                                    gapInRef = next_gap_in_ref;
                                    break;
                                }
                                // A match starting the alignment should have been taken
                                // care of in the within-node function.
                                
                                // If none of those work, try the next option for the previous
                                // node.
                            }
                            else if (UNLIKELY(deflection_idx == alt_alignment->num_deflections &&
                                              score_diff < alt_alignment->score &&
                                              diagonalSourceScore > 0)) {
                                // score is suboptimal or we have already chosen an optimal trace
                                uint16_t alt_score = alt_alignment->score - score_diff;
                                
                                if (alt_score > gssw_min_alt_alignment_score(alt_alignment_stack)
                                    || alt_alignment_stack->current_size < alt_alignment_stack->capacity) {
                                    gssw_add_alignment(alt_alignment_stack, alt_alignment, alt_score,
                                                       readEnd, refEnd, n, cn, Match, Match);
                                }
                            }
                        }
                        else {
#ifdef DEBUG_TRACEBACK
                            fprintf(stderr, "Comparing gap open across nodes: score here %d, penalty %d, source score %d\n", score, gap_open, gapOpenSourceScore);
#endif
                            
                            // If we are in a gap, it would have been a last resort in the node's traceback.
                            uint16_t score_diff = score - (gapOpenSourceScore - gap_open);
                            if (score_diff == 0 && !found_trace) {
                                found_trace = 1;
                                // This node is consistent with an open. Take it.
                                best_prev = cn;
                                gssw_cigar_push_front(nc->cigar, 'D', 1);
                                next_score += gap_open;
                                // Unset the gap flag
                                next_gap_in_read = 0;
#ifdef DEBUG_TRACEBACK
                                fprintf(stderr, "Gap open across nodes to %p\n", cn);
#endif
                                // is this the last alternate we will look for?
                                if (final_traceback) {
                                    // safe to stop looking for suboptimal scores
                                    score = next_score;
                                    readEnd = next_read_end;
                                    refEnd = next_ref_end;
                                    gapInRead = next_gap_in_read;
                                    gapInRef = next_gap_in_ref;
                                    break;
                                }
                            }
                            else if (UNLIKELY(deflection_idx == alt_alignment->num_deflections &&
                                              score_diff < alt_alignment->score &&
                                              gapOpenSourceScore > 0)) {
                                // score is suboptimal or we have already chosen an optimal trace
                                uint16_t alt_score = alt_alignment->score - score_diff;
                                if (alt_score > gssw_min_alt_alignment_score(alt_alignment_stack)
                                    || alt_alignment_stack->current_size < alt_alignment_stack->capacity) {
                                    gssw_add_alignment(alt_alignment_stack, alt_alignment, alt_score,
                                                       readEnd, refEnd, n, cn, ReadGap, Match);
                                }
                            }
                            
#ifdef DEBUG_TRACEBACK
                            fprintf(stderr, "Comparing gap extend across nodes: score here %d, penalty %d, source score %d\n", score, gap_extension, gapExtendSourceScore);
#endif
                            score_diff = score - (gapExtendSourceScore - gap_extension);
                            if (score_diff == 0 && !found_trace) {
                                found_trace = 1;
                                // This node is consistent with an extend. Take it.
                                best_prev = cn;
                                gssw_cigar_push_front(nc->cigar, 'D', 1);
                                next_score += gap_extension;
#ifdef DEBUG_TRACEBACK
                                fprintf(stderr, "Gap extend across nodes to %p\n", cn);
#endif
                                // is this the last alternate we will look for?
                                if (final_traceback) {
                                    // safe to stop looking for suboptimal scores
                                    score = next_score;
                                    readEnd = next_read_end;
                                    refEnd = next_ref_end;
                                    gapInRead = next_gap_in_read;
                                    gapInRef = next_gap_in_ref;
                                    break;
                                }
                            }
                            else if (UNLIKELY(deflection_idx == alt_alignment->num_deflections &&
                                              score_diff < alt_alignment->score &&
                                              gapExtendSourceScore > 0)) {
                                // score is suboptimal or we have already chosen an optimal trace
                                uint16_t alt_score = alt_alignment->score - score_diff;
                                if (alt_score > gssw_min_alt_alignment_score(alt_alignment_stack)
                                    || alt_alignment_stack->current_size < alt_alignment_stack->capacity) {
                                    gssw_add_alignment(alt_alignment_stack, alt_alignment, alt_score,
                                                       readEnd, refEnd, n, cn, ReadGap, ReadGap);
                                }
                            }
                        }
                    }
                    score = next_score;
                    readEnd = next_read_end;
                    refEnd = next_ref_end;
                    gapInRead = next_gap_in_read;
                    gapInRef = next_gap_in_ref;
                }
                
                // Once we go through all the possible previous nodes, we sure hope we found something consistent.
                if(best_prev == NULL) {
                    fprintf(stderr, "error:[gssw] Could not find a valid previous node\n");
                    assert(0);
                }
                
            }
            
#ifdef DEBUG_TRACEBACK
            fprintf(stderr, "best_prev = %p, node = %p\n", best_prev, n);
#endif
            if (best_prev) {
                // Update everything to move to the chosen node
                
                // Score was laready taken care of, as was readEnd, since they
                // depend on the path taken.
                
                n = best_prev;
                // update ref end repeat
                refEnd = n->len - 1;
                ++nc;
#ifdef DEBUG_TRACEBACK
                fprintf(stderr, "transitioning to new node at %p with id %llu at reference position %d with score %d\n", n, n->id, refEnd, score);
#endif
            }
            else {
                // Couldn't find somewhere to go
                
                if(score > 0) {
                    fprintf(stderr, "error:[gssw] Could not find a node to go to!\n");
                    assert(0);
                }
                
#ifdef DEBUG_TRACEBACK
                fprintf(stderr, "soft clip of %i\n", readEnd+1);
#endif
                if (readEnd > -1) {
                    gssw_cigar_push_front(nc->cigar, 'S', readEnd+1);
                }
                break;
            }
        }
        
        if (deflection_idx < alt_alignment->num_deflections) {
            fprintf(stderr, "error:[gssw] Alternate alignment did not find all of its deflections from optimal alignment\n");
            assert(0);
        }
        
#ifdef DEBUG_TRACEBACK
        fprintf(stderr, "at end of traceback loop\n");
        gssw_print_graph_mapping(gm, stderr);
#endif
        gssw_reverse_graph_cigar(gc);
        
        gm->position = (refEnd +1 < 0 ? 0 : refEnd +1); // drop last step by -1 on ref position
        
    }
    
    gssw_delete_multi_align_stack(alt_alignment_stack);
    free(qual_num);

    return gms;

}

gssw_graph_mapping* gssw_graph_trace_back (gssw_graph* graph,
                                           const char* read,
                                           int32_t readLen,
                                           int8_t* nt_table,
                                           int8_t* score_matrix,
                                           uint8_t gap_open,
                                           uint8_t gap_extension,
                                           int8_t start_full_length_bonus,
                                           int8_t end_full_length_bonus) {
    
    gssw_graph_mapping** gms = gssw_graph_trace_back_internal(graph,
                                                              0,
                                                              1,
                                                              0,
                                                              read,
                                                              NULL,
                                                              readLen,
                                                              NULL,
                                                              0,
                                                              nt_table,
                                                              score_matrix,
                                                              gap_open,
                                                              gap_extension,
                                                              start_full_length_bonus,
                                                              end_full_length_bonus);
    gssw_graph_mapping* gm = gms[0];
    free(gms);
    return(gm);
}

gssw_graph_mapping* gssw_graph_trace_back_qual_adj (gssw_graph* graph,
                                                    const char* read,
                                                    const char* qual,
                                                    int32_t readLen,
                                                    int8_t* nt_table,
                                                    int8_t* adj_score_matrix,
                                                    uint8_t gap_open,
                                                    uint8_t gap_extension,
                                                    int8_t start_full_length_bonus,
                                                    int8_t end_full_length_bonus) {

    gssw_graph_mapping** gms = gssw_graph_trace_back_internal(graph,
                                                              0,
                                                              1,
                                                              0,
                                                              read,
                                                              qual,
                                                              readLen,
                                                              NULL,
                                                              0,
                                                              nt_table,
                                                              adj_score_matrix,
                                                              gap_open,
                                                              gap_extension,
                                                              start_full_length_bonus,
                                                              end_full_length_bonus);
    gssw_graph_mapping* gm = gms[0];
    free(gms);
    return(gm);
}

gssw_graph_mapping* gssw_graph_trace_back_pinned (gssw_graph* graph,
                                                  const char* read,
                                                  int32_t readLen,
                                                  gssw_node** pinning_nodes,
                                                  int32_t num_pinning_nodes,
                                                  int8_t* nt_table,
                                                  int8_t* score_matrix,
                                                  uint8_t gap_open,
                                                  uint8_t gap_extension,
                                                  int8_t start_full_length_bonus,
                                                  int8_t end_full_length_bonus) {

    gssw_graph_mapping** gms = gssw_graph_trace_back_internal(graph,
                                                              1,
                                                              1,
                                                              0,
                                                              read,
                                                              NULL,
                                                              readLen,
                                                              pinning_nodes,
                                                              num_pinning_nodes,
                                                              nt_table,
                                                              score_matrix,
                                                              gap_open,
                                                              gap_extension,
                                                              start_full_length_bonus,
                                                              end_full_length_bonus);
    gssw_graph_mapping* gm = gms[0];
    free(gms);
    return(gm);
}

gssw_graph_mapping* gssw_graph_trace_back_pinned_qual_adj (gssw_graph* graph,
                                                           const char* read,
                                                           const char* qual,
                                                           int32_t readLen,
                                                           gssw_node** pinning_nodes,
                                                           int32_t num_pinning_nodes,
                                                           int8_t* nt_table,
                                                           int8_t* adj_score_matrix,
                                                           uint8_t gap_open,
                                                           uint8_t gap_extension,
                                                           int8_t start_full_length_bonus,
                                                           int8_t end_full_length_bonus) {
    
    gssw_graph_mapping** gms = gssw_graph_trace_back_internal(graph,
                                                              1,
                                                              1,
                                                              0,
                                                              read,
                                                              qual,
                                                              readLen,
                                                              pinning_nodes,
                                                              num_pinning_nodes,
                                                              nt_table,
                                                              adj_score_matrix,
                                                              gap_open,
                                                              gap_extension,
                                                              start_full_length_bonus,
                                                              end_full_length_bonus);
    gssw_graph_mapping* gm = gms[0];
    free(gms);
    return(gm);
}

gssw_graph_mapping** gssw_graph_trace_back_pinned_multi (gssw_graph* graph,
                                                         int32_t num_tracebacks,
                                                         int32_t find_internal_node_alts,
                                                         const char* read,
                                                         int32_t readLen,
                                                         gssw_node** pinning_nodes,
                                                         int32_t num_pinning_nodes,
                                                         int8_t* nt_table,
                                                         int8_t* score_matrix,
                                                         uint8_t gap_open,
                                                         uint8_t gap_extension,
                                                         int8_t start_full_length_bonus,
                                                         int8_t end_full_length_bonus) {
    
    return gssw_graph_trace_back_internal(graph,
                                          1,
                                          num_tracebacks,
                                          find_internal_node_alts,
                                          read,
                                          NULL,
                                          readLen,
                                          pinning_nodes,
                                          num_pinning_nodes,
                                          nt_table,
                                          score_matrix,
                                          gap_open,
                                          gap_extension,
                                          start_full_length_bonus,
                                          end_full_length_bonus);
}

gssw_graph_mapping** gssw_graph_trace_back_pinned_qual_adj_multi (gssw_graph* graph,
                                                                  int32_t num_tracebacks,
                                                                  int32_t find_internal_node_alts,
                                                                  const char* read,
                                                                  const char* qual,
                                                                  int32_t readLen,
                                                                  gssw_node** pinning_nodes,
                                                                  int32_t num_pinning_nodes,
                                                                  int8_t* nt_table,
                                                                  int8_t* adj_score_matrix,
                                                                  uint8_t gap_open,
                                                                  uint8_t gap_extension,
                                                                  int8_t start_full_length_bonus,
                                                                  int8_t end_full_length_bonus) {
    return gssw_graph_trace_back_internal(graph,
                                          1,
                                          num_tracebacks,
                                          find_internal_node_alts,
                                          read,
                                          qual,
                                          readLen,
                                          pinning_nodes,
                                          num_pinning_nodes,
                                          nt_table,
                                          adj_score_matrix,
                                          gap_open,
                                          gap_extension,
                                          start_full_length_bonus,
                                          end_full_length_bonus);
}

void gssw_cigar_push_back(gssw_cigar* c, char type, uint32_t length) {
    if (length == 0) {
        return;
    }
    
    if (c->length == 0) {
        c->length = 1;
        c->elements = (gssw_cigar_element*) malloc(c->length * sizeof(gssw_cigar_element));
        c->elements[0].type = type;
        c->elements[0].length = length;
    } else if (type != c->elements[c->length - 1].type) {
        c->length++;
        // change to not realloc every single freakin time
        // but e.g. on doubling
        c->elements = (gssw_cigar_element*) realloc(c->elements, c->length * sizeof(gssw_cigar_element));
        c->elements[c->length - 1].type = type;
        c->elements[c->length - 1].length = length;
    } else {
        c->elements[c->length - 1].length += length;
    }
}

// TODO: this could be made faster / less hacky
void gssw_cigar_push_front(gssw_cigar* c, char type, uint32_t length) {
    gssw_reverse_cigar(c);
    gssw_cigar_push_back(c, type, length);
    gssw_reverse_cigar(c);
    /*
    if (c->length == 0) {
        c->length = 1;
        c->elements = (gssw_cigar_element*) malloc(c->length * sizeof(gssw_cigar_element));
        c->elements[0].type = type;
        c->elements[0].length = length;
    } else if (type != c->elements[0].type) {
        c->length++;
        // change to not realloc every single freakin time
        // but e.g. on doubling
        c->elements = (gssw_cigar_element*) realloc(c->elements, c->length * sizeof(gssw_cigar_element));
        //gssw_cigar_element* new = (gssw_cigar_element*) malloc(c->length * sizeof(gssw_cigar_element));
        //(gssw_cigar_element*) memcpy(new + sizeof(gssw_cigar_element), c->elements, c->length-1 * sizeof(gssw_cigar_element));
        //free(c->elements);
        int32_t i;
        for (i = c->length-1; i > 1; --i) {
            c->elements[i].type = c->elements[i-1].type;
            c->elements[i].length = c->elements[i-1].length;
        }
        c->elements[0].type = type;
        c->elements[0].length = length;
    } else {
        c->elements[0].length += length;
    }
    */
}

void gssw_reverse_cigar(gssw_cigar* c) {
    if (!c->length) return; // bail out
    gssw_cigar* reversed = (gssw_cigar*)malloc(sizeof(gssw_cigar));
    reversed->length = c->length;
    reversed->elements = (gssw_cigar_element*) malloc(c->length * sizeof(gssw_cigar_element));
    gssw_cigar_element* c1 = c->elements;
    gssw_cigar_element* c2 = reversed->elements;
    int32_t s = 0;
    int32_t e = c->length - 1;
    while (LIKELY(s <= e)) {
        c2[s] = c1[e];
        c2[e] = c1[s];
        ++ s;
        -- e;
    }
    free(c->elements);
    c->elements = reversed->elements;
    free(reversed);
}

void gssw_print_cigar(gssw_cigar* c, FILE* out) {
    int i;
    int l = c->length;
    gssw_cigar_element* e = c->elements;
    for (i=0; LIKELY(i < l); ++i, ++e) {
        fprintf(out, "%i%c", e->length, e->type);
    }
}

void gssw_cigar_destroy(gssw_cigar* c) {
    free(c->elements);
    c->elements = NULL;
    free(c);
}

void gssw_seed_destroy(gssw_seed* s) {
    free(s->pvE);
    s->pvE = NULL;
    free(s->pvHStore);
    s->pvE = NULL;
    free(s);
}

//TODO: why is score_matrix even an argument here?
gssw_node* gssw_node_create(void* data,
                            const uint64_t id,
                            const char* seq,
                            const int8_t* nt_table,
                            const int8_t* score_matrix) {
    gssw_node* n = calloc(1, sizeof(gssw_node));
    int32_t len = strlen(seq);
    n->id = id;
    n->len = len;
    n->seq = (char*)malloc(len+1);
    strncpy(n->seq, seq, len); n->seq[len] = 0;
    n->data = data;
    n->num = gssw_create_num(seq, len, nt_table);
    n->count_prev = 0; // are these be set == 0 by calloc?
    n->count_next = 0;
    n->alignment = NULL;
    return n;
}

// for reuse of graph through multiple alignments
void gssw_node_clear_alignment(gssw_node* n) {
    gssw_align_destroy(n->alignment);
    n->alignment = NULL;
}

void gssw_profile_destroy(gssw_profile* prof) {
    free(prof->profile_byte);
    free(prof->profile_word);
    free(prof);
}

void gssw_node_destroy(gssw_node* n) {
    free(n->seq);
    free(n->num);
    free(n->prev);
    free(n->next);
    if (n->alignment) {
        gssw_align_destroy(n->alignment);
    }
    free(n);
}

//void node_clear_alignment(node* n) {
//    align_clear_matrix_and_seed(n->alignment);
//}

void gssw_node_add_prev(gssw_node* n, gssw_node* m) {
    ++n->count_prev;
    n->prev = (gssw_node**)realloc(n->prev, n->count_prev*sizeof(gssw_node*));
    n->prev[n->count_prev -1] = m;
}

void gssw_node_add_next(gssw_node* n, gssw_node* m) {
    ++n->count_next;
    n->next = (gssw_node**)realloc(n->next, n->count_next*sizeof(gssw_node*));
    n->next[n->count_next -1] = m;
}

void gssw_nodes_add_edge(gssw_node* n, gssw_node* m) {
    //fprintf(stderr, "connecting %u -> %u\n", n->id, m->id);
    // check that there isn't already an edge
    uint32_t k;
    // check to see if there is an edge from n -> m, and exit if so
    for (k=0; k<n->count_next; ++k) {
        if (n->next[k] == m) {
            return;
        }
    }
    gssw_node_add_next(n, m);
    gssw_node_add_prev(m, n);
}

void gssw_node_del_prev(gssw_node* n, gssw_node* m) {
    gssw_node** x = (gssw_node**)malloc(n->count_prev*sizeof(gssw_node*));
    int i = 0;
    gssw_node** np = n->prev;
    for ( ; i < n->count_prev; ++i, ++np) {
        if (*np != m) {
            x[i] = *np;
        }
    }
    free(n->prev);
    n->prev = x;
    --n->count_prev;
}

void gssw_node_del_next(gssw_node* n, gssw_node* m) {
    gssw_node** x = (gssw_node**)malloc(n->count_next*sizeof(gssw_node*));
    int i = 0;
    gssw_node** nn = n->next;
    for ( ; i < n->count_next; ++i, ++nn) {
        if (*nn != m) {
            x[i] = *nn;
        }
    }
    free(n->next);
    n->next = x;
    --n->count_next;
}

void gssw_nodes_del_edge(gssw_node* n, gssw_node* m) {
    gssw_node_del_next(n, m);
    gssw_node_del_prev(m, n);
}

void gssw_node_replace_prev(gssw_node* n, gssw_node* m, gssw_node* p) {
    int i = 0;
    gssw_node** np = n->prev;
    for ( ; i < n->count_prev; ++i, ++np) {
        if (*np == m) {
            *np = p;
        }
    }
}

void gssw_node_replace_next(gssw_node* n, gssw_node* m, gssw_node* p) {
    int i = 0;
    gssw_node** nn = n->next;
    for ( ; i < n->count_next; ++i, ++nn) {
        if (*nn == m) {
            *nn = p;
        }
    }
}

gssw_seed* gssw_create_seed_byte(int32_t readLen, gssw_node** prev, int32_t count) {
    int32_t j = 0, k = 0;
    for (k = 0; k < count; ++k) {
        if (!prev[k]->alignment) {
            fprintf(stderr, "error:[gssw] cannot align because node predecessors cannot provide seed\n");
            fprintf(stderr, "failing is node %llu\n", prev[k]->id);
            exit(1);
        }
    }
    __m128i vZero = _mm_set1_epi32(0);
    int32_t segLen = (readLen + 15) / 16;
    gssw_seed* seed = (gssw_seed*)calloc(1, sizeof(gssw_seed));
    if (!(!posix_memalign((void**)&seed->pvE,      sizeof(__m128i), segLen*sizeof(__m128i)) &&
          !posix_memalign((void**)&seed->pvHStore, sizeof(__m128i), segLen*sizeof(__m128i)))) {
        fprintf(stderr, "error:[gssw] Could not allocate memory for alignment seed\n"); exit(1);
        exit(1);
    }
    memset(seed->pvE,      0, segLen*sizeof(__m128i));
    memset(seed->pvHStore, 0, segLen*sizeof(__m128i));
    // take the max of all inputs
    __m128i pvE = vZero, pvH = vZero, ovE = vZero, ovH = vZero;
    for (j = 0; j < segLen; ++j) {
        pvE = vZero; pvH = vZero;
        for (k = 0; k < count; ++k) {
            ovE = _mm_load_si128(prev[k]->alignment->seed.pvE + j);
            ovH = _mm_load_si128(prev[k]->alignment->seed.pvHStore + j);
            pvE = _mm_max_epu8(pvE, ovE);
            pvH = _mm_max_epu8(pvH, ovH);
        }
        _mm_store_si128(seed->pvHStore + j, pvH);
        _mm_store_si128(seed->pvE + j, pvE);
    }
    return seed;
}

gssw_seed* gssw_create_seed_word(int32_t readLen, gssw_node** prev, int32_t count) {
    int32_t j = 0, k = 0;
    for (k = 0; k < count; ++k) {
        if (!prev[k]->alignment) {
            fprintf(stderr, "error:[gssw] cannot align because node predecessors cannot provide seed\n");
            fprintf(stderr, "failing is node %llu\n", prev[k]->id);
            exit(1);
        }
    }
    __m128i vZero = _mm_set1_epi32(0);
    int32_t segLen = (readLen + 7) / 8;
    gssw_seed* seed = (gssw_seed*)calloc(1, sizeof(gssw_seed));
    if (!(!posix_memalign((void**)&seed->pvE,      sizeof(__m128i), segLen*sizeof(__m128i)) &&
          !posix_memalign((void**)&seed->pvHStore, sizeof(__m128i), segLen*sizeof(__m128i)))) {
        fprintf(stderr, "error:[gssw] Could not allocate memory for alignment seed\n"); exit(1);
        exit(1);
    }
    memset(seed->pvE,      0, segLen*sizeof(__m128i));
    memset(seed->pvHStore, 0, segLen*sizeof(__m128i));
    // take the max of all inputs
    __m128i pvE = vZero, pvH = vZero, ovE = vZero, ovH = vZero;
    for (j = 0; j < segLen; ++j) {
        pvE = vZero; pvH = vZero;
        for (k = 0; k < count; ++k) {
            ovE = _mm_load_si128(prev[k]->alignment->seed.pvE + j);
            ovH = _mm_load_si128(prev[k]->alignment->seed.pvHStore + j);
            pvE = _mm_max_epu16(pvE, ovE);
            pvH = _mm_max_epu16(pvH, ovH);
        }
        _mm_store_si128(seed->pvHStore + j, pvH);
        _mm_store_si128(seed->pvE + j, pvE);
    }
    return seed;
}

gssw_graph*
gssw_graph_fill_internal (gssw_graph* graph,
                          const char* read_seq,
                          const char* read_qual,
                          const int8_t* nt_table,
                          const int8_t* score_matrix,
                          const uint8_t weight_gapO,
                          const uint8_t weight_gapE,
                          const int8_t start_full_length_bonus,
                          const int8_t end_full_length_bonus,
                          const int32_t maskLen,
                          const int8_t score_size,
                          bool save_matrixes) {
    int32_t read_length = strlen(read_seq);
    int8_t* read_num = gssw_create_num(read_seq, read_length, nt_table);
    int8_t* qual_num = gssw_create_qual_num(read_qual, read_length);
    gssw_profile* prof;
    if (read_qual) {
        prof = gssw_qual_adj_init (read_num, qual_num, read_length, score_matrix, 5, start_full_length_bonus,
                                   end_full_length_bonus, score_size);
    }
    else {
        prof = gssw_init(read_num, read_length, score_matrix, 5, start_full_length_bonus, end_full_length_bonus, score_size);
    }
    gssw_seed* seed = NULL;
    uint16_t max_score = 0;
    uint32_t i;
    gssw_node** npp = &graph->nodes[0];
    // seed the head nodes of the graph
    for (i = 0; i < graph->size; ++i, ++npp) {
        gssw_node* n = *npp;
        // head node condition
        if (!n->count_prev) {
            if (prof->profile_byte) {
                seed = gssw_create_seed_byte(prof->readLen, n->prev, n->count_prev);
            } else {
                seed = gssw_create_seed_word(prof->readLen, n->prev, n->count_prev);
            }
            gssw_node* filled_node = gssw_node_fill(n, prof, weight_gapO, weight_gapE, maskLen, save_matrixes, seed);
            gssw_seed_destroy(seed); seed = NULL; // cleanup seed
            // test if we have exceeded the score dynamic range
            if (prof->profile_byte && !filled_node) {
                free(prof->profile_byte);
                prof->profile_byte = NULL;
                free(read_num);
                free(qual_num);
                gssw_profile_destroy(prof);
                if (read_qual) {
                    return gssw_graph_fill_pinned_qual_adj(graph, read_seq, read_qual, nt_table, score_matrix, weight_gapO,
                                                           weight_gapE, start_full_length_bonus, end_full_length_bonus,
                                                           maskLen, 1, save_matrixes);
                } else {
                    return gssw_graph_fill_pinned(graph, read_seq, nt_table, score_matrix, weight_gapO, weight_gapE,
                                                  start_full_length_bonus, end_full_length_bonus, maskLen, 1, save_matrixes);
                }
            } else {
                if (!graph->max_node || n->alignment->score1 > max_score) {
                    graph->max_node = n;
                    max_score = n->alignment->score1;
                }
            }
        }
    }
    npp = &graph->nodes[0];
    // for each node, from start to finish in the partial order (which should be sorted topologically)
    // generate a seed from input nodes or use existing (e.g. for subgraph traversal here)
    for (i = 0; i < graph->size; ++i, ++npp) {
        gssw_node* n = *npp;
        // get seed from parents (max of multiple inputs)
        if (n->count_prev) {
            if (prof->profile_byte) {
                seed = gssw_create_seed_byte(prof->readLen, n->prev, n->count_prev);
            } else {
                seed = gssw_create_seed_word(prof->readLen, n->prev, n->count_prev);
            }
            gssw_node* filled_node = gssw_node_fill(n, prof, weight_gapO, weight_gapE, maskLen, save_matrixes, seed);
            gssw_seed_destroy(seed); seed = NULL; // cleanup seed
            // test if we have exceeded the score dynamic range
            if (prof->profile_byte && !filled_node) {
                free(prof->profile_byte);
                prof->profile_byte = NULL;
                free(read_num);
                free(qual_num);
                gssw_profile_destroy(prof);
                if (read_qual) {
                    return gssw_graph_fill_pinned_qual_adj(graph, read_seq, read_qual, nt_table, score_matrix, weight_gapO,
                                                           weight_gapE, start_full_length_bonus, end_full_length_bonus,
                                                           maskLen, 1, save_matrixes);
                } else {
                    return gssw_graph_fill_pinned(graph, read_seq, nt_table, score_matrix, weight_gapO, weight_gapE,
                                                  start_full_length_bonus, end_full_length_bonus, maskLen, 1, save_matrixes);
                }
            } else {
                if (!graph->max_node || n->alignment->score1 > max_score) {
                    graph->max_node = n;
                    max_score = n->alignment->score1;
                }
            }
        }
    }

    free(read_num);
    free(qual_num);
    gssw_profile_destroy(prof);

    return graph;

}

gssw_graph*
gssw_graph_fill (gssw_graph* graph,
                 const char* read_seq,
                 const int8_t* nt_table,
                 const int8_t* score_matrix,
                 const uint8_t weight_gapO,
                 const uint8_t weight_gapE,
                 const int8_t start_full_length_bonus,
                 const int8_t end_full_length_bonus,
                 const int32_t maskLen,
                 const int8_t score_size,
                 bool save_matrixes) {
    
    return gssw_graph_fill_internal(graph, read_seq, NULL, nt_table, score_matrix,
                                    weight_gapO, weight_gapE, start_full_length_bonus,
                                    end_full_length_bonus, maskLen, score_size, save_matrixes);
}


/* Assumes that offset has already been removed from read_qual */
gssw_graph*
gssw_graph_fill_qual_adj(gssw_graph* graph,
                         const char* read_seq,
                         const char* read_qual,
                         const int8_t* nt_table,
                         const int8_t* adj_score_matrix,
                         const uint8_t weight_gapO,
                         const uint8_t weight_gapE,
                         const int8_t start_full_length_bonus,
                         const int8_t end_full_length_bonus,
                         const int32_t maskLen,
                         const int8_t score_size,
                         bool save_matrixes) {

    return gssw_graph_fill_internal(graph, read_seq, read_qual, nt_table, adj_score_matrix,
                                    weight_gapO, weight_gapE, start_full_length_bonus,
                                    end_full_length_bonus, maskLen, score_size, save_matrixes);
}


gssw_graph*
gssw_graph_fill_pinned (gssw_graph* graph,
                        const char* read_seq,
                        const int8_t* nt_table,
                        const int8_t* score_matrix,
                        const uint8_t weight_gapO,
                        const uint8_t weight_gapE,
                        const int8_t start_full_length_bonus,
                        const int8_t end_full_length_bonus,
                        const int32_t maskLen,
                        const int8_t score_size,
                        bool save_matrixes) {
                        
    // TODO: now that we have full length bonuses for unpinned alignment, this
    // doesn't do anything different than the unpinned version...
    
    return gssw_graph_fill_internal(graph, read_seq, NULL, nt_table, score_matrix,
                                    weight_gapO, weight_gapE, start_full_length_bonus,
                                    end_full_length_bonus, maskLen, score_size, save_matrixes);
}

gssw_graph*
gssw_graph_fill_pinned_qual_adj(gssw_graph* graph,
                                const char* read_seq,
                                const char* read_qual,
                                const int8_t* nt_table,
                                const int8_t* adj_score_matrix,
                                const uint8_t weight_gapO,
                                const uint8_t weight_gapE,
                                const int8_t start_full_length_bonus,
                                const int8_t end_full_length_bonus,
                                const int32_t maskLen,
                                const int8_t score_size,
                                bool save_matrixes) {
    
    return gssw_graph_fill_internal(graph, read_seq, read_qual, nt_table, adj_score_matrix,
                                    weight_gapO, weight_gapE, start_full_length_bonus,
                                    end_full_length_bonus, maskLen, score_size, save_matrixes);
}


gssw_node*
gssw_node_fill (gssw_node* node,
                const gssw_profile* prof,
                const uint8_t weight_gapO,
                const uint8_t weight_gapE,
                const int32_t maskLen,
                bool save_matrixes,
                const gssw_seed* seed) {

    gssw_alignment_end* bests = NULL;
    int32_t readLen = prof->readLen;

    //alignment_end* best = (alignment_end*)calloc(1, sizeof(alignment_end));
    gssw_align* alignment = node->alignment;

    if (alignment) {
        // clear old alignment
        gssw_align_destroy(alignment);
    }
    // and build up a new one
    node->alignment = alignment = gssw_align_create();

    
    // if we have parents, we should generate a new seed as the max of each vector
    // if one of the parents has moved into uint16_t space, we need to account for this
    // otherwise, just use the single parent alignment result as seed
    // or, if no parents, run unseeded

    // to decrease code complexity, we assume the same stripe size for the entire graph
    // this is ensured by changing the stripe size for the entire graph in graph_fill if any node scores >= 255

    // Find the alignment scores and ending positions
    if (prof->profile_byte) {
        // Do a byte-sized fill
        
        if (gssw_sse2_enabled) {
            // Use SSE2
            bests = gssw_sw_sse2_byte((const int8_t*)node->num, 0, node->len, readLen, weight_gapO, weight_gapE, prof->profile_byte, -1, prof->bias, maskLen, alignment, save_matrixes, seed);
        } else {
            // Use pure software
            bests = gssw_sw_software_byte((const int8_t*)node->num, 0, node->len, readLen, weight_gapO, weight_gapE, prof->profile_byte, -1, prof->bias, maskLen, alignment, seed);
        }
        if (bests[0].score == 255) {
            free(bests);
            gssw_align_clear_matrix_and_seed(alignment);
            return 0; // re-run from external context
        }
    } else if (prof->profile_word) {
        if (gssw_sse2_enabled) {
            // Use SSE2
            bests = gssw_sw_sse2_word((const int8_t*)node->num, 0, node->len, readLen, weight_gapO, weight_gapE, prof->profile_word, -1, maskLen, alignment, save_matrixes, seed);
        } else {
            // Use software
            bests = gssw_sw_software_word((const int8_t*)node->num, 0, node->len, readLen, weight_gapO, weight_gapE, prof->profile_word, -1, maskLen, alignment, seed);
        }
    } else {
        fprintf(stderr, "Please call the function ssw_init before ssw_align.\n");
        return 0;
    }
    
    //fprintf(stderr, "### best: score(%d), ref_end(%d), read_end(%d); 2nd best: score(%d), ref_end(%d), read_end(%d)\n", bests[0].score, bests[0].ref, bests[0].read, bests[1].score, bests[1].ref, bests[1].read);
    
    alignment->score1 = bests[0].score;
    alignment->ref_end1 = bests[0].ref;
    alignment->read_end1 = bests[0].read;
    if (maskLen >= 15) {
        alignment->score2 = bests[1].score;
        alignment->ref_end2 = bests[1].ref;
    } else {
        alignment->score2 = 0;
        alignment->ref_end2 = -1;
    }
    free(bests);

    return node;

}

gssw_graph* gssw_graph_create(uint32_t size) {
    gssw_graph* g = calloc(1, sizeof(gssw_graph));
    g->nodes = malloc(size*sizeof(gssw_node*));
    if (!g || !g->nodes) { fprintf(stderr, "error:[gssw] Could not allocate memory for graph of %u nodes.\n", size); exit(1); }
    return g;
}

void gssw_graph_clear_alignment(gssw_graph* g) {
    g->max_node = NULL;
    uint32_t i;
    for (i = 0; i < g->size; ++i) {
        gssw_node_clear_alignment(g->nodes[i]);
    }
}

void gssw_graph_destroy(gssw_graph* g) {
    uint32_t i;
    for (i = 0; i < g->size; ++i) {
        gssw_node_destroy(g->nodes[i]);
    }
    g->max_node = NULL;
    free(g->nodes);
    g->nodes = NULL;
    free(g);
}

uint32_t gssw_graph_add_node(gssw_graph* graph, gssw_node* node) {
    if (UNLIKELY(graph->size % 1024 == 0)) {
        size_t old_size = graph->size * sizeof(void*);
        size_t increment = 1024 * sizeof(void*);
        if (UNLIKELY(!(graph->nodes = realloc((void*)graph->nodes, old_size + increment)))) {
            fprintf(stderr, "error:[gssw] could not allocate memory for graph\n"); exit(1);
        }
    }
    ++graph->size;
    graph->nodes[graph->size-1] = node;
    return graph->size;
}

int8_t* gssw_create_num(const char* seq,
                        const int32_t len,
                        const int8_t* nt_table) {
    int32_t m;
    int8_t* num = (int8_t*)malloc(len);
    for (m = 0; m < len; ++m) num[m] = nt_table[(int)seq[m]];
    return num;
}

int8_t* gssw_create_qual_num(const char* qual,
                             const int32_t len) {
    if (qual == NULL) {
        return NULL;
    }
    int32_t m;
    int8_t* num = (int8_t*)malloc(len);
    for (m = 0; m < len; ++m) num[m] = (int8_t) qual[m];
    return num;
}


int8_t* gssw_create_score_matrix(int32_t match, int32_t mismatch) {
    // initialize scoring matrix for genome sequences
    //  A  C  G  T    N (or other ambiguous code)
    //  2 -2 -2 -2     0    A
    // -2  2 -2 -2     0    C
    // -2 -2  2 -2     0    G
    // -2 -2 -2  2     0    T
    //    0  0  0  0  0    N (or other ambiguous code)
    int32_t l, k, m;
    int8_t* mat = (int8_t*)calloc(25, sizeof(int8_t));
    for (l = k = 0; l < 4; ++l) {
        for (m = 0; m < 4; ++m) mat[k++] = l == m ? match : - mismatch;    /* weight_match : -weight_mismatch */
        mat[k++] = 0; // ambiguous base: no penalty
    }
    for (m = 0; m < 5; ++m) mat[k++] = 0;
    return mat;
}

int8_t* gssw_create_nt_table(void) {
    int8_t* ret_nt_table = calloc(128, sizeof(int8_t));
    int8_t nt_table[128] = {
        4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
        4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
        4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
        4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
        4, 0, 4, 1,  4, 4, 4, 2,  4, 4, 4, 4,  4, 4, 4, 4,
        4, 4, 4, 4,  3, 0, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
        4, 0, 4, 1,  4, 4, 4, 2,  4, 4, 4, 4,  4, 4, 4, 4,
        4, 4, 4, 4,  3, 0, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4
    };
    memcpy(ret_nt_table, nt_table, 128*sizeof(int8_t));
    return ret_nt_table;
}

/* Rounds a double to nearest int8_t. */
int8_t gssw_round8_t(double x) {
    int8_t int_x = (int8_t) x;
    if (x >= 0.0) {
        if (x - int_x >= 0.5) {
            return int_x + (int8_t) 1;
        }
        else {
            return int_x;
        }
    }
    else {
        if (int_x - x >= 0.5) {
            return int_x - (int8_t) 1;
        }
        else {
            return int_x;
        }
    }
}

/* Simple (slow) algorithm for finding greatest common factor of the scores, not performance critical. */
int8_t gssw_score_gcf(const int8_t* score_matrix, int32_t alphabet_size) {
    int8_t* score_matrix_copy = (int8_t*) malloc(sizeof(int8_t) * alphabet_size * alphabet_size);
    int32_t i;
    for (i = 0; i < alphabet_size * alphabet_size; ++i) {
        score_matrix_copy[i] = score_matrix[i];
    }
    int8_t gcf = 1;
    int8_t factor = 2;
    int8_t min_score = 127;
    for (i = 0; i < alphabet_size * alphabet_size; ++i) {
        if (abs(score_matrix[i]) < min_score) {
            min_score = (int8_t) abs(score_matrix[i]);
        }
    }
    while (factor <= min_score / 2) {
        int8_t common_factor = 1;
        for (i = 0; i < alphabet_size * alphabet_size; ++i) {
            if (score_matrix_copy[i] % factor != 0) {
                common_factor = 0;
                break;
            }
        }
        if (common_factor) {
            gcf *= factor;
            for (i = 0; i < alphabet_size * alphabet_size; ++i) {
                score_matrix_copy[i] /= factor;
            }
            min_score /= factor;
        }
        else {
            factor++;
        }
    }
    free(score_matrix_copy);
    return gcf;
}

int8_t gssw_verify_valid_log_odds_score_matrix(const int8_t* score_matrix, const double* char_freqs,
                                               uint32_t alphabet_size) {
    int32_t i, j;
    int8_t contains_positive_score = 0.0;
    for (i = 0; i < alphabet_size * alphabet_size; i++) {
        if (score_matrix[i] > 0) {
            contains_positive_score = 1;
            break;
        }
    }
    if (!contains_positive_score) {
        return 0;
    }
    
    double expected_score = 0.0;
    for (i = 0; i < alphabet_size; i++) {
        for (j = 0; j < alphabet_size; j++) {
            expected_score += char_freqs[i] * char_freqs[j] * score_matrix[i * alphabet_size + j];
        }
    }
    return (int8_t) (expected_score < 0.0);
}

/* Returns the total probability in the distribution of aligned characters with a given logarithm base */
double gssw_alignment_partition_func(double lam, const int8_t* score_matrix, const double* char_freqs,
                                     uint32_t alphabet_size) {
    int32_t i, j;
    double partition = 0.0;
    for (i = 0; i < alphabet_size; i++) {
        for (j = 0; j < alphabet_size; j++) {
            partition += char_freqs[i] * char_freqs[j] * exp(lam * score_matrix[i * alphabet_size + j]);
        }
    }
    
    if (isnan(partition)) {
        fprintf(stderr, "error:[gssw] overflow error in log-odds base recovery subroutine.\n");
        exit(EXIT_FAILURE);
    }

    return partition;
}

/* Numerical routine to compute the base of the logarithm that translates alignment scores to log-odds */
double gssw_recover_log_base(const int8_t* score_matrix, const double* char_freqs, uint32_t alphabet_size, double tol) {

    if (!gssw_verify_valid_log_odds_score_matrix(score_matrix, char_freqs, alphabet_size)) {
        fprintf(stderr, "error:[gssw] score matrix does not correspond to log-odds of any distribution, cannot adjust for base quality.\n");
        exit(EXIT_FAILURE);
    }
    
    // searching for a positive value (because it's a base of a logarithm)
    double lower_bound;
    double upper_bound;
    
    // arbitrary starting point greater than zero
    double lam = 1.0;
    // search for a window containing lambda where total probability is 1
    double partition = gssw_alignment_partition_func(lam, score_matrix, char_freqs, alphabet_size);
    if (partition < 1.0) {
        lower_bound = lam;
        while (partition <= 1.0) {
            lower_bound = lam;
            lam *= 2.0;
            partition = gssw_alignment_partition_func(lam, score_matrix, char_freqs, alphabet_size);
        }
        upper_bound = lam;
    }
    else {
        upper_bound = lam;
        while (partition >= 1.0) {
            upper_bound = lam;
            lam /= 2.0;
            partition = gssw_alignment_partition_func(lam, score_matrix, char_freqs, alphabet_size);
        }
        lower_bound = lam;
    }
    
    // bisect to find a log base where total probability is 1
    while (upper_bound / lower_bound - 1.0 > tol) {
        lam = 0.5 * (lower_bound + upper_bound);
        if (gssw_alignment_partition_func(lam, score_matrix, char_freqs, alphabet_size) < 1.0) {
            lower_bound = lam;
        }
        else {
            upper_bound = lam;
        }
    }

    return 0.5 * (lower_bound + upper_bound);
}

double gssw_dna_recover_log_base(int8_t match, int8_t mismatch, double gc_content, double tol) {
    double gc_freq = gc_content / 2.0;
    double at_freq = 0.5 - gc_freq;
    double* nt_freqs = (double*) malloc(sizeof(double) * 4);
    nt_freqs[0] = at_freq; nt_freqs[1] = gc_freq; nt_freqs[2] = gc_freq; nt_freqs[3] = at_freq;
    int8_t* score_matrix = (int8_t*) malloc(sizeof(int8_t) * 16);
    int32_t i, j;
    for (i = 0; i < 4; i++) {
        for (j = 0; j < 4; j++) {
            score_matrix[i * 4 + j] = (i == j) ? match : -mismatch;
        }
    }
    double log_base = gssw_recover_log_base(score_matrix, nt_freqs, 4, 1e-12);
    free(nt_freqs);
    free(score_matrix);
    return log_base;
}

/* Returns a 3-dimensional matrix of quality-adjusted scores indexed by (qual score) x (ref base) x (query base). */
int8_t* gssw_adjusted_qual_matrix(uint8_t max_qual, const int8_t* score_matrix, const double* char_freqs,
                                  uint32_t alphabet_size, double tol){
    
    int32_t i, j, k, q;
    // recover base of logarithm used in log odds scores
    double lam;
    // factoring out GCF can avoid numerical problems without affecting correctness
    int8_t gcf = gssw_score_gcf(score_matrix, alphabet_size);
    int8_t* score_matrix_scaled = (int8_t*) malloc(sizeof(int8_t) * alphabet_size * alphabet_size);
    for (i = 0; i < alphabet_size * alphabet_size; i++) {
        score_matrix_scaled[i] = score_matrix[i] / gcf;
    }
    lam = gssw_recover_log_base(score_matrix_scaled, char_freqs, alphabet_size, tol) / gcf;
    free(score_matrix_scaled);
    
    // recover the emission probabilities of the align state of the HMM
    int32_t mat_size = alphabet_size * alphabet_size;
    double* align_prob = (double*) malloc(sizeof(double) * mat_size);

    for (i = 0; i < alphabet_size; i++) {
        for (j = 0; j < alphabet_size; j++) {
            align_prob[i * alphabet_size + j] = exp(lam * score_matrix[i * alphabet_size + j])
                                                      * char_freqs[i] * char_freqs[j];
        }
    }

    // compute the sum of the emission probabilities under a base error
    double* align_complement_prob = (double*) malloc(sizeof(double) * mat_size);
    for (i = 0; i < alphabet_size; i++) {
        for (j = 0; j < alphabet_size; j++) {
            align_complement_prob[i * alphabet_size + j] = 0.0;
            for (k = 0; k < alphabet_size; k++) {
                if (k != j) {
                    align_complement_prob[i * alphabet_size + j] += align_prob[i * alphabet_size + k];
                }
            }
        }
    }
    
    // quality score of random guessing
    int8_t lowest_meaningful_qual = gssw_round8_t(-10.0 * log10(1.0 - 1.0 / alphabet_size));
    
    // compute the adjusted alignment scores for each quality level
    int8_t* adj_qual_mat = (int8_t*) calloc(mat_size * (max_qual + (int8_t) 1), sizeof(int8_t));
    double score, err;
    for (q = lowest_meaningful_qual; q <= max_qual; q++) {
        err = pow(10.0, -q / 10.0);
        for (i = 0; i < alphabet_size; i++) {
            for (j = 0; j < alphabet_size; j++) {
                score = log(((1.0 - err) * align_prob[i * alphabet_size + j] + (err / (alphabet_size - 1.0)) * align_complement_prob[i * alphabet_size + j])
                            / (char_freqs[i] * ((1.0 - err) * char_freqs[j] + (err / (alphabet_size - 1.0)) * (1.0 - char_freqs[j]))));
                score /= lam;
                adj_qual_mat[q * mat_size + i * alphabet_size + j] = gssw_round8_t(score);
            }
        }
    }

    free(align_complement_prob);
    free(align_prob);

    return adj_qual_mat;
}

/* Returns a 3-dimensional matrix of quality-adjusted scores indexed by (qual score) x (ref base) x (query base)
 * that have been scaled up to (at most) a max score to accentuate differences, also adjusts value of gap penalties. */
int8_t* gssw_scaled_adjusted_qual_matrix(int8_t max_score, uint8_t max_qual, int8_t* gap_open_out, int8_t* gap_extend_out,
                                         const int8_t* score_matrix, const double* char_freqs, uint32_t alphabet_size,
                                         double tol) {

    int8_t gap_extend = *gap_extend_out;
    int8_t gap_open = *gap_open_out;

    // find largest integer multiplier that keeps all scores under maximum
    uint8_t multiplier = (uint8_t) abs(max_score);
    if (abs(max_score / gap_open) < multiplier) {
        multiplier = (uint8_t) max_score / gap_open;
    }
    if (abs(max_score / gap_extend) < multiplier) {
        multiplier = (uint8_t) max_score / gap_extend;
    }
    int32_t i;
    for (i = 0; i < alphabet_size * alphabet_size; i++) {
        if (abs(max_score / score_matrix[i]) < multiplier) {
            multiplier = (uint8_t) abs(max_score / score_matrix[i]);
        }
    }

    if (multiplier == 0) {
        fprintf(stderr, "error:[gssw] max scaled score is smaller than baseline score.\n");
        exit(EXIT_FAILURE);
    }

    // scale scores by multiplier
    int8_t* scaled_score_mat = (int8_t*) malloc(sizeof(int8_t) * alphabet_size * alphabet_size);

    for (i = 0; i < alphabet_size * alphabet_size; i++) {
        scaled_score_mat[i] = multiplier * score_matrix[i];
    }

    // compute adjusted score matrices
    int8_t* scaled_adj_qual_mat = gssw_adjusted_qual_matrix(max_qual, scaled_score_mat, char_freqs, alphabet_size, tol);

    free(scaled_score_mat);

    *gap_open_out = multiplier * gap_open;
    *gap_extend_out = multiplier * gap_extend;

    return scaled_adj_qual_mat;
}

int8_t* gssw_add_ambiguous_char_to_adjusted_matrix(int8_t* adj_mat, uint8_t max_qual, uint32_t alphabet_size) {
    int32_t mat_size = alphabet_size * alphabet_size;
    int32_t aug_alph_size = alphabet_size + 1;
    int32_t aug_mat_size = aug_alph_size * aug_alph_size;
    
    int8_t* aug_adj_mat = (int8_t*) malloc(sizeof(int8_t) * aug_mat_size * (max_qual + 1));
    
    int32_t q, i, j;
    for (q = 0; q <= max_qual; q++) {
        for (i = 0; i < aug_alph_size; i++) {
            for (j = 0; j < aug_alph_size; j++) {
                if (i == alphabet_size || j == alphabet_size) {
                    aug_adj_mat[q * aug_mat_size + i * aug_alph_size + j] = 0;
                }
                else {
                    aug_adj_mat[q * aug_mat_size + i * aug_alph_size + j] = adj_mat[q * mat_size + i * alphabet_size + j];
                }
            }
        }
    }
    
    return aug_adj_mat;
}

// automatically adds 0-scoring N to the final row and column
int8_t* gssw_dna_scaled_adjusted_qual_matrix(int8_t max_score, uint8_t max_qual, int8_t* gap_open_out,
                                             int8_t* gap_extend_out, int8_t match_score, int8_t mismatch_score,
                                             double gc_content, double tol) {
    
    double gc_freq = gc_content / 2.0;
    double at_freq = 0.5 - gc_freq;
    double* nt_freqs = (double*) malloc(sizeof(double) * 4);
    nt_freqs[0] = at_freq; nt_freqs[1] = gc_freq; nt_freqs[2] = gc_freq; nt_freqs[3] = at_freq;
    
    int32_t i, j;
    int8_t* score_matrix = (int8_t*) malloc(sizeof(int8_t) * 16);
    for (i = 0; i < 4; ++i) {
        for (j = 0; j < 4; ++j) {
            score_matrix[i * 4 + j] = (i == j) ? match_score : -mismatch_score;
        }
    }
    
    
    int8_t* adj_mat_init = gssw_scaled_adjusted_qual_matrix(max_score, max_qual, gap_open_out,
                                                            gap_extend_out, score_matrix,
                                                            nt_freqs, 4, tol);
    
    int8_t* adj_mat = gssw_add_ambiguous_char_to_adjusted_matrix(adj_mat_init, max_qual, 4);
    
    free(nt_freqs);
    free(score_matrix);
    free(adj_mat_init);
    
    return adj_mat;
}

gssw_multi_align_stack* gssw_new_multi_align_stack(int32_t capacity) {
    gssw_multi_align_stack* stack = (gssw_multi_align_stack*) malloc(sizeof(gssw_multi_align_stack));
    stack->current_size = 0;
    stack->capacity = capacity;
    stack->top_scoring = NULL;
    stack->bottom_scoring = NULL;
    
    return stack;
}

void gssw_delete_multi_align_stack(gssw_multi_align_stack* stack) {
    gssw_multi_align_stack_node* node = stack->bottom_scoring;
    while (node != NULL) {
        gssw_multi_align_stack_node* next = node->next;
        gssw_delete_multi_align_stack_node(node);
        node = next;
    }
    free(stack);
}

gssw_multi_align_stack_node* gssw_new_multi_align_stack_node(gssw_alternate_alignment_ends* alignment_suffix, int16_t score,
                                                             int32_t read_pos, int32_t ref_pos, gssw_node* from_node,
                                                             gssw_node* to_node, gssw_matrix_t from_matrix, gssw_matrix_t to_matrix) {
    
    gssw_alternate_alignment_ends* alt_alignment = (gssw_alternate_alignment_ends*) malloc(sizeof(gssw_alternate_alignment_ends));
    
    // add score of the alignment
    alt_alignment->score = score;
    
    // initialize list of deflections one longer than suffix
    int32_t num_suffix_deflections = alignment_suffix->num_deflections;
    alt_alignment->num_deflections = num_suffix_deflections + 1;
    alt_alignment->deflections = (gssw_trace_back_deflection*) malloc(sizeof(gssw_trace_back_deflection) * alt_alignment->num_deflections);
    
    // copy the preceding deflections
    int i;
    for (i = 0; i < num_suffix_deflections; i++) {
        alt_alignment->deflections[i] = alignment_suffix->deflections[i];
    }
    
    // add the last deflection
    alt_alignment->deflections[num_suffix_deflections].read_pos = read_pos;
    alt_alignment->deflections[num_suffix_deflections].ref_pos = ref_pos;
    alt_alignment->deflections[num_suffix_deflections].from_node = from_node;
    alt_alignment->deflections[num_suffix_deflections].to_node = to_node;
    alt_alignment->deflections[num_suffix_deflections].from_matrix = from_matrix;
    alt_alignment->deflections[num_suffix_deflections].to_matrix = to_matrix;
    
    // create stack node for this alignment
    gssw_multi_align_stack_node* stack_node = (gssw_multi_align_stack_node*) malloc(sizeof(gssw_multi_align_stack_node));
    stack_node->alt_alignment = alt_alignment;
    stack_node->next = NULL;
    stack_node->prev = NULL;
    
    return stack_node;
}

void gssw_delete_multi_align_stack_node(gssw_multi_align_stack_node* stack_node) {
    if (stack_node->next) {
        stack_node->next->prev = NULL;
    }
    if (stack_node->prev) {
        stack_node->prev->next = NULL;
    }
    free(stack_node->alt_alignment->deflections);
    free(stack_node->alt_alignment);
    free(stack_node);
}

// check if alignment is better than any of the top recorded alignments
void gssw_add_alignment(gssw_multi_align_stack* stack, gssw_alternate_alignment_ends* alignment_suffix, int16_t score,
                        int32_t read_pos, int32_t ref_pos, gssw_node* from_node, gssw_node* to_node, gssw_matrix_t from_matrix,
                        gssw_matrix_t to_matrix) {
    
#ifdef DEBUG_TRACEBACK
    if (from_node) {
        fprintf(stderr, "checking whether to add new alignment at read pos %d, ref pos %d, from node id %llu, to node id %llu, score %d, from matrix %s, to matrix %s\n", read_pos, ref_pos, from_node->id, to_node->id, score, from_matrix == Match ? "Match" : (from_matrix == RefGap ? "RefGap" : "ReadGap"), to_matrix == Match ? "Match" : (to_matrix == RefGap ? "RefGap" : "ReadGap"));
    }
    else {
        fprintf(stderr, "checking whether to add new alignment at read pos %d, ref pos %d, score %d, from matrix %s, to matrix %s\n", read_pos, ref_pos, score, from_matrix == Match ? "Match" : (from_matrix == RefGap ? "RefGap" : "ReadGap"), to_matrix == Match ? "Match" : (to_matrix == RefGap ? "RefGap" : "ReadGap"));
    }
#endif
    
    // edge case where stack is initialized to hold no alignments
    if (stack->capacity <= 0) {
        return;
    }
    
    gssw_multi_align_stack_node* next_stack_node = stack->bottom_scoring;
    
    // edge case of first node inserted
    if (next_stack_node == NULL) {
        stack->bottom_scoring = gssw_new_multi_align_stack_node(alignment_suffix, score, read_pos, ref_pos,
                                                                from_node, to_node, from_matrix, to_matrix);
        stack->top_scoring = stack->bottom_scoring;
        stack->current_size = 1;
        return;
    }
    
    // find the position where this stack node belongs
    while (score > next_stack_node->alt_alignment->score) {
        next_stack_node = next_stack_node->next;
        if (next_stack_node == NULL) {
            break;
        }
    }
    
    // should we add this alternate alignment to the stack?
    if (next_stack_node != stack->bottom_scoring || stack->current_size < stack->capacity) {
#ifdef DEBUG_TRACEBACK
        fprintf(stderr, "alignment is better than current lowest scoring saved alignment\n");
#endif
        // make a new node to insert
        gssw_multi_align_stack_node* new_node = gssw_new_multi_align_stack_node(alignment_suffix, score, read_pos, ref_pos,
                                                                                from_node, to_node, from_matrix, to_matrix);
        
        // get previous stack node
        gssw_multi_align_stack_node* prev_stack_node;
        if (next_stack_node == NULL) {
            prev_stack_node = stack->top_scoring;
        }
        else {
            prev_stack_node = next_stack_node->prev;
        }
        
        // insert and update links
        if (next_stack_node) {
            next_stack_node->prev = new_node;
            new_node->next = next_stack_node;
        }
        if (prev_stack_node) {
            prev_stack_node->next = new_node;
            new_node->prev = prev_stack_node;
        }
        
        // update top and bottom scoring nodes
        if (stack->top_scoring->next) {
            stack->top_scoring = stack->top_scoring->next;
        }
        if (stack->bottom_scoring->prev) {
            stack->bottom_scoring = stack->bottom_scoring->prev;
        }
        
        // do we need to remove the worst alignment?
        if (stack->current_size >= stack->capacity) {
#ifdef DEBUG_TRACEBACK
            fprintf(stderr, "popping off worst alignment with score %d\n", stack->bottom_scoring->alt_alignment->score);
#endif
            gssw_multi_align_stack_node* node_to_pop = stack->bottom_scoring;
            stack->bottom_scoring = node_to_pop->next;
            gssw_delete_multi_align_stack_node(node_to_pop);
        }
        else {
#ifdef DEBUG_TRACEBACK
            fprintf(stderr, "increasing size of stack by 1\n");
#endif
            (stack->current_size)++;
        }
    }
#ifdef DEBUG_TRACEBACK
    fprintf(stderr, "finished updating alignment stack, scores are now: ");
    if (stack->top_scoring) {
        gssw_multi_align_stack_node* n = stack->top_scoring;
        fprintf(stderr, "%d", n->alt_alignment->score);
        while (n->prev != NULL) {
            n = n->prev;
            fprintf(stderr, "->%d", n->alt_alignment->score);
        }
        fprintf(stderr, "\n");
    }
    else {
        fprintf(stderr, ".\n");
    }
#endif
}

int16_t gssw_min_alt_alignment_score(gssw_multi_align_stack* stack) {
    return stack->bottom_scoring->alt_alignment->score;
}
