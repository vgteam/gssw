/* The MIT License

   Copyright (c) 2012-1015 Boston College.

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

/* Contact: Mengyao Zhao <zhangmp@bc.edu> */
/* Contact: Erik Garrison <erik.garrison@bc.edu> */

/*
 *  ssw.c
 *
 *  Created by Mengyao Zhao on 6/22/10.
 *  Copyright 2010 Boston College. All rights reserved.
 *	Version 0.1.4
 *	Last revision by Erik Garrison 01/02/2014
 *
 */

#include <emmintrin.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <inttypes.h>
#include <assert.h>
#include "gssw.h"

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


/* Generate query profile rearrange query sequence & calculate the weight of match/mismatch. */
__m128i* gssw_qP_byte (const int8_t* read_num,
                       const int8_t* mat,
                       const int32_t readLen,
                       const int32_t n,	/* the edge length of the squre matrix mat */
                       uint8_t bias) {

	int32_t segLen = (readLen + 15) / 16; /* Split the 128 bit register into 16 pieces.
								     Each piece is 8 bit. Split the read into 16 segments.
								     Calculat 16 segments in parallel.
								   */
	__m128i* vProfile = (__m128i*)malloc(n * segLen * sizeof(__m128i));
	int8_t* t = (int8_t*)vProfile;
	int32_t nt, i, j, segNum;

	/* Generate query profile rearrange query sequence & calculate the weight of match/mismatch */
	for (nt = 0; LIKELY(nt < n); nt ++) {
		for (i = 0; i < segLen; i ++) {
			j = i;
			for (segNum = 0; LIKELY(segNum < 16) ; segNum ++) {
				*t++ = j>= readLen ? bias : mat[nt * n + read_num[j]] + bias;
				j += segLen;
			}
		}
	}
	return vProfile;
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

/* Striped Smith-Waterman
   Record the highest score of each reference position.
   Return the alignment score and ending position of the best alignment, 2nd best alignment, etc.
   Gap begin and gap extension are different.
   wight_match > 0, all other weights < 0.
   The returned positions are 0-based.
 */
gssw_alignment_end* gssw_sw_sse2_byte (const int8_t* ref,
                                       int8_t ref_dir,	// 0: forward ref; 1: reverse ref
                                       int32_t refLen,
                                       int32_t readLen,
                                       const uint8_t weight_gapO, /* will be used as - */
                                       const uint8_t weight_gapE, /* will be used as - */
                                       __m128i* vProfile,
                                       uint8_t terminate,	/* the best alignment score: used to terminate
                                                               the matrix calculation when locating the
                                                               alignment beginning point. If this score
                                                               is set to 0, it will not be used */
                                       uint8_t bias,  /* Shift 0 point to a positive value. */
                                       int32_t maskLen,
                                       gssw_align* alignment, /* to save seed and matrix */
                                       const gssw_seed* seed) {     /* to seed the alignment */

	uint8_t max = 0;		                     /* the max alignment score */
	int32_t end_read = readLen - 1;
	int32_t end_ref = -1; /* 0_based best alignment ending point; Initialized as isn't aligned -1. */
	int32_t segLen = (readLen + 15) / 16; /* number of segment */

    /* Initialize buffers used in alignment */
	__m128i* pvHStore;
    __m128i* pvHLoad;
    __m128i* pvHmax;
    __m128i* pvE;
    // We have a couple extra arrays for logging columns
    __m128i* pvEStore;
    __m128i* pvFStore;
    uint8_t* mH; // used to save matrices for external traceback: overall best score
    uint8_t* mE; // Gap in read best score
    uint8_t* mF; // Gap in ref best score
    /* Note use of aligned memory.  Return value of 0 means success for posix_memalign. */
    if (!(!posix_memalign((void**)&pvHStore,     sizeof(__m128i), segLen*sizeof(__m128i)) &&
          !posix_memalign((void**)&pvHLoad,      sizeof(__m128i), segLen*sizeof(__m128i)) &&
          !posix_memalign((void**)&pvHmax,       sizeof(__m128i), segLen*sizeof(__m128i)) &&
          !posix_memalign((void**)&pvE,          sizeof(__m128i), segLen*sizeof(__m128i)) &&
          !posix_memalign((void**)&pvEStore,     sizeof(__m128i), segLen*sizeof(__m128i)) &&
          !posix_memalign((void**)&pvFStore,     sizeof(__m128i), segLen*sizeof(__m128i)) &&
          !posix_memalign((void**)&alignment->seed.pvE,      sizeof(__m128i), segLen*sizeof(__m128i)) &&
          !posix_memalign((void**)&alignment->seed.pvHStore, sizeof(__m128i), segLen*sizeof(__m128i)) &&
          !posix_memalign((void**)&mH,           sizeof(__m128i), segLen*refLen*sizeof(__m128i)) &&
          !posix_memalign((void**)&mE,           sizeof(__m128i), segLen*refLen*sizeof(__m128i)) &&
          !posix_memalign((void**)&mF,           sizeof(__m128i), segLen*refLen*sizeof(__m128i)))) {
        fprintf(stderr, "error:[gssw] Could not allocate memory required for alignment buffers.\n");
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
    memset(mH,                       0, segLen*refLen*sizeof(__m128i));
    memset(mE,                       0, segLen*refLen*sizeof(__m128i));
    memset(mF,                       0, segLen*refLen*sizeof(__m128i));

    /* if we are running a seeded alignment, copy over the seeds */
    if (seed) {
        memcpy(pvE, seed->pvE, segLen*sizeof(__m128i));
        memcpy(pvHStore, seed->pvHStore, segLen*sizeof(__m128i));
    }

    /* Set external matrix pointers */
    alignment->mH = mH;
    alignment->mE = mE;
    alignment->mF = mF;

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
		int32_t cmp;
		__m128i e = vZero, vF = vZero, vMaxColumn = vZero; /* Initialize F value to 0.
							   Any errors to vH values will be corrected in the Lazy_F loop.
							 */
		//max16(maxColumn[i], vMaxColumn);
		//fprintf(stderr, "middle[%d]: %d\n", i, maxColumn[i]);

		//__m128i vH = pvHStore[segLen - 1];
        __m128i vH = _mm_load_si128 (pvHStore + (segLen - 1));
		vH = _mm_slli_si128 (vH, 1); /* Shift the 128-bit value in vH left by 1 byte. */
		__m128i* vP = vProfile + ref[i] * segLen; /* Right part of the vProfile */

		/* Swap the 2 H buffers. */
		__m128i* pv = pvHLoad;
		pvHLoad = pvHStore;
		pvHStore = pv;

		/* inner loop to process the query sequence */
		for (j = 0; LIKELY(j < segLen); ++j) {

			vH = _mm_adds_epu8(vH, _mm_load_si128(vP + j));
			vH = _mm_subs_epu8(vH, vBias); /* vH will be always > 0 */
	//	max16(maxColumn[i], vH);
	//	fprintf(stderr, "H[%d]: %d\n", i, maxColumn[i]);
            /*
            int8_t* t;
            int32_t ti;
            fprintf(stdout, "%d\n", i);
            for (t = (int8_t*)&vH, ti = 0; ti < 16; ++ti) fprintf(stdout, "%d\t", *t++);
            fprintf(stdout, "\n");
            */

			/* Get max from vH, vE and vF. */
			e = _mm_load_si128(pvE + j);
			//_mm_store_si128(vE + j, e);

			vH = _mm_max_epu8(vH, e);
			vH = _mm_max_epu8(vH, vF);
			vMaxColumn = _mm_max_epu8(vMaxColumn, vH);

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

			/* Update vE value. */
			vH = _mm_subs_epu8(vH, vGapO); /* saturation arithmetic, result >= 0 */
			e = _mm_subs_epu8(e, vGapE);
			e = _mm_max_epu8(e, vH);

			/* Update vF value. */
			vF = _mm_subs_epu8(vF, vGapE);
			vF = _mm_max_epu8(vF, vH);

            /* Save E */
			_mm_store_si128(pvE + j, e);

			/* Load the next vH. */
			vH = _mm_load_si128(pvHLoad + j);
		}


		/* Lazy_F loop: has been revised to disallow adjecent insertion and then deletion, so don't update E(i, j), learn from SWPS3 */
        /* reset pointers to the start of the saved data */
        j = 0;
        vH = _mm_load_si128 (pvHStore + j);

        /*  the computed vF value is for the given column.  since */
        /*  we are at the end, we need to shift the vF value over */
        /*  to the next column. */
        vF = _mm_slli_si128 (vF, 1);

        vTemp = _mm_subs_epu8 (vH, vGapO);
		vTemp = _mm_subs_epu8 (vF, vTemp);
		vTemp = _mm_cmpeq_epi8 (vTemp, vZero);
		cmp  = _mm_movemask_epi8 (vTemp);
        while (cmp != 0xffff)
        {
            // Update this stripe of the H matrix
            vH = _mm_max_epu8 (vH, vF);
			vMaxColumn = _mm_max_epu8(vMaxColumn, vH);
            _mm_store_si128 (pvHStore + j, vH);

            // Save the stripe of the F matrix
            // Only add in better F scores. Sometimes during this loop we'll
            // recompute worse ones.
            vTemp = _mm_load_si128 (pvFStore + j);
            vTemp = _mm_max_epu8 (vTemp, vF);
            _mm_store_si128(pvFStore + j, vTemp);

            vF = _mm_subs_epu8 (vF, vGapE);

            j++;
            if (j >= segLen)
            {
                j = 0;
                vF = _mm_slli_si128 (vF, 1);
            }

            vH = _mm_load_si128 (pvHStore + j);
            vTemp = _mm_subs_epu8 (vH, vGapO);
            vTemp = _mm_subs_epu8 (vF, vTemp);
            vTemp = _mm_cmpeq_epi8 (vTemp, vZero);
            cmp  = _mm_movemask_epi8 (vTemp);
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
				if (max + bias >= 255) break;	//overflow
				end_ref = i;

				/* Store the column with the highest alignment score in order to trace the alignment ending position on read. */
				for (j = 0; LIKELY(j < segLen); ++j) pvHmax[j] = pvHStore[j];

			}
		}

        // save the current column for traceback
        // Need to unswizzle all the stripes


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
				  const int32_t n) {

	int32_t segLen = (readLen + 7) / 8;
	__m128i* vProfile = (__m128i*)malloc(n * segLen * sizeof(__m128i));
	int16_t* t = (int16_t*)vProfile;
	int32_t nt, i, j;
	int32_t segNum;

	/* Generate query profile rearrange query sequence & calculate the weight of match/mismatch */
	for (nt = 0; LIKELY(nt < n); nt ++) {
		for (i = 0; i < segLen; i ++) {
			j = i;
			for (segNum = 0; LIKELY(segNum < 8) ; segNum ++) {
				*t++ = j>= readLen ? 0 : mat[nt * n + read_num[j]];
				j += segLen;
			}
		}
	}
	return vProfile;
}

gssw_alignment_end* gssw_sw_sse2_word (const int8_t* ref,
                                       int8_t ref_dir,	// 0: forward ref; 1: reverse ref
                                       int32_t refLen,
                                       int32_t readLen,
                                       const uint8_t weight_gapO, /* will be used as - */
                                       const uint8_t weight_gapE, /* will be used as - */
                                       __m128i* vProfile,
                                       uint16_t terminate,
                                       int32_t maskLen,
                                       gssw_align* alignment, /* to save seed and matrix */
                                       const gssw_seed* seed) {     /* to seed the alignment */
    

	uint16_t max = 0;		                     /* the max alignment score */
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
    uint16_t* mH; // used to save matrices for external traceback: overall best
    uint16_t* mE; // Read gap
    uint16_t* mF; // Ref gap
    /* Note use of aligned memory */

    if (!(!posix_memalign((void**)&pvHStore,     sizeof(__m128i), segLen*sizeof(__m128i)) &&
          !posix_memalign((void**)&pvHLoad,      sizeof(__m128i), segLen*sizeof(__m128i)) &&
          !posix_memalign((void**)&pvHmax,       sizeof(__m128i), segLen*sizeof(__m128i)) &&
          !posix_memalign((void**)&pvE,          sizeof(__m128i), segLen*sizeof(__m128i)) &&
          !posix_memalign((void**)&pvEStore,     sizeof(__m128i), segLen*sizeof(__m128i)) &&
          !posix_memalign((void**)&pvFStore,     sizeof(__m128i), segLen*sizeof(__m128i)) &&
          !posix_memalign((void**)&alignment->seed.pvE,      sizeof(__m128i), segLen*sizeof(__m128i)) &&
          !posix_memalign((void**)&alignment->seed.pvHStore, sizeof(__m128i), segLen*sizeof(__m128i)) &&
          !posix_memalign((void**)&mH,           sizeof(__m128i), segLen*refLen*sizeof(__m128i)) &&
          !posix_memalign((void**)&mE,           sizeof(__m128i), segLen*refLen*sizeof(__m128i)) &&
          !posix_memalign((void**)&mF,           sizeof(__m128i), segLen*refLen*sizeof(__m128i)))) {
        fprintf(stderr, "error:[gssw] Could not allocate memory required for alignment buffers.\n");
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
    memset(mH,                       0, segLen*refLen*sizeof(__m128i));
    memset(mE,                       0, segLen*refLen*sizeof(__m128i));
    memset(mF,                       0, segLen*refLen*sizeof(__m128i));

    /* if we are running a seeded alignment, copy over the seeds */
    if (seed) {
        memcpy(pvE, seed->pvE, segLen*sizeof(__m128i));
        memcpy(pvHStore, seed->pvHStore, segLen*sizeof(__m128i));
    }

    /* Set external matrix pointers */
    alignment->mH = mH;
    alignment->mE = mE;
    alignment->mF = mF;

    /* Record that we have done a word-order alignment */
    alignment->is_byte = 0;

	/* Define 16 byte 0 vector. */
	__m128i vZero = _mm_set1_epi32(0);

    /* Used for iteration */
	int32_t i, j, k;

	/* 16 byte insertion begin vector */
	__m128i vGapO = _mm_set1_epi16(weight_gapO);

	/* 16 byte insertion extension vector */
	__m128i vGapE = _mm_set1_epi16(weight_gapE);

	/* 16 byte bias vector */
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

		/* Swap the 2 H buffers. */
		__m128i* pv = pvHLoad;

		__m128i vMaxColumn = vZero; /* vMaxColumn is used to record the max values of column i. */

		__m128i* vP = vProfile + ref[i] * segLen; /* Right part of the vProfile */
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

		/* Lazy_F loop: has been revised to disallow adjecent insertion and then deletion, so don't update E(i, j), learn from SWPS3 */
		for (k = 0; LIKELY(k < 8); ++k) {
			vF = _mm_slli_si128 (vF, 2);
			for (j = 0; LIKELY(j < segLen); ++j) {
			    // Update this stripe of the H matrix
				vH = _mm_load_si128(pvHStore + j);
				vH = _mm_max_epi16(vH, vF);
				_mm_store_si128(pvHStore + j, vH);
				
				// Save the stripe of the F matrix
                // Only add in better F scores. Sometimes during this loop we'll
                // recompute worse ones.
                vTemp = _mm_load_si128 (pvFStore + j);
                vTemp = _mm_max_epi16 (vTemp, vF);
                _mm_store_si128(pvFStore + j, vTemp);
				
				vH = _mm_subs_epu16(vH, vGapO);
				vF = _mm_subs_epu16(vF, vGapE);
				if (UNLIKELY(! _mm_movemask_epi8(_mm_cmpgt_epi16(vF, vH)))) goto end;
			}
		}

end:
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

int8_t* gssw_seq_reverse(const int8_t* seq, int32_t end)	/* end is 0-based alignment ending position */
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

gssw_profile* gssw_init (const int8_t* read, const int32_t readLen, const int8_t* mat, const int32_t n, const int8_t score_size) {
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
		p->profile_byte = gssw_qP_byte (read, mat, readLen, n, bias);
	}
	if (score_size == 1 || score_size == 2) p->profile_word = gssw_qP_word (read, mat, readLen, n);
	p->read = read;
	p->mat = mat;
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
                       gssw_seed* seed) {

	gssw_alignment_end* bests = 0;
	int32_t readLen = prof->readLen;
    gssw_align* alignment = gssw_align_create();

	if (maskLen < 15) {
		fprintf(stderr, "When maskLen < 15, the function ssw_align doesn't return 2nd best alignment information.\n");
	}

	// Find the alignment scores and ending positions
	if (prof->profile_byte) {
		bests = gssw_sw_sse2_byte(ref, 0, refLen, readLen, weight_gapO, weight_gapE, prof->profile_byte, -1, prof->bias, maskLen,
                             alignment, seed);

		if (prof->profile_word && bests[0].score == 255) {
			free(bests);
            gssw_align_clear_matrix_and_seed(alignment);
            bests = gssw_sw_sse2_word(ref, 0, refLen, readLen, weight_gapO, weight_gapE, prof->profile_byte, -1, maskLen,
                                      alignment, seed);
        } else if (bests[0].score == 255) {
			fprintf(stderr, "Please set 2 to the score_size parameter of the function ssw_init, otherwise the alignment results will be incorrect.\n");
			return 0;
		}
	} else if (prof->profile_word) {
		bests = gssw_sw_sse2_word(ref, 0, refLen, readLen, weight_gapO, weight_gapE, prof->profile_word, -1, maskLen,
                                  alignment, seed);
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
        fprintf(stdout, "GRAPH // node %u %u %s\n", n->id, n->len, n->seq);
        uint32_t k;
        for (k=0; k<n->count_prev; ++k) {
            //fprintf(stdout, "GRAPH %u -> %u;\n", n->prev[k]->id, n->id);
            fprintf(stdout, "GRAPH \"%u %s\" -> \"%u %s\";\n", n->prev[k]->id, n->prev[k]->seq, n->id, n->seq);
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
        fprintf(stderr, "GRAPH // node %u %u %s\n", n->id, n->len, n->seq);
        uint32_t k;
        for (k=0; k<n->count_prev; ++k) {
            //fprintf(stdout, "GRAPH %u -> %u;\n", n->prev[k]->id, n->id);
            fprintf(stderr, "GRAPH \"%u %s\" -> \"%u %s\";\n", n->prev[k]->id, n->prev[k]->seq, n->id, n->seq);
        }
    }
    fprintf(stderr, "GRAPH }\n");
}

void gssw_graph_print_score_matrices(gssw_graph* graph, const char* read, int32_t readLen, FILE* out) {
    uint32_t i = 0, gs = graph->size;
    gssw_node** npp = graph->nodes;
    for (i=0; i<gs; ++i, ++npp) {
        gssw_node* n = *npp;
        fprintf(out, "node %u\n", n->id);
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

gssw_cigar* gssw_alignment_trace_back (gssw_align* alignment,
                                       uint16_t* score,
                                       int32_t* refEnd,
                                       int32_t* readEnd,
                                       int32_t* refGapFlag,
                                       int32_t* readGapFlag,
                                       const char* ref,
                                       int32_t refLen,
                                       const char* read,
                                       int32_t readLen,
                                       int32_t match,
                                       int32_t mismatch,
                                       int32_t gap_open,
                                       int32_t gap_extension) {
    if (LIKELY(gssw_is_byte(alignment))) {
        return gssw_alignment_trace_back_byte(alignment,
                                              score,
                                              refEnd,
                                              readEnd,
                                              refGapFlag,
                                              readGapFlag,
                                              ref,
                                              refLen,
                                              read,
                                              readLen,
                                              match,
                                              mismatch,
                                              gap_open,
                                              gap_extension);
    } else {
        return gssw_alignment_trace_back_word(alignment,
                                              score,
                                              refEnd,
                                              readEnd,
                                              refGapFlag,
                                              readGapFlag,
                                              ref,
                                              refLen,
                                              read,
                                              readLen,
                                              match,
                                              mismatch,
                                              gap_open,
                                              gap_extension);
    }
}

gssw_cigar* gssw_alignment_trace_back_byte (gssw_align* alignment,
                                            uint16_t* score,
                                            int32_t* refEnd,
                                            int32_t* readEnd,
                                            int32_t* refGapFlag,
                                            int32_t* readGapFlag,
                                            const char* ref,
                                            int32_t refLen,
                                            const char* read,
                                            int32_t readLen,
                                            int32_t match,
                                            int32_t mismatch,
                                            int32_t gap_open,
                                            int32_t gap_extension) {

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
    
#ifdef DEBUG_TRACEBACK
        fprintf(stderr, "score=%i at %i,%i with %c vs %c, gRef=%i gRead=%i\n", scoreHere, i, j, ref[i], read[j], gRef, gRead);
#endif
    
        if(gRead) {
            // We're in E
            
            // If we're in a gap matrix, see if we can leave the gap here (i.e.
            // gap open score is consistent). If so, do it. Otherwise, extend
            // the gap.
            if(i > 0 && mH[readLen*(i - 1) +  j] - gap_open == scoreHere) {
                // We are consistent with a gap open, bringing us back to the
                // main matrix. Take it.
                
                // D = gap in read
                gssw_cigar_push_back(result, 'D', 1);
                
                // Fix score
                scoreHere += gap_open;
                
                // Move in the reference
                --i;
                
                // Go back to the main matrix
                gRead = 0;
                
#ifdef DEBUG_TRACEBACK
                fprintf(stderr, "Read gap open\n");
#endif
                
            } else if(i > 0 && mE[readLen*(i - 1) +  j] - gap_extension == scoreHere) {
                // We are consistent with a gap extend
                gssw_cigar_push_back(result, 'D', 1);
                scoreHere += gap_extension;
                --i;
#ifdef DEBUG_TRACEBACK
                fprintf(stderr, "Read gap extend\n");
#endif
            } else if(i == 0) {
                // We are in a gap and at the very left edge. We need to look
                // left from here and trace into our previous node in order to
                // figure out if this is an open or an extend or what.
#ifdef DEBUG_TRACEBACK
                fprintf(stderr, "Read gap out left edge\n");
#endif
                break;
            } else {
                // Something has gone wrong. We're in this matrix but don't have
                // an open or an extend and can't leave left.
                fprintf(stderr, "error:[gssw] Stuck in read gap!\n");
                assert(0);
            }
        } else if(gRef) {
            // We're in F
            if(j > 0 && mH[readLen*i +  (j - 1)] - gap_open == scoreHere) {
                // We are consistent with a gap open, bringing us back to the
                // main matrix. Take it.
                
                // I = gap in ref
                gssw_cigar_push_back(result, 'I', 1);
                
                // Fix score
                scoreHere += gap_open;
                
                // Move in the read
                --j;
                
                // Go back to the main matrix
                gRef = 0;
#ifdef DEBUG_TRACEBACK
                fprintf(stderr, "Ref gap open\n");
#endif
            } else if(j > 0 && mF[readLen*i + (j - 1)] - gap_extension == scoreHere) {
                // We are consistent with a gap extend
                gssw_cigar_push_back(result, 'I', 1);
                scoreHere += gap_extension;
                --j;
#ifdef DEBUG_TRACEBACK
                fprintf(stderr, "Ref gap extend\n");
#endif
            } else if(j == 0) {
                // We have hit the end of the read in a gap, which should be
                // impossible because there's nowhere to open from.
                fprintf(stderr, "error:[gssw] Ref gap hit edge!\n");
                assert(0);
            } else {
                // Something has gone wrong. We're in this matrix but don't have
                // an open or an extend and can't leave left.
                fprintf(stderr, "error:[gssw] Ref gap stuck!\n");
                assert(0);
            }
        } else {
            // We're in H
            
            // If we're in the main matrix, see if we can do a match, mismatch,
            // or N-match. If so, do it.
            if(i > 0 && j > 0 && mH[readLen*(i-1) + (j-1)] == scoreHere && (ref[i] == 'N' || read[j] == 'N')) {
                // This is an N-match.
                gssw_cigar_push_back(result, 'N', 1);
                // Leave score unchanged
                --i; --j;
#ifdef DEBUG_TRACEBACK
                fprintf(stderr, "N-match\n");
#endif
            } else if(i > 0 && j > 0 && mH[readLen*(i-1) + (j-1)] + match == scoreHere && ref[i] == read[j]) {
                // This is a match
                gssw_cigar_push_back(result, 'M', 1);
                scoreHere -= match;
                --i; --j;
#ifdef DEBUG_TRACEBACK
                fprintf(stderr, "Match\n");
#endif
            } else if(i > 0 && j > 0 && mH[readLen*(i-1) + (j-1)] - mismatch == scoreHere && ref[i] != read[j]) {
                // This is a mismatch
                gssw_cigar_push_back(result, 'X', 1);
                scoreHere += mismatch;
                --i; --j;
#ifdef DEBUG_TRACEBACK
                fprintf(stderr, "Mismatch\n");
#endif
            } else if(scoreHere == match && ref[i] == read[j]) {
                // Handle traceback termination at the leading match, anywhere.
                gssw_cigar_push_back(result, 'M', 1);
                scoreHere -= match;
                // TODO: are we allowed to/do we need to let these go negative?
                --i; --j;
#ifdef DEBUG_TRACEBACK
                fprintf(stderr, "Alignment start match\n");
#endif
            } else if(j > 0 && mF[readLen*i + j] == scoreHere) {
                // We can't do anything diagonal. But we can become a ref gap, because there is more read.
                gRef = 1;
                //fprintf(stderr, "Ref gap close\n");
            } else if(mE[readLen*i + j] == scoreHere) {
                // We assume there's always more ref off to the left. We tried
                // everything else, so try a read gap.
                gRead = 1;
#ifdef DEBUG_TRACEBACK
                fprintf(stderr, "Read gap close\n");
#endif
            } else if(i == 0) {
                // We can't go anywhere, but we're at the left edge, so maybe we
                // can go to a previous node diagonally. Just let it slide.
#ifdef DEBUG_TRACEBACK                
                fprintf(stderr, "Match/mismatch out left edge\n");
#endif
                break;
            } else {
                // We're in H and can't go anywhere.
                fprintf(stderr, "error:[gssw] Stuck in main matrix!\n");
                assert(0);
            }
            
        }
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

gssw_cigar* gssw_alignment_trace_back_word (gssw_align* alignment,
                                            uint16_t* score,
                                            int32_t* refEnd,
                                            int32_t* readEnd,
                                            int32_t* refGapFlag,
                                            int32_t* readGapFlag,
                                            const char* ref,
                                            int32_t refLen,
                                            const char* read,
                                            int32_t readLen,
                                            int32_t match,
                                            int32_t mismatch,
                                            int32_t gap_open,
                                            int32_t gap_extension) {

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
    
        // We think differently than we did creating the matrix. When creating
        // the matrix, we care about the relative scores, because we want to
        // come from the places giving the best score.
        
        // When computing the traceback, the scores are already determined, and
        // only some places to come from are consistent with the facts. We can
        // just see what's consistent, and then pick the "best" consistent
        // thing, according to fixed priorities.
    
        if(gRead) {
            // We're in E
            
            // If we're in a gap matrix, see if we can leave the gap here (i.e.
            // gap open score is consistent). If so, do it. Otherwise, extend
            // the gap.
            if(i > 0 && mH[readLen*(i - 1) +  j] - gap_open == scoreHere) {
                // We are consistent with a gap open, bringing us back to the
                // main matrix. Take it.
                
                // D = gap in read
                gssw_cigar_push_back(result, 'D', 1);
                
                // Fix score
                scoreHere += gap_open;
                
                // Move in the reference
                --i;
                
                // Go back to the main matrix
                gRead = 0;
                
                //fprintf(stderr, "Read gap open\n");
                
            } else if(i > 0 && mE[readLen*(i - 1) +  j] - gap_extension == scoreHere) {
                // We are consistent with a gap extend
                gssw_cigar_push_back(result, 'D', 1);
                scoreHere += gap_extension;
                --i;
                
                //fprintf(stderr, "Read gap extend\n");
            } else if(i == 0) {
                // We are in a gap and at the very left edge. We need to look
                // left from here and trace into our previous node in order to
                // figure out if this is an open or an extend or what.
                //fprintf(stderr, "Read gap out left edge\n");
                break;
            } else {
                // Something has gone wrong. We're in this matrix but don't have
                // an open or an extend and can't leave left.
                fprintf(stderr, "error:[gssw] Stuck in read gap!\n");
                assert(0);
            }
        } else if(gRef) {
            // We're in F
            if(j > 0 && mH[readLen*i +  (j - 1)] - gap_open == scoreHere) {
                // We are consistent with a gap open, bringing us back to the
                // main matrix. Take it.
                
                // I = gap in ref
                gssw_cigar_push_back(result, 'I', 1);
                
                // Fix score
                scoreHere += gap_open;
                
                // Move in the read
                --j;
                
                // Go back to the main matrix
                gRef = 0;
                
                //fprintf(stderr, "Ref gap open\n");
            } else if(j > 0 && mF[readLen*i + (j - 1)] - gap_extension == scoreHere) {
                // We are consistent with a gap extend
                gssw_cigar_push_back(result, 'I', 1);
                scoreHere += gap_extension;
                --j;
                //fprintf(stderr, "Ref gap extend\n");
            } else if(j == 0) {
                // We have hit the end of the read in a gap, which should be
                // impossible because there's nowhere to open from.
                fprintf(stderr, "error:[gssw] Ref gap hit edge!\n");
                assert(0);
            } else {
                // Something has gone wrong. We're in this matrix but don't have
                // an open or an extend and can't leave left.
                fprintf(stderr, "error:[gssw] Ref gap stuck!\n");
                assert(0);
            }
        } else {
            // We're in H
            
            // If we're in the main matrix, see if we can do a match, mismatch,
            // or N-match. If so, do it.
            if(i > 0 && j > 0 && mH[readLen*(i-1) + (j-1)] == scoreHere && (ref[i] == 'N' || read[j] == 'N')) {
                // This is an N-match.
                gssw_cigar_push_back(result, 'N', 1);
                // Leave score unchanged
                --i; --j;
                //fprintf(stderr, "N-match\n");
                
            } else if(i > 0 && j > 0 && mH[readLen*(i-1) + (j-1)] + match == scoreHere && ref[i] == read[j]) {
                // This is a match
                gssw_cigar_push_back(result, 'M', 1);
                scoreHere -= match;
                --i; --j;
                //fprintf(stderr, "Match\n");
            } else if(i > 0 && j > 0 && mH[readLen*(i-1) + (j-1)] - mismatch == scoreHere && ref[i] != read[j]) {
                // This is a mismatch
                gssw_cigar_push_back(result, 'X', 1);
                scoreHere += mismatch;
                --i; --j;
                //fprintf(stderr, "Mismatch\n");
            } else if(scoreHere == match && ref[i] == read[j]) {
                // Handle traceback termination at the leading match, anywhere.
                gssw_cigar_push_back(result, 'M', 1);
                scoreHere -= match;
                // TODO: are we allowed to/do we need to let these go negative?
                --i; --j;
                //fprintf(stderr, "Alignment start match\n");
            } else if(j > 0 && mF[readLen*i + j] == scoreHere) {
                // We can't do anything diagonal. But we can become a ref gap, because there is more read.
                gRef = 1;
                //fprintf(stderr, "Ref gap close\n");
            } else if(mE[readLen*i + j] == scoreHere) {
                // We assume there's always more ref off to the left. We tried
                // everything else, so try a read gap.
                gRead = 1;
                //fprintf(stderr, "Read gap close\n");
            } else if(i == 0) {
                // We can't go anywhere, but we're at the left edge, so maybe we
                // can go to a previous node diagonally. Just let it slide.
                //fprintf(stderr, "Match/mismatch out left edge\n");
                break;
            } else {
                // We're in H and can't go anywhere.
                fprintf(stderr, "error:[gssw] Stuck in main matrix!\n");
                assert(0);
            }
        }
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
    gssw_graph_cigar* g = &m->cigar;
    for (i = 0; i < g->length; ++i) {
        gssw_cigar_destroy(g->elements[i].cigar);
    }
    free(g->elements);
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
        fprintf(out, "%u[", nc->node->id);
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

gssw_graph_mapping* gssw_graph_trace_back (gssw_graph* graph,
                                           const char* read,
                                           int32_t readLen,
                                           int32_t match,
                                           int32_t mismatch,
                                           int32_t gap_open,
                                           int32_t gap_extension) {

    // Make a graph and get its cigar
    gssw_graph_mapping* gm = gssw_graph_mapping_create();
    gssw_graph_cigar* gc = &gm->cigar;
    
    // The cigar needs to double every so often so it is big enough. This is how
    // long is thould be to start.
    uint32_t graph_cigar_bufsiz = 16;
    gc->elements = NULL;
    gc->elements = realloc((void*) gc->elements, graph_cigar_bufsiz * sizeof(gssw_node_cigar));
    // And how much of it is used
    gc->length = 0;

    // Find the best node to start the traceback from
    gssw_node* n = graph->max_node;
    if (!n) {
        fprintf(stderr, "error:[gssw] Cannot trace back because graph alignment has not been run.\n");
        fprintf(stderr, "error:[gssw] You must call graph_fill(...) before tracing back.\n");
        exit(1);
    }
    
    // And the score that we have there
    uint16_t score = n->alignment->score1;
    gm->score = score;
    
    // Are we using bytes, or did we have to do shorts?
    uint8_t score_is_byte = gssw_is_byte(n->alignment);
    
    // Where in the reference and the read is the best place to trace back from
    // in this node?
    int32_t refEnd = n->alignment->ref_end1;
    int32_t readEnd = n->alignment->read_end1;
    
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
    fprintf(stderr, "tracing back, max node = %p %u\n", n, n->id);
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
        fprintf(stderr, "id=%i\n", n->id);
#endif
        nc->cigar = gssw_alignment_trace_back (n->alignment,
                                               &score,
                                               &refEnd,
                                               &readEnd,
                                               &gapInRef,
                                               &gapInRead,
                                               n->seq,
                                               n->len,
                                               read,
                                               readLen,
                                               match,
                                               mismatch,
                                               gap_open,
                                               gap_extension);

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
        fprintf(stderr, "score is %u as we end node %p %u at position %i in read and %i in ref\n", score, n, n->id, readEnd, refEnd);
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
            for (i = 0; i < n->count_prev; ++i) {
                // Consider each node we could have come from
                gssw_node* cn = n->prev[i];
                
                // If we were to match/mismatch, what characters are we comparing?
                char refChar = n->seq[refEnd];
                char readChar = read[readEnd];
                
                // What if we came diagonally on a match or mismatch? What score would we come from?
                uint8_t diagonalSourceScore = ((uint8_t*)cn->alignment->mH)[readLen*(cn->len-1) + (readEnd-1)];
                
                // What if we came from the left, on a gap open in the read?
                uint8_t gapOpenSourceScore = ((uint8_t*)cn->alignment->mH)[readLen*(cn->len-1) + readEnd];
                
                // And what if we came on a gap extend instead?
                uint8_t gapExtendSourceScore = ((uint8_t*)cn->alignment->mE)[readLen*(cn->len-1) + readEnd];
                
                // If we could have entered a read gap before leaving our last
                // node, we would have.
                
#ifdef DEBUG_TRACEBACK
                fprintf(stderr, "Consider prev node %p with score %i, %c vs %c, %i diagonal, %i open, %i extend\n", cn, score, refChar, readChar, diagonalSourceScore, gapOpenSourceScore, gapExtendSourceScore);
#endif
                
                if(!gapInRead) {
                    // If we're not in a gap...
                    if(diagonalSourceScore == score && (refChar == 'N' || readChar == 'N')) {
                        // We're consistent with an N-match
                        
                        // Go to that node (which consumes a ref character), and
                        // consume a read character
                        best_prev = cn;
                        gssw_cigar_push_front(nc->cigar, 'N', 1);
                        --readEnd;
#ifdef DEBUG_TRACEBACK
                        fprintf(stderr, "N-match across nodes to %p\n", cn);
#endif
                        break;
                    } else if(diagonalSourceScore + match == score && refChar == readChar) {
                        // Then try a match
                        
                        // Go to that node (which consumes a ref character), and
                        // consume a read character
                        best_prev = cn;
                        gssw_cigar_push_front(nc->cigar, 'M', 1);
                        score -= match;
                        --readEnd;
#ifdef DEBUG_TRACEBACK
                        fprintf(stderr, "Match across nodes to %p\n", cn);
#endif
                        break;
                    } else if(diagonalSourceScore - mismatch == score && refChar != readChar) {
                        // Then try a mismatch
                        
                        // Go to that node (which consumes a ref character), and
                        // consume a read character
                        best_prev = cn;
                        gssw_cigar_push_front(nc->cigar, 'X', 1);
                        score += mismatch;
                        --readEnd;
#ifdef DEBUG_TRACEBACK
                        fprintf(stderr, "Mismatch across nodes to %p\n", cn);
#endif
                        break;
                    }
                    
                    // A match starting the alignment should have been taken
                    // care of in the within-node function.
                    
                    // If none of those work, try the next option for the previous
                    // node.
                } else {
                
                    // If we are in a gap, it would have been a last resort in the node's traceback.
                    
                    if(gapOpenSourceScore - gap_open == score) {
                        // This node is consistent with an open. Take it.
                        best_prev = cn;
                        gssw_cigar_push_front(nc->cigar, 'D', 1);
                        score += gap_open;
                        // Unset the gap flag
                        gapInRead = 0;
#ifdef DEBUG_TRACEBACK
                        fprintf(stderr, "Gap open across nodes to %p\n", cn);
#endif
                        break;
                    } else if(gapExtendSourceScore - gap_extension == score) {
                        // This node is consistent with an extend. Take it.
                        best_prev = cn;
                        gssw_cigar_push_front(nc->cigar, 'D', 1);
                        score += gap_extension;
#ifdef DEBUG_TRACEBACK
                        fprintf(stderr, "Gap extend across nodes to %p\n", cn);
#endif
                        break;
                    }
                }
            }
            // Once we go through all the possible previous nodes, we sure hope we found something consistent.
            if(best_prev == NULL) {
                fprintf(stderr, "error:[gssw] Could not find a valid previous node\n");
                assert(0);
            }
        } else {
            // Repeat the whole thing for words.
            for (i = 0; i < n->count_prev; ++i) {
                // Consider each node we could have come from
                gssw_node* cn = n->prev[i];
                
                // If we were to match/mismatch, what characters are we comparing?
                char refChar = n->seq[refEnd];
                char readChar = read[readEnd];
                
                // What if we came diagonally on a match or mismatch? What score would we come from?
                uint16_t diagonalSourceScore = ((uint16_t*)cn->alignment->mH)[readLen*(cn->len-1) + (readEnd-1)];
                
                // What if we came from the left, on a gap open in the read?
                uint16_t gapOpenSourceScore = ((uint16_t*)cn->alignment->mH)[readLen*(cn->len-1) + readEnd];
                
                // And what if we came on a gap extend instead?
                uint16_t gapExtendSourceScore = ((uint16_t*)cn->alignment->mE)[readLen*(cn->len-1) + readEnd];
                
                // If we could have entered a read gap before leaving our last
                // node, we would have.
                
#ifdef DEBUG_TRACEBACK
                fprintf(stderr, "Consider prev node %p with score %i, %c vs %c, %i diagonal, %i open, %i extend\n", cn, score, refChar, readChar, diagonalSourceScore, gapOpenSourceScore, gapExtendSourceScore);
#endif
                
                if(!gapInRead) {
                    // If we're not in a gap...
                    if(diagonalSourceScore == score && (refChar == 'N' || readChar == 'N')) {
                        // We're consistent with an N-match
                        
                        // Go to that node (which consumes a ref character), and
                        // consume a read character
                        best_prev = cn;
                        gssw_cigar_push_front(nc->cigar, 'N', 1);
                        --readEnd;
#ifdef DEBUG_TRACEBACK
                        fprintf(stderr, "N-match across nodes to %p\n", cn);
#endif
                        break;
                    } else if(diagonalSourceScore + match == score && refChar == readChar) {
                        // Then try a match
                        
                        // Go to that node (which consumes a ref character), and
                        // consume a read character
                        best_prev = cn;
                        gssw_cigar_push_front(nc->cigar, 'M', 1);
                        score -= match;
                        --readEnd;
#ifdef DEBUG_TRACEBACK
                        fprintf(stderr, "Match across nodes to %p\n", cn);
#endif
                        break;
                    } else if(diagonalSourceScore - mismatch == score && refChar != readChar) {
                        // Then try a mismatch
                        
                        // Go to that node (which consumes a ref character), and
                        // consume a read character
                        best_prev = cn;
                        gssw_cigar_push_front(nc->cigar, 'X', 1);
                        score += mismatch;
                        --readEnd;
#ifdef DEBUG_TRACEBACK
                        fprintf(stderr, "Mismatch across nodes to %p\n", cn);
#endif
                        break;
                    }
                    
                    // A match starting the alignment should have been taken
                    // care of in the within-node function.
                    
                    // If none of those work, try the next option for the previous
                    // node.
                } else {
                
                    // If we are in a gap, it would have been a last resort in the node's traceback.
                    
                    if(gapOpenSourceScore - gap_open == score) {
                        // This node is consistent with an open. Take it.
                        best_prev = cn;
                        gssw_cigar_push_front(nc->cigar, 'D', 1);
                        score += gap_open;
                        // Unset the gap flag
                        gapInRead = 0;
#ifdef DEBUG_TRACEBACK
                        fprintf(stderr, "Gap open across nodes to %p\n", cn);
#endif
                        break;
                    } else if(gapExtendSourceScore - gap_extension == score) {
                        // This node is consistent with an extend. Take it.
                        best_prev = cn;
                        gssw_cigar_push_front(nc->cigar, 'D', 1);
                        score += gap_extension;
#ifdef DEBUG_TRACEBACK
                        fprintf(stderr, "Gap extend across nodes to %p\n", cn);
#endif
                        break;
                    }
                }
            }
            // Once we go through all the possible previous nodes, we sure hope we found something consistent.
            if(best_prev == NULL) {
                fprintf(stderr, "error:[gssw] Could not find a valid previous node");
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
        } else {
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

#ifdef DEBUG_TRACEBACK
    fprintf(stderr, "at end of traceback loop\n");
#endif
    
    gssw_reverse_graph_cigar(gc);

    gm->position = (refEnd +1 < 0 ? 0 : refEnd +1); // drop last step by -1 on ref position

    return gm;

}

void gssw_cigar_push_back(gssw_cigar* c, char type, uint32_t length) {
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

gssw_node* gssw_node_create(void* data,
                            const uint32_t id,
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
            fprintf(stderr, "cannot align because node predecessors cannot provide seed\n");
            fprintf(stderr, "failing is node %u\n", prev[k]->id);
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
gssw_graph_fill (gssw_graph* graph,
                 const char* read_seq,
                 const int8_t* nt_table,
                 const int8_t* score_matrix,
                 const uint8_t weight_gapO,
                 const uint8_t weight_gapE,
                 const int32_t maskLen,
                 const int8_t score_size) {

    int32_t read_length = strlen(read_seq);
    int8_t* read_num = gssw_create_num(read_seq, read_length, nt_table);
	gssw_profile* prof = gssw_init(read_num, read_length, score_matrix, 5, score_size);
    gssw_seed* seed = NULL;
    uint16_t max_score = 0;

    // for each node, from start to finish in the partial order (which should be sorted topologically)
    // generate a seed from input nodes or use existing (e.g. for subgraph traversal here)
    uint32_t i;
    gssw_node** npp = &graph->nodes[0];
    for (i = 0; i < graph->size; ++i, ++npp) {
        gssw_node* n = *npp;
        // get seed from parents (max of multiple inputs)
        if (prof->profile_byte) {
            seed = gssw_create_seed_byte(prof->readLen, n->prev, n->count_prev);
        } else {
            seed = gssw_create_seed_word(prof->readLen, n->prev, n->count_prev);
        }
        gssw_node* filled_node = gssw_node_fill(n, prof, weight_gapO, weight_gapE, maskLen, seed);
        gssw_seed_destroy(seed); seed = NULL; // cleanup seed
        // test if we have exceeded the score dynamic range
        if (prof->profile_byte && !filled_node) {
            free(prof->profile_byte);
            prof->profile_byte = NULL;
            free(read_num);
            gssw_profile_destroy(prof);
            return gssw_graph_fill(graph, read_seq, nt_table, score_matrix, weight_gapO, weight_gapE, maskLen, 1);
        } else {
            if (!graph->max_node || n->alignment->score1 > max_score) {
                graph->max_node = n;
                max_score = n->alignment->score1;
            }
        }
    }

    free(read_num);
    gssw_profile_destroy(prof);

    return graph;

}

// TODO graph traceback


gssw_node*
gssw_node_fill (gssw_node* node,
                const gssw_profile* prof,
                const uint8_t weight_gapO,
                const uint8_t weight_gapE,
                const int32_t maskLen,
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
		bests = gssw_sw_sse2_byte((const int8_t*)node->num, 0, node->len, readLen, weight_gapO, weight_gapE, prof->profile_byte, -1, prof->bias, maskLen, alignment, seed);
		if (bests[0].score == 255) {
			free(bests);
            gssw_align_clear_matrix_and_seed(alignment);
            return 0; // re-run from external context
		}
	} else if (prof->profile_word) {
        bests = gssw_sw_sse2_word((const int8_t*)node->num, 0, node->len, readLen, weight_gapO, weight_gapE, prof->profile_word, -1, maskLen, alignment, seed);
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

int32_t gssw_graph_add_node(gssw_graph* graph, gssw_node* node) {
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

int8_t* gssw_create_score_matrix(int32_t match, int32_t mismatch) {
	// initialize scoring matrix for genome sequences
	//  A  C  G  T	N (or other ambiguous code)
	//  2 -2 -2 -2 	0	A
	// -2  2 -2 -2 	0	C
	// -2 -2  2 -2 	0	G
	// -2 -2 -2  2 	0	T
	//	0  0  0  0  0	N (or other ambiguous code)
    int32_t l, k, m;
	int8_t* mat = (int8_t*)calloc(25, sizeof(int8_t));
	for (l = k = 0; l < 4; ++l) {
		for (m = 0; m < 4; ++m) mat[k++] = l == m ? match : - mismatch;	/* weight_match : -weight_mismatch */
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
